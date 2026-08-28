"""
visualization.py  –  ZrMicro plotting and run-management module.

All figures are written as individual files into a uniquely-identified run
directory under ``output/``.  A provenance Markdown file is written alongside
the figures so that every run is fully reproducible.

Typical usage
-------------
>>> from dislocluster_code.post.visualization import ZrMicroVisualizer
>>> viz = ZrMicroVisualizer(simulation, results, output_dir='output',
...                         use_dpa=True)
>>> viz.plot_all()          # writes all figures + provenance.md to run_dir
"""

import platform
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Global plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 12,
    'lines.linewidth': 2,
})

_FIG_SIZE = (8, 5)   # width × height inches for every individual figure


# ── Run-directory helpers ──────────────────────────────────────────────────────

def _get_git_hash(path=None):
    """Return the short git commit hash for the repo at *path*, or 'unknown'.

    DisloCluster was assembled from two independent checkouts, so its root may
    not itself be a git repository. When the lookup at *path* fails, fall back
    to ``py_utils.paths.git_hash``, which walks the repo root and both nested
    checkouts, so run directories keep a meaningful provenance tag.
    """
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(path or Path.cwd()),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        h = out.decode().strip()
        if h:
            return h
    except Exception:
        pass
    try:
        from dislocluster_code import paths
        return paths.git_hash()
    except Exception:
        return 'unknown'


def create_run_directory(base_output_dir, repo_root=None):
    """
    Create ``<base_output_dir>/<timestamp>_<git-hash>/`` and return it.

    Parameters
    ----------
    base_output_dir : str or Path
    repo_root       : str or Path, optional  – directory used for git lookup.

    Returns
    -------
    run_dir : Path
    run_id  : str  – e.g. ``'20260311_142537_a1b2c3d'``
    """
    base_output_dir = Path(base_output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    git_hash  = _get_git_hash(repo_root or base_output_dir)
    run_id    = f'{timestamp}_{git_hash}'
    run_dir   = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def save_provenance(run_dir, input_data, results, sim_config=None):
    """
    Write ``provenance.md`` inside *run_dir*.

    The file contains one markdown table per parameter group:
    Material & Environment, Physical Properties, Model Parameters,
    Derived Parameters, Simulation Config (if provided), Solver Statistics,
    Run Parameters, and Software Environment.

    Returns the path to the written file.
    """
    run_dir = Path(run_dir)

    # Collect ordered groups: (heading, {param: value})
    groups = []

    groups.append(('Material & Environment', dict(input_data.material_params)))
    groups.append(('Physical Properties',    dict(input_data.physical_props)))
    groups.append(('Model Parameters',       dict(input_data.model_params)))
    groups.append(('Derived Parameters',     {k: v for k, v in input_data.derived.items()
                                              if not hasattr(v, '__len__')}))

    if sim_config:
        groups.append(('Simulation Config', dict(sim_config)))

    if 'concentrations' in results:
        groups.append(('Final Concentrations', {
            k: v[-1] for k, v in results['concentrations'].items()
        }))

    if 'loop_sizes' in results:
        groups.append(('Final Loop Sizes (m)', {
            k: v[-1] for k, v in results['loop_sizes'].items()
        }))

    if 'mechanical_properties' in results:
        mech = results['mechanical_properties']
        groups.append(('Final Mechanical Properties', {
            'strain_a':   mech['strain_a'][-1],
            'strain_c':   mech['strain_c'][-1],
            'creep_rate': mech['creep_rate'][-1],
            'hardening':  mech['hardening'][-1],
        }))

    if 'metadata' in results:
        meta = results['metadata']
        solver = {k: v for k, v in meta.get('solver_stats', {}).items() if v is not None}
        if solver:
            groups.append(('Solver Statistics', solver))
        run_params = meta.get('parameters', {})
        if run_params:
            groups.append(('Run Parameters', dict(run_params)))

    try:
        import numpy as _np
        import scipy as _scipy
        sw = {
            'python_version': platform.python_version(),
            'numpy_version':  _np.__version__,
            'scipy_version':  _scipy.__version__,
            'platform':       platform.platform(),
            'run_timestamp':  datetime.now().isoformat(),
        }
    except ImportError:
        sw = {'run_timestamp': datetime.now().isoformat()}
    groups.append(('Software Environment', sw))

    def _fmt_val(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.4e}'
        if isinstance(v, (int, np.integer)) and (abs(v) > 9999 or v != 0 and abs(v) < 1):
            return f'{v:.4e}'
        return str(v)

    def _md_table(mapping):
        lines = ['| Parameter | Value |', '|-----------|-------|']
        for k, v in mapping.items():
            lines.append(f'| {k} | {_fmt_val(v)} |')
        return '\n'.join(lines)

    run_id = run_dir.name
    md_lines = [f'# Provenance — run `{run_id}`', '']
    for heading, mapping in groups:
        if not mapping:
            continue
        md_lines.append(f'## {heading}')
        md_lines.append('')
        md_lines.append(_md_table(mapping))
        md_lines.append('')

    prov_file = run_dir / 'provenance.md'
    prov_file.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'  ✓ provenance.md')
    return prov_file


# ── Visualizer class ───────────────────────────────────────────────────────────

class ZrMicroVisualizer:
    """
    Generates and saves individual plot files for a completed ZrMicro simulation.

    Each call to a ``plot_*`` method writes one ``.png`` file to the run
    directory and returns its path.  ``plot_all()`` calls every method.

    Parameters
    ----------
    simulation  : ZrMicroSimulation
    results     : dict returned by ``simulation.run_simulation()``
    output_dir  : str or Path  – parent folder (e.g. ``ZrMicro/output``).
                  A unique ``<timestamp>_<git-hash>`` subdirectory is created
                  automatically inside it.
    use_dpa     : bool  – x-axis as DPA (True) or seconds (False)
    dpa_range   : tuple ``(dpa_min, dpa_max)`` or None
    sim_config  : dict of solver settings to include in provenance (optional)
    """

    #: What the four immobile slots are called on a figure. THE LEGACY NAMES
    #: ARE NOT UNIVERSAL. They describe the fitted `loop_model = 0`
    #: formulation, which pairs each family with an ALIGNED partner. The
    #: self-consistent model carries no such split: `field.to_legacy_layout`
    #: puts the prismatic interstitial variants in the first slot and the
    #: prismatic VACANCY variants in the second, and the faulted and perfect
    #: BASAL states in the third and fourth. Labelling those "aligned" is
    #: wrong in a way a reader cannot detect -- the curve is a different
    #: polarity, or a different stacking state, not a different orientation.
    SLOT_LABELS = {
        0: {"CiL":  "Interstitial loops",
            "CaiL": "Aligned interstitial loops",
            "CvL":  "Vacancy loops",
            "CavL": "Aligned vacancy loops"},
        1: {"CiL":  r"$\langle a\rangle$ interstitial (a1-a3)",
            "CaiL": r"$\langle a\rangle$ vacancy (a1v-a3v)",
            "CvL":  r"$\langle c\rangle$ faulted (c$_f$)",
            "CavL": r"$\langle c\rangle$ perfect (c$_p$)"},
    }

    def __init__(self, simulation, results, output_dir,
                 use_dpa=True, dpa_range=None, sim_config=None, run_dir=None,
                 n_atoms=None, open_system=False, loop_model=0):
        self.sim     = simulation
        self.results = results
        self.inp     = simulation.input_data
        self.use_dpa = use_dpa
        # Which formulation produced the state these figures draw. It decides
        # the legend, and nothing else: the four slots are plotted the same way
        # either way, they simply do not mean the same thing.
        self.loop_model = int(loop_model)
        self.labels = self.SLOT_LABELS[1 if self.loop_model else 0]
        # Number of lattice atoms in the simulation volume. Concentrations here
        # are atom fractions, so this is the factor that turns any of them into
        # a COUNT of defects -- "3.4e9 vacancies absorbed at the grain
        # boundary" rather than "a vacancy atom fraction of 5.1e-5 absorbed",
        # which is a number nobody can hold. None keeps the fraction axis.
        self.n_atoms = None if n_atoms is None else float(n_atoms)
        # True when the domain has an absorbing surface, i.e. when the balance
        # residual is grain-boundary absorption rather than solver error. See
        # `zerod.post_process._calculate_conservation`.
        self.open_system = bool(open_system)

        # Locate the repository root for the provenance git tag. Prefer the
        # DisloCluster root resolved by paths.py; fall back to walking up from
        # the input workbook (Simulations/input/x.xlsx -> input -> Simulations).
        try:
            from dislocluster_code import paths
            repo_root = paths.REPO_ROOT
        except Exception:
            repo_root = Path(simulation.input_file).resolve().parent.parent.parent

        # Create the unique run directory, unless the caller supplied one.
        # A caller that is placing this figure suite INSIDE a larger run --
        # the volume-average suite of a 3-D march, for instance -- already owns
        # a timestamped directory and wants the figures in a named subdirectory
        # of it, not in a second timestamped tree nested inside the first.
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = self.run_dir.name
        else:
            self.run_dir, self.run_id = create_run_directory(
                Path(output_dir), repo_root=repo_root
            )
        print(f'✓ Run directory: {self.run_dir}')

        # Write provenance immediately
        save_provenance(self.run_dir, self.inp, results, sim_config)

        # Prepare x-axis and filtered copies of the data arrays
        self._prepare_axes(dpa_range)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _prepare_axes(self, dpa_range):
        """Compute x-axis data and optionally filter to dpa_range (on copies)."""
        time = self.results['time'].copy()

        # Deep copy so we never mutate the original results dict
        self.conc  = {k: v.copy() for k, v in self.results['concentrations'].items()}
        self.loops = {k: v.copy() for k, v in self.results['loop_sizes'].items()}
        self.mech  = {k: v.copy() for k, v in self.results['mechanical_properties'].items()}
        # Conservation balance (nested {species: {channel: array}}); optional
        self.cons  = {
            sp: {k: v.copy() for k, v in ch.items()}
            for sp, ch in self.results.get('conservation', {}).items()
        }
        self.time  = time

        G = self.inp.material_params['G']
        if self.use_dpa:
            self.x_data  = time * G
            self.x_label = 'Dose (dpa)'
        else:
            self.x_data  = time
            self.x_label = 'Time (s)'

        if dpa_range is not None:
            x_min, x_max = (dpa_range if self.use_dpa
                            else (dpa_range[0] / G, dpa_range[1] / G))
            mask = (self.x_data >= x_min) & (self.x_data <= x_max)
            if np.sum(mask) == 0:
                warnings.warn(
                    f'No data in range [{x_min:.2e}, {x_max:.2e}]; '
                    f'using full range [{self.x_data[0]:.2e}, {self.x_data[-1]:.2e}]'
                )
            else:
                self.x_data = self.x_data[mask]
                self.time   = self.time[mask]
                for d in (self.conc, self.loops, self.mech):
                    for k in d:
                        d[k] = d[k][mask]
                for species in self.cons.values():
                    for k in species:
                        species[k] = species[k][mask]

    def _savefig(self, fig, name):
        """Save *fig* as ``<run_dir>/<name>.png``, close it, print confirmation."""
        # Uniform lower dose limit on every dose-axis figure (10^-6 dpa).
        for ax in fig.axes:
            try:
                if 'Dose' in ax.get_xlabel():
                    ax.set_xlim(left=1e-6)
            except Exception:
                pass
        path = self.run_dir / f'{name}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  ✓ {name}.png')
        return path

    # ── Individual plot methods ────────────────────────────────────────────────

    def plot_point_defects(self):
        """
        Point-defect and defect-in-loop concentration evolution.
        Units: cm⁻³  (concentration / atomic-volume-in-cm³).
        """
        conc  = self.conc
        Omega_cm3 = self.inp.physical_props['Omega'] * 1e6   # m³ → cm³

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.loglog(self.x_data, conc['Cv']     / Omega_cm3, 'b-',  label='Vacancies (Cv)')
        ax.loglog(self.x_data, conc['Ci']     / Omega_cm3, 'r-',  label='Interstitials (Ci)')
        ax.loglog(self.x_data, conc['C2i']    / Omega_cm3, 'g--', label='Di-interstitials (C2i)')
        ax.loglog(self.x_data, conc['C3i']    / Omega_cm3, 'm:',  label='Tri-interstitials (C3i)')
        ax.loglog(self.x_data, conc['CiL_i']  / Omega_cm3, 'b--', label='Interstitials in i-loops')
        ax.loglog(self.x_data, conc['CaiL_i'] / Omega_cm3, 'b:',  label='Interstitials in ai-loops')
        ax.loglog(self.x_data, conc['CvL_v']  / Omega_cm3, 'r--', label='Vacancies in v-loops')
        ax.loglog(self.x_data, conc['CavL_v'] / Omega_cm3, 'r:',  label='Vacancies in av-loops')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Concentration (cm⁻³)')
        ax.set_ylim(bottom=1e9)
        ax.set_title('Point Defect Evolution')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'point_defects')

    def _exp_targets_near_T(self, sheet, tol=50.0):
        """Experimental targets (T, dpa, N, d, Source) within ``tol`` K of the
        run temperature, for superimposing on the single-temperature figures."""
        import pandas as pd
        try:
            df = pd.read_excel(self.sim.input_file, sheet_name=sheet).iloc[:, :5]
        except Exception:
            return None
        df.columns = ['T', 'dpa', 'N', 'd', 'Source']
        for c in ('T', 'dpa', 'N', 'd'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['T'] = df['T'].ffill(); df['Source'] = df['Source'].ffill()
        df = df.dropna(subset=['T', 'dpa'])
        T0 = float(self.inp.material_params['T'])
        return df[(df['T'] - T0).abs() <= tol]

    def plot_loop_density(self):
        """Dislocation loop number density [m⁻³] vs dose/time, with the measured
        a-/c-loop densities (within ±50 K of the run T) superimposed (±20%)."""
        conc  = self.conc
        Omega = self.inp.physical_props['Omega']   # m³/atom

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.loglog(self.x_data, conc['CiL']  / Omega, 'b-',  label=self.labels['CiL'])
        ax.loglog(self.x_data, conc['CaiL'] / Omega, 'b--', label=self.labels['CaiL'])
        ax.loglog(self.x_data, conc['CvL']  / Omega, 'r-',  label=self.labels['CvL'])
        ax.loglog(self.x_data, conc['CavL'] / Omega, 'r--', label=self.labels['CavL'])

        _mk = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']; _si = 0
        for sheet, base, lt in (('Targets_A', 'tab:blue', 'a'),
                                ('Targets_C', 'tab:red',  'c')):
            ed = self._exp_targets_near_T(sheet)
            if ed is None:
                continue
            for (T, src), g in ed.dropna(subset=['N']).groupby(['T', 'Source']):
                ax.errorbar(g['dpa'], g['N'], yerr=0.2 * g['N'],
                            fmt=_mk[_si % len(_mk)], color=base, ms=7, capsize=3,
                            lw=1, mec='k', mew=0.4,
                            label=f'{lt}-exp {T:.0f} K, {src}')
                _si += 1

        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Number Density (m⁻³)')
        ax.set_ylim(bottom=1e19)
        ax.set_title('Dislocation Loop Density (model lines; exp points ±20%)')
        ax.legend(fontsize=8, ncol=2, loc='upper center',
                  bbox_to_anchor=(0.5, -0.13))
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'loop_density')

    def plot_loop_sizes(self):
        """Dislocation loop radius [nm] vs dose/time."""
        loops = self.loops

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.semilogx(self.x_data, loops['r_iL']  * 1e9, 'b-',  label=self.labels['CiL'])
        ax.semilogx(self.x_data, loops['r_aiL'] * 1e9, 'b--', label=self.labels['CaiL'])
        ax.semilogx(self.x_data, loops['r_vL']  * 1e9, 'r-',  label=self.labels['CvL'])
        ax.semilogx(self.x_data, loops['r_avL'] * 1e9, 'r--', label=self.labels['CavL'])
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Loop Radius (nm)')
        ax.set_title('Dislocation Loop Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'loop_sizes')

    def plot_loop_sizes_vs_experiment(self):
        """Mean loop DIAMETER [nm] vs dose/time with the measured a-/c-loop sizes
        (within +/-50 K of the run T) superimposed (+/-20%).

        Companion to :meth:`plot_loop_density`. NOTE the unit convention: the
        ``Mean Size (nm)`` column of Targets_A / Targets_C is a **diameter**,
        whereas the model state carries loop **radii**, so the model curves are
        multiplied by 2 here. Plotting radii against these points (as the plain
        ``plot_loop_sizes`` axes would) understates the model by a factor of 2.

        The per-family curve is the number-weighted mean over the pure and
        aligned variants, matching the observation operator used in the fit:
            d_A = 2 (C_iL r_iL + C_aiL r_aiL) / (C_iL + C_aiL)
        """
        conc, loops = self.conc, self.loops

        Na = conc['CiL'] + conc['CaiL']
        Nc = conc['CvL'] + conc['CavL']
        d_a = 2e9 * (conc['CiL'] * loops['r_iL'] +
                     conc['CaiL'] * loops['r_aiL']) / np.maximum(Na, 1e-30)
        d_c = 2e9 * (conc['CvL'] * loops['r_vL'] +
                     conc['CavL'] * loops['r_avL']) / np.maximum(Nc, 1e-30)

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.semilogx(self.x_data, d_a, 'b-', lw=2,
                    label=r'$\langle a\rangle$ loops (model, N-weighted)')
        ax.semilogx(self.x_data, d_c, 'r-', lw=2,
                    label=r'$\langle c\rangle$ loops (model, N-weighted)')

        _mk = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']; _si = 0
        n_pts = 0
        for sheet, base, lt in (('Targets_A', 'tab:blue', 'a'),
                                ('Targets_C', 'tab:red',  'c')):
            ed = self._exp_targets_near_T(sheet)
            if ed is None:
                continue
            for (T, src), g in ed.dropna(subset=['d']).groupby(['T', 'Source']):
                ax.errorbar(g['dpa'], g['d'], yerr=0.2 * g['d'],
                            fmt=_mk[_si % len(_mk)], color=base, ms=7, capsize=3,
                            lw=1, mec='k', mew=0.4,
                            label=f'{lt}-exp {T:.0f} K, {src}')
                _si += 1
                n_pts += len(g)
        if n_pts == 0:
            T0 = float(self.inp.material_params['T'])
            ax.text(0.02, 0.96, f'no size data within 50 K of {T0:.0f} K',
                    transform=ax.transAxes, va='top', fontsize=8, color='0.35')

        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Mean Loop Diameter (nm)')
        ax.set_title('Loop Size (model lines; exp points +/-20%)')
        ax.legend(fontsize=8, ncol=2, loc='upper center',
                  bbox_to_anchor=(0.5, -0.13))
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'loop_sizes_vs_experiment')

    def plot_irradiation_growth(self):
        """Irradiation growth strains ε_a and ε_c [%] vs dose/time."""
        mech = self.mech

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(self.x_data, mech['strain_a'] * 100, 'b-', label='ε_a  (a-direction)')
        ax.plot(self.x_data, mech['strain_c'] * 100, 'r-', label='ε_c  (c-direction)')
        ax.set_xscale('log')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Strain (%)')
        ax.set_title('Irradiation Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'irradiation_growth')

    def plot_creep(self):
        """Irradiation creep strain magnitude [%] vs dose/time."""
        mech = self.mech

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(self.x_data, np.abs(mech['creep_rate']) * 100, 'g-')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('|Creep Strain| (%)')
        ax.set_title('Irradiation Creep')
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'irradiation_creep')

    def plot_hardening(self):
        """Radiation hardening – yield strength increase [MPa] vs dose/time."""
        mech = self.mech

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(self.x_data, mech['hardening'] / 1e6, 'k-')
        ax.set_xscale('log')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Yield Strength Increase (MPa)')
        ax.set_title('Radiation Hardening')
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'radiation_hardening')

    def plot_flux_evolution(self):
        """
        Total vacancy and interstitial fluxes over dose/time.

        Flux definitions (s⁻¹):
          J_v = ω_v · Cv
          J_i = ω_i · (Ci + 2·C2i + 3·C3i)
        """
        rr     = self.sim.reaction_rates
        conc   = self.conc
        flux_v = rr.omega_v * conc['Cv']
        flux_i = rr.omega_i * (conc['Ci'] + 2*conc['C2i'] + 3*conc['C3i'])

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.loglog(self.x_data, flux_v, 'r-', label='Vacancy flux  J_v')
        ax.loglog(self.x_data, flux_i, 'b-', label='Interstitial flux  J_i')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Flux (s⁻¹)')
        ax.set_ylim(bottom=1e-4)
        ax.set_title('Point Defect Flux Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'flux_evolution')

    def plot_net_flux_balance(self):
        """
        Net flux driving force for each loop type (s⁻¹) vs dose/time.

        Positive → loop grows; negative → loop shrinks.

        Interstitial loops: Z_iL·J_i − J_v
        Vacancy loops:      J_v − Z_vL·J_i
        """
        rr   = self.sim.reaction_rates
        inp  = self.inp
        conc = self.conc

        flux_v = rr.omega_v * conc['Cv']
        flux_i = rr.omega_i * (conc['Ci'] + 2*conc['C2i'] + 3*conc['C3i'])

        Z_iL      = inp.model_params.get('Z_iL', 1.05)
        Z_vL_na   = inp.model_params.get('Z_vL', 1.2)   # non-aligned vacancy loops
        Z_vL_a    = inp.model_params.get('Z_vL', 0.7)   # aligned vacancy loops (different default)

        net_iL  = Z_iL   * flux_i - flux_v          # i-loop driving force
        net_aiL = Z_iL   * flux_i - flux_v          # ai-loop (same bias)
        net_vL  = flux_v - Z_vL_na * flux_i         # v-loop driving force
        net_avL = flux_v - Z_vL_a  * flux_i         # av-loop driving force

        if self.loop_model:
            # THE SECOND SLOT IS NOT AN ALIGNED PARTNER HERE, IT IS THE OTHER
            # POLARITY. `to_legacy_layout` puts the prismatic VACANCY variants
            # in the `CaiL` slot, so its driving force is the vacancy form and
            # not the interstitial one -- keeping the legacy expression would
            # draw that curve with the sign inverted, which is a worse error
            # than a wrong label because it is a plausible-looking line.
            #
            # The fourth slot holds the PERFECT basal state c_p, still a
            # vacancy family, so its expression is unchanged and only its name
            # was ever wrong.
            net_aiL = flux_v - Z_vL_na * flux_i

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.semilogx(self.x_data, net_iL,  'b-',  label=self.labels['CiL'])
        ax.semilogx(self.x_data, net_aiL, 'b--', label=self.labels['CaiL'])
        ax.semilogx(self.x_data, net_vL,  'r-',  label=self.labels['CvL'])
        ax.semilogx(self.x_data, net_avL, 'r--', label=self.labels['CavL'])
        ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Net Flux Driving Force (s⁻¹)')
        ax.set_title('Net Flux Balance by Loop Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'net_flux_balance')

    # ── Additional analysis plots ──────────────────────────────────────────────

    def _total_defects(self):
        """
        Return (total_vacancies, total_interstitials) arrays, both as the
        total number of point defects of each type (atom fraction) summed
        over all storage channels: free point defects, small clusters, and
        defects stored in loops. Loop terms use the *defects stored* in the
        loops (CiL_i/CaiL_i, CvL_v/CavL_v), NOT the loop number densities
        (CiL/CaiL/CvL/CavL), so the two totals are on a consistent
        atoms-conserved basis.
        """
        conc = self.conc
        total_v = conc['Cv'] + conc['CvL_v'] + conc['CavL_v']
        total_i = (conc['Ci'] + 2*conc['C2i'] + 3*conc['C3i']
                   + conc['CiL_i'] + conc['CaiL_i'])
        return total_v, total_i

    def plot_total_defects_analysis(self):
        """Total vacancy and interstitial concentrations vs dose/time."""
        total_v, total_i = self._total_defects()

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.loglog(self.x_data, total_v, 'b-', label='Total Vacancies')
        ax.loglog(self.x_data, total_i, 'r-', label='Total Interstitials')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Total Concentration')
        ax.set_ylim(bottom=1e-5)
        ax.set_title('Total Defect Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'total_defect_evolution')

    def plot_defect_imbalance(self):
        """
        Net defect imbalance (vacancies − interstitials) vs dose/time.

        Each term is the *total* concentration of that defect type summed
        over all channels — free point defects, di/tri clusters, and defects
        stored in loops — expressed as an atom fraction (defects per lattice
        site). It is NOT a fraction-in-clusters; it is the net excess of
        retained vacancy-type over interstitial-type defects. A positive value
        means more vacancies than interstitials are retained in the
        microstructure.
        """
        total_v, total_i = self._total_defects()
        imbalance = total_v - total_i

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(self.x_data, imbalance, 'g-')
        ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Vacancies − Interstitials (atom fraction)')
        ax.set_title('Net Defect Imbalance')
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'defect_imbalance')

    def _counts(self, arr):
        """An atom-fraction array as a count of defects, when the volume is known."""
        return np.asarray(arr) * self.n_atoms if self.n_atoms else np.asarray(arr)

    @property
    def _count_label(self):
        return ('Defects in the simulation volume' if self.n_atoms
                else 'Cumulative atoms (atom fraction)')

    def plot_conservation(self):
        """
        Where the produced point defects ended up, as counts over the whole
        simulation volume: recombined, absorbed at network sinks, absorbed at
        the grain boundary, or still stored in the microstructure.

        On an open system this is the useful form of the balance. The old
        version plotted (stored − [production − recombination − sink]) /
        production and called it a conservation error, which on a domain with
        an absorbing surface reads ~-70% and looks like a catastrophic bug; it
        is the boundary channel, and naming it is the whole fix.

        Returns None if the run carries no conservation accumulators.
        """
        if not self.cons:
            return None
        if not self.open_system:
            return self._plot_conservation_error()

        fig, axes = plt.subplots(1, 2, figsize=(2 * _FIG_SIZE[0], _FIG_SIZE[1]))
        for ax, (sp, title) in zip(axes, (('interstitial', 'Interstitial'),
                                          ('vacancy', 'Vacancy'))):
            c = self.cons[sp]
            prod = self._counts(c['production'])
            chans = [
                ('Recombined', self._counts(c['recombination']), 'g'),
                ('Absorbed at network sinks', self._counts(c['sink']), 'C1'),
                ('Absorbed at grain boundary',
                 self._counts(c['grain_boundary']), 'r'),
            ] + ([
                ('Loops absorbed at grain boundary',
                 self._counts(self._gb_loops(c)), 'C4'),
            ] if self._gb_loops(c) is not None else []) + [
                ('Stored in microstructure',
                 self._counts(c.get('immobile_change', c['stored_change'])),
                 'b'),
            ]
            ax.loglog(self.x_data, np.abs(prod), 'k-', lw=2, label='Produced')
            for lab, v, col in chans:
                ax.loglog(self.x_data, np.abs(v), color=col, ls='--', label=lab)
            # Drawn so the split can be read against the total, NOT as a check:
            # the boundary channel is defined as production minus the others,
            # so this lies on the production curve by construction. It would
            # only depart from it if a channel were dropped from the sum.
            tot = sum(np.abs(v) for _, v, _ in chans)
            ax.loglog(self.x_data, tot, 'w:', lw=1.4,
                      label='Sum (= produced, by construction)')
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self._count_label)
            ax.set_title(f'{title} fate')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._savefig(fig, 'conservation_error')

    def _plot_conservation_error(self):
        """The closed-system form: relative residual, which should be ~0."""
        i_err = self.cons['interstitial']['rel_error']
        v_err = self.cons['vacancy']['rel_error']

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.semilogx(self.x_data, i_err, 'r-', label='Interstitial balance')
        ax.semilogx(self.x_data, v_err, 'b-', label='Vacancy balance')
        ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Relative conservation error')
        ax.set_title('Point-Defect Conservation Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'conservation_error')

    @staticmethod
    def _gb_loops(c):
        """The MEASURED loop-absorption channel, or None.

        `grain_boundary` beside it is closure by difference and carries every
        other channel's error; this one is an integral the solver performed. A
        run without gb_absorption has the key absent or identically zero, and
        every figure then draws exactly what it drew before.
        """
        g = c.get('gb_loops')
        if g is None:
            return None
        import numpy as _np
        return g if _np.any(_np.abs(g) > 0.0) else None

    def plot_conservation_channels(self):
        """
        Cumulative production and loss channels (production, recombination, sink
        absorption) plus net stored atoms, for interstitials and vacancies.
        Lets the production be compared directly against the sum of loss channels.

        Returns None if the run carries no conservation accumulators.
        """
        if not self.cons:
            return None

        # Separate figure per species (interstitial / vacancy).
        last = None
        for sp, title, tag in (('interstitial', 'Interstitial', 'i'),
                               ('vacancy', 'Vacancy', 'v')):
            c = self.cons[sp]
            f = self._counts
            fig, ax = plt.subplots(figsize=_FIG_SIZE)
            ax.loglog(self.x_data, np.abs(f(c['production'])),    'k-',  label='Production ∫P')
            ax.loglog(self.x_data, np.abs(f(c['recombination'])), 'g--', label='Recombination')
            ax.loglog(self.x_data, np.abs(f(c['sink'])),          'm-.', label='Sink absorption')
            ax.loglog(self.x_data, np.abs(f(c['stored_change'])), 'b-',  label='|Δ stored|')
            if self.open_system and 'grain_boundary' in c:
                ax.loglog(self.x_data, np.abs(f(c['grain_boundary'])), 'r:',
                          label='Grain-boundary absorption (mobile, by closure)')
            _gb = self._gb_loops(c)
            if _gb is not None:
                ax.loglog(self.x_data, np.abs(f(_gb)), 'C4-',
                          label='Loops absorbed at grain boundary (measured)')
            if self.open_system and 'grain_boundary' in c:
                pass
            else:
                ax.loglog(self.x_data, np.abs(f(c['residual'])), 'r:', label='|Residual|')
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self._count_label)
            # A count floor of 1 defect is meaningful; an atom-fraction floor is
            # the old 1e-12.
            ax.set_ylim(bottom=1.0 if self.n_atoms else 1e-12)
            ax.set_title(f'{title} Balance Channels')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            last = self._savefig(fig, f'conservation_channels_{tag}')
        return last

    def plot_interstitial_fractions(self):
        """
        Fraction of produced interstitials residing in / lost to each channel
        vs dose/time, normalised by cumulative production. Channels:
        recombination, network sinks, free interstitials, di/tri clusters, and
        loop content. Their sum (dashed) equals 1 — a visual conservation check.

        Returns None if the run carries no conservation accumulators.
        """
        if not self.cons:
            return None

        c    = self.cons['interstitial']
        conc = self.conc
        prod = c['production']
        denom = np.where(prod > 0, prod, np.nan)

        free = conc['Ci']
        clus = 2.0 * conc['C2i'] + 3.0 * conc['C3i']
        loop = conc['CiL_i'] + conc['CaiL_i']

        f_recomb = c['recombination'] / denom
        f_sink   = c['sink']          / denom
        f_free   = (free - free[0])   / denom
        f_clus   = (clus - clus[0])   / denom
        f_loop   = (loop - loop[0])   / denom
        f_sum    = f_recomb + f_sink + f_free + f_clus + f_loop
        # Without the boundary channel this sum is not 1 on an open system --
        # it was reaching ~0.3 while the legend promised 1.
        f_gb = None
        if self.open_system and 'grain_boundary' in c:
            f_gb = c['grain_boundary'] / denom
            f_sum = f_sum + f_gb
        # The loop channel is a separate destination for the atoms, so it is a
        # separate term in the sum. Leaving it out would put the "Sum (= 1)"
        # line back below 1 by exactly the loops absorbed, which is the defect
        # that adding the boundary channel fixed in the first place.
        f_gbl = None
        _gb_meas = self._gb_loops(c)
        if _gb_meas is not None:
            f_gbl = _gb_meas / denom
            f_sum = f_sum + f_gbl

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(self.x_data, f_recomb, 'r-',  label='Recombination')
        ax.plot(self.x_data, f_sink,   'C1-', label='Network sinks')
        ax.plot(self.x_data, f_free,   'b-',  label='Free interstitials')
        ax.plot(self.x_data, f_clus,   'g-',  label='Di/tri clusters')
        ax.plot(self.x_data, f_loop,   'm-',  label='Loop content')
        if f_gb is not None:
            ax.plot(self.x_data, f_gb, 'k-', lw=1.6, label='Grain boundary')
        if f_gbl is not None:
            ax.plot(self.x_data, f_gbl, 'C4-', lw=1.6,
                    label='Loops absorbed at GB')
        ax.plot(self.x_data, f_sum,    'k--', label='Sum (=1)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 2.0)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Fraction of produced interstitials')
        ax.set_title('Interstitial Fate by Channel')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'interstitial_fractions')

    def plot_vacancy_fractions(self):
        """
        Fraction of produced vacancies residing in / lost to each channel vs
        dose/time, normalised by cumulative production. Channels: recombination,
        network sinks, free vacancies, and vacancy-loop content. Their sum
        (dashed) equals 1 — a visual conservation check.

        Returns None if the run carries no conservation accumulators.
        """
        if not self.cons:
            return None

        c    = self.cons['vacancy']
        conc = self.conc
        prod = c['production']
        denom = np.where(prod > 0, prod, np.nan)

        free = conc['Cv']
        loop = conc['CvL_v'] + conc['CavL_v']

        f_recomb = c['recombination'] / denom
        f_sink   = c['sink']          / denom
        f_free   = (free - free[0])   / denom
        f_loop   = (loop - loop[0])   / denom
        f_sum    = f_recomb + f_sink + f_free + f_loop
        f_gb = None
        if self.open_system and 'grain_boundary' in c:
            f_gb = c['grain_boundary'] / denom
            f_sum = f_sum + f_gb
        # The loop channel is a separate destination for the atoms, so it is a
        # separate term in the sum. Leaving it out would put the "Sum (= 1)"
        # line back below 1 by exactly the loops absorbed, which is the defect
        # that adding the boundary channel fixed in the first place.
        f_gbl = None
        _gb_meas = self._gb_loops(c)
        if _gb_meas is not None:
            f_gbl = _gb_meas / denom
            f_sum = f_sum + f_gbl

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(self.x_data, f_recomb, 'r-',  label='Recombination')
        ax.plot(self.x_data, f_sink,   'C1-', label='Network sinks')
        ax.plot(self.x_data, f_free,   'b-',  label='Free vacancies')
        ax.plot(self.x_data, f_loop,   'm-',  label='Loop content')
        if f_gb is not None:
            ax.plot(self.x_data, f_gb, 'k-', lw=1.6, label='Grain boundary')
        if f_gbl is not None:
            ax.plot(self.x_data, f_gbl, 'C4-', lw=1.6,
                    label='Loops absorbed at GB')
        ax.plot(self.x_data, f_sum,    'k--', label='Sum (=1)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 2.0)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Fraction of produced vacancies')
        ax.set_title('Vacancy Fate by Channel')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'vacancy_fractions')

    def plot_loop_density_analysis(self):
        """The first slot of each pair, i- and v-loop number density [m⁻³].

        Under ``loop_model = 0`` those are the NON-ALIGNED interstitial and
        vacancy loops, which is what this figure was written for. Under
        ``loop_model = 1`` there is no alignment split at all and the same two
        slots hold the prismatic interstitial variants and the faulted basal
        state -- so the figure is "the first of each pair" in general and
        `self.labels` says which.
        """
        conc  = self.conc
        Omega = self.inp.physical_props['Omega']

        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.loglog(self.x_data, conc['CiL'] / Omega, 'b-', label=self.labels['CiL'])
        ax.loglog(self.x_data, conc['CvL'] / Omega, 'r-', label=self.labels['CvL'])
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('Loop Density (m⁻³)')
        ax.set_ylim(bottom=1e19)
        ax.set_title('Loop Density Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'loop_density_analysis')

    def plot_strain_rates(self):
        """Absolute strain rates |dε/dt| [s⁻¹] for a- and c-directions."""
        mech = self.mech
        time = self.time

        rate_a = np.gradient(mech['strain_a'], time)
        rate_c = np.gradient(mech['strain_c'], time)

        # drop first point (can be noisy at very small t)
        x = self.x_data[1:]
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.loglog(x, np.abs(rate_a[1:]), 'b-', label='|ε̇_a|')
        ax.loglog(x, np.abs(rate_c[1:]), 'r-', label='|ε̇_c|')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel('|Strain Rate| (s⁻¹)')
        ax.set_title('Strain Rate Evolution')
        ax.set_ylim(bottom=1e-10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._savefig(fig, 'strain_rates')

    def save_analysis_data(self):
        """
        Save additional analysis quantities to ``additional_analysis.csv``
        inside the run directory.

        Columns: x_axis (dose or time), total_vacancies, total_interstitials,
        defect_imbalance, swelling_estimate.
        """
        conc  = self.conc
        loops = self.loops

        total_v, total_i = self._total_defects()
        imbalance = total_i - total_v
        swelling  = (conc['CvL'] + conc['CavL']) * np.pi * loops['r_vL']**2 * 1e-2

        x_col = self.x_label.replace(' ', '_').replace('(', '').replace(')', '')
        df = pd.DataFrame({
            x_col:                 self.x_data,
            'total_vacancies':     total_v,
            'total_interstitials': total_i,
            'defect_imbalance':    imbalance,
            'swelling_estimate':   swelling,
        })

        path = self.run_dir / 'additional_analysis.csv'
        df.to_csv(path, index=False)
        print('  ✓ additional_analysis.csv')
        return path

    # ── Convenience ────────────────────────────────────────────────────────────

    def plot_all(self):
        """Generate and save all figures and analysis data. Returns list of output paths."""
        print('Generating figures...')
        paths = [
            self.plot_point_defects(),
            self.plot_loop_density(),
            self.plot_loop_sizes(),
            self.plot_loop_sizes_vs_experiment(),
            self.plot_irradiation_growth(),
            self.plot_hardening(),
            self.plot_flux_evolution(),
            self.plot_net_flux_balance(),
            self.plot_total_defects_analysis(),
            self.plot_defect_imbalance(),
            self.plot_loop_density_analysis(),
            self.plot_strain_rates(),
            self.plot_conservation(),
            self.plot_conservation_channels(),
            self.plot_interstitial_fractions(),
            self.plot_vacancy_fractions(),
        ]
        paths = [p for p in paths if p is not None]
        print('Saving analysis data...')
        paths.append(self.save_analysis_data())
        print(f'✓ All outputs saved to {self.run_dir}')

        # Append post-processing summary to provenance
        prov_file = self.run_dir / 'provenance.md'
        file_list = '\n'.join(f'- {f.name}' for f in sorted(self.run_dir.iterdir()))
        with prov_file.open('a', encoding='utf-8') as fh:
            fh.write('\n## Output Files\n\n')
            fh.write(file_list)
            fh.write('\n')

        return paths


# ── Multi-simulation comparison ────────────────────────────────────────────────

def compare_simulations(results_list, labels=None, use_dpa=False):
    """
    Overlay key outputs from multiple simulation results on a 2×2 figure.

    Parameters
    ----------
    results_list : list[dict]  – results dicts from ZrMicroSimulation.run_simulation()
    labels       : list[str], optional  – legend labels (default: "Simulation N")
    use_dpa      : bool  – if True, divide time by dose_rate to get dpa on x-axis

    Returns
    -------
    fig : matplotlib Figure
    """
    if labels is None:
        labels = [f'Simulation {i + 1}' for i in range(len(results_list))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Simulation Comparison', fontsize=14, fontweight='bold')

    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']

    for i, (results, label) in enumerate(zip(results_list, labels)):
        color = colors[i % len(colors)]
        time  = results['time']
        conc  = results['concentrations']
        mech  = results['mechanical_properties']

        if use_dpa and 'metadata' in results:
            G    = results['metadata']['parameters'].get('dose_rate', 1.0)
            xval = time * G
            xlabel = 'Dose (dpa)'
        else:
            xval   = time
            xlabel = 'Time (s)'

        axes[0, 0].loglog(xval, conc['Cv'],            color=color, label=label, linewidth=2)
        axes[0, 1].loglog(xval, conc['CiL'],           color=color, label=label, linewidth=2)
        axes[1, 0].semilogx(xval, mech['strain_a']*100, color=color, label=label, linewidth=2)
        axes[1, 1].semilogx(xval, mech['hardening']/1e6, color=color, label=label, linewidth=2)

    titles   = ['Vacancy Evolution', 'Loop Evolution', 'Irradiation Growth', 'Radiation Hardening']
    ylabels  = ['Vacancy Concentration', 'Interstitial Loop Concentration',
                'Strain (%)', 'Hardening (MPa)']

    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Convenience entry-point ────────────────────────────────────────────────────

def plot_results(simulation, results, save_plots=True, output_dir=None,
                 dpa_range=None, use_dpa=True, sim_config=None):
    """
    Construct a ZrMicroVisualizer and optionally write all figures to disk.

    Parameters
    ----------
    simulation : ZrMicroSimulation
    results    : dict  – from run_simulation()
    save_plots : bool  – write PNG files (False for quick preview)
    output_dir : str or Path  – base output folder (default: output/)
    dpa_range  : tuple (dpa_min, dpa_max) or None
    use_dpa    : bool  – DPA on x-axis (True) or seconds (False)
    sim_config : dict  – optional solver settings for provenance

    Returns
    -------
    viz : ZrMicroVisualizer
    """
    if output_dir is None:
        output_dir = Path.cwd() / 'output'

    viz = ZrMicroVisualizer(
        simulation, results,
        output_dir=Path(output_dir),
        use_dpa=use_dpa,
        dpa_range=dpa_range,
        sim_config=sim_config,
    )
    if save_plots:
        viz.plot_all()
    return viz
