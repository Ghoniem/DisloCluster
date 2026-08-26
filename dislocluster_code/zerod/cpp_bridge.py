"""
cpp_bridge.py – Python bridge for the ZrMicro C++ SUNDIALS solver.

Responsibilities
----------------
1. Collect all pre-computed parameters from InputData / ReactionRates and
   convert them to --key=value CLI arguments for solver.exe.
2. Invoke solver.exe as a subprocess (mirrors Creep/py_utils/read_data.py).
3. Parse the solver's stdout and reconstruct the standard results dict via
   post_process.calculate_derived_quantities — so the existing visualization
   and post-processing pipeline works unchanged.

The C++ solver output format is one row per time point:
  t  Cv  Ci  C2i  C3i  CiL  CaiL  CvL  CavL  CiL_i  CaiL_i  CvL_v  CavL_v
  (13 space-separated values, scientific notation)
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from dislocluster_code import paths as _paths

log = logging.getLogger(__name__)

# Matches the state-vector layout in RateEquations / parameters.h:
# 12 physical species + 6 conservation accumulators + 1 evolving rho_N
_N_CONC = 19


# ── Parameter collection ─────────────────────────────────────────────────────

#: Which crystallographic family each of the eight immobile slots is. Slots 4-6
#: are the prismatic VACANCY variants step 1 added and slot 7 the second basal
#: state step 4 will fill; both are already addressable here so that step 4 has
#: nothing to change on this side.
_EMIS_SLOT_FAMILY = ('c_f', 'a_i', 'a_i', 'a_i', 'a_v', 'a_v', 'a_v', 'c_p',
                     'c_f')   # slot 8 is the pyramid; it emits with c_f's fault
_EMIS_SLOT_KEY = ('c', 'a1', 'a2', 'a3', 'a1v', 'a2v', 'a3v', 'cp', 'c0')


def _emission_params(material_file, solver_config):
    """Per-family geometry for the step-3 emission channel.

    Taken from `studies.loop_annealing`, which builds every length from the
    material file's own constants rather than quoting any of them, and which the
    manuscript's Sec. 4.1-4.2 tables are produced by. The C++ reproduces its
    `dEdm` / `binding_energy` / `c_eq_vacancy` exactly; this is where the two are
    tied to one source.
    """
    from dislocluster_code.studies import loop_annealing as la

    mat = la.load_material(material_file)
    fams = la.families(mat)
    T = float(solver_config.get('temperature_K', 0.0) or 0.0)
    if T <= 0.0:
        raise ValueError("emission_model=1 needs solver_config['temperature_K']")

    out = {
        'emission_model': 1,
        'kT_eV': la.KB_EV * T,
        'Ef_v': mat.Ef_v,
    }
    sigma = solver_config.get('emission_sigma_Pa') or {}
    for slot, (fname, key) in enumerate(zip(_EMIS_SLOT_FAMILY, _EMIS_SLOT_KEY)):
        f = fams[fname]
        out[f'emis_gamma_{key}'] = f.gamma
        out[f'emis_Kbar_{key}'] = f.Kbar
        out[f'emis_bmag_{key}'] = f.bmag
        out[f'emis_bdotn_{key}'] = f.bdotn
        out[f'emis_lam_{key}'] = f.lam
        out[f'emis_sigma_{key}'] = float(sigma.get(key, 0.0))
    return out


def collect_solver_args(sim, solver_config):
    """
    Build a list of --key=value CLI strings for solver.exe.

    All quantities that the C++ RHS needs are pre-computed here on the Python
    side using the already-initialised InputData and ReactionRates objects, so
    the C++ solver carries no Excel-reading or derivation logic.

    Parameters
    ----------
    sim           : ZrMicroSimulation  – fully initialised
    solver_config : dict               – t_begin, t_end, n_points, rtol, atol,
                                         log_time

    Returns
    -------
    list[str]  – ['--omega_i=1.23e+08', ...]
    """
    inp = sim.input_data
    rr  = sim.reaction_rates
    re  = sim.rate_equations

    # Make sure stress-dependent emissions are initialised (they are computed
    # lazily inside update_state, not during __init__).
    y0_init = re.get_initial_conditions()
    rr.update_state(y0_init, 0.0)

    params = {}

    # ── Pre-computed jump frequencies ──────────────────────────────────────
    params['omega_i']  = rr.omega_i
    params['omega_v']  = rr.omega_v
    params['omega_2i'] = rr.omega_2i

    # ── Thermal emission probabilities ─────────────────────────────────────
    params['e_v']           = rr.e_v
    params['e_2i']          = rr.e_2i
    params['e_3i']          = rr.e_3i
    params['e_v_iL']        = rr.e_v_iL
    params['e_v_vL']        = rr.e_v_vL
    params['e_sigma_v_iL']  = rr.e_sigma_v_iL
    params['e_sigma_v_vL']  = rr.e_sigma_v_vL
    params['e_sigma_v_avL'] = rr.e_sigma_v_avL

    # ── Sink / material parameters ─────────────────────────────────────────
    params['rho_N'] = inp.material_params['rho']
    params['a']     = inp.physical_props['a']
    params['z_c']   = inp.physical_props['z_c']
    params['Omega'] = inp.physical_props['Omega']
    params['Z_N']   = inp.model_params.get('Z_N',   1.05)
    # Orientation/species-resolved DAD loop capture efficiencies (derived from
    # delta_DAD in InputData). These replace the old single Z_iL/Z_vL bias on
    # the loop growth/absorption terms; Z_N still sets the network bias.
    params['Z_i_a'] = inp.derived['Z_i_a']
    params['Z_v_a'] = inp.derived['Z_v_a']
    params['Z_i_c'] = inp.derived['Z_i_c']
    params['Z_v_c'] = inp.derived['Z_v_c']
    params['recom'] = inp.physical_props.get('recom', 1.0)

    # ── Pre-computed length scales ─────────────────────────────────────────
    params['l']   = inp.derived['l']
    params['l_a'] = inp.derived['l_a']
    params['l_c'] = inp.derived['l_c']

    # ── Loop growth ────────────────────────────────────────────────────────
    params['Q'] = inp.model_params['Q']

    # ── Effective generation rates (matching Python G_2i()/G_3i() methods) ─
    params['G_v']   = inp.derived['G_v']
    params['G_i']   = inp.derived['G_i']
    params['G_2i']  = inp.derived['G_2i'] / 2.0   # "divided by 2 for pairs"
    params['G_3i']  = inp.derived['G_3i'] / 3.0   # "divided by 3 for pairs"
    params['G_iL']  = inp.derived['G_iL']     # cascade a-loop atom rate (non-aligned)
    params['G_aiL'] = inp.derived['G_aiL']    # cascade a-loop atom rate (aligned)
    params['G_vL']  = inp.derived['G_vL']
    params['G_avL'] = inp.derived['G_avL']

    # ── Loop fractions ─────────────────────────────────────────────────────
    params['f_a']  = inp.derived['f_a']
    params['f_na'] = inp.derived['f_na']

    # ── Annealing lifetimes ────────────────────────────────────────────────
    # Both lifetimes are derived (T-dependent):
    #   tau_vL  = tau_vL0  * exp(E_a_vL  / kT)
    #   tau_avL = tau_avL0 * exp(E_a_avL / kT)   (defaults to tau_vL0/E_a_vL)
    # The legacy constant model_params['tau_avL'] override has been removed so
    # the fitted tau_vL0/E_a_vL fully control the c-loop steady-state density.
    params['tau_vL']  = inp.derived['tau_vL']
    params['tau_avL'] = inp.derived['tau_avL']
    params['n_vL_nuc'] = inp.model_params.get('n_vL_nuc', 20.0)
    params['n_iL_nuc'] = inp.model_params.get('n_iL_nuc', 20.0)

    # ── Geometric loop coalescence coefficients ────────────────────────────
    params['c_LL']     = inp.model_params.get('c_LL',     1.0)
    params['kappa_LL'] = inp.model_params.get('kappa_LL', 1.0)
    params['c_LN']     = inp.model_params.get('c_LN',     1.0)
    params['kappa_LN'] = inp.model_params.get('kappa_LN', 1.0)
    # Orientation-resolved coalescence (a-loops = iL/aiL, c-loops = vL/avL);
    # absent in the sheet -> fall back to the shared c_LL/c_LN.
    params['c_LL_a']   = inp.model_params.get('c_LL_a', params['c_LL'])
    params['c_LN_a']   = inp.model_params.get('c_LN_a', params['c_LN'])
    params['c_LL_c']   = inp.model_params.get('c_LL_c', params['c_LL'])
    params['c_LN_c']   = inp.model_params.get('c_LN_c', params['c_LN'])

    # ── Evolving network density rho_N (defaults match reaction_rates.py) ──
    params['c_rhoN']     = inp.model_params.get('c_rhoN',     0.1)
    params['k_rhoN_rec'] = inp.model_params.get('k_rhoN_rec', 1e-7)

    # ── Minimum stable a-loop radius (halts shrinkage; prevents the a-loop ──
    #    mean diameter collapsing to zero at high T/dose). Match a_shrink_gate.
    params['r_min_a'] = inp.model_params.get('r_min_a', 1.0e-9)
    params['w_rmin']  = inp.model_params.get('w_rmin',  0.3)

    # ── Concentration floor ────────────────────────────────────────────────
    params['C_floor'] = inp.model_params.get('C_floor', 1e-20)

    # ── Initial conditions (y0_0 … y0_18: 12 species + 6 accumulators + rho_N) ─
    for k, val in enumerate(y0_init):
        params[f'y0_{k}'] = val

    # ── Solver settings ────────────────────────────────────────────────────
    params['t_begin']  = solver_config.get('t_begin',  1e-1)
    params['t_end']    = solver_config.get('t_end',    1e9)
    params['n_points'] = int(solver_config.get('n_points', 1000))
    params['log_time'] = 1.0 if solver_config.get('log_time', True) else 0.0
    params['rtol']     = solver_config.get('rtol',  1e-6)
    params['atol']     = solver_config.get('atol',  1e-20)

    # ── Operator-split QSSA: freeze the mobile species (two-time-scale coupling) ─
    # When set, the C++ RHS holds Cv,Ci,C2i,C3i fixed at y0[0:4] (their
    # derivatives are zeroed) so the solver advances only the immobile state.
    # Used by py_utils/modelib_coupling.py; default off -> standalone runs
    # are bit-identical.
    params['freeze_mobile'] = 1.0 if solver_config.get('freeze_mobile', False) else 0.0

    # ── Reduced implicit block + analytic AD Jacobian ──────────────────────
    # 'reduced'      : carry the six conservation accumulators as CVODES
    #                  quadrature variables instead of ODE state, shrinking the
    #                  Newton system from 19 to 13 (or to 9 with freeze_mobile).
    #                  It also removes the accumulators from the error test,
    #                  which is what stops their zero initial value from forcing
    #                  a roundoff-limited first step on every coupling substep.
    # 'analytic_jac' : exact dense Jacobian by forward-mode AD instead of
    #                  SUNDIALS' difference quotients (dense linear solver only).
    #
    # Both default OFF here so a standalone 0-D run remains bit-identical to the
    # pre-existing solver. modelib_coupling.build_immobile_cases turns them on
    # for the operator-split march, where they are measurably better.
    # acc_mode: 0 = accumulators in the state and in the error test (legacy,
    #           bit-identical), 1 = CVODES quadrature, 2 = in the state with
    #           their atol neutralized. See modelib_coupling for the
    #           measurements behind the coupling default of 2.
    params['acc_mode']     = int(solver_config.get(
        'acc_mode', 1 if solver_config.get('reduced', False) else 0))
    params['analytic_jac'] = 1.0 if solver_config.get('analytic_jac', False) else 0.0
    if solver_config.get('stats', False):
        params['stats'] = 1.0

    # ── Integration method options ─────────────────────────────────────────
    # Optional 'solver_method' sub-dict in solver_config:
    #   backend   : 'cvode' (default) or 'arkode'
    #   lmm       : 'bdf' (default, CVODE only) or 'adams'
    #   linsol    : 'dense' (default), 'band', or 'gmres'
    #   mu        : upper bandwidth for band solver (default N_EQ-1=11)
    #   ml        : lower bandwidth for band solver (default N_EQ-1=11)
    #   max_order : max solver order; 0 = solver default
    #   ark_table : ARKODE DIRK table name or integer ID (default 'ARK548L2SA_DIRK_8_4_5')
    #               Key stiff-system choices (order):
    #                 'SDIRK_2_1_2'             (2) — lightweight SDIRK
    #                 'SDIRK_5_3_4'             (4) — L-stable SDIRK
    #                 'KVAERNO_7_4_5'           (5) — A-stable, 7-stage
    #                 'ARK548L2SA_DIRK_8_4_5'   (5) — L-stable, default
    #                 'ESDIRK547L2SA_7_4_5'     (5) — stiffly accurate ESDIRK
    _ark_table_map = {
        'SDIRK_2_1_2':              100,
        'BILLINGTON_3_3_2':         101,
        'TRBDF2_3_3_2':             102,
        'KVAERNO_4_2_3':            103,
        'ARK324L2SA_DIRK_4_2_3':    104,
        'CASH_5_2_4':               105,
        'CASH_5_3_4':               106,
        'SDIRK_5_3_4':              107,
        'KVAERNO_5_3_4':            108,
        'ARK436L2SA_DIRK_6_3_4':    109,
        'KVAERNO_7_4_5':            110,
        'ARK548L2SA_DIRK_8_4_5':    111,  # default
        'ARK437L2SA_DIRK_7_3_4':    112,
        'ARK548L2SAb_DIRK_8_4_5':   113,
        'ESDIRK324L2SA_4_2_3':      114,
        'ESDIRK325L2SA_5_2_3':      115,
        'ESDIRK32I5L2SA_5_2_3':     116,
        'ESDIRK436L2SA_6_3_4':      117,
        'ESDIRK43I6L2SA_6_3_4':     118,
        'QESDIRK436L2SA_6_3_4':     119,
        'ESDIRK437L2SA_7_3_4':      120,
        'ESDIRK547L2SA_7_4_5':      121,
        'ESDIRK547L2SA2_7_4_5':     122,
        'ARK2_DIRK_3_1_2':          123,
        'BACKWARD_EULER_1_1':       124,
        'IMPLICIT_MIDPOINT_1_2':    125,
        'IMPLICIT_TRAPEZOIDAL_2_2': 126,
    }
    method_opts = solver_config.get('solver_method', {})
    _backend_map = {'cvode': 0, 'arkode': 1}
    _lmm_map     = {'bdf': 2, 'adams': 1}
    _linsol_map  = {'dense': 0, 'band': 1, 'gmres': 2}

    params['backend']   = _backend_map.get(
                              str(method_opts.get('backend', 'cvode')).lower(), 0)
    params['lmm']       = _lmm_map.get(
                              str(method_opts.get('lmm', 'bdf')).lower(), 2)
    params['linsol']    = _linsol_map.get(
                              str(method_opts.get('linsol', 'dense')).lower(), 0)
    params['mu']        = int(method_opts.get('mu',        _N_CONC - 1))
    params['ml']        = int(method_opts.get('ml',        _N_CONC - 1))
    params['max_order'] = int(method_opts.get('max_order', 0))
    # ark_table: accept name string or direct integer
    _ark_raw = method_opts.get('ark_table', 'ARK548L2SA_DIRK_8_4_5')
    if isinstance(_ark_raw, int):
        params['ark_table'] = _ark_raw
    else:
        params['ark_table'] = _ark_table_map.get(str(_ark_raw).upper(), 111)

    # ── Self-consistent loop model (optional) ───────────────────────────────
    # solver_config['loop_model'] = 1 switches the slow step to the same
    # family structure and the same capture physics as the fast solve: one
    # basal <c> and three prismatic <a> variants -- no aligned/non-aligned --
    # with Woo efficiencies built from the SAME p_m that sets the diffusion
    # tensor. Absent or 0 keeps the fitted legacy formulation, which the C++
    # reproduces bit-for-bit.
    #
    # The parameters are read from the MoDELib material file rather than the
    # workbook, deliberately: that file is the single source both sides already
    # share, and `staging/anisotropy.py` writes dadAnisotropy there together
    # with the migration energies. Taking them from anywhere else would
    # reintroduce exactly the drift this change exists to remove.
    if int(solver_config.get('loop_model', 0)):
        from dislocluster_code import paths
        from dislocluster_code.coupling.field import read_material_vector
        mat = solver_config.get('material_file') or paths.MODELIB_MATERIAL
        names = ('v', 'i', '2i', '3i')
        fams = ('c', 'a1', 'a2', 'a3')
        params['loop_model'] = 1
        pm = read_material_vector(mat, 'dadAnisotropy', 4)
        z0 = read_material_vector(mat, 'dadZ0', 4)
        for j, nm in enumerate(names):
            params[f'dad_p_{nm}'] = float(pm[j])
            params[f'dad_Z0_{nm}'] = float(z0[j])
        # loopSinkScale is per FAMILY, and step 1 took the material file from
        # four families to eight. Read whatever it carries rather than demanding
        # a fixed length: asking for 4 against an 8-column file raised, the
        # blanket `except` below turned that into a silent fall back to 1.0, and
        # the calibrated 0.291528 / 0.792317 quietly became 1.0 for a whole run.
        # The step-0 baseline caught it by diffing the resolved CLI, which is the
        # single reason it compares parameters before it compares trajectories.
        try:
            ls = read_material_vector(mat, 'loopSinkScale')
        except Exception as exc:
            # Not silent. A missing sink-strength calibration is a different
            # model, not a default.
            raise RuntimeError(
                f"loopSinkScale unreadable from {mat}: {exc}") from exc
        all_fams = ('c', 'a1', 'a2', 'a3', 'a1v', 'a2v', 'a3v', 'cp')
        for j, fm in enumerate(all_fams[:len(ls)]):
            params[f'loop_sink_scale_{fm}'] = float(ls[j])
        w = solver_config.get('variant_weights', (1 / 3, 1 / 3, 1 / 3))
        for j, fm in enumerate(fams[1:]):
            params[f'variant_frac_{fm}'] = float(w[j])

        # ── Step 1: the VACANCY variants weight the other way ───────────────
        # w_k for an interstitial prism loop is exp(+sigma_k Omega/kT)/Z: it
        # inserts material, so a tensile resolved stress on its habit normal
        # favours it. A vacancy loop REMOVES material on that plane, so the same
        # stress disfavors it by exactly the reciprocal factor. Supplying one
        # set of weights for both characters would have made the two populations
        # respond to load identically, which is the single thing the variant
        # resolution exists to distinguish.
        #
        # Reciprocate-then-normalize IS the Boltzmann weight with the sign
        # flipped: with w_k = e^{x_k}/Z, 1/w_k = Z e^{-x_k}, and the Z cancels on
        # renormalization to leave e^{-x_k}/sum_j e^{-x_j} exactly.
        #
        # At zero deviatoric stress both are (1/3, 1/3, 1/3) and the two
        # coincide, so nothing an existing run does is changed by this.
        wv = solver_config.get('variant_weights_vac')
        if wv is None:
            a = np.asarray(w, dtype=float)
            inv = np.where(a > 0.0, 1.0 / np.maximum(a, 1e-300), 0.0)
            s = float(inv.sum())
            wv = (inv / s) if s > 0 else np.full(3, 1.0 / 3.0)
        for j, fm in enumerate(fams[1:]):
            params[f'variant_frac_v_{fm}'] = float(wv[j])
        if int(solver_config.get('n_fam', 4)) != 4:
            params['n_fam'] = int(solver_config['n_fam'])
        # Step 2's character factor. Emitted only when it is not the degenerate
        # value, so an unchanged command line stays byte-for-byte what it was
        # and the step-0/1 artifacts keep comparing.
        if float(solver_config.get('chi', 1.0)) != 1.0:
            params['chi'] = float(solver_config['chi'])

        # ── Step 3: peripheral emission ─────────────────────────────────────
        # The geometry of every family comes from studies/loop_annealing.py,
        # which is the reference implementation of Sec. 4.1-4.2 and reads its
        # constants from the SAME material file. Deriving them here a second
        # time would be a second place for them to drift.
        if int(solver_config.get('emission_model', 0)) != 0:
            params.update(_emission_params(mat, solver_config))

        # ── Step 4: the basal chain c_0 -> c_f -> c_p ───────────────────────
        # Emitted only when the chain is switched on, so a step-3 command line
        # is byte-for-byte what it was.
        if int(solver_config.get('basal_chain', 0)) != 0:
            params['basal_chain'] = 1
            for key in ('eps_sfp', 'n_sfp_nuc', 'tau_sfp',
                        'nu_col', 'nu_uf', 'alpha_sfp'):
                if key in solver_config:
                    params[key] = float(solver_config[key])

    return [f'--{k}={v}' for k, v in params.items()]


# ── Output parsing ────────────────────────────────────────────────────────────

def _parse_stdout(text):
    """
    Convert solver stdout (space-separated, n_points × 13) to a numpy array.

    Returns
    -------
    numpy.ndarray  shape (n_points, 13)
    """
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if parts:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass   # skip malformed lines (shouldn't happen)
    if not rows:
        return np.empty((0, 13))
    return np.array(rows)


# ── Executable resolution ─────────────────────────────────────────────────────

def _find_solver_exe(base_dir):
    """Locate the compiled solver binary, or return None if it is missing.

    MSVC / multi-config generators place the binary in a config sub-folder
    (Debug/ or Release/); single-config generators (Makefile, Ninja) put it
    directly in build/. Debug is preferred when present (matches the historical
    build instructions), then Release, then the flat build/ layout.
    """
    exe_name  = 'solver.exe' if sys.platform == 'win32' else 'solver'
    build_dir = Path(base_dir) / 'build'
    for candidate in (build_dir / 'Debug'   / exe_name,
                      build_dir / 'Release' / exe_name,
                      build_dir / exe_name):
        if candidate.exists():
            return candidate
    return None


def _solver_popen_kwargs():
    """Launch the child in its own process group/session so a kernel interrupt
    (SIGINT) is not delivered straight into the solver mid-output."""
    if sys.platform == 'win32':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


# ── Main entry point ──────────────────────────────────────────────────────────

def run_cpp_solver(sim, solver_config, base_dir=None):
    """
    Run the compiled ZrMicro C++ solver and return the standard results dict.

    Parameters
    ----------
    sim           : ZrMicroSimulation  – fully initialised simulation object
    solver_config : dict               – same keys as SIMULATION_CONFIG in the
                                         notebook
    base_dir      : Path or None       – ZrMicro/ root; auto-detected if None

    Returns
    -------
    dict  – same format as post_process.process_solution(), or None on failure
    """
    from dislocluster_code.zerod.post_process import calculate_derived_quantities

    if base_dir is None:
        # ZrMicro/ -- where build/ and input/ live. This was
        # `Path(__file__).parent.parent`, true only while cpp_bridge sat in
        # ZrMicro/py_utils/; from dislocluster_code/zerod/ it resolved to the
        # package root and silently found no solver.
        base_dir = _paths.ZRMICRO_DIR

    exe_path = _find_solver_exe(base_dir)
    if exe_path is None:
        print(f"❌ solver executable not found under {Path(base_dir) / 'build'}")
        print("   Build it with:")
        print(f"     cd {Path(base_dir) / 'cpp_utils'}")
        print( "     cmake -S . -B ../build -DCMAKE_BUILD_TYPE=Release")
        print( "     cmake --build ../build --config Release")
        return None

    cli_args = collect_solver_args(sim, solver_config)

    print(f"Running C++ solver ({exe_path.name}) ...")

    # On KeyboardInterrupt we terminate the child ourselves and return None,
    # so the notebook's normal "no results" path handles the interruption.
    proc = subprocess.Popen(
        [str(exe_path)] + cli_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **_solver_popen_kwargs(),
    )

    try:
        stdout, stderr = proc.communicate()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted — stopping C++ solver...")
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.kill()
            proc.communicate()
        print("✓ C++ solver stopped cleanly (no results).")
        return None

    if proc.returncode != 0:
        print(f"❌ C++ solver failed (exit code {proc.returncode}):")
        print(stderr)
        return None

    if stderr.strip():
        # CVODE may emit non-fatal warnings to stderr
        print("C++ solver warnings:\n" + stderr)

    sol_array = _parse_stdout(stdout)
    if sol_array.shape[0] == 0:
        print("❌ C++ solver produced no output")
        return None

    n_pts = sol_array.shape[0]
    print(f"✓ C++ solver completed — {n_pts} time points")

    t = sol_array[:, 0]           # shape (n_pts,)
    y = sol_array[:, 1:].T        # shape (12, n_pts)  — matches scipy sol.y layout

    # Reuse the existing post-processing pipeline
    results = calculate_derived_quantities(t, y, sim.input_data, sim.rate_equations)

    method_opts   = solver_config.get('solver_method', {})
    backend_name  = str(method_opts.get('backend', 'cvode')).lower()
    linsol_label  = str(method_opts.get('linsol', 'dense')).upper()
    if backend_name == 'arkode':
        ark_tbl   = str(method_opts.get('ark_table', 'ARK548L2SA_DIRK_8_4_5'))
        backend_str = f'C++ SUNDIALS ARKODE ARKStep DIRK ({ark_tbl}) / {linsol_label}'
    else:
        lmm_label   = str(method_opts.get('lmm', 'bdf')).upper()
        backend_str = f'C++ SUNDIALS CVODE {lmm_label} / {linsol_label}'

    results['metadata'] = {
        'solver_stats': {
            'success':  True,
            'message':  backend_str,
            'nfev':     None,
            'njev':     None,
            'nlu':      None,
            'status':   0,
            't_events': None,
        },
        'parameters': {
            'temperature':         sim.input_data.material_params['T'],
            'dose_rate':           sim.input_data.material_params['G'],
            'dislocation_density': sim.input_data.material_params['rho'],
            'end_time':            t[-1],
            'n_time_points':       n_pts,
        },
        'input_file':      str(sim.input_file),
        'solver_backend':  backend_str,
    }

    print("✓ Results processing complete!")
    return results


# ── Batch entry point (parallel multi-case solve) ─────────────────────────────

def _parse_batch_stdout(text, ncases):
    """Split batch solver stdout into per-case (t, y) arrays.

    The solver writes, for each case in input order:
        === CASE <i> status=<s> ===
        <rows ...>
    Rows are kept only for cases that reported status=0; failed cases (and any
    case missing from the output) yield None, mirroring the single-case
    "integration failed → None" contract.

    Returns
    -------
    list  – length `ncases`; each entry is (t[n_pts], y[N_EQ, n_pts]) or None.
    """
    results = [None] * ncases
    cur_idx, cur_status, rows = None, None, []

    def _flush():
        if cur_idx is not None and 0 <= cur_idx < ncases and cur_status == 0 and rows:
            arr = np.array(rows)
            results[cur_idx] = (arr[:, 0], arr[:, 1:].T)

    for line in text.splitlines():
        if line.startswith('=== CASE'):
            _flush()
            rows = []
            parts = line.split()
            try:
                cur_idx    = int(parts[2])
                cur_status = int(parts[3].split('=')[1])
            except (IndexError, ValueError):
                cur_idx, cur_status = None, None
        elif line.strip():
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                pass   # skip malformed lines
    _flush()
    return results


def _batch_lines(cases_cli):
    """Serialize cases as an '@BASE' line plus one delta line per case.

    Returns the list of lines to write. With a single case, or when nothing is
    shared, this degrades gracefully to the original full-line format.
    """
    toks = [[a[2:] if a.startswith('--') else a for a in cli] for cli in cases_cli]
    dicts = []
    for t in toks:
        d = {}
        for tok in t:
            k, _, v = tok.partition('=')
            d[k] = v
        dicts.append(d)

    if len(dicts) < 2:
        return [' '.join(f'{k}={v}' for k, v in d.items()) for d in dicts]

    # A key belongs in the base only if every case has it with the same text.
    first = dicts[0]
    shared = {k: v for k, v in first.items()
              if all(d.get(k) == v for d in dicts[1:])}
    if not shared:
        return [' '.join(f'{k}={v}' for k, v in d.items()) for d in dicts]

    lines = ['@BASE ' + ' '.join(f'{k}={v}' for k, v in shared.items())]
    for d in dicts:
        delta = ' '.join(f'{k}={v}' for k, v in d.items() if k not in shared)
        # A case identical to the base has an EMPTY delta, and the solver skips
        # blank lines -- so a batch of duplicates reduced to "@BASE ..." plus
        # nothing and came back "Batch file contained no cases". Two nodes in a
        # Dirichlet boundary shell are byte-identical often enough for this to
        # bite. Re-state one token so the case line is never empty.
        if not delta:
            k, v = next(iter(d.items()))
            delta = f'{k}={v}'
        lines.append(delta)
    return lines


def run_cpp_solver_batch(cases_cli, base_dir=None):
    """Solve many independent cases in ONE solver subprocess (OpenMP-parallel).

    This amortises the (Windows-dominant) process-spawn + DLL-load cost over all
    cases instead of paying it per case, and runs the integrations concurrently.
    It returns the RAW (t, y) trajectories; the caller post-processes each case
    with its own T-specific InputData (e.g. via calculate_derived_quantities),
    exactly as run_cpp_solver does for a single case.

    Parameters
    ----------
    cases_cli : list[list[str]]
        One entry per case — each is the `collect_solver_args(...)` output
        (a list of '--key=value' strings) for that case.
    base_dir  : Path or None
        ZrMicro/ root; auto-detected if None.

    Returns
    -------
    list  – same length as `cases_cli`; each entry is (t, y) with y shaped
            (N_EQ, n_pts), or None if that case failed or the solver was
            unavailable.

    Raises
    ------
    KeyboardInterrupt
        Propagated rather than converted into an all-None result, so an
        interrupt cannot be mistaken for "every case failed".
    """
    if base_dir is None:
        # ZrMicro/ -- where build/ and input/ live. This was
        # `Path(__file__).parent.parent`, true only while cpp_bridge sat in
        # ZrMicro/py_utils/; from dislocluster_code/zerod/ it resolved to the
        # package root and silently found no solver.
        base_dir = _paths.ZRMICRO_DIR

    n = len(cases_cli)
    if n == 0:
        return []

    exe_path = _find_solver_exe(base_dir)
    if exe_path is None:
        print(f"❌ solver executable not found under {Path(base_dir) / 'build'}")
        return [None] * n

    # ── Write the batch file in the base+delta protocol ──────────────────────
    # Keys whose value is identical across every case go once into an "@BASE"
    # line; each case line then carries only what differs. In the coupling march
    # that is the 19 y0 values and the time window, roughly 22 tokens per case
    # instead of ~150. The solver accepts both formats, so an older case file
    # still parses.
    lines = _batch_lines(cases_cli)

    tf = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False,
                                     encoding='ascii')
    try:
        tf.write('\n'.join(lines))
        tf.close()

        proc = subprocess.Popen(
            [str(exe_path), f'--batch_file={tf.name}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_solver_popen_kwargs(),
        )
        try:
            stdout, stderr = proc.communicate()
        except KeyboardInterrupt:
            # Re-raise, never return [None] * n. A caller that treats a None
            # entry as "this point failed, keep its previous state" -- which is
            # exactly what run_coupled does -- would otherwise take the whole
            # batch as a no-op, advance the dose coordinate by one substep with
            # zero physics applied, and carry on. Ctrl-C has to STOP the march,
            # not silently corrupt it.
            print("\n⚠ Interrupted — stopping C++ batch solver...")
            proc.terminate()
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                proc.kill()
                proc.communicate()
            raise

        if proc.returncode != 0:
            print(f"❌ C++ batch solver failed (exit code {proc.returncode}):")
            print(stderr)
            return [None] * n

        return _parse_batch_stdout(stdout, n)
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass
