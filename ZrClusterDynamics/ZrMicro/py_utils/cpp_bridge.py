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

log = logging.getLogger(__name__)

# Matches the state-vector layout in RateEquations / parameters.h:
# 12 physical species + 6 conservation accumulators + 1 evolving rho_N
_N_CONC = 19


# ── Parameter collection ─────────────────────────────────────────────────────

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
    params['reduced']      = 1.0 if solver_config.get('reduced', False) else 0.0
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
    from py_utils.post_process import calculate_derived_quantities

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

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
            (N_EQ, n_pts), or None if that case failed / the solver was
            unavailable or interrupted.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    n = len(cases_cli)
    if n == 0:
        return []

    exe_path = _find_solver_exe(base_dir)
    if exe_path is None:
        print(f"❌ solver executable not found under {Path(base_dir) / 'build'}")
        return [None] * n

    # Write the batch file: one case per line, '--' prefix stripped for clarity.
    lines = []
    for cli in cases_cli:
        toks = [a[2:] if a.startswith('--') else a for a in cli]
        lines.append(' '.join(toks))

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
            print("\n⚠ Interrupted — stopping C++ batch solver...")
            proc.terminate()
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                proc.kill()
                proc.communicate()
            return [None] * n

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
