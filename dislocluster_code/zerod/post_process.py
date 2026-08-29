"""
post_process.py – Solution quality checks, derived-quantity calculation, and I/O.

Called after solve_ivp returns.  All physics calculations that transform the raw
ODE solution into physically meaningful quantities live here.

This module is intentionally decoupled from the solver so that the ODE integration
can later be migrated to a C++ back-end without touching the post-processing logic.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ── Solution quality ───────────────────────────────────────────────────────────

def check_solution_quality(sol, concentration_names):
    """
    Inspect the ODE solution for numerical issues.

    Parameters
    ----------
    sol                : scipy OdeResult
    concentration_names: list[str]

    Returns
    -------
    warnings_found : list[str]  (empty → no issues detected)
    """
    warnings_found = []

    # Only inspect the physical species; trailing rows (if any) are monotonic
    # conservation accumulators and are not subject to these quality checks.
    n_phys = len(concentration_names)

    min_values = np.min(sol.y[:n_phys], axis=1)
    if np.any(min_values < -1e-12):
        neg_idx   = np.where(min_values < -1e-12)[0]
        neg_names = [concentration_names[i] for i in neg_idx]
        warnings_found.append(f"Significant negative concentrations: {neg_names}")

    if len(sol.t) > 10:
        early = np.sum(sol.y[:4, :10],  axis=0)
        late  = np.sum(sol.y[:4, -10:], axis=0)
        rel_change = np.abs(np.mean(late) - np.mean(early)) / (np.mean(early) + 1e-30)
        if rel_change > 1.0:
            warnings_found.append(f"Large total defect change: {rel_change*100:.1f}%")

    if len(sol.t) > 20:
        for i in range(n_phys):
            recent = sol.y[i, -20:]
            mean   = np.mean(recent)
            if mean > 1e-20:
                rel_std = np.std(recent) / mean
                if rel_std > 0.5:
                    warnings_found.append(
                        f"Oscillations in {concentration_names[i]} (rel_std={rel_std:.2f})"
                    )

    if warnings_found:
        print("⚠️  Solution quality warnings:")
        for w in warnings_found:
            print(f"    - {w}")
    else:
        print("✓ Solution quality: Good")

    return warnings_found


# ── Derived quantities ─────────────────────────────────────────────────────────

def process_solution(sol, input_data, rate_equations, input_file):
    """
    Convert a raw ODE solution into the standard results dict.

    Parameters
    ----------
    sol            : scipy OdeResult
    input_data     : InputData
    rate_equations : RateEquations
    input_file     : Path

    Returns
    -------
    results : dict
    """
    print("Processing results...")

    results = calculate_derived_quantities(sol.t, sol.y, input_data, rate_equations)
    results['metadata'] = {
        'solver_stats': {
            'success':  sol.success,
            'message':  sol.message,
            'nfev':     sol.nfev,
            'njev':     getattr(sol, 'njev', None),
            'nlu':      getattr(sol, 'nlu',  None),
            'status':   sol.status,
            't_events': getattr(sol, 't_events', None),
        },
        'parameters': {
            'temperature':         input_data.material_params['T'],
            'dose_rate':           input_data.material_params['G'],
            'dislocation_density': input_data.material_params['rho'],
            'end_time':            sol.t[-1],
            'n_time_points':       len(sol.t),
        },
        'input_file': str(input_file),
    }

    print("✓ Results processing complete!")
    return results


def calculate_derived_quantities(time, concentrations, input_data, rate_equations):
    """
    Compute loop radii, strains, creep, and hardening (fully vectorized).

    Parameters
    ----------
    time           : numpy.ndarray  [n_time]
    concentrations : numpy.ndarray  [n_species, n_time]
    input_data     : InputData
    rate_equations : RateEquations

    Returns
    -------
    dict with keys: time, concentrations, loop_sizes, mechanical_properties
    """
    print("Calculating derived quantities (vectorized)...")

    conc = np.maximum(concentrations, 1e-20)

    l_a = input_data.derived['l_a']
    l_c = input_data.derived['l_c']

    CiL   = conc[4];  CiL_i  = conc[8]
    CaiL  = conc[5];  CaiL_i = conc[9]
    CvL   = conc[6];  CvL_v  = conc[10]
    CavL  = conc[7];  CavL_v = conc[11]

    r_iL  = np.where(CiL  > 1e-20, l_a * np.sqrt(CiL_i  / np.maximum(CiL,  1e-20)), 5e-9)
    r_aiL = np.where(CaiL > 1e-20, l_a * np.sqrt(CaiL_i / np.maximum(CaiL, 1e-20)), 5e-9)
    r_vL  = np.where(CvL  > 1e-20, l_c * np.sqrt(CvL_v  / np.maximum(CvL,  1e-20)), 5e-9)
    r_avL = np.where(CavL > 1e-20, l_c * np.sqrt(CavL_v / np.maximum(CavL, 1e-20)), 5e-9)

    A_a = input_data.model_params['A_a']
    A_c = input_data.model_params['A_c']

    strain_iL  =  A_a * CiL_i
    strain_aiL =  A_a * CaiL_i
    strain_vL  = -A_c * CvL_v
    strain_avL = -A_c * CavL_v

    strain_a   = strain_iL  + strain_aiL
    strain_c   = strain_vL  + strain_avL
    creep_rate = (strain_aiL - 0.5 * strain_iL) + (strain_avL - 0.5 * strain_vL)

    hardening = _calculate_hardening(conc, r_iL, r_aiL, r_vL, r_avL, input_data)

    print("✓ Derived quantities calculated.")

    results = {
        'time': time,
        'concentrations': {
            name: concentrations[idx, :]
            for idx, name in enumerate(rate_equations.concentration_names)
        },
        'loop_sizes': {
            'r_iL': r_iL, 'r_vL': r_vL, 'r_aiL': r_aiL, 'r_avL': r_avL,
        },
        'mechanical_properties': {
            'strain_a':   strain_a,
            'strain_c':   strain_c,
            'creep_rate': creep_rate,
            'hardening':  hardening,
        },
    }

    conservation = _calculate_conservation(time, concentrations, rate_equations)
    if conservation is not None:
        results['conservation'] = conservation

    # Evolving network dislocation density rho_N [m^-2] (state row idx_rhoN).
    # Falls back to the constant grown-in seed for legacy solutions that do not
    # carry the rho_N row.
    idx_rhoN = getattr(rate_equations, 'idx_rhoN', None)
    if idx_rhoN is not None and concentrations.shape[0] > idx_rhoN:
        results['rho_N'] = concentrations[idx_rhoN, :]
    else:
        results['rho_N'] = np.full_like(time, input_data.material_params['rho'])

    return results


def _calculate_conservation(time, concentrations, rate_equations):
    """
    Point-defect atom-conservation balance for interstitials and vacancies.

    Uses the integrated accumulators (state rows >= n_physical) together with the
    stored atom content reconstructed from the physical species:

        I_stored = Ci + 2*C2i + 3*C3i + CiL_i + CaiL_i
        V_stored = Cv + CvL_v + CavL_v

    For each species the conservation law is

        d(stored)/dt = production - recombination - sink_absorption

    so the integrated residual

        residual = (stored - stored[0]) - (production - recombination - sink)

    is ~0 (solver tolerance) for a strictly atom-conserving model. A non-zero
    residual measures the model's conservation error; ``rel_error`` normalises it
    by cumulative production.

    THAT READING ONLY HOLDS FOR A CLOSED SYSTEM. The balance above has no
    transport term, so on a spatially-resolved run it cannot close and the
    residual is not an error: it is the atoms that left through the Dirichlet
    surface. Two things guarantee it,

      * the domains this repository meshes are Dirichlet over their whole
        surface unless faces are made periodic, so the boundary is a sink;
      * the coupling freezes the mobile species through every slow substep,
        while production keeps accumulating into them -- the mobile pool's own
        balance is closed by the FAST solve, which is where the flux to the
        boundary lives,

    and the size of it settles the argument: the 200 nm march reports a
    residual of -71% of cumulative production. Nothing in a solver run at
    rtol=1e-6 is 71% wrong.

    So the residual is also published as ``grain_boundary`` (= -residual),
    positive when atoms have left. Callers that know the system is open --
    `post.volume_average` -- present it as a physical channel; a standalone 0-D
    run leaves it as the numerical residual it is there.

    Returns None when the solution carries no accumulator rows (e.g. an old run).
    """
    n_phys   = rate_equations.n_physical
    acc_names = rate_equations.accumulator_names
    if concentrations.shape[0] < n_phys + len(acc_names):
        return None

    y   = concentrations
    acc = {name: y[n_phys + k] for k, name in enumerate(acc_names)}

    I_stored = y[1] + 2.0 * y[2] + 3.0 * y[3] + y[8] + y[9]
    V_stored = y[0] + y[10] + y[11]

    # In a spatially-resolved run the mobile species are NOT part of the local
    # balance: they are frozen through every slow substep, and what they do
    # between substeps -- diffuse to the Dirichlet faces and be absorbed there
    # -- is the fast solve's business, not this one's. Splitting `stored` lets
    # the open-system balance attribute the difference instead of burying it.
    I_mobile = y[1] + 2.0 * y[2] + 3.0 * y[3]
    V_mobile = y[0]

    def _balance(stored, prod, recomb, sink, mobile=None, gb_loops=None):
        # Reference every cumulative quantity to the first output point. The C++
        # solver starts integrating the accumulators at t_begin, whereas scipy's
        # solve_ivp integrates from t=0; subtracting the first value makes both
        # backends consistent and references the balance to stored[0].
        prod   = prod   - prod[0]
        recomb = recomb - recomb[0]
        sink   = sink   - sink[0]
        stored_change = stored - stored[0]
        # LOOPS SWALLOWED BY A FREE SURFACE ARE A SEVENTH CHANNEL, and a
        # MEASURED one -- the solver integrates it, where `grain_boundary`
        # below is closure by difference. When gb_absorption is on it enters
        # the balance explicitly, so what remains in the residual is still just
        # the MOBILE flux to the surface and the two do not contaminate each
        # other. Absent, it is exactly zero and every earlier balance is
        # reproduced term for term.
        gb = (gb_loops - gb_loops[0]) if gb_loops is not None             else np.zeros_like(prod)
        net_expected  = prod - recomb - sink - gb
        residual      = stored_change - net_expected
        # Relative error vs cumulative production (0 at the reference point).
        rel_error     = np.where(prod > 0, residual / np.maximum(prod, 1e-300), 0.0)
        out = {
            'production':    prod,
            'recombination': recomb,
            'sink':          sink,
            'gb_loops':      gb,
            'stored':        stored,
            'stored_change': stored_change,
            'net_expected':  net_expected,
            'residual':      residual,
            'rel_error':     rel_error,
        }
        if mobile is not None:
            mob_change = mobile - mobile[0]
            out['mobile_change'] = mob_change
            out['immobile_change'] = stored_change - mob_change
            # THE named channel. In a closed 0-D system `residual` is numerical
            # error and this is meaningless; in an open one -- every domain this
            # repository meshes, whose whole surface is Dirichlet unless faces
            # are made periodic -- it is the only remaining place for the atoms
            # to have gone, so it IS the boundary absorption.
            #
            # It is closure by difference, not an independent measurement: it
            # inherits every other channel's error. Its credibility rests on
            # those channels being exact integrals from the solver rather than
            # reconstructions, which they are.
            out['grain_boundary'] = -residual
        return out

    # The grain-boundary loop ledger, present only when the state carries it.
    # y has one row per state component, so its presence IS the switch: a run
    # without the channel simply has no ledger rows and `gb_loops` stays None.
    #
    # IT IS ALWAYS THE LAST TWO ROWS, and the widths that carry it are exactly
    # `immobile.state_width`'s: 21 / 31 / 40 against 19 / 29 / 38. Testing for
    # row 38 alone was right for a solver state and wrong for the one this
    # function is usually handed -- `field.to_legacy_layout` re-expresses a
    # self-consistent march in the 19-slot layout, so the ledger arrives at 19
    # and 20, the test failed, and the channel vanished from the balance.
    gb_i = y[-2] if y.shape[0] in (21, 31, 40) else None
    gb_v = y[-1] if y.shape[0] in (21, 31, 40) else None

    interstitial = _balance(I_stored, acc['cum_prod_i'], acc['cum_recomb_i'],
                            acc['cum_sink_i'], mobile=I_mobile, gb_loops=gb_i)
    vacancy      = _balance(V_stored, acc['cum_prod_v'], acc['cum_recomb_v'],
                            acc['cum_sink_v'], mobile=V_mobile, gb_loops=gb_v)

    print(f"  Conservation (final): interstitial rel.err = "
          f"{interstitial['rel_error'][-1]:+.2e}, "
          f"vacancy rel.err = {vacancy['rel_error'][-1]:+.2e}")

    return {'interstitial': interstitial, 'vacancy': vacancy}


def _calculate_hardening(conc, r_iL, r_aiL, r_vL, r_avL, input_data):
    """Vectorized radiation hardening via dispersed-barrier model."""
    mu    = 30e9   # shear modulus [Pa]
    b     = input_data.physical_props['b_a']
    M     = input_data.physical_props['M']
    Omega = input_data.physical_props['Omega']

    alpha_v   = input_data.model_params['alpha_v']
    alpha_i   = input_data.model_params['alpha_i']
    alpha_iL  = input_data.model_params['alpha_iL']
    alpha_aiL = input_data.model_params['alpha_aiL']
    alpha_vL  = input_data.model_params['alpha_vL']
    alpha_avL = input_data.model_params['alpha_avL']

    Cv, Ci, C2i, C3i = conc[0], conc[1], conc[2], conc[3]
    CiL, CaiL, CvL, CavL = conc[4], conc[5], conc[6], conc[7]

    def tau(alpha, N, d):
        nd = N * d
        return np.where(nd > 0, alpha * mu * b * np.sqrt(nd), 0.0)

    Delta_tau_SR = np.sqrt(
        tau(alpha_v,   Cv   / Omega, b      ) ** 2 +
        tau(alpha_i,   Ci   / Omega, b      ) ** 2 +
        tau(alpha_i,   C2i  / Omega, 2 * b  ) ** 2 +
        tau(alpha_i,   C3i  / Omega, 3 * b  ) ** 2 +
        tau(alpha_iL,  CiL  / Omega, 2*r_iL ) ** 2 +
        tau(alpha_aiL, CaiL / Omega, 2*r_aiL) ** 2 +
        tau(alpha_vL,  CvL  / Omega, 2*r_vL ) ** 2 +
        tau(alpha_avL, CavL / Omega, 2*r_avL) ** 2
    )

    rho_N        = input_data.material_params['rho']
    Delta_tau_LR = 0.3 * mu * b * np.sqrt(rho_N)   # scalar, broadcasts

    return (Delta_tau_SR + Delta_tau_LR) / M


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_results(results, reaction_rates, filename=None):
    """
    Persist results to a pickle file and write a summary CSV alongside it.

    Parameters
    ----------
    results        : dict  – from process_solution()
    reaction_rates : ReactionRates  – needed for flux columns in the CSV
    filename       : str or Path, optional  – defaults to output/zrmicro_results.pkl
    """
    if filename is None:
        filename = Path.cwd() / 'output' / 'zrmicro_results.pkl'
    else:
        filename = Path(filename)

    filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Results saved to {filename}")

        csv_file = filename.with_name(filename.stem + '_summary.csv')
        _save_summary_csv(results, reaction_rates, csv_file)
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def _save_summary_csv(results, reaction_rates, filename):
    """Write key time-series columns to a CSV file."""
    try:
        conc  = results['concentrations']
        loops = results['loop_sizes']
        mech  = results['mechanical_properties']
        rr    = reaction_rates

        df = pd.DataFrame({
            'time':             results['time'],
            'Cv':               conc['Cv'],
            'Ci':               conc['Ci'],
            'CiL':              conc['CiL'],
            'CvL':              conc['CvL'],
            'CiL_i':            conc['CiL_i'],
            'CvL_v':            conc['CvL_v'],
            'r_iL_nm':          loops['r_iL'] * 1e9,
            'r_vL_nm':          loops['r_vL'] * 1e9,
            'flux_i':           rr.omega_i * (conc['Ci'] + 2*conc['C2i'] + 3*conc['C3i']),
            'flux_v':           rr.omega_v * conc['Cv'],
            'strain_a_percent': mech['strain_a'] * 100,
            'strain_c_percent': mech['strain_c'] * 100,
            'hardening_MPa':    mech['hardening'] / 1e6,
        })
        df.to_csv(filename, index=False)
        print(f"✓ Summary saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving summary CSV: {e}")


def load_results(filename):
    """
    Load previously saved simulation results from a pickle file.

    Parameters
    ----------
    filename : str or Path

    Returns
    -------
    dict or None
    """
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"✓ Results loaded from {filename}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None
