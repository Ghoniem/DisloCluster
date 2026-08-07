"""
fit_vacancy_loops.py — calibrate the vacancy-loop (c-loop) parameters
{tau_vL0, E_a_vL, n_vL_nuc} to measured vacancy-loop density AND size.

WHY A SEPARATE FIT.  The notebook's main parameter-identification objective uses
the TOTAL loop density/size, which is dominated by interstitial (a-)loops
(~1e23). Vacancy loops sit 1-3 decades below, so their parameters have almost no
leverage there and the notebook deliberately pre-fits + freezes them
(section 2b). This script is that calibration, but model-in-the-loop and against
vacancy-SPECIFIC observables, so n_vL_nuc (which sets both the nucleated size
r = l_c*sqrt(n_vL_nuc) and, with tau_vL, the saturation number density) and the
annealing lifetime tau_vL(T) = tau_vL0*exp(E_a_vL/kT) are pinned by real c-loop
density + size data.

TARGETS (vacancy-dominated subset of the experimental database).  From the
Jostons sheet (JNM 66/68, 1977), whose "Interstitial loops (%)" column shows the
high-T large loops are mostly VACANCY in character:
    668 K : 9-55% interstitial  -> predominantly vacancy c-loops, 33-62 nm
    623 K, 0.088 dpa : ~47-63% interstitial -> ~40-50% vacancy
N_vL is estimated as N_total*(1 - f_interstitial); d is the measured mean loop
diameter (the large high-T loops ARE the vacancy population). Edit VL_TARGETS to
refine the split or add points.

OBJECTIVE (same scale-invariant form as the notebook fit):
    J = w_N * mean[(log10 N_vL^sim - log10 N_vL^exp)^2]
      + w_d * mean[((d_vL^sim - d_vL^exp)/d_vL^exp)^2]

Run:  python3 fit_vacancy_loops.py
Writes the calibrated values to output/fit_vL_<stamp>/optimal_parameters_vL.csv
and prints the model-vs-experiment table. Copy the result into the
Model_Parameters sheet (tau_vL0, E_a_vL, n_vL_nuc).
"""
import sys, io, contextlib, time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from py_utils.simulation import ZrMicroSimulation
from py_utils.cpp_bridge import run_cpp_solver

INPUT_FILE = BASE_DIR / 'input' / 'Zr_input_parameters.xlsx'
OUTPUT_DIR = BASE_DIR / 'output'
_DEVNULL = contextlib.redirect_stdout(io.StringIO())

# ── Vacancy-loop targets:  T [K], dose [dpa], N_vL [m^-3], d_vL [nm] ──────────
# (Jostons; N_vL = N_total*(1-f_interstitial). Edit freely.)
VL_TARGETS = [
    {'T': 623.0, 'dpa': 0.0883, 'N_vL': 1.2e21, 'd': 23.0},
    {'T': 668.0, 'dpa': 0.055,  'N_vL': 5.0e20, 'd': 33.0},
    {'T': 668.0, 'dpa': 0.107,  'N_vL': 1.6e20, 'd': 56.0},
    {'T': 668.0, 'dpa': 0.300,  'N_vL': 1.1e20, 'd': 62.0},
]

# ── Fitted parameters:  name, [min, max], seed, log-search? ──────────────────
# Flux-growth regime: the vacancy-loop LIFETIME (E_a_vL) is the gate that lets
# c-loops bootstrap and grow by excess vacancy flux; recom and delta_DAD set the
# magnitude/direction of that excess. n_vL_nuc sets the birth size.
FIT_PARAMS = [
    ('E_a_vL',     0.3,  1.5,   1.25,   False),  # lifetime activation energy [eV] — GATE (number sat. + survival)
    ('tau_vL0',    1e-5, 1e2,   1e-4,   True),   # lifetime prefactor [s]
    ('n_vL_nuc',   5.0,  1000., 70.0,   True),   # birth size; also sets density via N_vL ∝ 1/n_vL0
    ('epsilon_vL', 1e-4, 2e-2,  1e-3,   True),   # production bias: free-vacancy / seed split (sets density)
    ('recom',      0.3,  10.0,  3.0,    True),   # recombination factor (amplifier)
    ('delta_DAD',  0.1,  0.45,  0.30,   False),  # DAD bias (amplifier)
]
W_DENSITY, W_DIAMETER = 1.0, 1.0   # size trend now correct; balance density and size
N_LHS, NM_MAXITER = 16, 50          # coarse design + Nelder-Mead polish budget

NAMES = [p[0] for p in FIT_PARAMS]
LO    = np.array([p[1] for p in FIT_PARAMS], float)
HI    = np.array([p[2] for p in FIT_PARAMS], float)
SEED  = np.array([p[3] for p in FIT_PARAMS], float)
LOGM  = np.array([p[4] for p in FIT_PARAMS], bool)

def to_t(v):   return np.where(LOGM, np.log10(np.maximum(v, 1e-300)), v)
def from_t(x): return np.where(LOGM, 10.0**x, x)
LO_T, HI_T = to_t(LO), to_t(HI)

print("=" * 70)
print("VACANCY-LOOP CALIBRATION  —  fit {tau_vL0, E_a_vL, n_vL_nuc}")
print("=" * 70)
print(f"targets: {len(VL_TARGETS)} points over "
      f"{sorted({t['T'] for t in VL_TARGETS})} K")

with _DEVNULL:
    SIM = ZrMicroSimulation(str(INPUT_FILE))
G_RATE = float(SIM.input_data.material_params['G'])
SOLVER_METHOD = {'backend': 'cvode', 'lmm': 'bdf', 'linsol': 'dense'}

# group targets by temperature (one ODE solve per temperature per eval)
TEMPS = sorted({t['T'] for t in VL_TARGETS})
BYT = {T: [t for t in VL_TARGETS if t['T'] == T] for T in TEMPS}


def predict_vL(pdict, T, dpas):
    """Model vacancy-loop (N_vL [m^-3], d_vL [nm]) at requested doses."""
    inp = SIM.input_data
    for k, v in pdict.items():
        # route each fitted key to the sheet that owns it
        if k in inp.physical_props:
            inp.physical_props[k] = float(v)
        elif k in inp.material_params:
            inp.material_params[k] = float(v)
        else:
            inp.model_params[k] = float(v)
    inp.material_params['T'] = float(T)
    with _DEVNULL:
        inp.calculate_derived_parameters()
        from py_utils.reaction_rates import ReactionRates
        from py_utils.rate_equations import RateEquations
        SIM.reaction_rates = ReactionRates(inp)
        SIM.rate_equations = RateEquations(inp, SIM.reaction_rates)
    max_dpa = float(np.max(dpas))
    t_end = min(10.0**np.ceil(np.log10(max(max_dpa / G_RATE * 3.0, 1e6))), 1e10)
    cfg = {'t_begin': 1e-1, 't_end': t_end, 'n_points': 120,
           'rtol': 1e-6, 'atol': 1e-20, 'log_time': True,
           'solver_method': SOLVER_METHOD}
    with _DEVNULL:
        r = run_cpp_solver(SIM, cfg, base_dir=BASE_DIR)
    if r is None:
        return None, None
    c, ls = r['concentrations'], r['loop_sizes']
    Om = inp.physical_props['Omega']
    dose = G_RATE * r['time']
    Nvl_frac = c['CvL'] + c['CavL']
    N_vL = np.maximum(Nvl_frac / Om, 1e-30)
    d_vL = 2.0 * (c['CvL'] * ls['r_vL'] + c['CavL'] * ls['r_avL']) \
           / np.maximum(Nvl_frac, 1e-30) * 1e9
    ld = np.log10(np.maximum(dose, 1e-30))
    q = np.log10(np.clip(np.asarray(dpas, float), dose.min(), dose.max()))
    return 10.0**np.interp(q, ld, np.log10(N_vL)), np.interp(q, ld, d_vL)


_HIST = {'best_J': np.inf, 'best_x': to_t(SEED).copy(), 'n': 0, 't0': time.time()}

def objective(xt):
    xt = np.clip(np.asarray(xt, float), LO_T, HI_T)
    pdict = dict(zip(NAMES, from_t(xt)))
    rN, rD = [], []
    try:
        for T in TEMPS:
            g = BYT[T]
            N_vL, d_vL = predict_vL(pdict, T, [t['dpa'] for t in g])
            if N_vL is None:
                return 50.0
            for i, t in enumerate(g):
                rN.append(np.log10(N_vL[i]) - np.log10(t['N_vL']))
                rD.append((d_vL[i] - t['d']) / t['d'])
        J = W_DENSITY * np.mean(np.square(rN)) + W_DIAMETER * np.mean(np.square(rD))
    except Exception:
        return 50.0
    _HIST['n'] += 1
    if J < _HIST['best_J']:
        _HIST['best_J'], _HIST['best_x'] = J, xt.copy()
    if _HIST['n'] % 5 == 0 or J <= _HIST['best_J']:
        print(f"  eval {_HIST['n']:3d}  J={J:8.4f}  best={_HIST['best_J']:8.4f}"
              f"  ({time.time()-_HIST['t0']:4.0f}s)")
    return float(J)


# ── coarse Latin-hypercube design, then Nelder-Mead polish ───────────────────
rng = np.random.default_rng(0)
print("\n[1] coarse search ...")
cut = np.linspace(0, 1, N_LHS + 1)
U = np.empty((N_LHS, len(NAMES)))
for j in range(len(NAMES)):
    U[:, j] = rng.permutation(cut[:-1] + rng.random(N_LHS) / N_LHS)
for u in [to_t(SEED)] + list(LO_T + U * (HI_T - LO_T)):
    objective(u)

print("\n[2] Nelder-Mead polish from best ...")
res = minimize(objective, _HIST['best_x'].copy(), method='Nelder-Mead',
               options={'maxiter': NM_MAXITER, 'xatol': 1e-3, 'fatol': 1e-4})

best = from_t(np.clip(_HIST['best_x'], LO_T, HI_T))
best_d = dict(zip(NAMES, best))
print("\n" + "=" * 70)
print(f"BEST  J = {_HIST['best_J']:.4f}")
for k, v in best_d.items():
    print(f"  {k:<10} = {v:.6g}")

print("\nmodel vs experiment at the optimum:")
print(f"{'T':>5} {'dpa':>7} | {'N_vL exp':>10} {'N_vL sim':>10} | {'d exp':>6} {'d sim':>6}")
for T in TEMPS:
    g = BYT[T]
    N_vL, d_vL = predict_vL(best_d, T, [t['dpa'] for t in g])
    for i, t in enumerate(g):
        print(f"{T:>5.0f} {t['dpa']:>7.3f} | {t['N_vL']:>10.2e} {N_vL[i]:>10.2e} | "
              f"{t['d']:>6.1f} {d_vL[i]:>6.1f}")

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = OUTPUT_DIR / f"fit_vL_{stamp}"
run_dir.mkdir(parents=True, exist_ok=True)
import csv
with open(run_dir / 'optimal_parameters_vL.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['parameter', 'min', 'optimal', 'max'])
    for k, lo, v, hi in zip(NAMES, LO, best, HI):
        w.writerow([k, lo, v, hi])
print(f"\n✓ written {run_dir / 'optimal_parameters_vL.csv'}")
