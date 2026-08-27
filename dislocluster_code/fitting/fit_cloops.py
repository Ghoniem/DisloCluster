"""
fit_cloops.py — refit the C-LOOP (vacancy-loop) parameters ONLY, holding every
a-loop / network / shared parameter FROZEN at the current joint optimum, so the
a-loop fit is provably undisturbed.  Then search for the SMALLEST subset of
c-loop levers that reproduces the best c-loop fit.

WHY.  In the joint notebook fit (cell 1) the c-loop levers are optimised together
with the a-loops; at the current optimum the c-loops are starved — density ~5x
low (J_C density 0.197) and size ~4-5x low (J_C diameter 0.587).  The clean
c-loop-only levers are:

    epsilon_vL  cascade vacancy fraction -> SUPPLY  (density, total content)
    Z_v_c       c-loop vacancy capture bias        -> net vacancy flux (SIZE)   [NEW lever]
    n_vL_nuc    vacancies per nucleated loop        -> density(1/n) <-> birth size
    tau_vL0     annealing-lifetime prefactor        -> density level
    E_a_vL      annealing-lifetime activation       -> density T-trend
    Q           c-loop absorption efficiency        -> size (already ~maxed)
    c_LL_c,c_LN_c  c-loop coalescence               -> size <-> density (c_LN_c ~maxed)

NONE of these enter the a-loop (iL/aiL) equations.  Z_v_c is set as an explicit
override so it does NOT change a-loop Z_v_a (= 1 - delta_DAD_v); delta_DAD stays
frozen.  The only residual coupling is the shared free-vacancy pool, which we
verify is negligible by reporting J_A before/after (it must not move).

Objective is a faithful replica of notebook cell-1's J (per-(loop,T) group misfit
+ overshoot penalty, balance-by-temperature, high-T boost), so the reported J is
directly comparable to the joint fit's 0.609.

Run:  .DisloClusterVenv/Scripts/python.exe ZrClusterDynamics/ZrMicro/py_utils/fit_cloops.py
"""
import sys, io, contextlib, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Every path below is relative to ZrMicro/ (the workbook, the output tree and
# the compiled solver all live there). This used to be spelled
# `Path(__file__).parent.parent`, which was ZrMicro/ only while this file sat in
# ZrMicro/py_utils/; ask `paths` instead so it survives being moved.
from dislocluster_code import paths as _paths
BASE = _paths.ZRMICRO_DIR
from dislocluster_code.zerod.simulation import ZrMicroSimulation
from dislocluster_code.zerod.reaction_rates import ReactionRates
from dislocluster_code.zerod.rate_equations import RateEquations
from dislocluster_code.zerod.cpp_bridge import (run_cpp_solver_batch, collect_solver_args)
from dislocluster_code.zerod.post_process import calculate_derived_quantities
from dislocluster_code.coupling import field as _field

XL  = BASE / 'input' / 'Zr_input_parameters.xlsx'
DEV = contextlib.redirect_stdout(io.StringIO())

# ── objective config (mirror notebook OPT_CONFIG) ────────────────────────────
W_DENSITY = W_DIAM = 1.0
W_OVERSHOOT = 1.0
OS_DOSE_LO = 1e-3
HIGHT_THR  = 650.0
HIGHT_BOOST = 2.0
WLOOP = {'A': 1.0, 'C': 1.0}
NPTS, RTOL, ATOL = 70, 1e-6, 1e-20
SOLVER = {'backend': 'cvode', 'lmm': 'bdf', 'linsol': 'dense'}

# ── which FORMULATION the objective evaluates ────────────────────────────────
# Empty is the legacy model, which is what every existing fit was made against
# and what this harness reproduces byte-for-byte with MODEL untouched.
#
# THE 0-D AND THE MARCH RUN THE SAME EQUATIONS. `rate_equations_core.h` is one
# templated function; the march's slow step and this objective are the same
# binary with the same switches. The only difference is `freeze_mobile`: the
# march pins the mobile species to the FEM solve's C_M*(x) and integrates 34
# equations per point, while a standalone 0-D leaves them free and integrates
# all 38. So a parameter set fitted here is a parameter set the 3-D code runs,
# term for term, with no second implementation to drift.
#
# Set it with `set_model(...)`, e.g. for the self-consistent nine-family model
#     set_model(loop_model=1, n_fam=9, emission_model=1, temperature_K=573.0,
#               basal_chain=1, moments=1, m_min=10.0, ...)
# and note that the chain's rates are NOT calibrated -- they are among the
# things a refit has to produce, not consume.
MODEL = {}


def set_model(**kw):
    """Choose the formulation the objective evaluates. See MODEL above."""
    MODEL.clear()
    MODEL.update(kw)
    return dict(MODEL)

# ── current joint optimum — loaded from the latest fit CSV (freeze point) ─────
# THE FREEZE POINT FALLS BACK TO THE CALIBRATED SET. This used to be
# `sorted(glob('fit_*/optimal_parameters.csv'))[-1]`, which raises IndexError at
# IMPORT time when no previous fit output is present -- and none is in a fresh
# checkout, because the fit tree was never committed. The module was therefore
# unimportable, which is a poor way to discover that a refit harness exists.
#
# calibration.REFERENCE_OVERRIDES is the same 26-parameter set every driver
# builds its model from, so falling back to it starts the search exactly where
# `build_sim()` would. A previous fit's CSV still wins when one is there.
_fits = sorted((BASE / 'output').glob('fit_*/optimal_parameters.csv'))
if _fits:
    OPT_CSV = _fits[-1]
    _optdf = pd.read_csv(OPT_CSV)
    OPT = {str(r['parameter']): float(r['optimal']) for _, r in _optdf.iterrows()}
    print(f"freeze point: {OPT_CSV.relative_to(BASE)}  ({len(OPT)} params)")
else:
    from dislocluster_code.zerod.calibration import REFERENCE_OVERRIDES
    OPT_CSV = None
    OPT = dict(REFERENCE_OVERRIDES)
    print(f"freeze point: calibration.REFERENCE_OVERRIDES  ({len(OPT)} params) "
          f"-- no previous fit output found")

# Derive the frozen Z_v_c implied by the current delta_DAD, so adding Z_v_c as an
# explicit lever starts EXACTLY where the joint fit left it (continuity check).
OPT.setdefault('Z_v_c', 1.0 + OPT.get('delta_DAD', 0.2))

# ── c-loop-only candidate levers:  name -> (lo, hi, log?) ────────────────────
CAND = {
    'epsilon_vL': (5e-3, 2e-1, True),
    'Z_v_c':      (1.0,  2.0,  False),
    'n_vL_nuc':   (5.0,  400., False),
    'tau_vL0':    (1e-2, 5e1,  True),
    'E_a_vL':     (0.2,  0.8,  False),
    'Q':          (0.1,  1.0,  False),
    # THE UPPER BOUNDS HAD TO MOVE, and by a lot. Fitted at loop_model = 0,
    # `tau_cvL` removed the basal loops and these two never had to carry that
    # channel, so 1e2 / 2e2 was ample. Step 3 deletes the lifetime and
    # coalescence becomes the ONLY number sink for every vacancy family, which
    # the measurement in CLAUDE.md puts at needing ~1e3 x the current
    # 0.162 / 2.96 -- i.e. ~160 / ~2960, near the PRISMATIC values 121 / 1131.
    # `c_LN_c` could not have reached that: its old ceiling of 200 sits below
    # the answer, so a search would have stopped at the wall and reported it as
    # an optimum.
    'c_LL_c':     (1e-2, 1e4,  True),
    'c_LN_c':     (1e-2, 1e4,  True),
}

# ── targets (replica of notebook _load_targets) ──────────────────────────────
def load_targets(sheet):
    df = pd.read_excel(XL, sheet).iloc[:, :4]
    df.columns = ['T', 'dpa', 'N_L', 'd']
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['T'] = df['T'].ffill()
    df = df.dropna(subset=['T', 'dpa'])
    df = df[~(df['N_L'].isna() & df['d'].isna())].reset_index(drop=True)
    return df

TARG = {'A': load_targets('Targets_A'), 'C': load_targets('Targets_C')}
TARG['A'].loc[TARG['A']['T'] >= 650.0, 'd'] = np.nan   # high-T 'a' sizes are vacancy loops
GROUPS = {k: {float(T): g.reset_index(drop=True)
             for T, g in TARG[k].groupby('T')} for k in ('A', 'C')}
EVAL_TEMPS = sorted(set(GROUPS['A']) | set(GROUPS['C']))

with DEV:
    SIM = ZrMicroSimulation(str(XL))
G_DEFAULT = float(SIM.input_data.material_params['G'])


def _configure(pdict, T):
    inp = SIM.input_data
    for k, v in pdict.items():
        for d in (inp.material_params, inp.physical_props, inp.model_params):
            if k in d:
                d[k] = float(v); break
        else:
            inp.model_params[k] = float(v)
    inp.material_params['T'] = float(T)
    with DEV:
        inp.calculate_derived_parameters()
        SIM.reaction_rates = ReactionRates(inp)
        SIM.rate_equations = RateEquations(inp, SIM.reaction_rates)


def model_history_batch(pdict, temps_maxdpa):
    cases, meta = [], []
    for T, max_dpa in temps_maxdpa:
        _configure(pdict, T)
        G = G_DEFAULT
        t_end = min(10.0 ** np.ceil(np.log10(max(max_dpa / G * 3.0, 1e6))), 1e10)
        cfg = {'t_begin': 1e-1, 't_end': t_end, 'n_points': NPTS,
               'rtol': RTOL, 'atol': ATOL, 'log_time': True,
               'solver_method': SOLVER}
        cfg.update(MODEL)
        if MODEL.get('emission_model'):
            # Emission needs the temperature it is being evaluated at, and this
            # harness sweeps temperature -- so it is taken from the case and
            # never from a fixed entry in MODEL.
            cfg['temperature_K'] = float(T)
        with DEV:
            cases.append(collect_solver_args(SIM, cfg))
        meta.append((T, G))
    with DEV:
        raw = run_cpp_solver_batch(cases, base_dir=BASE)
    out = {}
    for (T, G), ty in zip(meta, raw):
        if ty is None:
            out[T] = None; continue
        t, y = ty
        _configure(pdict, T)
        if MODEL.get('loop_model'):
            # THE SLOTS DO NOT MEAN THE SAME THING. Under loop_model >= 1,
            # y[4] is the basal <c> family and y[6..7] are prismatic <a>
            # variants, where the legacy names below read them as CiL and
            # CvL/CavL -- so `Na` would be built from a basal density and `Nc`
            # from a prismatic one. `to_legacy_layout` lumps by HABIT into the
            # slots those names expect and truncates to the legacy 19, which is
            # exactly what calculate_derived_quantities consumes.
            y = _field.to_legacy_layout(y.T, 1).T
        with DEV:
            r = calculate_derived_quantities(t, y, SIM.input_data, SIM.rate_equations)
        c, ls = r['concentrations'], r['loop_sizes']
        Om = SIM.input_data.physical_props['Omega']
        dose = G * r['time']
        Na = c['CiL'] + c['CaiL']; Nc = c['CvL'] + c['CavL']
        d_A = 2.0 * (c['CiL'] * ls['r_iL'] + c['CaiL'] * ls['r_aiL']) / np.maximum(Na, 1e-30) * 1e9
        d_C = 2.0 * (c['CvL'] * ls['r_vL'] + c['CavL'] * ls['r_avL']) / np.maximum(Nc, 1e-30) * 1e9
        out[T] = (dose, {'A': (np.maximum(Na / Om, 1e-30), d_A),
                         'C': (np.maximum(Nc / Om, 1e-30), d_C)})
    return out


def _sample(dose, N, d, dpas):
    ld = np.log10(np.maximum(dose, 1e-30))
    q = np.log10(np.clip(np.asarray(dpas, float), dose.min(), dose.max()))
    return 10.0 ** np.interp(q, ld, np.log10(N)), np.interp(q, ld, d)


def _overshoot(dose, y, logscale):
    m = np.asarray(dose) >= OS_DOSE_LO
    if int(np.sum(m)) < 3:
        return 0.0
    v = np.log10(np.maximum(np.asarray(y)[m], 1e-30)) if logscale else np.asarray(y)[m]
    dv = np.diff(v)
    down = float(-np.sum(dv[dv < 0]))
    rise = max(float(v.max() - v.min()), 1e-9)
    return down / rise


def objective_full(pdict, want_breakdown=False):
    """Faithful replica of notebook cell-1 J (returns J, and optional per-type)."""
    tk = {k: {'t': 0.0, 'w': 0.0, 'sN': 0.0, 'wN': 0.0, 'sD': 0.0, 'wD': 0.0}
          for k in ('A', 'C')}
    terms = wsum = 0.0
    os_terms = []
    _tm = [(T, max(float(GROUPS[k][T]['dpa'].max())
                   for k in ('A', 'C') if T in GROUPS[k])) for T in EVAL_TEMPS]
    hist_all = model_history_batch(pdict, _tm)
    for T in EVAL_TEMPS:
        hist = hist_all.get(T)
        if hist is None:
            return 50.0, {}
        dose, series = hist
        nT = HIGHT_BOOST if T >= HIGHT_THR else 1.0
        for k in ('A', 'C'):
            Ntr, dtr = series[k]
            os_terms.append(_overshoot(dose, Ntr, True))
            os_terms.append(_overshoot(dose, dtr, False))
            if T not in GROUPS[k]:
                continue
            g = GROUPS[k][T]
            N_sim, d_sim = _sample(dose, series[k][0], series[k][1], g['dpa'].to_numpy())
            mN, md = g['N_L'].to_numpy(), g['d'].to_numpy()
            okN, okD = ~np.isnan(mN), ~np.isnan(md)
            res = []
            if okN.any():
                dMSE = float(np.mean((np.log10(N_sim[okN]) - np.log10(mN[okN])) ** 2))
                res.append(W_DENSITY * dMSE); tk[k]['sN'] += dMSE * nT; tk[k]['wN'] += nT
            if okD.any():
                sMSE = float(np.mean(((d_sim[okD] - md[okD]) / md[okD]) ** 2))
                res.append(W_DIAM * sMSE); tk[k]['sD'] += sMSE * nT; tk[k]['wD'] += nT
            if not res:
                continue
            score = float(np.mean(res)); w = WLOOP[k] * nT
            terms += score * w; wsum += w
            tk[k]['t'] += score * nT; tk[k]['w'] += nT
    if not wsum:
        return 50.0, {}
    J = terms / wsum + W_OVERSHOOT * (float(np.mean(os_terms)) if os_terms else 0.0)
    if want_breakdown:
        bd = {k: {'J': tk[k]['t'] / tk[k]['w'] if tk[k]['w'] else 0.0,
                  'lossN': tk[k]['sN'] / tk[k]['wN'] if tk[k]['wN'] else 0.0,
                  'lossD': tk[k]['sD'] / tk[k]['wD'] if tk[k]['wD'] else 0.0}
              for k in ('A', 'C')}
        bd['J_os'] = float(np.mean(os_terms)) if os_terms else 0.0
        return J, bd
    return J, None


# ── RUNNING THIS MODULE MUST NOT RUN A FIT ──────────────────────────────────
# The baseline evaluation and the subset search below used to execute at
# IMPORT, so `import fit_cloops` spent ten minutes doing a full backward
# elimination before returning. That makes the harness unusable as a
# library -- which is what it has to be to refit a formulation chosen with
# `set_model`. Both regions are guarded; running the file as a script is
# unchanged.
_RUN_AS_SCRIPT = (__name__ == '__main__')

if _RUN_AS_SCRIPT:
    # ── 0. Baseline: reproduce the joint optimum J (validation of the replica) ────
    print("=" * 72)
    print("BASELINE  (all params frozen at joint optimum)")
    print("=" * 72)
    t0 = time.time()
    J0, bd0 = objective_full(dict(OPT), want_breakdown=True)
    print(f"J = {J0:.4f}   J_A = {bd0['A']['J']:.4f}  J_C = {bd0['C']['J']:.4f}  "
          f"J_os = {bd0['J_os']:.4f}   ({time.time()-t0:.0f}s)")
    print(f"   c-loop  density-loss {bd0['C']['lossN']:.4f}  diameter-loss {bd0['C']['lossD']:.4f}")
    print(f"   a-loop  density-loss {bd0['A']['lossN']:.4f}  diameter-loss {bd0['A']['lossD']:.4f}")
    J_A_FROZEN = bd0['A']['J']


def cloop_table(pdict, label):
    """Print c-loop N_sim/d_sim vs experiment at every Targets_C point."""
    print(f"\n  c-loop model vs experiment  [{label}]")
    print(f"  {'T':>5} {'dpa':>6} | {'N_exp':>9} {'N_sim':>9} {'N x':>6} | "
          f"{'d_exp':>6} {'d_sim':>6} {'d x':>5}")
    tm = [(T, float(GROUPS['C'][T]['dpa'].max())) for T in GROUPS['C']]
    hist = model_history_batch(pdict, tm)
    for T in GROUPS['C']:
        g = GROUPS['C'][T]
        dose, series = hist[T]
        N_sim, d_sim = _sample(dose, series['C'][0], series['C'][1], g['dpa'].to_numpy())
        for i in range(len(g)):
            Ne, de = g['N_L'][i], g['d'][i]
            print(f"  {T:>5.0f} {g['dpa'][i]:>6.1f} | {Ne:>9.2e} {N_sim[i]:>9.2e} "
                  f"{N_sim[i]/Ne:>6.2f} | {de:>6.0f} {d_sim[i]:>6.1f} {d_sim[i]/de:>5.2f}")


import os as _os
if _os.environ.get('BASELINE_ONLY'):
    cloop_table(dict(OPT), 'baseline / joint optimum')
    sys.exit(0)

if _os.environ.get('SWEEP_ONLY'):
    # Can the model reach 95-150 nm c-loops AT ALL? Sweep each size lever far
    # past its physical bound at 583 K / 35 dpa (target d=150 nm, N=1.17e21).
    T, dpa = 583.0, 35.0
    def d_at(over):
        p = dict(OPT); p.update(over)
        h = model_history_batch(p, [(T, dpa)])[T]
        dose, series = h
        N, d = _sample(dose, series['C'][0], series['C'][1], [dpa])
        return float(N[0]), float(d[0])
    print("\nSWEEP at 583 K / 35 dpa   (target  N=1.17e21 m^-3,  d=150 nm)")
    print("  lever sweeps (others frozen at joint optimum):")
    for nv in [50, 200, 400, 1000, 3000, 8000, 20000]:
        N, d = d_at({'n_vL_nuc': nv}); print(f"   n_vL_nuc={nv:>6} : N={N:.2e}  d={d:6.1f} nm")
    for zc in [1.0, 1.2, 1.5, 2.0, 3.0]:
        N, d = d_at({'Z_v_c': zc}); print(f"   Z_v_c   ={zc:>6.1f} : N={N:.2e}  d={d:6.1f} nm")
    for q in [0.1, 0.5, 1.0, 3.0, 10.0]:
        N, d = d_at({'Q': q}); print(f"   Q       ={q:>6.1f} : N={N:.2e}  d={d:6.1f} nm")
    for cl in [0.5, 10, 100, 1000, 1e4]:
        N, d = d_at({'c_LN_c': cl}); print(f"   c_LN_c  ={cl:>6g} : N={N:.2e}  d={d:6.1f} nm")
    # combined push: big birth + strong flux + strong coalescence + long life
    for combo in [
        {'n_vL_nuc': 2000, 'Z_v_c': 2.0, 'Q': 3.0},
        {'n_vL_nuc': 2000, 'Z_v_c': 2.0, 'Q': 3.0, 'c_LN_c': 1000, 'tau_vL0': 5.0},
        {'n_vL_nuc': 8000, 'Z_v_c': 3.0, 'Q': 10.0, 'c_LN_c': 1e4, 'tau_vL0': 20.0},
    ]:
        N, d = d_at(combo); print(f"   combo {combo}\n      -> N={N:.2e}  d={d:6.1f} nm")
    sys.exit(0)

if _os.environ.get('CTEST_ONLY'):
    cands = [
        {'c_LN_c': 2.0},
        {'c_LN_c': 3.0, 'epsilon_vL': 0.04},
        {'c_LN_c': 2.0, 'epsilon_vL': 0.05},
        {'c_LN_c': 2.0, 'epsilon_vL': 0.05, 'E_a_vL': 0.45},
        {'c_LN_c': 2.5, 'epsilon_vL': 0.06, 'E_a_vL': 0.30},
    ]
    for ov in cands:
        p = dict(OPT); p.update(ov)
        J, bd = objective_full(p, want_breakdown=True)
        print(f"\n>>> {ov}")
        print(f"    J={J:.4f}  J_A={bd['A']['J']:.4f}  J_C={bd['C']['J']:.4f}  "
              f"(densN {bd['C']['lossN']:.4f}  diamD {bd['C']['lossD']:.4f})")
        cloop_table(p, str(ov))
    sys.exit(0)

if _os.environ.get('JOINT'):
    # Joint refit of ALL params, seeded at the current optimum but with the
    # c-loop levers pushed into the newly-found good basin (c_LN_c down for size,
    # epsilon_vL up for density), letting a-loop params re-adjust to recover J_A.
    import csv as _csv
    from datetime import datetime as _dt
    pdf = pd.read_excel(XL, 'Params')
    pdf = pdf.rename(columns={pdf.columns[0]: 'parameter', pdf.columns[1]: 'min',
                              pdf.columns[2]: 'max'})
    pdf = pdf.dropna(subset=['parameter', 'min', 'max'])
    jn = pdf['parameter'].astype(str).str.strip().tolist()
    jlo = pdf['min'].astype(float).to_numpy()
    jhi = pdf['max'].astype(float).to_numpy()
    jlogm = np.array([(l > 0 and h / l >= 20.0) for l, h in zip(jlo, jhi)])

    seed = np.array([OPT.get(n, np.sqrt(jlo[i] * jhi[i])) for i, n in enumerate(jn)])
    OVR = {'c_LN_c': 2.5, 'epsilon_vL': 0.06, 'E_a_vL': 0.30}
    for k, v in OVR.items():
        if k in jn:
            seed[jn.index(k)] = v
    seed = np.clip(seed, jlo, jhi)

    def jto(v):   return np.where(jlogm, np.log10(np.maximum(v, 1e-300)), v)
    def jfrom(x): return np.where(jlogm, 10.0 ** np.clip(x, -300, 300), x)
    jlo_t, jhi_t = jto(jlo), jto(jhi)

    from datetime import datetime as _dt0
    _stamp0 = _dt0.now().strftime('%Y%m%d_%H%M%S')
    _ckdir = (BASE / 'output' / f"fit_cloops_joint_{_stamp0}")
    _ckdir.mkdir(parents=True, exist_ok=True)

    _J = {'best': np.inf, 'x': jto(seed).copy(), 'n': 0, 't0': time.time()}

    def _checkpoint():
        bd = dict(zip(jn, jfrom(_J['x'])))
        with open(_ckdir / 'optimal_parameters.csv', 'w', newline='') as fh:
            import csv as _c
            w = _c.writer(fh); w.writerow(['parameter', 'min', 'optimal', 'max'])
            for n, l, h in zip(jn, jlo, jhi):
                w.writerow([n, l, bd[n], h])

    def jf(xt):
        xt = np.clip(xt, jlo_t, jhi_t)
        J, _ = objective_full(dict(zip(jn, jfrom(xt))))
        _J['n'] += 1
        if J < _J['best']:
            _J['best'], _J['x'] = J, xt.copy(); _checkpoint()
        if _J['n'] % 25 == 0:
            print(f"   eval {_J['n']:4d}  best={_J['best']:.4f}  "
                  f"({time.time()-_J['t0']:.0f}s)", flush=True)
        return J

    print("\n" + "=" * 72)
    print("JOINT REFIT  (all params; seed = current optimum + c-loop overrides)")
    print("=" * 72)
    print("seed overrides:", OVR, flush=True)
    x = jto(seed)
    for rnd in range(2):
        res = minimize(jf, x, method='Nelder-Mead',
                       options={'maxiter': 800, 'maxfev': 800, 'xatol': 1e-3,
                                'fatol': 1e-4, 'adaptive': True})
        x = np.clip(_J['x'], jlo_t, jhi_t)   # restart from best-so-far
        print(f"  round {rnd}: best J={_J['best']:.4f}  "
              f"({time.time()-_J['t0']:.0f}s)", flush=True)
    bestd = dict(zip(jn, jfrom(_J['x'])))

    # Faithful report at NPTS=100 (matches the notebook / the 0.609 baseline).
    NPTS = 100
    Jseed_full, _ = objective_full(dict(zip(jn, jfrom(jto(seed)))))  # not used; warm
    Jb, bdb = objective_full(dict(OPT), want_breakdown=True)
    Jf, bdf = objective_full(bestd, want_breakdown=True)
    print("\n" + "=" * 72)
    print(f"BASELINE (joint optimum) @NPTS=100 : J={Jb:.4f}  J_A={bdb['A']['J']:.4f}  J_C={bdb['C']['J']:.4f}")
    print(f"JOINT REFIT              @NPTS=100 : J={Jf:.4f}  J_A={bdf['A']['J']:.4f}  J_C={bdf['C']['J']:.4f}")
    print(f"   a-loop  densN {bdf['A']['lossN']:.4f} (was {bdb['A']['lossN']:.4f})  "
          f"diamD {bdf['A']['lossD']:.4f} (was {bdb['A']['lossD']:.4f})")
    print(f"   c-loop  densN {bdf['C']['lossN']:.4f} (was {bdb['C']['lossN']:.4f})  "
          f"diamD {bdf['C']['lossD']:.4f} (was {bdb['C']['lossD']:.4f})")
    cloop_table(bestd, 'JOINT REFIT optimum')
    print("\nchanged parameters (|rel|>2%):")
    for n in jn:
        old, new = OPT.get(n, float('nan')), bestd[n]
        if old and abs(new / old - 1) > 0.02:
            print(f"   {n:<16} {old:.5g}  ->  {new:.5g}")

    stamp = _dt.now().strftime('%Y%m%d_%H%M%S')
    rd = (BASE / 'output' / f"fit_cloops_joint_{stamp}")
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / 'optimal_parameters.csv', 'w', newline='') as fh:
        w = _csv.writer(fh); w.writerow(['parameter', 'min', 'optimal', 'max'])
        for n, l, h in zip(jn, jlo, jhi):
            w.writerow([n, l, bestd[n], h])
    print(f"\nsaved {rd / 'optimal_parameters.csv'}")
    print("\nPARAMETER_OVERRIDES_BEST = {")
    for n in jn:
        print(f"    {n!r:<18}: {bestd[n]:.6g},")
    print("}")
    sys.exit(0)


# ── fit one subset of c-loop levers, everything else frozen at OPT ───────────
def fit_subset(names, n_starts=2, maxiter=45):
    lo = np.array([CAND[n][0] for n in names])
    hi = np.array([CAND[n][1] for n in names])
    logm = np.array([CAND[n][2] for n in names])

    def to_t(v):   return np.where(logm, np.log10(np.maximum(v, 1e-300)), v)
    def from_t(x): return np.where(logm, 10.0 ** np.clip(x, -300, 300), x)
    lo_t, hi_t = to_t(lo), to_t(hi)
    seed = np.clip(to_t(np.array([OPT.get(n, np.sqrt(lo[i]*hi[i]))
                                  for i, n in enumerate(names)])), lo_t, hi_t)

    def f(xt):
        xt = np.clip(xt, lo_t, hi_t)
        p = dict(OPT); p.update(dict(zip(names, from_t(xt))))
        J, _ = objective_full(p)
        return J

    best_x, best_J = seed.copy(), f(seed)
    rng = np.random.default_rng(0)
    starts = [seed] + [lo_t + rng.random(len(names)) * (hi_t - lo_t)
                       for _ in range(n_starts - 1)]
    for s in starts:
        res = minimize(f, s, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-4,
                                'adaptive': True})
        if res.fun < best_J:
            best_J, best_x = res.fun, np.clip(res.x, lo_t, hi_t)
    p = dict(OPT); p.update(dict(zip(names, from_t(best_x))))
    J, bd = objective_full(p, want_breakdown=True)
    return J, bd, dict(zip(names, from_t(best_x)))


if _RUN_AS_SCRIPT:
    # ── 1. Full candidate set, then backward elimination to the smallest set ─────
    ALL = list(CAND)
    print("\n" + "=" * 72)
    print("FULL c-loop candidate set:", ', '.join(ALL))
    print("=" * 72)
    Jf, bdf, pf = fit_subset(ALL, n_starts=3, maxiter=70)
    print(f"J = {Jf:.4f}  (J_A {bdf['A']['J']:.4f}  J_C {bdf['C']['J']:.4f}  "
          f"densN {bdf['C']['lossN']:.4f} diamD {bdf['C']['lossD']:.4f})")
    for k, v in pf.items():
        print(f"   {k:<11} = {v:.5g}")
    cloop_table(dict(OPT, **pf), 'FULL set optimum')

    # Candidate minimal sets ------------------------------------------------------
    # A: my original recommendation (supply + flux).  B/C: the levers that actually
    # moved in the full fit (birth size + annealing lifetime).
    TRIALS = [
        ['epsilon_vL', 'Z_v_c', 'n_vL_nuc'],            # A — supply + flux + birth
        ['n_vL_nuc', 'E_a_vL'],                          # B — birth + lifetime (2)
        ['n_vL_nuc', 'tau_vL0', 'E_a_vL'],              # B+ — birth + lifetime (3)
        ['n_vL_nuc', 'tau_vL0', 'E_a_vL', 'c_LN_c'],    # B+ + coalescence
        ['epsilon_vL', 'n_vL_nuc', 'tau_vL0', 'E_a_vL'],# supply + birth + lifetime
    ]
    print("\n" + "=" * 72)
    print("SUBSET SEARCH (smallest optimal set)")
    print("=" * 72)
    results = [('FULL(' + str(len(ALL)) + ')', Jf, bdf, pf)]
    for names in TRIALS:
        J, bd, p = fit_subset(names, n_starts=2, maxiter=45)
        tag = '{' + ','.join(names) + '}'
        results.append((tag, J, bd, p))
        print(f"\n{tag}\n  J = {J:.4f}  (J_A {bd['A']['J']:.4f}  J_C {bd['C']['J']:.4f}  "
              f"densN {bd['C']['lossN']:.4f} diamD {bd['C']['lossD']:.4f})")
        for k, v in p.items():
            print(f"   {k:<11} = {v:.5g}")
        cloop_table(dict(OPT, **p), tag)

    print("\n" + "=" * 72)
    print(f"{'set':<48}{'J':>8}{'J_A':>8}{'J_C':>8}")
    print("-" * 72)
    print(f"{'BASELINE (joint optimum, frozen)':<48}{J0:>8.4f}{bd0['A']['J']:>8.4f}{bd0['C']['J']:>8.4f}")
    for tag, J, bd, p in results:
        print(f"{tag:<48}{J:>8.4f}{bd['A']['J']:>8.4f}{bd['C']['J']:>8.4f}")
    print("=" * 72)
    print(f"a-loop J_A must stay ~{J_A_FROZEN:.4f} (frozen) — confirms a-loops undisturbed.")
