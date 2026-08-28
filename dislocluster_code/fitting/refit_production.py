"""Refit the parameter set against experiment, in the PRODUCTION formulation.

Three stale-fit warnings had accumulated and they compound: the atomic volume
was corrected (Omega, loop densities x0.5), the <c> Burgers vector was corrected
(|b| = c/2, l_c x1.4142), and `loop_model = 1` replaced the phenomenological
capture efficiencies with Woo's. On top of those, steps 3 and 5 changed which
channels exist at all -- peripheral emission deleted the two annealing
lifetimes, and the second moment added a floor current and a small-end leak.
The parameter set every driver still loads was fitted against NONE of that.

WHAT THIS FITS

    loop_model = 1, n_fam = 9, moments = 1, emission_model = 1,
    variant_weights = (1/3, 1/3, 1/3)

which is the march's slow step term for term -- there is no second
implementation, `rate_equations_core.h` is one templated function and the only
difference is `freeze_mobile`. So a set fitted here is a set the 3-D code runs.

WHAT IT DOES NOT FIT, and why

    the basal chain (`basal_chain`). Switching it on changes J by less than
    1e-4, because every one of its rates defaults to zero and nothing in the
    workbook or the material file supplies one. It is not inert as physics; it
    is inert as a CALIBRATION, and giving its rates values makes the fit choose
    numbers no measurement pins. `nu_uf` in particular is not separately
    identifiable from a steady c_p density until it is settled whether c_p has
    a sink of its own.

    nine workbook parameters this formulation does not read -- see the note
    above `CAND.update` in fit_cloops.

The staging is deliberate. Twelve levers at once from a single seed is how the
earlier attempt produced a superset scoring worse than its own subset; each
stage here starts from the previous stage's optimum and is REJECTED if it does
not improve on it, so the sequence is monotone by construction.

Run it:  python -m dislocluster_code.fitting.refit_production [--diam-log]

`--diam-log` scores the diameter as a log ratio instead of a relative residual.
It is not a cosmetic choice: as written, the diameter term saturates at 1 as
d_sim -> 0 while the density term is unbounded, so the fit can always pay for
density with diameter and every <c> fit ends at d_c ~ 0.1x experiment. See the
note on `DIAM_LOG` in fit_cloops. Both are run and both are reported, because
which one is right is a modelling judgement and not something the fit can
settle for itself.
"""
from __future__ import annotations

import csv as _csv
import json
import sys
import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

import dislocluster_code.fitting.fit_cloops as F
from dislocluster_code import paths

#: The formulation being fitted. See the module docstring.
MODEL = dict(loop_model=1, n_fam=9, moments=1, emission_model=1,
             variant_weights=(1 / 3, 1 / 3, 1 / 3))

#: One reporting point, the same one every leak measurement was quoted at.
T_REF, DPA_REF = 573.0, 24.5
N_EXP, D_EXP = 8.0e20, 95.0

#: The stages, in the order they are run.
#:
#: 1. the <c> zeroth moment and its leak. The screen puts `epsilon_vL` and
#:    `n_vL_nuc` at the two largest sensitivities in the whole set (dJ = 1.13
#:    and 1.55 over a factor of 3), and the small-end leak is the only channel
#:    that takes density down while raising diameter.
#: 2. the <c> removal channels, which is where a missing sink would show.
#: 3. the <a> side and the moment floor. The vacancy pool is shared, so stages
#:    1-2 move <a> whether or not it is a lever.
#: 4. everything, from there.
BIAS = ['dad_p_v', 'dad_p_i', 'dad_Z0_v', 'dad_Z0_i', 'loop_sink_scale_c']

# THE BIAS IS FITTED BEFORE THE REMOVAL CHANNELS, and the order is the whole
# difference between a usable fit and a useless one. There are two ways to
# bring the <c> density down: crank COALESCENCE, which removes loops by
# shrinking the survivors, or open the small-end leak and let the BIAS grow
# them, which removes loops by enlarging the survivors. The data wants the
# second -- it asks for FEWER and LARGER. A search that fits removal first
# commits to the coalescence basin and never leaves: run that way it ended with
# c_LL_c on its 1e4 ceiling, kappa_LL and kappa_LN on their bounds, and a <c>
# diameter of 8.6 nm against the 95 nm measured. Fitting the bias first reaches
# d/d_exp ~ 1.0 with the density still within a factor of 2.
STAGES = [
    ("<c> nucleation + small-end leak",
     ['epsilon_vL', 'n_vL_nuc', 'nu_vanish', 'm_vanish']),
    ("bias", BIAS),
    ("<c> removal",
     ['c_LL_c', 'c_LN_c', 'kappa_LL', 'kappa_LN']),
    ("<a> side",
     ['epsilon_iL', 'n_iL_nuc', 'c_LN_a', 'm_min']),
    ("joint polish",
     ['epsilon_vL', 'n_vL_nuc', 'nu_vanish', 'm_vanish',
      'c_LL_c', 'c_LN_c', 'kappa_LL', 'kappa_LN',
      'epsilon_iL', 'n_iL_nuc', 'c_LN_a', 'm_min'] + BIAS),
]


def _transform(names):
    lo = np.array([F.CAND[n][0] for n in names], float)
    hi = np.array([F.CAND[n][1] for n in names], float)
    lg = np.array([F.CAND[n][2] for n in names], bool)

    def to_t(v):
        v = np.asarray(v, float)
        return np.where(lg, np.log10(np.maximum(v, 1e-300)), v)

    def from_t(x):
        return np.where(lg, 10.0 ** np.clip(x, -300, 300), x)

    return to_t, from_t, to_t(lo), to_t(hi)


def fit_stage(names, base, n_starts=3, maxiter=None, seed=0, log=print):
    """Nelder-Mead over `names`, everything else frozen at `base`.

    Returns (J, breakdown, params) for the BEST point seen, which is the
    incoming `base` itself when nothing beats it. Restarts are perturbations of
    the seed rather than uniform draws from the box: a uniform draw in a
    four-decade log box lands nowhere near the physical scale and simply burns
    iterations.

    A LEVER THAT STARTS AT ZERO HAS NO LOG SCALE. `nu_vanish` is 0 in the
    reference set and log10(0) is -inf, which the clip turns into the lower
    bound -- so the seed silently becomes 1e-9 rather than "off". That is the
    behaviour wanted here (the leak has to be able to switch on), but it means
    the seed evaluation is NOT the baseline, and the baseline is evaluated
    separately and kept in the comparison.
    """
    to_t, from_t, lo_t, hi_t = _transform(names)
    x0 = np.clip(to_t([base.get(n, 0.0) for n in names]), lo_t, hi_t)
    maxiter = maxiter or 120 * len(names)

    n_eval = [0]

    def f(xt):
        xt = np.clip(xt, lo_t, hi_t)
        p = dict(base)
        p.update(dict(zip(names, from_t(xt))))
        n_eval[0] += 1
        return F.objective_full(p)[0]

    J_base = F.objective_full(base)[0]
    best_J, best_x, best_is_base = J_base, x0.copy(), True

    J_seed = f(x0)
    if J_seed < best_J:
        best_J, best_x, best_is_base = J_seed, x0.copy(), False

    rng = np.random.default_rng(seed)
    span = hi_t - lo_t
    starts = [x0] + [np.clip(x0 + rng.normal(0.0, 0.15, len(names)) * span,
                             lo_t, hi_t) for _ in range(n_starts - 1)]
    for i, s in enumerate(starts):
        t0 = time.time()
        res = minimize(f, s, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-3,
                                'fatol': 1e-5, 'adaptive': True})
        log(f"      start {i}: J {res.fun:.4f}  ({res.nit} iters, "
            f"{time.time() - t0:.0f} s)")
        if res.fun < best_J:
            best_J = res.fun
            best_x = np.clip(res.x, lo_t, hi_t)
            best_is_base = False

    p = dict(base)
    if not best_is_base:
        p.update(dict(zip(names, from_t(best_x))))
    J, bd = F.objective_full(p, want_breakdown=True)
    log(f"      {n_eval[0]} evaluations; "
        f"{'kept the incoming point' if best_is_base else 'improved'}")
    return J, bd, p


def reference_point(pdict):
    """N/N_exp and d/d_exp for the <c> family at T_REF / DPA_REF."""
    hist = F.model_history_batch(pdict, [(T_REF, DPA_REF)])[T_REF]
    if hist is None:
        return float('nan'), float('nan')
    dose, series = hist
    N, d = F._sample(dose, series['C'][0], series['C'][1], [DPA_REF])
    return float(N[0] / N_EXP), float(d[0] / D_EXP)


def _write_params(out, params, stale, cand):
    """The fitted set as a CSV a later run can start from. Called per stage.

    WRITTEN AFTER EVERY STAGE, not only at the end. A staged fit runs for
    hours -- the joint polish alone took 1.6 h per start -- and until this the
    fitted values existed only in the running process's memory. Killing a run
    whose last stage had converged to no improvement therefore threw away every
    stage before it too. The file is rewritten in place, so the directory
    always holds the best set reached so far.
    """
    with open(out / 'optimal_parameters.csv', 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(['parameter', 'stale', 'refitted', 'min', 'max'])
        for k in sorted(params):
            lo, hi = (cand[k][0], cand[k][1]) if k in cand else ('', '')
            w.writerow([k, stale.get(k, ''), params[k], lo, hi])


def load_refit(where):
    """The `refitted` column of a previous run's optimal_parameters.csv.

    A refit is staged and each stage costs minutes, so re-running the whole
    sequence to add one stage wastes the part that already converged. This
    reads a finished run back as the starting point.
    """
    import csv
    path = paths.OUTPUT_DIR / where if not str(where).endswith('.csv') else where
    if path.is_dir():
        path = path / 'optimal_parameters.csv'
    with open(path, newline='') as fh:
        return {r['parameter']: float(r['refitted']) for r in csv.DictReader(fh)}


def main(diam_log=False, stages=None, start_from=None, c_weight=1.0):
    """`c_weight` raises the <c> family's weight in the objective.

    THE TWO FAMILIES ARE NOT EQUALLY CONSTRAINED, and at c_weight = 1 they are
    not equally WEIGHTED either. J averages over (temperature, family) pairs;
    <a> is measured at 12 temperatures and <c> at 2, so <c> carries about a
    seventh of the objective however the per-family weights are set. Every
    lever that helps both families -- the bias above all -- is therefore spent
    on <a>, and the <c> diameter is left where it was.

    c_weight = 6 brings the two to parity. It is a statement about what the fit
    is FOR, not a correction: a fit that has to predict ///c/// loop sizes and
    one that has to predict the <a> population are different fits, and no
    single weighting serves both.
    """
    t_start = time.time()
    F.DIAM_LOG = bool(diam_log)
    F.WLOOP['C'] = float(c_weight)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'production_diamlog' if diam_log else 'production'
    if c_weight != 1.0:
        tag += f'_cw{c_weight:g}'
    out = paths.OUTPUT_DIR / f"refit_{stamp}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    F.set_model(**MODEL)
    log("=" * 74)
    log("REFIT -- production formulation "
        "(loop_model 1, 9 families, moments, emission)")
    log(f"diameter term: {'log ratio' if diam_log else 'relative residual'}")
    log(f"<c> family weight: {c_weight:g}"
        f"{'  (parity with <a>)' if c_weight != 1.0 else ''}")
    log("=" * 74)

    base = dict(F.OPT)
    if start_from:
        base.update(load_refit(start_from))
        log(f"continuing from {start_from}")
    J0, bd0 = F.objective_full(base, want_breakdown=True)
    n0, d0 = reference_point(base)
    log(f"\nbaseline (the stale set):  J {J0:.4f}   "
        f"J_A {bd0['A']['J']:.4f}   J_C {bd0['C']['J']:.4f}")
    log(f"   <c> at {T_REF:.0f} K / {DPA_REF} dpa:  "
        f"N/N_exp {n0:.2f}   d/d_exp {d0:.3f}")

    history = [("baseline", J0, bd0, n0, d0)]
    for tag, names in (stages if stages is not None else STAGES):
        log(f"\n--- {tag}:  {', '.join(names)}")
        J, bd, base = fit_stage(names, base, log=log)
        n_r, d_r = reference_point(base)
        log(f"      J {J:.4f}   J_A {bd['A']['J']:.4f}   J_C {bd['C']['J']:.4f}"
            f"   N/N_exp {n_r:.2f}   d/d_exp {d_r:.3f}")
        history.append((tag, J, bd, n_r, d_r))
        _write_params(out, base, F.OPT, F.CAND)

    J, bd = F.objective_full(base, want_breakdown=True)
    log("")
    log("=" * 74)
    log(f"  {'stage':<34}{'J':>8}{'J_A':>8}{'J_C':>8}"
        f"{'N/N_exp':>10}{'d/d_exp':>9}")
    log("-" * 74)
    for tag, Jv, bdv, n_r, d_r in history:
        log(f"  {tag:<34}{Jv:>8.4f}{bdv['A']['J']:>8.4f}{bdv['C']['J']:>8.4f}"
            f"{n_r:>10.2f}{d_r:>9.3f}")
    log("=" * 74)

    log("")
    log("  parameters that moved:")
    log(f"  {'parameter':<16}{'stale':>14}{'refitted':>14}{'ratio':>9}")
    log("  " + "-" * 51)
    moved = {}
    for k in sorted(set(base) | set(F.OPT)):
        old, new = F.OPT.get(k), base.get(k)
        if old is None or new is None:
            continue
        if old == 0.0 and new == 0.0:
            continue
        if old != 0.0 and abs(new / old - 1.0) < 0.02:
            continue
        moved[k] = (old, new)
        ratio = f"{new / old:>9.3g}" if old else f"{'from 0':>9}"
        log(f"  {k:<16}{old:>14.6g}{new:>14.6g}{ratio}")

    F.cloop_table(base, 'refitted')

    _write_params(out, base, F.OPT, F.CAND)
    (out / 'refit.md').write_text(
        "# Refit -- production formulation\n\n```\n"
        + "\n".join(lines) + "\n```\n", encoding='utf-8')
    with open(out / 'refit.json', 'w') as fh:
        json.dump({'model': dict(MODEL), 'diam_log': bool(diam_log),
                   'J': J, 'J_A': bd['A']['J'], 'J_C': bd['C']['J'],
                   'baseline_J': J0,
                   'params': base, 'moved': moved,
                   'wall_clock_s': time.time() - t_start}, fh, indent=2,
                  default=float)
    print(f"\nwrote {out}")
    print("\nPARAMETER_OVERRIDES_REFIT = {")
    for k in sorted(base):
        print(f"    {k!r:<16}: {base[k]:.6g},")
    print("}")


if __name__ == '__main__':
    _argv = sys.argv[1:]
    _from = None
    if '--from' in _argv:
        _from = _argv[_argv.index('--from') + 1]
    _stages = None
    if '--bias-only' in _argv:
        _stages = [s for s in STAGES if s[0] in ('bias', 'joint polish')]
    _cw = 1.0
    if '--c-weight' in _argv:
        _cw = float(_argv[_argv.index('--c-weight') + 1])
    main(diam_log='--diam-log' in _argv, stages=_stages, start_from=_from,
         c_weight=_cw)
