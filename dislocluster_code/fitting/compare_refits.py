"""Score several fitted sets on ONE objective, so they can be compared at all.

Each refit is run under the weighting it was fitted with, and those weightings
differ -- an <a>-dominated fit and a <c>-at-parity fit do not report comparable
J values, because the number they minimize is not the same number. Declaring a
winner from the J each run printed for itself would be comparing two different
questions' answers.

This re-evaluates every set under one common objective and prints what each one
actually predicts, which is the comparison that means something:

    python -m dislocluster_code.fitting.compare_refits <dir> [<dir> ...]

The last columns are the ones to read. J_A and J_C are per-family errors and do
not depend on the weighting at all; N/N_exp and d/d_exp are the model's
prediction at 573 K / 24.5 dpa against the measurement. A fit is not better
because its J is lower under its own weights.
"""
from __future__ import annotations

import sys

import dislocluster_code.fitting.fit_cloops as F
from dislocluster_code.fitting.refit_report import load

#: The common scale. Log-ratio diameter (so both terms are commensurate) and
#: equal family weights -- NOT because that weighting is right, but because a
#: comparison needs one fixed scale and this is the neutral one.
COMMON = dict(diam_log=True, c_weight=1.0)

T_REF, DPA_REF = 573.0, 24.5
N_EXP, D_EXP = 8.0e20, 95.0


def evaluate(params):
    J, bd = F.objective_full(params, want_breakdown=True)
    hist = F.model_history_batch(params, [(T_REF, DPA_REF)])[T_REF]
    if hist is None:
        return J, bd, float('nan'), float('nan')
    dose, ser = hist
    N, d = F._sample(dose, ser['C'][0], ser['C'][1], [DPA_REF])
    return J, bd, float(N[0] / N_EXP), float(d[0] / D_EXP)


def main(argv=None):
    dirs = list(argv if argv is not None else sys.argv[1:])
    if not dirs:
        raise SystemExit(__doc__)
    F.set_model(loop_model=1, n_fam=9, moments=1, emission_model=1,
                variant_weights=(1 / 3, 1 / 3, 1 / 3))
    F.DIAM_LOG = COMMON['diam_log']
    F.WLOOP['C'] = COMMON['c_weight']

    print("  scored on ONE objective: log-ratio diameter, equal family weights")
    print(f"  <c> prediction quoted at {T_REF:.0f} K / {DPA_REF} dpa "
          f"(N {N_EXP:.1e} m^-3, d {D_EXP:.0f} nm)\n")
    print(f"  {'set':<34}{'J':>8}{'J_A':>8}{'J_C':>8}{'J_os':>8}"
          f"{'N/Nexp':>9}{'d/dexp':>9}")
    print("  " + "-" * 82)

    rows = []
    base = dict(F.OPT)
    J, bd, n, d = evaluate(base)
    print(f"  {'stale set (reference)':<34}{J:>8.4f}{bd['A']['J']:>8.4f}"
          f"{bd['C']['J']:>8.4f}{bd['J_os']:>8.4f}{n:>9.2f}{d:>9.3f}")
    for where in dirs:
        params, meta = load(where)
        J, bd, n, d = evaluate(params)
        tag = where if len(where) <= 33 else '...' + where[-30:]
        print(f"  {tag:<34}{J:>8.4f}{bd['A']['J']:>8.4f}{bd['C']['J']:>8.4f}"
              f"{bd['J_os']:>8.4f}{n:>9.2f}{d:>9.3f}")
        rows.append((where, params, J, bd, n, d))

    print("\n  J_A and J_C are per-family and weighting-independent; J is not.")
    print("  A <c> diameter near 1.0 is the thing no earlier fit could reach.")
    return rows


if __name__ == '__main__':
    main()
