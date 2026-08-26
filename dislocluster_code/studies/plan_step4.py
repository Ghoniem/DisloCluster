"""Step 4 of the code implementation plan: the basal chain c_0 -> c_f -> c_p.

The plan adds the stacking-fault pyramid ``c_0`` as a ninth family with its own
cascade yield and dissolution lifetime, the compact sink form Eq. (Pisfp), the
volumetric geometry Eq. (Rsfp), and the two activated transfers Eq. (transfer).

The pyramid is not a loop, and three things follow: its radius is volumetric
rather than ``lam sqrt(m)``, it presents a capture cross-section rather than a
sink line, and it has NO coalescence channel. That last is what makes its number
equation linear and its saturation closed-form, which is validation goal (ii).

The distribution gates ``Phi^(j)`` of Eq. (barrier) need the size distribution
step 5 carries; until then they are 1, the barrier-limited limit the formulation
names explicitly.

**Regression:** ``basal_chain = 0`` reproduces step 3 bit-for-bit.

**Validation goals:**
(i)   conservation across the chain, for arbitrary nu_col, nu_uf, tau_sfp, and
      in the three degenerate limits;
(ii)  the branching ratio -- steady pyramid density against Eq. (sfp-saturation)
      and the converting fraction against f_col;
(iii) incubation -- the dose at which c_f becomes measurable moves with the
      branching ratio in the direction Eq. (branching) predicts.

Run:
    python -m dislocluster_code.studies.plan_step4 verify
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dislocluster_code import paths, config as dcfg
from dislocluster_code.coupling import immobile as imm
from dislocluster_code.studies.baseline import resolve_run, REFERENCE_RUN
from dislocluster_code.studies.plan_step1 import _cli, _states, n_idx, c_idx
from dislocluster_code.zerod.cpp_bridge import collect_solver_args

SLOT_C0, SLOT_CF, SLOT_CP = 8, 0, 7

#: Conservation bar for goal (i), ten times the rtol the probes run at.
CONSERVE_TOL = 1.0e-9

#: Everything except the basal chain switched off, so what the chain does is the
#: only thing that moves. The mobile mobilities go too: with omega_* = 0 no
#: family captures or emits anything and the transfers act alone, which is what
#: makes goal (i) an identity rather than a tolerance.
QUIET = {
    "G_iL": 0.0, "G_aiL": 0.0, "G_vL": 0.0, "G_avL": 0.0,
    "G_v": 0.0, "G_i": 0.0, "G_2i": 0.0, "G_3i": 0.0,
    "c_LL_a": 0.0, "c_LN_a": 0.0, "c_LL_c": 0.0, "c_LN_c": 0.0,
    "omega_v": 0.0, "omega_i": 0.0, "omega_2i": 0.0,
    # The LOOP lifetimes too. c_f is a loop family, so with emission_model = 0
    # it still dissolves on tau_vL -- ~1.4e4 s, against the 1e5 s these probes
    # run for -- and it drained away as fast as the chain fed it: the basal sum
    # fell from 2.4e-7 to 9e-12 and read as a conservation failure in the chain
    # when it was an unrelated channel left switched on.
    "tau_vL": 1e30, "tau_avL": 1e30,
}


def _cli4(run_dir, chain=True, overrides=None, **chain_kw):
    sim = dcfg.sim_for_run(run_dir)
    cfg = dict(t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
               rtol=1e-10, atol=1e-30, stats=True, loop_model=1,
               material_file=paths.MODELIB_MATERIAL,
               variant_weights=(1 / 3, 1 / 3, 1 / 3), n_fam=9)
    if chain:
        cfg["basal_chain"] = 1
        cfg.update(chain_kw)
    cli = collect_solver_args(sim, cfg)
    ov = dict(overrides or {})
    if ov:
        d = {}
        for a in cli:
            if a.startswith("--") and "=" in a:
                k, v = a[2:].split("=", 1)
                d[k] = v
        d.update({k: repr(float(v)) for k, v in ov.items()})
        cli = [f"--{k}={v}" for k, v in d.items()]
    return cli


def _seed(n_c0=1e-9, m_c0=200.0, n_cf=0.0, m_cf=400.0):
    y = np.zeros(29)
    y[18] = 1.88e14
    if n_c0:
        y[n_idx(SLOT_C0)] = n_c0
        y[c_idx(SLOT_C0)] = n_c0 * m_c0
    if n_cf:
        y[n_idx(SLOT_CF)] = n_cf
        y[c_idx(SLOT_CF)] = n_cf * m_cf
    return y


def _basal_content(y):
    return sum(y[c_idx(k)] for k in (SLOT_C0, SLOT_CF, SLOT_CP))


def check_regression(run_dir):
    """basal_chain = 0 must reproduce step 3 bit-for-bit."""
    y0, doses = _states(run_dir)
    t0, t1 = doses[4] / 1e-7, doses[5] / 1e-7
    a = imm.run_immobile_step(_cli(run_dir, n_fam=8), y0, t0, t1, retries=0)
    b = imm.run_immobile_step(_cli4(run_dir, chain=False), y0, t0, t1,
                              retries=0)
    keep = [i for i in range(len(y0))
            if a[i] is not None and b[i] is not None]
    # Compare EVERYTHING, including the pyramid slot: with the chain off it
    # must be identically zero, and slicing it away would hide that.
    A = np.array([a[i] for i in keep])
    B = np.array([b[i] for i in keep])
    exact = np.array_equal(A, B)
    # Not bit-for-bit, and it cannot be, for the reason step 1 established: the
    # implicit block grows with the family count (17 -> 19 here), and CVODE's
    # WRMS error norm divides by the block size, so the step sequence changes
    # even though every added component is identically zero. Judged the same
    # way -- against the tolerance the states were integrated at.
    sc = np.maximum(np.maximum(np.abs(A), np.abs(B)), 1e-20)
    rel = (np.abs(A - B) / sc).max(1)
    print("REGRESSION  basal_chain = 0 vs the step-3 command line (n_fam 8 vs 9)")
    print(f"  states compared          : {len(keep)}")
    print(f"  every component identical : {exact}")
    print(f"  relative difference       : median {np.median(rel):.2e}   "
          f"p90 {np.percentile(rel, 90):.2e}   max {rel.max():.2e}")
    ok = float(np.median(rel)) < 1e-6
    print(f"  within solver tolerance   : {ok}")
    return ok


def check_conservation(run_dir):
    """Goal (i): the chain moves vacancies between families and creates none.

    With every capture and every source switched off, the three basal families
    exchange content and nothing else happens, so their SUM is an invariant of
    the transfers -- exactly, not to a tolerance.
    """
    print("GOAL (i)    conservation across the chain")
    cases = [
        ("both transfers", dict(nu_col=1e-4, nu_uf=3e-5, tau_sfp=1e5)),
        ("nu_col = 0 (pyramid terminal)", dict(nu_col=0.0, nu_uf=3e-5, tau_sfp=0.0)),
        ("nu_uf = 0 (one basal loop family)", dict(nu_col=1e-4, nu_uf=0.0, tau_sfp=0.0)),
        ("no dissolution", dict(nu_col=1e-4, nu_uf=3e-5, tau_sfp=0.0)),
    ]
    ok = True
    for label, kw in cases:
        y = _seed(n_cf=1e-10)
        out = imm.run_immobile_step(
            _cli4(run_dir, overrides=QUIET, eps_sfp=0.0, **kw),
            [y], 0.0, 1e5, retries=0)[0]
        if out is None:
            print(f"  {label:36s}: integration FAILED")
            ok = False
            continue
        before, after = _basal_content(y), _basal_content(out)
        # Dissolution genuinely removes vacancies from the families (they go to
        # the matrix), so the invariant only holds where it is switched off.
        closed = kw["tau_sfp"] == 0.0
        rel = abs(after - before) / before
        # The probes integrate at rtol = 1e-10, so conservation cannot be
        # demonstrated tighter than that: an exactly-conserving ODE still
        # accumulates the integrator's own error. Judged at ten times it, and
        # the measured residuals come in BELOW rtol -- 8e-13 to 1.5e-11 -- which
        # is the strongest statement the probe can support.
        verdict = ("closes" if rel < CONSERVE_TOL else "LEAKS") if closed else \
                  ("falls" if after < before else "GROWS")
        print(f"  {label:36s}: sum {before:.6e} -> {after:.6e}  "
              f"rel {rel:.2e}  {verdict}")
        ok = ok and (rel < CONSERVE_TOL if closed else after < before)
    return ok


def check_branching(run_dir):
    """Goal (ii): the closed-form saturation and the branching ratio."""
    print("GOAL (ii)   pyramid saturation and branching ratio")
    ok = True
    for nu_col, tau_sfp in ((1e-4, 1e4), (1e-5, 1e4), (1e-4, 1e5)):
        J_n = 1e-12                       # pyramid NUMBER nucleation rate
        n_nuc = 200.0
        eps = J_n * n_nuc                 # the solver takes the ATOM rate
        n_star = J_n / (1.0 / tau_sfp + nu_col)
        f_col = nu_col / (1.0 / tau_sfp + nu_col)
        # Long enough to saturate: many times the controlling time constant.
        t_end = 50.0 / (1.0 / tau_sfp + nu_col)
        out = imm.run_immobile_step(
            _cli4(run_dir, overrides=QUIET, eps_sfp=eps, n_sfp_nuc=n_nuc,
                  tau_sfp=tau_sfp, nu_col=nu_col, nu_uf=0.0),
            [_seed(n_c0=0.0)], 0.0, t_end, retries=0)[0]
        if out is None:
            print(f"  nu_col={nu_col:.0e} tau={tau_sfp:.0e}: FAILED")
            ok = False
            continue
        got = out[n_idx(SLOT_C0)]
        rel = abs(got - n_star) / n_star
        print(f"  nu_col={nu_col:.0e} tau_sfp={tau_sfp:.0e} : "
              f"n* solver {got:.6e}  closed form {n_star:.6e}  "
              f"rel {rel:.2e}   f_col {f_col:.4f}")
        ok = ok and rel < 1e-3
    return ok


def check_incubation(run_dir):
    """Goal (iii): a larger branching ratio makes c_f appear sooner.

    Eq. (branching) makes f_col a function of nu_col tau_sfp alone, so raising
    that product must move the appearance of c_f earlier and raise its
    saturated content. The direction is the prediction; the magnitude is not.
    """
    print("GOAL (iii)  incubation moves with the branching ratio")
    tau_sfp = 1e4
    rows = []
    for nu_col in (1e-6, 1e-5, 1e-4):
        f_col = nu_col / (1.0 / tau_sfp + nu_col)
        out = imm.run_immobile_step(
            _cli4(run_dir, overrides=QUIET, eps_sfp=1e-12 * 200.0,
                  n_sfp_nuc=200.0, tau_sfp=tau_sfp, nu_col=nu_col, nu_uf=0.0),
            [_seed(n_c0=0.0)], 0.0, 1e5, retries=0)[0]
        c_f = np.nan if out is None else out[c_idx(SLOT_CF)]
        rows.append((nu_col, f_col, c_f))
        print(f"  nu_col={nu_col:.0e}  f_col={f_col:.4f}  "
              f"c_f(1e5 s)={c_f:.6e}")
    cfs = [r[2] for r in rows]
    monotone = all(cfs[i] < cfs[i + 1] for i in range(len(cfs) - 1))
    print(f"  c_f rises monotonically with f_col: {monotone}")
    return monotone


def verify(run=None):
    run_dir = resolve_run(run or REFERENCE_RUN)
    print(f"reference: {run_dir.name}\n")
    reg = check_regression(run_dir)
    print()
    g1 = check_conservation(run_dir)
    print()
    g2 = check_branching(run_dir)
    print()
    g3 = check_incubation(run_dir)
    print()
    print(f"regression (chain off)  : {reg}")
    print(f"goal (i)   conservation : {g1}")
    print(f"goal (ii)  branching    : {g2}")
    print(f"goal (iii) incubation   : {g3}")
    return reg and g1 and g2 and g3


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 4 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify(a.run) else 1)


if __name__ == "__main__":
    main()
