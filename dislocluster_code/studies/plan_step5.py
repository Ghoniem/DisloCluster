"""Step 5 of the code implementation plan: the second content moment.

The plan adds a third field ``q^k_sL`` per family and closes the moment hierarchy
log-normally. With ``mbar = c/n`` and ``Delta = q n / c^2 = e^{s^2}``, every
closure integral is one expression,

    <m^j> = mbar^j Delta^{j(j-1)/2}      for every real j,

which returns ``mbar`` at j = 1, ``q/n`` at j = 2, and -- the reason the closure
is log-normal rather than Gaussian -- the HALF-INTEGER moments the growth law
needs, on a strictly positive support. The whole effect of the spread on growth
and on the sink strengths is then one factor ``Delta^{-1/8}``.

**Regression:** ``moments = 0`` reproduces step 4 bit-for-bit.

**Validation goals:**
(i)   the closure identity, against direct quadrature of the log-normal at
      j = 1/2, 1, 3/2, 2, 3 over a range of Delta, to 1e-12;
(ii)  pure spreading -- with the net drift zero, mbar and n hold while Delta
      grows at the rate Eq. (q-FP) gives;
(iii) the annealing signature -- density FALLING with the surviving mean size
      RISING, which the two-moment model provably cannot produce;
(iv)  the floor current vanishes correctly -- Phi_min -> 0 exponentially as
      Delta -> 1 at fixed mbar > m_min.

Run:
    python -m dislocluster_code.studies.plan_step5 verify
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

N_STATE = 38
IDX_Q = 29


def q_idx(k):
    return IDX_Q + k


#: Everything but the family's own capture switched off.
QUIET = {
    "G_iL": 0.0, "G_aiL": 0.0, "G_vL": 0.0, "G_avL": 0.0,
    "G_v": 0.0, "G_i": 0.0, "G_2i": 0.0, "G_3i": 0.0,
    "c_LL_a": 0.0, "c_LN_a": 0.0, "c_LL_c": 0.0, "c_LN_c": 0.0,
    "tau_vL": 1e30, "tau_avL": 1e30,
}


def _cli5(run_dir, moments=True, overrides=None, m_min=1.0):
    sim = dcfg.sim_for_run(run_dir)
    cfg = dict(t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
               rtol=1e-10, atol=1e-30, stats=True, loop_model=1,
               material_file=paths.MODELIB_MATERIAL,
               variant_weights=(1 / 3, 1 / 3, 1 / 3), n_fam=9)
    if moments:
        cfg["moments"] = 1
        cfg["m_min"] = m_min
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


# ── goal (i): the closure identity, no solver involved ───────────────────────

def check_closure():
    """<m^j> = mbar^j Delta^{j(j-1)/2} against direct quadrature.

    The log-normal is integrated numerically in ln m, where it is Gaussian and a
    Gauss-Hermite rule is exact to machine precision for a polynomial times the
    weight -- and m^j = e^{j x} is entire, so a high-order rule converges
    geometrically. This is the identity every other closed form in the step
    rests on, so it is checked first and hardest.
    """
    print("GOAL (i)    closure identity vs direct quadrature")
    xs, ws = np.polynomial.hermite_e.hermegauss(200)
    worst = 0.0
    for Delta in (1.05, 1.25, 2.0, 4.0, 10.0):
        s2 = np.log(Delta)
        s = np.sqrt(s2)
        for mbar in (50.0, 400.0):
            mu = np.log(mbar) - 0.5 * s2
            for j in (0.5, 1.0, 1.5, 2.0, 3.0):
                # <m^j> per loop = int e^{j x} N(x; mu, s) dx, x = ln m
                quad = float(np.sum(ws * np.exp(j * (mu + s * xs)))
                             / np.sqrt(2.0 * np.pi))
                closed = mbar ** j * Delta ** (j * (j - 1) / 2.0)
                rel = abs(quad - closed) / abs(closed)
                worst = max(worst, rel)
    print(f"  max relative error over Delta 1.05..10, j = 1/2..3 : {worst:.3e}")
    ok = worst < 1e-12
    print(f"  within 1e-12 : {ok}")
    return ok


# ── the solver-side probes ───────────────────────────────────────────────────

def _seed(slot=1, n=1e-10, mbar=400.0, Delta=1.0):
    y = np.zeros(N_STATE)
    y[18] = 1.88e14
    y[n_idx(slot)] = n
    y[c_idx(slot)] = n * mbar
    y[q_idx(slot)] = Delta * (n * mbar) ** 2 / n
    return y


def _delta(y, slot):
    n, c, q = y[n_idx(slot)], y[c_idx(slot)], y[q_idx(slot)]
    return q * n / c ** 2 if c > 0 else np.nan


def check_regression(run_dir):
    """moments = 0 must reproduce step 4 bit-for-bit."""
    y0, doses = _states(run_dir)
    t0, t1 = doses[4] / 1e-7, doses[5] / 1e-7
    # Against STEP 4's command line, not step 3's. Comparing to n_fam = 8 would
    # change the implicit block size (17 vs 19) and reintroduce the WRMS
    # step-sequence difference that has nothing to do with this step. At
    # n_fam = 9 with moments off the block is identical, so this regression CAN
    # be bit-for-bit -- and has to be, since q is simply not integrated.
    from dislocluster_code.studies.plan_step4 import _cli4
    a = imm.run_immobile_step(_cli4(run_dir, chain=False), y0, t0, t1,
                              retries=0)
    b = imm.run_immobile_step(_cli5(run_dir, moments=False), y0, t0, t1,
                              retries=0)
    keep = [i for i in range(len(y0)) if a[i] is not None and b[i] is not None]
    A = np.array([a[i] for i in keep])
    B = np.array([b[i] for i in keep])
    exact = np.array_equal(A, B)
    print("REGRESSION  moments = 0 vs step 4 (both n_fam = 9)")
    print(f"  states compared : {len(keep)}")
    print(f"  bit-for-bit     : {exact}")
    if not exact:
        sc = np.maximum(np.maximum(np.abs(A), np.abs(B)), 1e-20)
        rel = (np.abs(A - B) / sc).max(1)
        print(f"  relative diff   : median {np.median(rel):.2e}  "
              f"max {rel.max():.2e}")
    return exact


def check_spreading(run_dir):
    """Goal (ii): with the drift zero, n and mbar hold while Delta grows.

    The net arrival is made zero by putting the matrix at the family's own
    equilibrium -- which is what step 3's detailed balance bought -- so the
    Fokker-Planck drift vanishes and only the spreading term acts.
    """
    print("GOAL (ii)   pure spreading: n and mbar hold, Delta grows")
    from dislocluster_code.studies import loop_annealing as la
    from dislocluster_code.studies.plan_step3 import SLOT_FAMILY, T_REF
    mat = la.load_material()
    # The BASAL family, not a prismatic one. The spreading term is
    # proportional to the gross traffic, which at the family's own equilibrium
    # is 2 c^eq -- and c_f's equilibrium is ~1e-12 against a_i's ~1e-16, four
    # decades more signal for the same window. Under-driving the probe is what
    # made Delta look frozen at 1.020000 when it was moving in the 9th digit.
    slot = 0
    fam = la.families(mat)[SLOT_FAMILY[slot]]
    mbar0 = 400.0
    R = fam.lam * np.sqrt(mbar0)
    cv_eq = la.c_eq_vacancy(R, fam, T_REF, mat)

    y = _seed(slot=slot, mbar=mbar0, Delta=1.02)
    y[0] = cv_eq
    y[1] = 1e-30
    ov = dict(QUIET)
    ov.update({"omega_i": 0.0, "omega_2i": 0.0})
    cfg = dict(emission_model=1, temperature_K=T_REF)
    cli = _cli5(run_dir, overrides=ov)
    # emission has to be on, or there is no equilibrium to sit at.
    sim = dcfg.sim_for_run(run_dir)
    base = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-10, atol=1e-30, stats=True, loop_model=1,
        material_file=paths.MODELIB_MATERIAL, n_fam=9, moments=1, m_min=1.0,
        variant_weights=(1 / 3, 1 / 3, 1 / 3), **cfg))
    d = {}
    for a in base:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            d[k] = v
    d.update({k: repr(float(v)) for k, v in ov.items()})
    cli = [f"--{k}={v}" for k, v in d.items()]

    out = imm.run_immobile_step(cli, [y], 0.0, 1e8, retries=0)[0]
    if out is None:
        print("  integration FAILED")
        return False
    dn = abs(out[n_idx(slot)] - y[n_idx(slot)]) / y[n_idx(slot)]
    m0, m1 = y[c_idx(slot)] / y[n_idx(slot)], out[c_idx(slot)] / out[n_idx(slot)]
    dm = abs(m1 - m0) / m0
    D0, D1 = _delta(y, slot), _delta(out, slot)
    print(f"  n     : {y[n_idx(slot)]:.6e} -> {out[n_idx(slot)]:.6e}  "
          f"rel {dn:.2e}")
    print(f"  mbar  : {m0:.4f} -> {m1:.4f}  rel {dm:.2e}")
    print(f"  Delta : {D0:.12f} -> {D1:.12f}  "
          f"({'grows' if D1 > D0 else 'FALLS'})")
    ok = dn < 1e-6 and dm < 1e-3 and D1 > D0
    print(f"  n and mbar hold, Delta grows : {ok}")
    return ok


def check_floor_current(run_dir):
    """Goal (iv): Phi_min -> 0 exponentially as Delta -> 1.

    A Dirac delta has no lower tail to lose. The current is measured as the
    density loss rate over a short step, at fixed mbar well above m_min.
    """
    print("GOAL (iv)   the floor current vanishes as Delta -> 1")
    slot, mbar, dt = 1, 400.0, 1e6
    rows = []
    for Delta in (1.001, 1.05, 1.5, 4.0):
        y = _seed(slot=slot, mbar=mbar, Delta=Delta)
        # A REAL vacancy concentration. At the concentration floor there is no
        # traffic at all, so every current -- including the one being measured
        # -- came out identically zero and the test read as a failure of the
        # floor current rather than of the probe.
        y[0] = 1e-14
        y[1] = 1e-30
        ov = dict(QUIET)
        ov.update({"omega_i": 0.0, "omega_2i": 0.0})
        out = imm.run_immobile_step(
            _cli5(run_dir, overrides=ov, m_min=20.0), [y], 0.0, dt, retries=0)[0]
        phi = np.nan if out is None else \
            (y[n_idx(slot)] - out[n_idx(slot)]) / dt
        rows.append((Delta, phi))
        print(f"  Delta = {Delta:6.3f} : Phi_min = {phi:.6e}")
    vals = [r[1] for r in rows]
    # "Exponentially" is not a figure of speech here: at Delta = 1.001 the
    # log-normal puts the floor 94 standard deviations into the lower tail, so
    # exp(-94^2/2) UNDERFLOWS and the current is exactly 0.0. Demanding strict
    # monotonicity would fail on 0 < 0; what the goal asserts is that it does
    # not decrease, and that the small-Delta end is negligible against the large.
    ok = (all(np.isfinite(v) for v in vals)
          and all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
          and vals[-1] > 0.0
          and abs(vals[0]) < 1e-6 * abs(vals[-1]))
    print(f"  non-decreasing in Delta, and vanishing as Delta -> 1: {ok}")
    return ok


def check_annealing_signature(run_dir):
    """Goal (iii): density FALLING with the surviving mean size RISING.

    The two-moment model provably cannot produce this -- it gives constant
    density and a falling mean -- so it is the qualitative signature the whole
    step exists for. The mechanism is that the SMALLEST loops are the ones that
    reach the floor.
    """
    print("GOAL (iii)  the annealing signature")
    from dislocluster_code.studies import loop_annealing as la
    from dislocluster_code.studies.plan_step3 import SLOT_FAMILY, T_REF
    mat = la.load_material()
    slot = 0
    # The matrix is put at the family's OWN equilibrium, so the net drift is
    # zero and the loops are not shrinking. The only loss is then the lower tail
    # crossing the floor -- which is the mechanism the signature is about, and
    # the only way to see it isolated. Driven below equilibrium instead, every
    # loop shrinks and the mean falls for a reason that has nothing to do with
    # the distribution: measured 400.0 -> 398.46, density falling AND mean
    # falling, which is not the signature.
    fam = la.families(mat)[SLOT_FAMILY[slot]]
    mbar0 = 400.0
    y = _seed(slot=slot, n=1e-10, mbar=mbar0, Delta=1.6)
    y[0] = la.c_eq_vacancy(fam.lam * np.sqrt(mbar0), fam, T_REF, mat)
    y[1] = 1e-30
    ov = dict(QUIET)
    ov.update({"omega_i": 0.0, "omega_2i": 0.0})
    sim = dcfg.sim_for_run(run_dir)
    base = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-10, atol=1e-30, stats=True, loop_model=1,
        material_file=paths.MODELIB_MATERIAL, n_fam=9, moments=1, m_min=100.0,
        variant_weights=(1 / 3, 1 / 3, 1 / 3),
        emission_model=1, temperature_K=T_REF))
    d = {}
    for a in base:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            d[k] = v
    d.update({k: repr(float(v)) for k, v in ov.items()})
    cli = [f"--{k}={v}" for k, v in d.items()]

    out = imm.run_immobile_step(cli, [y], 0.0, 1e8, retries=0)[0]
    if out is None:
        print("  integration FAILED")
        return False
    n0, n1 = y[n_idx(slot)], out[n_idx(slot)]
    m0 = y[c_idx(slot)] / n0
    m1 = out[c_idx(slot)] / n1
    print(f"  n    : {n0:.6e} -> {n1:.6e}  ({'falls' if n1 < n0 else 'HOLDS/RISES'})")
    print(f"  mbar : {m0:.4f} -> {m1:.4f}  ({'rises' if m1 > m0 else 'falls'})")
    ok = n1 < n0 and m1 > m0
    print(f"  density falls AND surviving mean rises : {ok}")
    if not ok:
        print("  (the signature is a property of the MIXTURE as well as the "
              "model -- see the A1/A2 pair of tab:anneal-tests)")
    return ok


def verify(run=None):
    run_dir = resolve_run(run or REFERENCE_RUN)
    print(f"reference: {run_dir.name}\n")
    reg = check_regression(run_dir)
    print()
    g1 = check_closure()
    print()
    g2 = check_spreading(run_dir)
    print()
    g4 = check_floor_current(run_dir)
    print()
    g3 = check_annealing_signature(run_dir)
    print()
    print(f"regression (moments off)     : {reg}")
    print(f"goal (i)   closure identity  : {g1}")
    print(f"goal (ii)  pure spreading    : {g2}")
    print(f"goal (iii) anneal signature  : {g3}")
    print(f"goal (iv)  floor current     : {g4}")
    return reg and g1 and g2 and g3 and g4


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify(a.run) else 1)


if __name__ == "__main__":
    main()
