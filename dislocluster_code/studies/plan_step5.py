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


def check_realizability(run_dir):
    """Goal (vi): Delta_k >= 1 survives a step with EVERY channel active.

    Delta = q n / c^2 is a ratio of moments of a non-negative measure, so
    Delta < 1 violates Cauchy-Schwarz and q < 0 is impossible outright. Nothing
    in an independently integrated moment set enforces either, so it has to be
    tested -- and the earlier goals cannot see it, because each of them switches
    the other channels OFF to isolate one term.

    This runs the real states from the reference march with coalescence,
    nucleation, emission and the basal chain all live. It is the test that
    catches an inconsistent channel: coalescence MERGES loops rather than
    removing them, so it must raise the second moment, and subtracting
    coal_num <m^2> instead drove q through zero on 40% of the c_p nodes of a
    nine-family march.

    EMISSION IS ON, which is the configuration the plan specifies -- step 3
    comes before step 5. It matters: emission_model = 1 deletes the annealing
    LIFETIME surrogate, which debits content at the fixed size n_vL_nuc and is
    incompatible with any moment closure once the family mean drifts away from
    it. See the note at `qann` in rate_equations_core.h; carrying a second
    moment against the pre-step-3 annealing is not a combination the plan asks
    for, and this goal does not pretend otherwise.
    """
    print("GOAL (vi)   Delta >= 1 with every channel live")
    from dislocluster_code.studies.plan_step4 import _seed as _seed4
    y = _seed4(n_c0=1e-9, m_c0=200.0, n_cf=2e-10, m_cf=500.0)
    y = np.concatenate([y, np.zeros(N_STATE - y.shape[0])])
    # Seed every family with a spread population and a mobile matrix that makes
    # them shrink, which is the direction that stresses q.
    for slot, n, mb, D in ((0, 1e-9, 500.0, 1.5), (1, 5e-10, 400.0, 1.8),
                           (4, 3e-10, 200.0, 2.0), (7, 2e-10, 600.0, 1.2),
                           (8, 1e-9, 200.0, 1.4)):
        y[n_idx(slot)] = n
        y[c_idx(slot)] = n * mb
        y[q_idx(slot)] = D * (n * mb) ** 2 / n
    y[0], y[1] = 1e-8, 1e-12
    y[18] = 1.88e14

    from dislocluster_code.studies.plan_step3 import T_REF
    sim = dcfg.sim_for_run(run_dir)
    cli = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-8, atol=1e-30, stats=True, loop_model=1,
        material_file=paths.MODELIB_MATERIAL, n_fam=9, moments=1, m_min=10.0,
        variant_weights=(1 / 3, 1 / 3, 1 / 3), basal_chain=1,
        emission_model=1, temperature_K=T_REF,
        eps_sfp=1e-10, n_sfp_nuc=100.0, tau_sfp=1e6,
        nu_col=1e-6, nu_uf=3e-7, m_col=150.0, m_uf=400.0))

    ok, worst = True, {}
    ys = [y]
    for dt in (1e5, 1e6, 1e7, 1e7, 1e7):
        out = imm.run_immobile_step(cli, ys[-1:], 0.0, dt, retries=0)[0]
        if out is None:
            print("  integration FAILED")
            return False
        ys.append(out)
    for i, out in enumerate(ys[1:], 1):
        for k in range(9):
            n, c, q = out[n_idx(k)], out[c_idx(k)], out[q_idx(k)]
            if n <= 0 or c <= 0:
                continue
            D = q * n / c ** 2
            if q < 0.0 or D < 1.0 - 1e-9:
                ok = False
            key = f"slot {k}"
            if key not in worst or D < worst[key]:
                worst[key] = D
    for key in sorted(worst, key=lambda s: int(s.split()[1])):
        print(f"  {key}: min Delta over the sequence = {worst[key]:.6f}")
    print(f"  q >= 0 and Delta >= 1 everywhere : {ok}")
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


def _gate_closed(m_theta, mbar, Delta, j):
    """Eq. (gatefraction), the closed form the C++ evaluates."""
    from math import erfc, log, sqrt
    s2 = log(Delta)
    s = sqrt(s2)
    mu = log(mbar) - 0.5 * s2
    return 0.5 * erfc((log(m_theta) - mu - j * s2) / (s * sqrt(2.0)))


def _gate_quadrature(m_theta, mbar, Delta, j):
    """The same fraction by direct quadrature of the log-normal.

    Independent of erfc -- it integrates m^j f(m) over [m_theta, inf) on a grid
    in x = ln m, where the density is Gaussian -- so it tests the CLOSED FORM
    and not merely the C++ transcription of it.
    """
    s2 = np.log(Delta)
    s = np.sqrt(s2)
    mu = np.log(mbar) - 0.5 * s2
    def _int(lo, hi):
        # The truncation point is made a GRID ENDPOINT rather than being masked
        # out of a common grid: a np.where step lands mid-cell and trapezoid
        # across the discontinuity is O(h) wrong, which showed up as a 5e-6
        # disagreement with the closed form and read as a physics discrepancy.
        x = np.linspace(lo, hi, 200001)
        w = np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
        return float(np.trapezoid(np.exp(j * x) * w, x))

    lo, hi = mu - 16.0 * s, mu + 16.0 * s
    return _int(max(np.log(m_theta), lo), hi) / _int(lo, hi)


def check_transfer_gates(run_dir):
    """Goal (v): the moment-weighted transfer gates of Eq. (gatefraction).

    The three gates are measured OPERATIONALLY -- as the fraction of each moment
    a short transfer actually removes from the source family -- and compared
    with the closed form and with direct quadrature. The sharp part is the
    ordering Phi2 > Phi1 > Phi0: it says the converting subpopulation is larger
    than the family mean, which is what a size-gated reaction does and what one
    rate multiplying all three moments cannot express.
    """
    print("GOAL (v)    the moment-weighted transfer gates")
    from dislocluster_code.studies.plan_step4 import SLOT_C0, SLOT_CF

    mbar, Delta, m_theta = 400.0, 1.6, 800.0
    # nu*dt sets the measurement, and it is squeezed from both sides: too large
    # and the source's own mu and Delta drift within the step, so the gate that
    # acted is not the one evaluated at the seed; too small and the difference
    # is lost in the solver's 11-significant-digit stdout. 1e-4 leaves 7 digits
    # on the difference and moves the gate by ~1e-4 of itself.
    nu, dt = 1.0e-4, 1.0
    n0 = 1e-10

    # The pyramid alone, with every other channel off: no cascade source, no
    # dissolution, no unfaulting, and a mobile matrix at zero so nothing is
    # captured. What is left is the one transfer this goal is about.
    ov = dict(QUIET)
    ov.update({"omega_i": 0.0, "omega_2i": 0.0, "omega_3i": 0.0})
    sim = dcfg.sim_for_run(run_dir)
    base = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-13, atol=1e-40, stats=True, loop_model=1,
        material_file=paths.MODELIB_MATERIAL, n_fam=9,
        moments=1, m_min=1.0, basal_chain=1,
        variant_weights=(1 / 3, 1 / 3, 1 / 3),
        eps_sfp=0.0, n_sfp_nuc=1.0, tau_sfp=0.0, nu_col=nu, nu_uf=0.0,
        m_col=m_theta, m_uf=0.0))
    d = {}
    for a in base:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            d[k] = v
    d.update({k: repr(float(v)) for k, v in ov.items()})
    cli = [f"--{k}={v}" for k, v in d.items()]

    y = _seed(slot=SLOT_C0, n=n0, mbar=mbar, Delta=Delta)
    out = imm.run_immobile_step(cli, [y], 0.0, dt, retries=0)[0]
    if out is None:
        print("  integration FAILED")
        return False

    ok = True
    meas, closed, quad = {}, {}, {}
    for j, idx in ((0, n_idx), (1, c_idx), (2, q_idx)):
        src = idx(SLOT_C0)
        # The transfer is a pure decay at rate nu*Phi, so the fraction removed
        # is 1 - exp(-nu Phi dt) and the logarithm inverts it exactly. Taking
        # the fraction itself would leave an O(nu Phi dt/2) bias -- a systematic
        # UNDER-reading of every gate, in the same direction for all three, so
        # the ordering would survive it and the agreement would not.
        meas[j] = -np.log(1.0 - (y[src] - out[src]) / y[src]) / (nu * dt)
        closed[j] = _gate_closed(m_theta, mbar, Delta, j)
        quad[j] = _gate_quadrature(m_theta, mbar, Delta, j)
        rel = abs(meas[j] - closed[j]) / closed[j]
        rq = abs(quad[j] - closed[j]) / closed[j]
        print(f"  Phi^({j})  measured {meas[j]:.9f}   closed {closed[j]:.9f}"
              f"   quadrature {quad[j]:.9f}")
        print(f"          solver vs closed {rel:.2e}   closed vs quadrature "
              f"{rq:.2e}")
        ok = ok and rel < 1e-3 and rq < 1e-7

    order = closed[2] > closed[1] > closed[0] and meas[2] > meas[1] > meas[0]
    print(f"  Phi^(2) > Phi^(1) > Phi^(0) : {order}")
    ok = ok and order

    # The transferred population is larger than the family it left, which is the
    # physical content of the ordering.
    m_x = ((y[c_idx(SLOT_C0)] - out[c_idx(SLOT_C0)])
           / (y[n_idx(SLOT_C0)] - out[n_idx(SLOT_C0)]))
    print(f"  transferred mean size {m_x:.2f} against family mean {mbar:.2f}"
          f"  ({m_x / mbar:.3f}x)")
    ok = ok and m_x > mbar

    # ...and the source mean therefore FALLS even though nothing shrank: what
    # left was above average. A single rate on all three moments leaves it flat.
    m_src = out[c_idx(SLOT_C0)] / out[n_idx(SLOT_C0)]
    print(f"  source mean {mbar:.6f} -> {m_src:.6f} "
          f"({'falls' if m_src < mbar else 'HOLDS/RISES'})")
    ok = ok and m_src < mbar

    # Conservation: whatever left c_0 arrived at c_f, moment for moment. This is
    # what makes the gates a transfer and not a loss.
    worst = 0.0
    for idx in (n_idx, c_idx, q_idx):
        lost = y[idx(SLOT_C0)] - out[idx(SLOT_C0)]
        gained = out[idx(SLOT_CF)] - y[idx(SLOT_CF)]
        worst = max(worst, abs(lost - gained) / abs(lost))
    print(f"  worst moment-for-moment conservation error : {worst:.3e}")
    # The ODE is conservative by construction -- the same rate on both sides --
    # so this measures the solver and the 11-digit stdout, not the model.
    ok = ok and worst < 1e-5

    print(f"  gates measured, ordered, and conservative : {ok}")
    return ok


def check_gates_off(run_dir):
    """m_col = m_uf = 0 must reproduce step 4's chain bit-for-bit.

    The gates are the one part of step 5 that touches a channel step 4 already
    ran, so this is the regression that says switching them off is a switch and
    not an approximation. It is run WITH moments on, so the only difference from
    the compared command line is the two thresholds.
    """
    print("REGRESSION  gates off (m_col = m_uf = 0) vs the ungated chain")
    from dislocluster_code.studies.plan_step4 import _seed as _seed4
    CHAIN = dict(eps_sfp=1e-12 * 200.0, n_sfp_nuc=200.0, tau_sfp=1e5,
                 nu_col=1e-5, nu_uf=3e-6)
    y = _seed4(n_c0=1e-9, m_c0=200.0, n_cf=2e-10, m_cf=500.0)
    y = np.concatenate([y, np.zeros(N_STATE - y.shape[0])])
    for slot, D in ((8, 1.4), (0, 1.8)):
        if y[c_idx(slot)] > 0:
            y[q_idx(slot)] = D * y[c_idx(slot)] ** 2 / y[n_idx(slot)]

    def _cli(gated):
        sim = dcfg.sim_for_run(run_dir)
        cfg = dict(t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
                   rtol=1e-10, atol=1e-30, stats=True, loop_model=1,
                   material_file=paths.MODELIB_MATERIAL, n_fam=9,
                   moments=1, m_min=1.0, basal_chain=1,
                   variant_weights=(1 / 3, 1 / 3, 1 / 3), **CHAIN)
        if gated:
            cfg.update(m_col=0.0, m_uf=0.0)
        return collect_solver_args(sim, cfg)

    a = imm.run_immobile_step(_cli(False), [y], 0.0, 1e5, retries=0)[0]
    b = imm.run_immobile_step(_cli(True), [y], 0.0, 1e5, retries=0)[0]
    if a is None or b is None:
        print("  integration FAILED")
        return False
    same = bool(np.array_equal(a, b))
    print(f"  bit-for-bit : {same}")
    return same


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
    g5 = check_transfer_gates(run_dir)
    print()
    reg2 = check_gates_off(run_dir)
    print()
    g6 = check_realizability(run_dir)
    print()
    print(f"regression (moments off)     : {reg}")
    print(f"regression (gates off)       : {reg2}")
    print(f"goal (i)   closure identity  : {g1}")
    print(f"goal (ii)  pure spreading    : {g2}")
    print(f"goal (iii) anneal signature  : {g3}")
    print(f"goal (iv)  floor current     : {g4}")
    print(f"goal (v)   transfer gates    : {g5}")
    print(f"goal (vi)  Delta >= 1 always  : {g6}")
    return reg and reg2 and g1 and g2 and g3 and g4 and g5 and g6


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
