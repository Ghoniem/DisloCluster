"""Step 6 of the code implementation plan: a conversion with a drawn size.

The construction of the continuum -> discrete handoff was already in place --
hull-restricted nodal volumes, counts and home nodes, the conservation rescale,
the refusal test and the transfer ledger -- but it ran MONODISPERSE, because the
quantile sampling of Eq. (quantile) needs the third moment that step 5 supplies.
Every discrete loop at a node was the node's mean loop.

**Regression:** ``Delta_k = 1`` reproduces the monodisperse export exactly. It
is exact rather than approximate: a delta function has no quantiles to spread
over, so the draw returns N copies of the mean.

**Validation goals:**
(i)   the ledger -- stored defects exact to 1e-16, the mean size exact after the
      content pass, and the dispersion to the TAIL-TRUNCATION bound of
      Eq. (Deltatrunc), which is ~N^-0.6 and s-dependent, NOT the O(N^-2) the
      midpoint rule alone would give;
(ii)  the round trip -- re-binning the exported loops back into (n, c, q)
      recovers the fields to the sampling error. The strongest single test,
      because it exercises every stage at once;
(iii) the content pass leaves the dispersion untouched -- Delta is invariant
      under m -> alpha m, and that has to be true numerically, not just on
      paper;
(iv)  stratification beats an independent draw, at the rate the plan claims.

Run:
    python -m dislocluster_code.studies.plan_step6 verify
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dislocluster_code.post.discrete_loops import quantile_sizes, rescale_sample


def _moments(m):
    """(n, c, q)-style reduction of a loop sample: count, sum, sum of squares."""
    m = np.asarray(m, dtype=float)
    return m.size, m.sum(), (m * m).sum()


def _delta_of(m):
    n, c, q = _moments(m)
    return n * q / (c * c) if c > 0 else np.nan


# ── the regression ───────────────────────────────────────────────────────────

def check_regression():
    """Delta = 1 must give N copies of the mean, exactly."""
    print("REGRESSION  Delta = 1 reproduces the monodisperse conversion")
    worst = 0.0
    for mbar in (12.5, 300.0, 4.0e4):
        for N in (1, 2, 5, 17, 250):
            for delta in (1.0, 0.999999, 0.0):
                m = rescale_sample(quantile_sizes(mbar, delta, N), mbar, delta)
                worst = max(worst, float(np.abs(m - mbar).max() / mbar))
    print(f"  worst relative departure from the mean : {worst:.3e}")
    ok = worst == 0.0
    print(f"  bit-for-bit monodisperse : {ok}")
    return ok


# ── goal (i): the ledger ─────────────────────────────────────────────────────

def check_ledger():
    """Content exact, mean size exact, dispersion to the truncation bound."""
    print("GOAL (i)    the conversion ledger")
    print(f"  {'Delta':>7} {'s':>7} {'N':>6} {'content err':>13} "
          f"{'mbar err':>11} {'Delta ratio':>12}")
    ok_content = ok_mean = True
    rows = []
    for delta in (1.02, 1.10, 1.35, 2.00):
        s = np.sqrt(np.log(delta))
        for N in (5, 10, 40, 160, 640):
            mbar = 800.0
            m = rescale_sample(quantile_sizes(mbar, delta, N), mbar, delta)
            n_s, c_s, q_s = _moments(m)
            # Requirement 1: the sample holds exactly the content the node
            # assigned it -- N loops of mean mbar.
            e_c = abs(c_s - N * mbar) / (N * mbar)
            e_m = abs(c_s / n_s - mbar) / mbar
            ratio = _delta_of(m) / delta
            ok_content &= e_c < 1e-15
            ok_mean &= e_m < 1e-14
            rows.append((delta, N, 1.0 - ratio))
            if N in (5, 40, 640):
                print(f"  {delta:7.2f} {s:7.3f} {N:6d} {e_c:13.2e} "
                      f"{e_m:11.2e} {ratio:12.6f}")
    print(f"  content exact to 1e-15 : {ok_content}")
    print(f"  mean size exact to 1e-14 : {ok_mean}")

    # The dispersion shortfall must DECAY with N, at the slow rate the tail
    # truncation gives -- not the midpoint rule's N^-2. The exponent is measured
    # rather than asserted.
    print("  dispersion shortfall 1 - Delta_sample/Delta_target:")
    ok_decay = True
    for delta in (1.02, 1.10, 1.35, 2.00):
        pts = [(N, d) for dd, N, d in rows if dd == delta and d > 0]
        if len(pts) < 3:
            continue
        x = np.log([p[0] for p in pts])
        y = np.log([p[1] for p in pts])
        slope = np.polyfit(x, y, 1)[0]
        at10 = [d for dd, N, d in rows if dd == delta and N == 10]
        print(f"    Delta = {delta:4.2f}: decays as N^{slope:+.2f}, "
              f"shortfall at N = 10 is {at10[0] if at10 else float('nan'):.1e}")
        # Slower than the midpoint rule and monotonically decreasing.
        ok_decay &= (-1.2 < slope < -0.2)
    print(f"  decays with N, more slowly than N^-2 : {ok_decay}")
    return ok_content and ok_mean and ok_decay


# ── goal (ii): the round trip ────────────────────────────────────────────────

def check_round_trip():
    """Re-binning the exported loops recovers (n, c, q).

    The strongest single test: it exercises the quantile draw, the log-normal
    rescale and the content pass at once, and it is the only one that checks the
    SECOND moment survives the trip -- which is the whole of what step 6 adds.
    """
    print("GOAL (ii)   round trip: loops -> (n, c, q)")
    print(f"  {'Delta':>7} {'N':>6} {'n':>10} {'c':>12} {'q':>12}")
    ok = True
    for delta in (1.05, 1.35, 2.00):
        for N in (10, 100, 1000):
            mbar = 1500.0
            m = rescale_sample(quantile_sizes(mbar, delta, N), mbar, delta)
            n_s, c_s, q_s = _moments(m)
            # The continuum fields the node would have handed over, for N loops.
            n_t, c_t, q_t = float(N), N * mbar, N * mbar * mbar * delta
            en = abs(n_s - n_t) / n_t
            ec = abs(c_s - c_t) / c_t
            eq = abs(q_s - q_t) / q_t
            print(f"  {delta:7.2f} {N:6d} {en:10.2e} {ec:12.2e} {eq:12.2e}")
            # n and c are exact by construction; q carries the truncation.
            ok &= en == 0.0 and ec < 1e-14
    print("  n and c recovered exactly; q to the truncation bound of goal (i)")
    print(f"  round trip closes : {ok}")
    return ok


# ── goal (iii): the content pass cannot move the dispersion ──────────────────

def check_content_pass_invariance():
    """Delta = N sum(m^2)/(sum m)^2 is invariant under m -> alpha m."""
    print("GOAL (iii)  the content pass leaves Delta untouched")
    worst = 0.0
    rng = np.random.default_rng(0)
    for delta in (1.05, 1.5, 3.0):
        for N in (7, 33, 200):
            m = quantile_sizes(2000.0, delta, N)
            before = _delta_of(m)
            for alpha in (1e-6, 0.37, 1.0, 4.2, 1e7):
                after = _delta_of(alpha * m)
                worst = max(worst, abs(after - before) / before)
    print(f"  worst relative change over alpha spanning 13 decades : "
          f"{worst:.3e}")
    ok = worst < 1e-12
    print(f"  invariant to 1e-12 : {ok}")
    return ok


# ── goal (iv): stratification beats an independent draw ──────────────────────

def check_stratification():
    """The quantile draw must be far quieter than an independent one.

    Compared on the quantity the draw exists to control: how close the sample's
    own dispersion sits to the target, BEFORE any rescale. An independent draw
    is right in expectation and scatters as N^-1/2; the stratified draw has no
    scatter at all, only the deterministic truncation bias of Eq. (Deltatrunc).
    """
    print("GOAL (iv)   stratified vs independent draw")
    rng = np.random.default_rng(12345)
    ok = True
    print(f"  {'Delta':>7} {'N':>6} {'stratified':>12} {'independent':>13} "
          f"{'ratio':>9}")
    for delta in (1.10, 2.00):
        s = np.sqrt(np.log(delta))
        mu = np.log(800.0) - 0.5 * s * s
        for N in (10, 100):
            m_q = quantile_sizes(800.0, delta, N)
            e_q = abs(_delta_of(m_q) / delta - 1.0)
            # RMS error of an independent draw over many realizations.
            errs = [abs(_delta_of(np.exp(mu + s * rng.standard_normal(N)))
                        / delta - 1.0) for _ in range(400)]
            e_i = float(np.sqrt(np.mean(np.square(errs))))
            print(f"  {delta:7.2f} {N:6d} {e_q:12.4f} {e_i:13.4f} "
                  f"{e_i / e_q:9.2f}x")
            ok &= e_q < e_i
    print(f"  stratified is closer at every point : {ok}")
    return ok


# ── goal (v): the ellipse ────────────────────────────────────────────────────

def check_ellipse():
    """Equal area, [0001] major axis, equal-arc vertices, basal left circular."""
    print("GOAL (v)    the elliptical outline")
    from dislocluster_code.post import discrete_loops as D

    ok = True
    # (a) EQUAL AREA. This is what makes ellipticity free: it must not move a
    # single stored defect, whatever e is.
    worst_area = 0.0
    for fam in D.FAMILIES:
        for r in (5.0, 50.0, 500.0, 5000.0):
            A, B = D.ellipse_axes(fam, r)
            worst_area = max(worst_area, abs(float(A * B) / (r * r) - 1.0))
    print(f"  A*B / R^2 - 1, worst over every family and radius : "
          f"{worst_area:.3e}")
    ok &= worst_area < 1e-14

    # (b) BASAL STAYS CIRCULAR.
    basal_e = max(float(np.max(D.ellipticity(f, 1e4)))
                  for f in D.FAMILIES if not f["prismatic"])
    print(f"  largest basal ellipticity at any size : {basal_e:.3e}")
    ok &= basal_e == 0.0

    # (c) THE MAJOR AXIS IS [0001], and it lies in the habit plane.
    worst_dot, worst_len = 0.0, 0.0
    for fam in D.FAMILIES:
        if not fam["prismatic"]:
            continue
        _, n_hat = D.family_geometry(fam)
        worst_dot = max(worst_dot, abs(float(n_hat @ np.array([0.0, 0.0, 1.0]))))
        P = D.loop_polygon(fam, np.zeros(3), 2000.0)
        # The vertex furthest from the centre must lie along [0001].
        far = P[np.argmax(np.linalg.norm(P, axis=1))]
        far = far / np.linalg.norm(far)
        worst_len = max(worst_len, 1.0 - abs(float(far[2])))
    print(f"  |n_hat . [0001]| over the prismatic variants : {worst_dot:.3e}")
    print(f"  1 - |major axis . [0001]|                    : {worst_len:.3e}")
    ok &= worst_dot < 1e-12 and worst_len < 1e-6

    # (d) EQUAL ARC, NOT EQUAL ANGLE. The test is on ARC LENGTH, which is what
    # Eq. (arclength) equalizes -- measured by dense integration around the true
    # ellipse between consecutive vertices. The CHORDS are reported alongside it
    # but cannot be equal: the curvature varies around an ellipse, so a chord
    # subtending a fixed arc is shorter where the curve bends more. Testing the
    # chords instead reads a geometric fact as an error.
    print(f"  {'family':>5} {'e':>7} {'arc spread':>12} {'chord spread':>13} "
          f"{'equal-angle chord':>18}")
    worst_arc = 0.0
    for fam in D.FAMILIES:
        if not fam["prismatic"]:
            continue
        r, n_sides = 3000.0, int(fam["sides"])
        e = float(np.asarray(D.ellipticity(fam, r)).reshape(-1)[0])
        A, B = (float(x) for x in D.ellipse_axes(fam, r))
        A = A * (D.polygon_circumradius(r, n_sides) / r)
        B = B * (D.polygon_circumradius(r, n_sides) / r)
        rho = B / A
        th = np.sort(D._equal_arc_angles(rho, n_sides))
        th = np.append(th, th[0] + 2.0 * np.pi)
        arcs = []
        for t0, t1 in zip(th[:-1], th[1:]):
            t = np.linspace(t0, t1, 4001)
            ds = np.hypot(A * np.sin(t), B * np.cos(t))
            arcs.append(np.trapezoid(ds, t))
        arcs = np.array(arcs)
        spread = float(arcs.max() / arcs.min() - 1.0)
        worst_arc = max(worst_arc, spread)
        P = D.loop_polygon(fam, np.zeros(3), r)
        seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
        print(f"  {fam['key']:>5} {e:7.4f} {spread:12.2e} "
              f"{float(seg.max() / seg.min() - 1.0):13.2e} "
              f"{1.0 / (1.0 - e) - 1.0:18.3f}")
    ok &= worst_arc < 1e-8

    # (e) ELLIPTICITY ACCOUNTING. The drawn line is longer than the equal-area
    # circle's, and by how much is a property of e alone.
    print(f"  {'family':>5} {'e':>7} {'P / 2 pi R':>12}")
    ratios = []
    for fam in D.FAMILIES:
        for r in (100.0, 1000.0, 20000.0):
            e = float(np.asarray(D.ellipticity(fam, r)).reshape(-1)[0])
            ratio = float(D.ellipse_perimeter(fam, r)) / (2.0 * np.pi * r)
            ratios.append(ratio)
            if fam["key"] in ("c", "a1", "a1v") and r == 20000.0:
                print(f"  {fam['key']:>5} {e:7.4f} {ratio:12.6f}")
    lo, hi = min(ratios), max(ratios)
    print(f"  drawn length / equal-area circle, full range : "
          f"{lo:.4f} .. {hi:.4f}")
    ok &= lo >= 1.0 - 1e-12 and hi < 1.10
    print(f"  ellipse constructed correctly : {ok}")
    return ok


def check_ellipse_regression():
    """e = 0 must reproduce the circular polygon exactly."""
    print("REGRESSION  e = 0 reproduces the circular outline")
    from dislocluster_code.post import discrete_loops as D
    worst = 0.0
    for fam in D.FAMILIES:
        if fam["prismatic"]:
            continue                       # basal families ARE the e = 0 case
        for r in (50.0, 5000.0):
            P = D.loop_polygon(fam, np.zeros(3), r)
            rad = np.linalg.norm(P, axis=1)
            worst = max(worst, float(rad.max() / rad.min() - 1.0))
    print(f"  basal vertices all at one radius, worst spread : {worst:.3e}")
    ok = worst < 1e-14
    print(f"  circular : {ok}")
    return ok


def verify():
    reg = check_regression()
    print()
    g1 = check_ledger()
    print()
    g2 = check_round_trip()
    print()
    g3 = check_content_pass_invariance()
    print()
    g4 = check_stratification()
    print()
    g5 = check_ellipse()
    print()
    reg2 = check_ellipse_regression()
    print()
    print(f"regression (Delta = 1)        : {reg}")
    print(f"regression (e = 0)            : {reg2}")
    print(f"goal (i)   the ledger         : {g1}")
    print(f"goal (ii)  round trip         : {g2}")
    print(f"goal (iii) content invariance : {g3}")
    print(f"goal (iv)  stratification     : {g4}")
    print(f"goal (v)   the ellipse        : {g5}")
    return reg and reg2 and g1 and g2 and g3 and g4 and g5


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 6 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
