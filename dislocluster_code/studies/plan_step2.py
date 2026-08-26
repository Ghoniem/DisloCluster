"""Step 2 of the code implementation plan: character-resolved capture.

The plan inserts the factor ``X_{s,m}`` of Eq. (Zsk), so that every capture
efficiency is

    Z_{sk,m} = Z^0_m  *  A_h(k)(p_m)  *  X_{s,m}  *  [SIPA]

with ``A_h`` breaking the ORIENTATION degeneracy (already carried) and ``X``
breaking the CHARACTER one. ``X`` is parameterised by the single
character-splitting factor ``chi = X_iI X_vV /(X_iV X_vI)``, taking
``chi^(+1/4)`` for like pairs and ``chi^(-1/4)`` for unlike ones.

**Regression:** ``chi = 1`` reproduces step 1 bit-for-bit.

**Validation goal:** the coexistence window opens. Sweeping the mobile flux
ratio ``x = Phi_V / Phi_I`` there must be a finite interval over which the
prismatic interstitial and prismatic vacancy families grow *simultaneously*, of
logarithmic width ``ln chi``.

Run:
    python -m dislocluster_code.studies.plan_step2 verify
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dislocluster_code.coupling import immobile as imm
from dislocluster_code.studies.baseline import resolve_run, REFERENCE_RUN
from dislocluster_code.studies.plan_step1 import _cli, _states, n_idx, c_idx

#: The character splitting used for the window test. Any value > 1 opens a
#: window; this one is large enough that the interval is comfortably wider than
#: the sweep resolution and small enough to stay a perturbation.
CHI_TEST = 4.0

#: Tolerance on the measured window width against ln(chi).
WIDTH_RTOL = 0.02


def _cli_chi(run_dir, chi, n_fam=8, overrides=None):
    ov = dict(overrides or {})
    if chi != 1.0:
        ov["chi"] = chi
    return _cli(run_dir, n_fam=n_fam, overrides=ov)


def check_regression(run_dir, G=1e-7):
    """chi = 1 must reproduce step 1 bit-for-bit.

    Bit-for-bit is achievable here, unlike step 1's, and the reason is that this
    step adds no state: X enters as a multiplicative factor that is EXACTLY 1.0
    when chi is, and multiplying a double by exactly 1.0 returns it unchanged.
    The implicit block keeps its size, so CVODE takes the same steps.
    """
    y0, doses = _states(run_dir)
    t0, t1 = doses[4] / G, doses[5] / G

    # No --chi at all (the pre-step-2 command line) against an explicit 1.0.
    a = imm.run_immobile_step(_cli(run_dir, n_fam=8), y0, t0, t1, retries=0)
    b = imm.run_immobile_step(_cli_chi(run_dir, 1.0), y0, t0, t1, retries=0)
    keep = [i for i in range(len(y0)) if a[i] is not None and b[i] is not None]
    A = np.array([a[i] for i in keep])
    B = np.array([b[i] for i in keep])
    exact = np.array_equal(A, B)
    print("REGRESSION  chi = 1 vs no chi")
    print(f"  states compared : {len(keep)}")
    print(f"  bit-for-bit     : {exact}")
    if not exact:
        d = np.abs(A - B)
        print(f"  max abs diff    : {d.max():.3e}")
    return exact


def efficiency_window(run_dir, chi):
    """The window bounds in closed form, from the efficiencies themselves.

    An interstitial <a> loop grows while its interstitial gain beats its vacancy
    loss, Z_aI,I Phi_I > Z_aI,V Phi_V, i.e. x < Z_aI,I/Z_aI,V. A vacancy <a>
    loop grows while x > Z_aV,I/Z_aV,V. The ratio of the two bounds is chi by
    construction, so the window is non-empty exactly when chi > 1.
    """
    d = {}
    for a in _cli(run_dir, n_fam=8):
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            d[k] = v
    p_v, p_i = float(d["dad_p_v"]), float(d["dad_p_i"])
    Z0_v, Z0_i = float(d["dad_Z0_v"]), float(d["dad_Z0_i"])
    A_a = lambda p: 0.5 * (p + p ** -2)          # noqa: E731  prismatic Woo factor
    base = (Z0_i * A_a(p_i)) / (Z0_v * A_a(p_v))
    hi = base * chi ** 0.5      # upper bound: interstitial loops still grow
    lo = base * chi ** -0.5     # lower bound: vacancy loops start to grow
    return lo, hi


def check_window(run_dir, chi=CHI_TEST):
    """Goal: the coexistence window opens, of logarithmic width ln(chi)."""
    print(f"GOAL        coexistence window at chi = {chi:g}")
    lo1, hi1 = efficiency_window(run_dir, 1.0)
    lo, hi = efficiency_window(run_dir, chi)
    print(f"  chi = 1 : window [{lo1:.6f}, {hi1:.6f}]  "
          f"log-width {np.log(hi1 / lo1):.3e}   (must be EMPTY)")
    print(f"  chi = {chi:g} : window [{lo:.6f}, {hi:.6f}]  "
          f"log-width {np.log(hi / lo):.6f}")
    print(f"  ln(chi)                                = {np.log(chi):.6f}")

    degenerate_closed = abs(np.log(hi1 / lo1)) < 1e-12
    width_ok = abs(np.log(hi / lo) - np.log(chi)) < WIDTH_RTOL * np.log(chi)
    print(f"  chi = 1 closes the window : {degenerate_closed}")
    print(f"  width matches ln(chi)     : {width_ok}")
    return degenerate_closed and width_ok


def check_window_numeric(run_dir, chi=CHI_TEST, G=1e-7, n_x=25):
    """The same window, read off the INTEGRATED sign rather than the formula.

    A closed form that agrees with itself proves nothing. This drives the actual
    ODE: the mobile species are frozen (the operator split freezes them anyway),
    so x = D_v c_v / (D_i c_i) is set by choosing c_v against a fixed c_i, and
    the sign of each family's content change over a short step is read directly.
    """
    print(f"NUMERIC     window swept on the integrated sign, chi = {chi:g}")
    d = {}
    for a in _cli(run_dir, n_fam=8):
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            d[k] = v
    om_v, om_i = float(d["omega_v"]), float(d["omega_i"])

    y0_ref, doses = _states(run_dir, n=1)
    base = np.asarray(y0_ref[0], dtype=float).copy()
    # A state with BOTH prismatic characters populated, and the basal family and
    # clusters out of the way so the sweep measures the prismatic pair alone.
    y = np.zeros(29)
    y[1] = 1e-10                       # Ci
    for k in (1, 2, 3, 4, 5, 6):
        y[n_idx(k)] = 1e-9
        y[c_idx(k)] = 3e-7
    y[18] = base[18]                   # rho_N

    lo, hi = efficiency_window(run_dir, chi)
    xs = np.exp(np.linspace(np.log(lo) - 1.0, np.log(hi) + 1.0, n_x))
    # x = Phi_V/Phi_I = (om_v c_v)/(om_i c_i)  ->  c_v = x c_i om_i/om_v
    states = []
    for x in xs:
        s = y.copy()
        s[0] = x * y[1] * om_i / om_v
        states.append(s)

    # Short step: the SIGN of the content change is what is being read, and a
    # long one would let the populations reshape and change the sign for
    # reasons that have nothing to do with capture.
    t0 = doses[4] / G
    dt = 1e-3 * (doses[5] - doses[4]) / G

    # EVERY OTHER CONTENT CHANNEL OFF. The window is a statement about the
    # capture balance -- gain beats loss -- and nothing else. Left on, cascade
    # nucleation deposits content at a rate independent of x and both families
    # "grow" at every x: measured 25 of 25 sweep points before these overrides
    # went in, which is a vacuous test that looks like a passing one.
    #
    # So: no cascade sources, no homogeneous clustering (C2i = C3i = 0 in the
    # state above), no thermal annealing, no coalescence. What remains in
    # dc/dt is exactly gain - loss.
    quiet = {
        "G_iL": 0.0, "G_aiL": 0.0, "G_vL": 0.0, "G_avL": 0.0,
        "c_LL_a": 0.0, "c_LN_a": 0.0, "c_LL_c": 0.0, "c_LN_c": 0.0,
        "tau_vL": 1e30, "tau_avL": 1e30,
    }
    out = imm.run_immobile_step(
        _cli_chi(run_dir, chi, overrides=quiet), states, t0, t0 + dt,
        retries=0)

    grow_i, grow_v = [], []
    for s, r in zip(states, out):
        if r is None:
            grow_i.append(None), grow_v.append(None)
            continue
        di = sum(r[c_idx(k)] - s[c_idx(k)] for k in (1, 2, 3))
        dv = sum(r[c_idx(k)] - s[c_idx(k)] for k in (4, 5, 6))
        grow_i.append(di > 0.0)
        grow_v.append(dv > 0.0)

    both = [i for i in range(len(xs))
            if grow_i[i] and grow_v[i]]
    print(f"  x swept over [{xs[0]:.4f}, {xs[-1]:.4f}] in {n_x} points")
    print(f"  predicted window            : [{lo:.4f}, {hi:.4f}]")
    if both:
        print(f"  measured co-growth interval : "
              f"[{xs[both[0]]:.4f}, {xs[both[-1]]:.4f}]  "
              f"({len(both)} of {n_x} points)")
    else:
        print("  measured co-growth interval : EMPTY")
    inside = all(lo * 0.5 <= xs[i] <= hi * 2.0 for i in both) if both else False
    print(f"  non-empty and inside the predicted bounds: {bool(both) and inside}")
    return bool(both) and inside


def verify(run=None):
    run_dir = resolve_run(run or REFERENCE_RUN)
    print(f"reference: {run_dir.name}\n")
    reg = check_regression(run_dir)
    print()
    w = check_window(run_dir)
    print()
    try:
        wn = check_window_numeric(run_dir)
    except SystemExit:
        raise
    except Exception as exc:                       # noqa: BLE001
        print(f"  numeric sweep failed: {exc}")
        wn = False
    print()
    print(f"regression bit-for-bit : {reg}")
    print(f"window (closed form)   : {w}")
    print(f"window (integrated)    : {wn}")
    return reg and w


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 2 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify(a.run) else 1)


if __name__ == "__main__":
    main()
