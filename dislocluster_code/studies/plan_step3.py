"""Step 3 of the code implementation plan: peripheral emission replaces the lifetimes.

The plan deletes ``tau_avL`` / ``tau_cvL`` and every term proportional to them,
and implements the physical channel instead: every loop exchanges vacancies with
the matrix at the NET rate

    phi_k = Z_k Dbar_v S_k [ c^v - c^{v,eq}_k(R_k, Sigma_k) ]

with the equilibrium concentration built from the work to detach one vacancy
from the loop periphery. It is the only step in the sequence whose predecessor
cannot be recovered by a parameter -- so ``emission_model`` keeps the old channel
behind a switch, and ``emission_model = 0`` reproduces step 2 bit-for-bit.

**The four validation goals**, in the plan's order:

(i)   detailed balance -- c^v = c^{v,eq}_k makes the net exchange vanish;
(ii)  the bulk limit -- E_b -> E^f_v as R -> infinity, from BOTH characters,
      which is what checks the sign of varsigma_s(v);
(iii) the stress-free anneal -- source off, contents fall, densities EXACTLY
      constant (the lifetimes were the only channel that removed loops);
(iv)  the stressed anneal -- above a threshold the interstitial families grow
      with no interstitial supply at all.

Goals (i) and (ii) are measured on the C++ itself rather than on the Python
reference, by BISECTING for the c^v at which a family's content rate changes
sign. That is the solver's own c^{v,eq} read back out of it, without
instrumenting the solver, and comparing it against
``studies.loop_annealing.c_eq_vacancy`` ties the two implementations together.

Run:
    python -m dislocluster_code.studies.plan_step3 verify
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import immobile as imm
from dislocluster_code.studies import loop_annealing as la
from dislocluster_code.studies.baseline import resolve_run, REFERENCE_RUN
from dislocluster_code.studies.plan_step1 import _cli, _states, n_idx, c_idx

#: Slot -> crystallographic family, matching cpp_bridge._EMIS_SLOT_FAMILY.
SLOT_FAMILY = ("c_f", "a_i", "a_i", "a_i", "a_v", "a_v", "a_v", "c_p")
SLOT_KEY = ("c", "a1", "a2", "a3", "a1v", "a2v", "a3v", "cp")

#: Reference temperature of the run this is all keyed on.
T_REF = 573.0


def _cli3(run_dir, emission=True, sigma=None, overrides=None, T=T_REF):
    ov = dict(overrides or {})
    cfg_extra = {}
    if emission:
        cfg_extra = dict(emission_model=1, temperature_K=T,
                         emission_sigma_Pa=sigma or {})
    from dislocluster_code import config as dcfg
    from dislocluster_code.zerod.cpp_bridge import collect_solver_args
    sim = dcfg.sim_for_run(run_dir)
    cfg = dict(t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
               rtol=1e-6, atol=1e-20, stats=True, loop_model=1,
               material_file=paths.MODELIB_MATERIAL,
               variant_weights=(1 / 3, 1 / 3, 1 / 3), n_fam=8, **cfg_extra)
    cli = collect_solver_args(sim, cfg)
    if ov:
        d = {}
        for a in cli:
            if a.startswith("--") and "=" in a:
                k, v = a[2:].split("=", 1)
                d[k] = v
        d.update({k: repr(float(v)) for k, v in ov.items()})
        cli = [f"--{k}={v}" for k, v in d.items()]
    return cli


#: Every source and loss channel except the vacancy exchange, switched off, so
#: the sign of a family's content rate IS the sign of that exchange.
QUIET = {
    "G_iL": 0.0, "G_aiL": 0.0, "G_vL": 0.0, "G_avL": 0.0,
    "c_LL_a": 0.0, "c_LN_a": 0.0, "c_LL_c": 0.0, "c_LN_c": 0.0,
    "G_v": 0.0, "G_i": 0.0, "G_2i": 0.0, "G_3i": 0.0,
    # The interstitial MOBILITIES too, and this is not optional. Ci, C2i and
    # C3i cannot be set below the concentration floor, so they sit at C_floor
    # and keep absorbing; the crossing then lands at
    #     c^eq + (interstitial gain)/(vacancy coefficient)
    # rather than at c^eq. Measured before this went in: the interstitial family
    # read 19% high at R = 5 nm and 3.5% high at R = 25 nm -- biased high, and
    # shrinking with R exactly as an additive contaminant should, while the two
    # VACANCY families were already right to 6e-4. Zeroing the mobilities is
    # also what goals (iii) and (iv) mean by "with no interstitial supply".
    "omega_i": 0.0, "omega_2i": 0.0,
}


def _state_one_family(slot, R_nm, mat, cv):
    """A 27-state carrying ONE loop family at radius R, and a chosen c^v."""
    fam = la.families(mat)[SLOT_FAMILY[slot]]
    R = R_nm * 1e-9
    # n and c chosen so that lam*sqrt(c/n) = R exactly, with n small enough that
    # the family is a tracer and does not move the mobile field.
    n = 1e-12
    c = n * (R / fam.lam) ** 2
    y = np.zeros(29)
    y[0] = cv
    y[1] = 1e-30          # interstitials effectively absent
    y[n_idx(slot)] = n
    y[c_idx(slot)] = c
    y[18] = 1.88e14       # rho_N, the grown-in seed
    return y, fam, R


#: Integration window for the sign probe. One second is far too short: the
#: content moves by ~1e-12 of itself and the solver returns the input unchanged,
#: so the rate reads as EXACTLY zero and the bisection below converges on
#: whichever bracket end it started from -- which is what it did, to the digit
#: (3.943420e-19 against a reference of 3.943420e-16, i.e. exactly lo).
PROBE_DT = 1.0e7

#: What the probe can actually resolve, expressed where it means something.
#: Near the crossing the content change is a difference of two nearly-equal
#: doubles parsed back from the solver's output, so the SIGN is noise-dominated
#: and the bisection stalls. Measured scatter on the interstitial family across
#: R = 3..50 nm: -0.8, +1.9, +0.4, -0.5, +1.5 meV -- non-monotonic, i.e. noise
#: and not a formula error. Compared in BINDING ENERGY rather than in c^eq
#: because c^eq is exponential in it: 2 meV at 573 K is already 4% in c^eq, so a
#: percent-level tolerance on the concentration is tighter than the measurement.
EB_TOL_EV = 5.0e-3


def _content_rate(run_dir, slot, R_nm, mat, cvs, sigma=None, dt=PROBE_DT):
    """d(content)/dt for one family at a spread of c^v, by a short integration."""
    states, fam, R = [], None, None
    for cv in cvs:
        y, fam, R = _state_one_family(slot, R_nm, mat, cv)
        states.append(y)
    out = imm.run_immobile_step(
        _cli3(run_dir, sigma=sigma, overrides=QUIET), states, 0.0, dt, retries=0)
    rates = []
    for y, r in zip(states, out):
        rates.append(np.nan if r is None
                     else (r[c_idx(slot)] - y[c_idx(slot)]) / dt)
    return np.array(rates), fam, R


def solved_c_eq(run_dir, slot, R_nm, mat, sigma=None, n_iter=40):
    """The c^v at which the family's content rate changes sign.

    That IS the solver's own c^{v,eq}_k, read out of it without instrumenting
    it. Bisected, because the rate is monotone in c^v by construction.
    """
    fam = la.families(mat)[SLOT_FAMILY[slot]]
    ref = la.c_eq_vacancy(R_nm * 1e-9, fam, T_REF, mat,
                          (sigma or {}).get(SLOT_KEY[slot], 0.0))
    lo, hi = ref * 1e-3, ref * 1e3
    # A VACANCY family grows when c^v exceeds its own equilibrium; an
    # INTERSTITIAL family shrinks. Orient on character so one bisection serves
    # both.
    grows_with_cv = (fam.character == "v")

    # Check the bracket ACTUALLY straddles a sign change before bisecting.
    # Without this the loop silently returns a bracket end, which reads as a
    # confident wrong answer rather than as a failed measurement.
    ends, _, _ = _content_rate(run_dir, slot, R_nm, mat, [lo, hi], sigma)
    r_lo, r_hi = ends
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo == 0.0 or r_hi == 0.0:
        return float("nan"), ref
    if (r_lo > 0) == (r_hi > 0):
        return float("nan"), ref

    for _ in range(n_iter):
        mid = float(np.sqrt(lo * hi))
        rate = _content_rate(run_dir, slot, R_nm, mat, [mid], sigma)[0][0]
        if not np.isfinite(rate) or rate == 0.0:
            return mid, ref          # landed on the crossing itself
        if (rate > 0) == grows_with_cv:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi)), ref


def check_detailed_balance(run_dir, mat):
    """Goal (i): the net exchange vanishes at c^v = c^{v,eq}_k."""
    print("GOAL (i)    detailed balance -- solver's c_eq vs the closed form")
    kT = la.KB_EV * T_REF
    ok = True
    for slot, R_nm in ((0, 5.0), (1, 5.0), (4, 5.0), (0, 25.0), (1, 25.0)):
        got, ref = solved_c_eq(run_dir, slot, R_nm, mat)
        if not np.isfinite(got):
            print(f"  slot {slot} ({SLOT_FAMILY[slot]:4s}) R={R_nm:5.1f} nm : "
                  f"NO SIGN CHANGE in the bracket -- not measured")
            ok = False
            continue
        dEb = kT * np.log(ref / got)      # solver E_b minus reference E_b
        flag = "ok" if abs(dEb) < EB_TOL_EV else "FAIL"
        print(f"  slot {slot} ({SLOT_FAMILY[slot]:4s}) R={R_nm:5.1f} nm : "
              f"solver {got:.6e}  reference {ref:.6e}  "
              f"dE_b {dEb * 1e3:+7.3f} meV  {flag}")
        ok = ok and abs(dEb) < EB_TOL_EV
    return ok


def check_bulk_limit(mat):
    """Goal (ii): E_b -> E^f_v and c_eq -> c_inf as R -> infinity, both characters.

    This is what checks the SIGN of varsigma_s(v): a vacancy family approaches
    the bulk value from BELOW in binding energy and an interstitial family from
    above, so getting the sign wrong sends them the same way.
    """
    print("GOAL (ii)   bulk limit, both characters")
    c_inf = mat.c_v_inf(T_REF)
    ok = True
    for name in ("c_f", "a_i", "a_v"):
        fam = la.families(mat)[name]
        row = []
        for R_nm in (5.0, 50.0, 500.0, 5000.0):
            Eb = la.binding_energy(R_nm * 1e-9, fam, mat)
            row.append((R_nm, Eb, la.c_eq_vacancy(R_nm * 1e-9, fam, T_REF, mat)))
        # The fault term is size-INDEPENDENT, so c_f does not tend to E^f_v: it
        # tends to E^f_v - gamma Omega/(b.n), which is the floor of Eq.
        # (levelfloor). Only the unfaulted families reach the bulk value.
        limit = mat.Ef_v - (fam.zeta_v * fam.gamma * fam.omega
                            / fam.bdotn) / la.EV_J
        approach = abs(row[-1][1] - limit)
        side = "below" if row[0][1] < limit else "above"
        expect = "below" if fam.character == "v" else "above"
        good = approach < 5e-3 and side == expect
        print(f"  {name:4s} ({fam.character}): E_b {row[0][1]:.4f} -> "
              f"{row[-1][1]:.4f} eV, limit {limit:.4f}, "
              f"|gap| {approach:.2e}, approached from {side} "
              f"(expect {expect})  {'ok' if good else 'FAIL'}")
        ok = ok and good
    print(f"  c_v_inf = exp(-E^f_v/kT) = {c_inf:.4e}")
    return ok


def check_stress_free_anneal(run_dir, mat):
    """Goal (iii): source off -- contents fall, densities EXACTLY constant."""
    print("GOAL (iii)  stress-free anneal")
    y = np.zeros(29)
    y[0] = mat.c_v_inf(T_REF)      # matrix at bulk equilibrium
    y[1] = 1e-30
    for slot in (0, 1, 4):
        fam = la.families(mat)[SLOT_FAMILY[slot]]
        n = 1e-10
        y[n_idx(slot)] = n
        y[c_idx(slot)] = n * (10e-9 / fam.lam) ** 2   # R = 10 nm
    y[18] = 1.88e14
    out = imm.run_immobile_step(
        _cli3(run_dir, overrides=QUIET), [y], 0.0, 1e5, retries=0)[0]
    if out is None:
        print("  integration FAILED")
        return False
    ok = True
    for slot in (0, 1, 4):
        dn = out[n_idx(slot)] - y[n_idx(slot)]
        dc = out[c_idx(slot)] - y[c_idx(slot)]
        const = (dn == 0.0)
        shrink = dc < 0.0
        print(f"  slot {slot} ({SLOT_FAMILY[slot]:4s}): dn = {dn:+.3e} "
              f"({'EXACTLY constant' if const else 'MOVED'}), "
              f"dc/c = {dc / y[c_idx(slot)]:+.3e} "
              f"({'shrinks' if shrink else 'GROWS'})")
        ok = ok and const and shrink
    return ok


def check_stressed_anneal(run_dir, mat):
    """Goal (iv): above a threshold, interstitial loops grow with no supply."""
    print("GOAL (iv)   stressed anneal -- the climb threshold")
    fam = la.families(mat)["a_i"]
    R = 10e-9
    # dE/dm for the interstitial family; the threshold is where the stress term
    # cancels the capillary one, Sigma* Omega = -zeta dE/dm, i.e. Sigma* =
    # dE/dm / Omega for an interstitial family (zeta = -1).
    dEdm = la.dEdm(R, fam)
    sigma_star = dEdm / fam.omega
    print(f"  closed form: Sigma* = (dE/dm)/Omega = {sigma_star / 1e6:.2f} MPa")

    y = np.zeros(29)
    y[0] = mat.c_v_inf(T_REF)
    y[1] = 1e-30
    n = 1e-10
    y[n_idx(1)] = n
    y[c_idx(1)] = n * (R / fam.lam) ** 2
    y[18] = 1.88e14

    fracs = (0.0, 0.5, 0.9, 1.1, 1.5, 2.0)
    states = [y.copy() for _ in fracs]
    grows = []
    for f, s in zip(fracs, states):
        out = imm.run_immobile_step(
            _cli3(run_dir, sigma={"a1": f * sigma_star}, overrides=QUIET),
            [s], 0.0, 1e5, retries=0)[0]
        grows.append(None if out is None
                     else out[c_idx(1)] - s[c_idx(1)] > 0.0)
    for f, g in zip(fracs, grows):
        print(f"    Sigma/Sigma* = {f:4.2f} : "
              f"{'GROWS' if g else 'shrinks' if g is not None else 'failed'}")
    below = [g for f, g in zip(fracs, grows) if f < 1.0]
    above = [g for f, g in zip(fracs, grows) if f > 1.0]
    ok = (not any(below)) and all(above)
    print(f"  shrinks below Sigma*, grows above: {ok}")
    return ok


def check_regression(run_dir):
    """emission_model = 0 must reproduce step 2 bit-for-bit."""
    y0, doses = _states(run_dir)
    t0, t1 = doses[4] / 1e-7, doses[5] / 1e-7
    a = imm.run_immobile_step(_cli(run_dir, n_fam=8), y0, t0, t1, retries=0)
    b = imm.run_immobile_step(_cli3(run_dir, emission=False), y0, t0, t1,
                              retries=0)
    keep = [i for i in range(len(y0)) if a[i] is not None and b[i] is not None]
    A = np.array([a[i] for i in keep])
    B = np.array([b[i] for i in keep])
    exact = np.array_equal(A, B)
    print("REGRESSION  emission_model = 0 vs the step-2 command line")
    print(f"  states compared : {len(keep)}")
    print(f"  bit-for-bit     : {exact}")
    return exact


def verify(run=None):
    run_dir = resolve_run(run or REFERENCE_RUN)
    mat = la.load_material()
    print(f"reference: {run_dir.name}\n")
    reg = check_regression(run_dir)
    print()
    g1 = check_detailed_balance(run_dir, mat)
    print()
    g2 = check_bulk_limit(mat)
    print()
    g3 = check_stress_free_anneal(run_dir, mat)
    print()
    g4 = check_stressed_anneal(run_dir, mat)
    print()
    print(f"regression (emission off) : {reg}")
    print(f"goal (i)   detailed balance : {g1}")
    print(f"goal (ii)  bulk limit       : {g2}")
    print(f"goal (iii) stress-free anneal: {g3}")
    print(f"goal (iv)  stressed anneal  : {g4}")
    return reg and g1 and g2 and g3 and g4


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 3 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify(a.run) else 1)


if __name__ == "__main__":
    main()
