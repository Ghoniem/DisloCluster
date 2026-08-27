"""Step 11 of the plan: the discrete observables, and the Upsilon measurement.

Goal (iii) is described in the plan as "the cheapest unclaimed measurement in
the document": ``Upsilon_h(k)`` has never been measured, and comparing
``beta^P_D`` of Eq. (betaPdisc) against Eq. (betaP) on one exported population
is supposed to determine it, because the discrete side carries no strain factor.

IT IS CHEAP, AND WHAT IT MEASURES IS THE CHAIN. Both tensors rest on the same
identity -- the conversion gives a loop the area that stores its defects,
``A |b| = m Omega``, and the continuum content of a family is by construction
``sum_l A_l (b.n)/V`` -- so the ratio is one by construction for a climb loop,
whatever the sizes are. That makes it a stringent END-TO-END check on the
handoff (Burgers conventions, polygon area, Omega, refusal) and NOT a
determination of a material property. See post/observables.py.

**Validation goals:**
(i)   the trace separates climb from glide -- a b-parallel-to-n population has
      a non-zero trace, a glide one has zero, to round-off;
(ii)  the two tensors agree to round-off on a synthetic population, which is the
      Upsilon measurement returning 1 for the reason above;
(iii) it is INSENSITIVE to the size distribution -- the same content spread over
      many small loops or few large ones gives the same tensor, which is what
      says the comparison cannot see the polydispersity step 6 added and hence
      cannot be measuring a size-dependent physical factor;
(iv)  it DOES see a broken handoff -- a deliberate mismatch in |b| or in Omega
      moves the ratio by exactly the amount injected.

Run:
    python -m dislocluster_code.studies.plan_step11 verify
    python -m dislocluster_code.studies.plan_step11 measure <run> [--dose 1]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dislocluster_code.post import observables as obs
from dislocluster_code.post.discrete_loops import (
    FAMILIES, LoopPopulation, OMEGA_B3)


def _pop(fam, n_loops, m_each, rng=None):
    """A synthetic population of ``n_loops`` loops each storing ``m_each``."""
    bmag = fam["b_dd"]
    r = np.sqrt(np.asarray(m_each, float) * OMEGA_B3 / (np.pi * bmag))
    r = np.broadcast_to(r, (n_loops,)).copy()
    rng = np.random.default_rng(0) if rng is None else rng
    return LoopPopulation(fam, rng.random((n_loops, 3)) * 1000.0, r)


def _content(pop, volume):
    """The continuum content the population represents: sum A|b| / (V)."""
    area = float(np.pi * np.sum(pop.radii ** 2))
    return area * pop.fam["b_dd"] / volume


# ── goal (i) ─────────────────────────────────────────────────────────────────

def check_trace():
    """Climb has a trace; glide does not."""
    print("GOAL (i)    the trace separates climb from glide")
    V = 1000.0 ** 3
    fam = [f for f in FAMILIES if f["key"] == "a1"][0]
    p = _pop(fam, 400, 800.0)
    b_climb = obs.beta_p_discrete([p], V)
    tr = obs.swelling_from_beta(b_climb)
    print(f"  climb population: tr beta^P = {tr:+.6e}")

    # The same loops made GLISSILE: b in the habit plane instead of along its
    # normal. Eq. (betaPdisc) is symmetrized, so this is a pure shear.
    n_hat, b_vec = obs.habit_frame(fam)
    glide_b = np.cross(n_hat, np.array([0.0, 0.0, 1.0]))
    glide_b = glide_b / np.linalg.norm(glide_b) * np.linalg.norm(b_vec)
    area = float(np.pi * np.sum(p.radii ** 2))
    A = area * n_hat
    b_gl = 0.5 * (np.outer(glide_b, A) + np.outer(A, glide_b)) / V
    tr_g = float(np.trace(b_gl))
    print(f"  glide population: tr beta^P = {tr_g:+.3e}")
    ok = abs(tr) > 1e-12 and abs(tr_g) < 1e-18 * max(abs(tr), 1e-30) + 1e-20
    print(f"  climb non-zero, glide zero : {ok}")
    return ok


# ── goals (ii) and (iii) ─────────────────────────────────────────────────────

def check_upsilon_identity():
    """The two tensors agree, and do not care how the content is divided."""
    print("GOAL (ii)   the two tensors agree on one population")
    print("GOAL (iii)  ...and are blind to the size distribution")
    V = 1000.0 ** 3
    print(f"  {'family':>5} {'loops':>7} {'m each':>10} {'discrete':>13} "
          f"{'continuum':>13} {'ratio':>18}")
    ok = True
    ratios = []
    for key in ("c", "a1", "a1v", "cp"):
        fam = [f for f in FAMILIES if f["key"] == key][0]
        # The SAME total content, split three ways: many small, balanced, few
        # large. If the ratio moved with this, the comparison would be seeing
        # the size distribution and could be a size-dependent physical factor.
        for n_loops, m_each in ((4000, 50.0), (500, 400.0), (25, 8000.0)):
            p = _pop(fam, n_loops, m_each)
            c = _content(p, V)
            bd = obs.beta_p_discrete([p], V)
            bc = obs.beta_p_continuum({key: c}, [fam])
            rd, rc = np.linalg.norm(bd), np.linalg.norm(bc)
            ratio = rd / rc
            ratios.append(ratio)
            print(f"  {key:>5} {n_loops:7d} {m_each:10.1f} {rd:13.5e} "
                  f"{rc:13.5e} {ratio:18.15f}")
            ok &= abs(ratio - 1.0) < 1e-13
    spread = max(ratios) - min(ratios)
    print(f"  spread of the ratio over every case : {spread:.3e}")
    ok &= spread < 1e-13
    print(f"  Upsilon = 1 to round-off, and size-independent : {ok}")
    return ok


# ── goal (iv) ────────────────────────────────────────────────────────────────

def check_detects_breakage():
    """A deliberately broken handoff must move the ratio by what was injected.

    This is the test that says goal (ii) is a check and not a tautology: the
    ratio is one only when the two sides really do share |b| and Omega, and it
    departs by exactly the injected factor when they do not. It is the same
    class of error the <c> Burgers correction removed -- CD and DD disagreeing
    by sqrt(2) on one family, silently.
    """
    print("GOAL (iv)   a broken handoff moves the ratio, by exactly the break")
    V = 1000.0 ** 3
    fam = dict([f for f in FAMILIES if f["key"] == "c"][0])
    ok = True
    print(f"  {'b_dd / b_cd':>12} {'ratio':>18} {'expected':>18}")
    for f_b in (1.0, np.sqrt(2.0), 0.5, 2.0):
        broken = dict(fam)
        broken["b_dd"] = fam["b_dd"] * f_b
        p = _pop(broken, 500, 400.0)
        # The CONTINUUM side still believes the cluster-dynamics magnitude, so
        # it converts the same loops with b_cd -- which is the whole failure.
        area = float(np.pi * np.sum(p.radii ** 2))
        c = area * fam["b_cd"] / V
        bd = obs.beta_p_discrete([p], V)
        bc = obs.beta_p_continuum({fam["key"]: c}, [fam])
        ratio = float(np.linalg.norm(bd) / np.linalg.norm(bc))
        print(f"  {f_b:12.6f} {ratio:18.12f} {f_b:18.12f}")
        ok &= abs(ratio - f_b) < 1e-12
    print(f"  the ratio reports the break exactly : {ok}")
    return ok


# ── measurement on a real run ────────────────────────────────────────────────

def measure(run_dir, dose=1.0, region="interior"):
    """Run the comparison on an exported population from a finished march."""
    from dislocluster_code.post import discrete_loops as D
    from dislocluster_code.post.fields import domain_volume
    print(f"Upsilon on {run_dir}, {dose:g} dpa, region={region}\n")
    dose_actual, pops, _stats, _box, P = D.build(run_dir, dose, region=region,
                                                 verbose=False)
    print(f"  (nearest solved snapshot: {dose_actual:g} dpa)\n")
    V = domain_volume(P)
    content = {p.fam["key"]: _content(p, V) for p in pops}
    res = obs.measure_upsilon(pops, content, V, FAMILIES)
    for habit, r in res.items():
        if r["n_loops"] == 0:
            print(f"  {habit:<10} no loops"); continue
        print(f"  {habit:<10} {r['n_loops']:6d} loops   "
              f"|beta_D| {r['norm_discrete']:.6e}   "
              f"|beta_C| {r['norm_continuum']:.6e}   "
              f"ratio {r['upsilon']:.12f}")
    print("\n  The ratio is a check on the handoff, not a material property:")
    print("  both sides rest on A|b| = m Omega. immobileSpeciesRelRelaxVol")
    print("  (0.405 basal, 1.2 prismatic interstitial) enters NEITHER, so a")
    print("  physical Upsilon != 1 is not available from this comparison.")
    return res


def verify():
    g1 = check_trace()
    print()
    g23 = check_upsilon_identity()
    print()
    g4 = check_detects_breakage()
    print()
    print(f"goal (i)   trace separates climb : {g1}")
    print(f"goal (ii)  tensors agree         : {g23}")
    print(f"goal (iii) size-independent      : {g23}")
    print(f"goal (iv)  detects a broken |b|  : {g4}")
    return g1 and g23 and g4


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 11 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    m = sub.add_parser("measure")
    m.add_argument("run")
    m.add_argument("--dose", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify() else 1)
    measure(a.run, a.dose)


if __name__ == "__main__":
    main()
