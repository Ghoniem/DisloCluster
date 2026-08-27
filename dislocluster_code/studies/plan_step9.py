"""Step 9 of the code implementation plan: the smeared discrete sink.

Eq. (kdiscrete) returns a transferred population's sink strength to the
continuum as a nodal field built from where the loops actually are. It fills the
lower-left cell of the climb-selection table -- the ``Theta ~ 1, Xi <~ 2`` regime
the basal family occupies -- and costs O(N_seg) against the O(N_seg^2) of the
pair assembly.

**Regression:** an empty population contributes exactly zero at every node, so a
run with no transfer is untouched.

**Validation goals:**
(i)   partition of unity -- ``sum_j V_j W = 1`` for every loop, INCLUDING loops
      whose support is cut by the domain surface, which is exactly where the
      refused loops are;
(ii)  conservation -- ``sum_j V_j k2_j`` equals ``sum_l coeff*P_l`` exactly, on
      an irregular node cloud;
(iii) consistency -- loops drawn from a UNIFORM field reproduce the continuum
      ``n 2 pi r`` to the sampling error, so the two routes are one channel;
(iv)  the spatial structure the scalar compensation loses -- a boundary-shell
      refusal must leave a deficit IN THE SHELL, where ``1 - frac_kept`` spreads
      it uniformly over the whole domain;
(v)   the width floor -- the kernel may not be narrower than max(h, L_s).

Run:
    python -m dislocluster_code.studies.plan_step9 verify
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from dislocluster_code.coupling.smeared_sink import (
    SUPPORT, kernel_width, partition_weights, discrete_sink_field,
    continuum_sink_field)


def _cloud(n_side=14, seed=0, jitter=0.35):
    """An irregular node cloud in a cube, with its nodal volumes.

    Jittered on purpose: a perfect lattice would let an analytically normalized
    kernel pass goal (i) by accident, and the whole point of the per-loop
    normalization is that it does not need one.
    """
    rng = np.random.default_rng(seed)
    g = (np.arange(n_side) + 0.5) / n_side
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    p = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    p = p + jitter / n_side * (rng.random(p.shape) - 0.5)
    p = np.clip(p, 0.0, 1.0) * 1000.0                       # a 1000 b cube
    vol = np.full(len(p), 1000.0 ** 3 / len(p))
    return p, vol


# ── regression ───────────────────────────────────────────────────────────────

def check_regression():
    print("REGRESSION  an empty population contributes exactly zero")
    nodes, vol = _cloud()
    w = kernel_width(nodes)
    f = discrete_sink_field(nodes, vol, np.zeros((0, 3)), np.zeros(0), w)
    ok = f.shape == (len(nodes),) and np.all(f == 0.0)
    print(f"  field shape {f.shape}, max |k2| {np.abs(f).max():.1e}")
    print(f"  identically zero : {ok}")
    return ok


# ── goal (i): partition of unity ─────────────────────────────────────────────

def check_partition():
    """sum_j V_j W = 1 per loop, including against the surface."""
    print("GOAL (i)    partition of unity, per loop")
    nodes, vol = _cloud()
    w = kernel_width(nodes)
    rng = np.random.default_rng(1)
    # Deliberately include loops ON the faces, in the corners, and outside --
    # the refused loops live in a boundary shell, so the surface is the case
    # that matters, not the exception.
    interior = 200.0 + 600.0 * rng.random((200, 3))
    face = np.column_stack([rng.random(60) * 1000.0,
                            rng.random(60) * 1000.0,
                            np.zeros(60)])
    corner = np.zeros((4, 3))
    corner[1] = [1000.0, 0.0, 0.0]
    corner[2] = [1000.0, 1000.0, 1000.0]
    corner[3] = [0.0, 1000.0, 0.0]
    outside = np.array([[-3.0 * w, 500.0, 500.0], [500.0, 1000.0 + w, 500.0]])
    groups = (("interior", interior), ("on a face", face),
              ("at a corner", corner), ("outside", outside))
    ok = True
    for name, cen in groups:
        r, c, ww = partition_weights(nodes, vol, cen, w)
        tot = np.bincount(r, weights=vol[c] * ww, minlength=len(cen))
        err = float(np.abs(tot - 1.0).max())
        print(f"  {name:<12} {len(cen):4d} loops   worst |sum V W - 1| = "
              f"{err:.3e}")
        ok &= err < 1e-12
    print(f"  unity everywhere : {ok}")
    return ok


# ── goal (ii): conservation ──────────────────────────────────────────────────

def check_conservation():
    """The population total survives the smearing, exactly."""
    print("GOAL (ii)   total sink strength conserved")
    nodes, vol = _cloud(seed=2)
    w = kernel_width(nodes)
    rng = np.random.default_rng(3)
    ok = True
    for tag, cen in (("bulk", 1000.0 * rng.random((500, 3))),
                     ("boundary shell", np.column_stack([
                         1000.0 * rng.random(300),
                         1000.0 * rng.random(300),
                         np.where(rng.random(300) < 0.5,
                                  40.0 * rng.random(300),
                                  1000.0 - 40.0 * rng.random(300))]))):
        P = 10.0 + 90.0 * rng.random(len(cen))
        for coeff in (1.0, 0.317):
            f = discrete_sink_field(nodes, vol, cen, P, w, coefficient=coeff)
            got = float(np.dot(vol, f))
            want = float(coeff * P.sum())
            rel = abs(got - want) / want
            print(f"  {tag:<15} coeff {coeff:6.3f}  "
                  f"sum V k2 / sum coeff P - 1 = {got / want - 1.0:+.3e}")
            ok &= rel < 1e-12
    print(f"  conserved to 1e-12 : {ok}")
    return ok


# ── goal (iii): consistency with the continuum ───────────────────────────────

def check_consistency():
    """A uniform draw must reproduce n * 2 pi r."""
    print("GOAL (iii)  uniform draw reproduces the continuum k^2")
    nodes, vol = _cloud(n_side=12, seed=4)
    box = 1000.0 ** 3
    w = kernel_width(nodes)
    rng = np.random.default_rng(5)
    print(f"  {'loops':>7} {'k2 discrete':>13} {'k2 continuum':>13} "
          f"{'interior mean err':>18}")
    ok = True
    for n_loops in (2000, 20000, 200000):
        r_loop = 25.0
        cen = 1000.0 * rng.random((n_loops, 3))
        P = np.full(n_loops, 2.0 * np.pi * r_loop)
        f = discrete_sink_field(nodes, vol, cen, P, w)
        n_dens = n_loops / box
        k_cont = continuum_sink_field(n_dens, r_loop)
        # Compare in the INTERIOR: the kernel is normalized per loop, so a node
        # near a face legitimately receives more from each of the fewer loops
        # that reach it, and the field is not meant to be flat there.
        d = np.minimum(nodes.min(axis=1), 1000.0 - nodes.max(axis=1))
        inner = d > SUPPORT * w
        err = float(abs(f[inner].mean() / k_cont - 1.0))
        print(f"  {n_loops:7d} {f[inner].mean():13.5e} {k_cont:13.5e} "
              f"{err:18.2e}")
        ok &= err < 0.05
    print(f"  agrees with the continuum to 5% : {ok}")
    return ok


# ── goal (iv): the structure the scalar compensation loses ───────────────────

def check_spatial_structure():
    """A boundary-shell refusal leaves its deficit IN the shell.

    This is the concrete reason Eq. (kdiscrete) exists. `transition.transfer`
    returns the refused defects by scaling the whole family field by a single
    `1 - frac_kept`, at every node alike -- but the loops were refused BECAUSE
    they were near a face, so the truth is a deficit confined to a shell. The
    two differ by construction; what this measures is by how much.
    """
    print("GOAL (iv)   refusal is spatial, and the scalar loses it")
    nodes, vol = _cloud(n_side=12, seed=6)
    w = kernel_width(nodes)
    rng = np.random.default_rng(7)
    n_loops, r_loop, shell = 40000, 25.0, 150.0
    cen = 1000.0 * rng.random((n_loops, 3))
    P = np.full(n_loops, 2.0 * np.pi * r_loop)
    d_loop = np.minimum(cen.min(axis=1), 1000.0 - cen.max(axis=1))
    kept = d_loop > shell                       # the refusal, as a shell test
    frac_kept = float(kept.mean())

    k_all = discrete_sink_field(nodes, vol, cen, P, w)
    k_kept = discrete_sink_field(nodes, vol, cen[kept], P[kept], w)
    k_scalar = frac_kept * k_all                # what the scalar route gives

    d_node = np.minimum(nodes.min(axis=1), 1000.0 - nodes.max(axis=1))
    in_shell = d_node < shell
    print(f"  frac_kept = {frac_kept:.4f}   ({(~kept).sum()} of {n_loops} "
          f"refused, all within {shell:g} b of a face)")
    for tag, sel in (("in the shell", in_shell), ("interior", ~in_shell)):
        a, b = k_kept[sel].mean(), k_scalar[sel].mean()
        print(f"  {tag:<13} spatial {a:11.4e}   scalar {b:11.4e}   "
              f"scalar/spatial {b / a:6.2f}x")
    # The scalar route must OVERSTATE the sink in the shell (it kept a share of
    # loops that are not there) and UNDERSTATE it in the interior.
    over = k_scalar[in_shell].mean() / k_kept[in_shell].mean()
    under = k_scalar[~in_shell].mean() / k_kept[~in_shell].mean()
    ok = over > 1.5 and under < 0.95
    print(f"  scalar overstates the shell and understates the interior : {ok}")
    # ...and both routes still hold the same TOTAL, which is why the ledger
    # closes on one and the field is wrong anyway.
    t1, t2 = float(np.dot(vol, k_kept)), float(np.dot(vol, k_scalar))
    print(f"  totals: spatial {t1:.5e}   scalar {t2:.5e}   "
          f"ratio {t2 / t1:.4f}")
    return ok


# ── goal (v): the width floor ────────────────────────────────────────────────

def check_width_floor():
    """max(h, L_s), and h taken from the cloud when it is not given."""
    print("GOAL (v)    the kernel width is max(h, L_s)")
    nodes, _ = _cloud(n_side=10)
    h = kernel_width(nodes, L_s=0.0)
    cases = [("L_s = 0", 0.0, h), ("L_s < h", 0.5 * h, h),
             ("L_s > h", 3.0 * h, 3.0 * h)]
    ok = True
    for tag, L_s, want in cases:
        got = kernel_width(nodes, L_s=L_s)
        print(f"  {tag:<9} -> {got:9.3f}  (want {want:9.3f})")
        ok &= abs(got - want) < 1e-12
    try:
        kernel_width(L_s=1.0)
        print("  refuses to guess h with no cloud : False")
        ok = False
    except ValueError:
        print("  refuses to guess h with no cloud : True")
    print(f"  width floor honoured : {ok}")
    return ok


def verify():
    reg = check_regression()
    print()
    g1 = check_partition()
    print()
    g2 = check_conservation()
    print()
    g3 = check_consistency()
    print()
    g4 = check_spatial_structure()
    print()
    g5 = check_width_floor()
    print()
    print(f"regression (empty population) : {reg}")
    print(f"goal (i)   partition of unity : {g1}")
    print(f"goal (ii)  conservation       : {g2}")
    print(f"goal (iii) continuum agreement: {g3}")
    print(f"goal (iv)  spatial structure  : {g4}")
    print(f"goal (v)   width floor        : {g5}")
    return reg and g1 and g2 and g3 and g4 and g5


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 9 acceptance tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
