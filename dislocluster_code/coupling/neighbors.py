"""
neighbors.py -- the screened cutoff and the neighbor search behind it (plan 4d).

WHY THIS EXISTS BEFORE THE CLIMB SOLVER
---------------------------------------
`GalerkinClimbSolver` assembles `clusterStiffnessMatrix(fieldSegment,
sourceSegment)` over every ORDERED PAIR of segments, each pair costing an
`mSize`-wide `concentrationMatrices` evaluation summed over `periodicShifts`.
The linear solve costs nothing by comparison -- the sparse path is dead code and
the live path is one scalar division per node -- so **100% of the cost is the
pairwise assembly**, and the only lever that matters is the number of pairs.
Plan 4.6 puts this sub-step before the solver deliberately: with all-pairs
assembly the 500 nm case cannot be run long enough to debug.

WHY TRUNCATION IS LEGITIMATE HERE
---------------------------------
A bare `1/r` kernel could not be truncated. A growing loop is a net sink, so its
monopole does not vanish, and a shell at radius `r` contributes `~r^2 * (1/r) =
r` -- the sum grows with the cutoff instead of converging.

The physical kernel is not bare. It is screened by the sink field as
`exp(-k r)/r`, with `k^2` the total sink strength that `ImmobileSinks.h` already
assembles:

    k2_m = sum_k Z(row(k), m) * S_k * loopSinkScale_k  +  otherSinks_m

per MOBILE SPECIES m -- which is worth stating, because the screening length is
therefore species-dependent and the plan's single quoted `L_s` per dose is a
representative value rather than the whole story. This module reports all of
them and takes the cutoff from the LONGEST, which is the conservative choice.

There is a second reason truncation is right rather than merely tolerable: the
continuum field already carries the mean-field response of the whole loop
population through `ImmobileSinks`, so the discrete Green's-function sum must
supply ONLY the near-field correction the mean field misses. Extending it
further would double count -- the same failure mode the transfer ledger guards
against in `transition.py`.

WHAT IS AND IS NOT DECIDED HERE
-------------------------------
The cutoff `R_c = n_L * L_s` with `n_L = 3` is a starting value, not a result.
Plan 5.5 requires a convergence sweep over `R_c/L_s` in [2,4] with the loop
kinetics required to be flat; the truncation error of a screened kernel shows up
as `R_c` sensitivity and nowhere else. `cutoff_report` prints what fraction of
the pair budget each choice costs so that sweep can be priced before it is run.

The search itself uses `scipy.spatial.cKDTree`, not a hand-rolled cell list. It
is already a dependency, it has the same asymptotics, and `boxsize` gives the
periodic images for free. A C++ port would want an explicit cell list; this is
the Python prototype the plan asks for.

USAGE
-----
    from dislocluster_code.coupling import neighbors as nb
    L = nb.screening_lengths(F, material)      # per species, in b
    R = nb.cutoff(L, n_L=3.0)
    pairs = nb.neighbor_pairs(centers, R)
    print(nb.cutoff_report(centers, L))
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from dislocluster_code.coupling.field import read_material_vector as _vec
from dislocluster_code.post.discrete_loops import FAMILIES, OMEGA_B3
from dislocluster_code.post.fields import B_SI

N_L_DEFAULT = 3.0          # R_c = N_L * L_s, plan 4.2; swept in plan 5.5


def sink_strength(F, fam, scale=1.0):
    """Geometric sink strength ``2 pi r N`` per node, in 1/b^2.

    The radius is the CONTINUUM's, from `b_cd` -- this reproduces what
    `ImmobileSinks` forms, not what the discrete population would carry. The
    two differ by ``sqrt(b_cd/b_dd)``; see `transition.sink_discontinuity`.
    """
    n = np.asarray(F[:, fam["ncol"]], float)
    c = np.asarray(F[:, fam["ccol"]], float)
    ok = np.isfinite(n) & np.isfinite(c) & (n > 0) & (c > 0)
    m = np.zeros_like(n)
    m[ok] = c[ok] / (n[ok] * OMEGA_B3)
    r = np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * fam["b_cd"]))
    return np.where(ok, 2.0 * np.pi * r * n, 0.0) * scale


def k_squared(F, material):
    """``(n_nodes, n_species)`` total sink strength, in 1/b^2.

    Mirrors `ImmobileSinks::operator()` plus the network term `getR1` adds:
    the loop part is summed over families with the DAD capture efficiency of
    the family's polarity row, and `otherSinks_SI` supplies the constant
    network part.
    """
    from dislocluster_code.studies.dad_sweep import capture_efficiencies
    p = _vec(material, "dadAnisotropy")
    z0 = _vec(material, "dadZ0", len(p))
    Zc, Za = capture_efficiencies(p, z0)          # row 0 vacancy, row 1 interstitial
    scale = _vec(material, "loopSinkScale", len(FAMILIES))

    k2 = np.zeros((F.shape[0], len(p)), float)
    for j, fam in enumerate(FAMILIES):
        S = sink_strength(F, fam, scale[j] if j < len(scale) else scale[0])
        Z = Zc if fam.get("vacancy") else Za
        k2 += S[:, None] * Z[None, :]

    try:                                          # network sinks, m^-2 -> b^-2
        other = _vec(material, "otherSinks_SI", len(p)) * (B_SI ** 2)
        k2 += other[None, :]
    except Exception:
        pass
    return k2


def screening_lengths(F, material, reduce="median"):
    """``L_s = 1/k`` per mobile species, in b. ``reduce`` over the nodes."""
    k2 = k_squared(F, material)
    k2 = np.maximum(k2, 1e-300)
    L = 1.0 / np.sqrt(k2)
    if reduce == "median":
        return np.median(L, axis=0)
    if reduce == "max":
        return L.max(axis=0)
    if reduce is None:
        return L
    raise ValueError(f"reduce must be median/max/None, got {reduce!r}")


def cutoff(L_s, n_L=N_L_DEFAULT):
    """``R_c`` from the LONGEST screening length -- the conservative choice."""
    return float(n_L) * float(np.max(L_s))


def neighbor_pairs(centers, R_c, boxsize=None):
    """``(M,2)`` index pairs within ``R_c``, i < j. Periodic if ``boxsize``.

    The SELF term is deliberately absent from this list and must always be
    retained by the caller: a segment's own contribution is not an interaction
    and is not subject to the cutoff.
    """
    c = np.ascontiguousarray(np.asarray(centers, float))
    tree = cKDTree(c, boxsize=boxsize)
    pairs = tree.query_pairs(float(R_c), output_type="ndarray")
    return pairs.reshape(-1, 2)


def brute_force_pairs(centers, R_c):
    """All-pairs reference, for validating :func:`neighbor_pairs` on small N."""
    c = np.asarray(centers, float)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    i, j = np.triu_indices(len(c), k=1)
    keep = d[i, j] <= R_c
    return np.column_stack([i[keep], j[keep]])


def validate(n=400, R_frac=0.18, seed=0):
    """``(ok, n_tree, n_brute)`` -- the tree must reproduce all-pairs exactly."""
    rng = np.random.default_rng(seed)
    c = rng.random((n, 3)) * 100.0
    R = R_frac * 100.0
    a = neighbor_pairs(c, R)
    b = brute_force_pairs(c, R)
    sa = {tuple(sorted(p)) for p in a}
    sb = {tuple(sorted(p)) for p in b}
    return sa == sb, len(sa), len(sb)


def cutoff_report(centers, L_s, n_L_values=(2.0, 3.0, 4.0), boxsize=None):
    """What each cutoff costs, against the all-pairs budget it replaces.

    Prices plan 5.5's `R_c/L_s` convergence sweep before it is run: the pair
    count is the cost, since assembly is 100% of the climb solve.
    """
    c = np.asarray(centers, float)
    N = len(c)
    all_pairs = N * (N - 1) // 2
    L = ["neighbor cutoff", "",
         f"segments (loop centres here): {N:,}",
         f"all ordered pairs: {all_pairs:,}",
         f"screening length L_s per species [b]: "
         + ", ".join(f"{x:.4g}" for x in np.atleast_1d(L_s)),
         f"longest L_s: {float(np.max(L_s)):.4g} b "
         f"= {float(np.max(L_s)) * B_SI * 1e9:.1f} nm", "",
         f"{'n_L':>5} {'R_c [b]':>12} {'R_c [nm]':>10} {'pairs':>12} "
         f"{'frac of all':>12} {'per segment':>12}"]
    for n_L in n_L_values:
        R = cutoff(L_s, n_L)
        p = neighbor_pairs(c, R, boxsize=boxsize)
        L.append(f"{n_L:>5.1f} {R:>12.4g} {R * B_SI * 1e9:>10.1f} "
                 f"{len(p):>12,} {len(p) / max(all_pairs, 1):>12.4%} "
                 f"{2 * len(p) / max(N, 1):>12.1f}")
    L += ["", "The self term is not in these counts and is never truncated.",
          "Plan 5.5 requires the loop kinetics to be flat across this range;",
          "the truncation error of a screened kernel appears as R_c",
          "sensitivity and nowhere else."]
    return "\n".join(L)
