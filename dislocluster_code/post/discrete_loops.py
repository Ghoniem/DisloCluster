"""
discrete_loops.py -- convert the continuum cluster-dynamics loop fields into a
DISCRETE loop population that dislocation dynamics can consume.

WHAT THIS REPLACES
------------------
``modelib_fields._overlay_family`` draws platelets whose COUNT is set by
geometric packing and whose radii are exaggerated by a fixed factor. It is a
size map, not a microstructure: at 1e-4 dpa it draws 2500 platelets where the
domain physically holds 2.7 loops. Nothing about it can be handed to DD.

This module instead produces the ACTUAL loop population -- the real number, at
the real size, at positions drawn from the real spatial density -- together with
the Burgers vector and habit-plane normal of every loop, in the form MoDELib3's
own ``aLoop`` microstructure generator reads.

THE CONVERSION, AND WHERE IT COMES FROM
---------------------------------------
MoDELib-fullCD does this in ``ClusterDynamics<dim>::initializeDiscreteClimbLoops``
(src/ClusterDynamics/ClusterDynamics.cpp). Its algorithm, in one sentence: rank
elements by defect content, greedily absorb neighbouring elements until the group
holds ``clusterDiscretizationFactor`` loops, then emit ONE 12-sided loop carrying
the group's ENTIRE defect content, centred on a random tetrahedron in the group
and snapped to the nearest crystallographic plane.

Two things are worth extracting from that.

1. The radius rule is defect-count conservation,

       pi * r^2 * |b| = N_defects * Omega     ->     r = sqrt(N*Omega/(pi*|b|))

   which is fullCD's ``sqrt(NiCiK.second*cdp.omega/(M_PI*cdp.b*
   cdp.immobileSpeciesBurgersMagnitude(k)))`` verbatim.

2. ``discretizationFactor`` makes one discrete loop stand in for a bundle of
   continuum loops. That is ALREADY an area-conserving coalescence, applied up
   front. Here it is applied as a second pass instead, where it can be driven by
   actual geometric overlap rather than by a user-chosen bundle size -- so the
   population is exact wherever the loops fit, and coarsens only where they do
   not.

WHY THE COUNT IS QUOTED TWICE
-----------------------------
The whole-domain loop count is dominated by the near-boundary artifact: cascade
nucleation is spatially uniform but the only loop-removal channel is coalescence
driven by the absorbed mobile flux, which vanishes where Dirichlet pins the
mobile concentrations. At 10 dpa the domain holds 19940 <a>1 loops and the
interior holds 205 -- a factor of 97. The interior number is the physical one;
``region='interior'`` is therefore the default for anything destined for DD.

THE <c> BURGERS VECTOR IS NOT THE SAME ON BOTH SIDES
----------------------------------------------------
The cluster-dynamics material file gives the <c> family a FULL <0001> Burgers
vector, |b| = c/a = 1.632993 b. MoDELib3's ``aLoop`` basal branch builds a HALF
<0001> loop, |b| = 0.8165 b, which is the physically standard basal vacancy loop
in Zr. The invariant handed from CD to DD is the number of vacancies stored, not
the radius, so the discrete radius is computed from the DD Burgers magnitude.
The consequence -- <c> radii come out sqrt(2) larger than the CD-internal value
-- is reported by ``summary()`` rather than hidden.

USAGE
-----
    python -m py_utils.discrete_loops <run_dir> [--doses 1e-4 0.01 1 10]
                                      [--region interior|domain]
                                      [--no-coalesce] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy.spatial import cKDTree


from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.post.fields import (                        # noqa: E402
    OMEGA_B3, B_SI, domain_faces, domain_edges, domain_volume,
    draw_domain_wireframe, _draw_orientation_triad,
)
from dislocluster_code.post.gb import gb_distance                  # noqa: E402
from dislocluster_code.post import movies as movies_mod                    # noqa: E402
from dislocluster_code.post.volume_average import voronoi_weights          # noqa: E402

# ── crystallography ──────────────────────────────────────────────────────────
# HCP lattice basis in units of b, as MoDELib3 builds it for crystalStructure=HEX
# (see Library/Materials/Zr3d_ghoniem.txt). Columns are the primitive vectors.
LATTICE_BASIS = np.array([[1.0, 0.5,       0.0],
                          [0.0, 0.8660254, 0.0],
                          [0.0, 0.0,       1.632993]])

# One entry per immobile family, in the column order the CD field block uses.
#
#   key          short name used in filenames
#   ncol, ccol   columns of the 12-wide CD block: number density, stored content
#   b_lattice    Burgers direction in LATTICE components, from the material file's
#                `immobileSpeciesBurgers`
#   b_cd         |b| the CLUSTER DYNAMICS side assumes, in units of b
#   b_dd         |b| MoDELib3's aLoopGenerator actually builds, in units of b
#   plane_id     index into singleCrystal->slipSystems(). With
#                enabledSlipSystems=fullBasal fullPrismatic the list is
#                6 basal systems (0-5), then prismatic in pairs: plane (a1,c)
#                -> 6,7, plane (a3,c) -> 8,9, plane (-a2,c) -> 10,11.
#   vacancy      1 vacancy-type, 0 interstitial-type (material immobileSpeciesVector)
#   sides        polygon sides used to draw and to export the loop; see
#                CIRCLE_SIDES below

# Sides used to draw a CIRCULAR loop. 64 is indistinguishable from a circle at
# every radius these runs reach and at every output resolution used, while
# remaining a polygon -- which it has to be, because MoDELib3's aLoopGenerator
# builds loops from vertices. Both loop types now read as circles: <c> uses
# this value and <a> keeps the 16 it has always had, already round enough that
# no pixel distinguishes it from a disc at these radii. <a> is left at 16 so
# that its exported radii and CSVs are unchanged by this edit.
#
# The <c> family used 6 until 16 August 2026. That was not arbitrary: basal
# vacancy loops in Zr facet on <10-10>-type edges, so a hexagon is a defensible
# equilibrium shape and the drawn hexagon was oriented with its edges along
# those directions. The physics has not changed -- only the representation, by
# request, to circles. The faceting note is kept here so the reason for the
# original choice is not lost if it is ever revisited.
#
# Nothing downstream assumes a particular side count. polygon_circumradius()
# matches the polygon AREA to the disc the continuum field assigned, for any n,
# so the number of point defects stored in a loop is conserved either way; the
# circumradius correction is 0.08% at n=64 against 10.0% for a hexagon. The
# drawn line is resampled to 1.5 nm steps before clipping regardless of n, so
# the higher count costs nothing to render.
CIRCLE_SIDES = 64

FAMILIES = [
    dict(key="c",  label="<c>",   ncol=4, ccol=8,  b_lattice=(0.0, 0.0, 1.0),
         b_cd=1.632993, b_dd=0.8164966, plane_id=0,  vacancy=1,
         sides=CIRCLE_SIDES,
         d_plane=1.632993, color="#1f4fbf", b_label="1/2[0001]",
         b_tex=r"$\frac{1}{2}[0001]$"),
    dict(key="a1", label="<a>1",  ncol=5, ccol=9,  b_lattice=(1.0, 0.0, 0.0),
         b_cd=1.0, b_dd=1.0, plane_id=6,  vacancy=0, sides=16,
         d_plane=0.8660254, color="#c62828", b_label="1/3[2-1-10]",
         b_tex=r"$\frac{1}{3}[2\bar{1}\bar{1}0]$"),
    dict(key="a2", label="<a>2",  ncol=6, ccol=10, b_lattice=(0.0, 1.0, 0.0),
         b_cd=1.0, b_dd=1.0, plane_id=8,  vacancy=0, sides=16,
         d_plane=0.8660254, color="#2e7d32", b_label="1/3[11-20]",
         b_tex=r"$\frac{1}{3}[11\bar{2}0]$"),
    dict(key="a3", label="<a>3",  ncol=7, ccol=11, b_lattice=(-1.0, 1.0, 0.0),
         b_cd=1.0, b_dd=1.0, plane_id=10, vacancy=0, sides=16,
         d_plane=0.8660254, color="#e6b800", b_label="1/3[-12-10]",
         b_tex=r"$\frac{1}{3}[\bar{1}2\bar{1}0]$"),
]

# Line thickness of the drawn tubes, in nm. A dislocation line has no thickness,
# so this is purely a drawing width -- but a single width across families makes
# the <c> loops, which are ten times larger, look like wire. The width is
# therefore proportional to the loop's own radius, clamped so that the small
# <a> loops stay visible and the large <c> loops do not fill in.
TUBE_FRACTION = 0.05
TUBE_MIN_NM = 1.2
TUBE_MAX_NM = 3.6


def tube_radius_nm(r_nm):
    """Drawing width for loops of radius ``r_nm`` (nm), proportional and clamped."""
    return np.clip(TUBE_FRACTION * np.asarray(r_nm, dtype=float),
                   TUBE_MIN_NM, TUBE_MAX_NM)


def family_geometry(fam):
    """``(b_cartesian, unit_normal)`` in units of b, both Cartesian.

    These loops are pure climb: the habit-plane normal is parallel to the
    Burgers vector. MoDELib3 encodes the same thing -- the prismatic branch of
    ``aLoopGenerator::generateSingle`` sets ``loopNorm = slipSystem.s.normalized()``,
    the Burgers direction itself.
    """
    b_cart = LATTICE_BASIS @ np.asarray(fam["b_lattice"], dtype=float)
    n_hat = b_cart / np.linalg.norm(b_cart)
    return n_hat * fam["b_dd"], n_hat


def polygon_circumradius(r_area, sides):
    """Circumradius of an ``sides``-gon whose AREA equals that of a disc ``r_area``.

    ``aLoopGenerator`` places vertices on a circle of the radius it is given, so
    passing the disc radius directly would under-fill the loop: a regular n-gon
    of circumradius R has area (n/2)R^2 sin(2pi/n), which for n=6 is only 82.7%
    of pi R^2. Since the quantity being conserved is the number of point defects
    stored in the loop, the polygon area is what has to match, not the radius.
    """
    n = int(sides)
    return float(r_area * np.sqrt(2.0 * np.pi / (n * np.sin(2.0 * np.pi / n))))


# ── the population ───────────────────────────────────────────────────────────
class LoopPopulation:
    """Discrete loops of one family at one dose."""

    def __init__(self, fam, centers, radii, n_merged=None):
        self.fam = fam
        self.centers = np.asarray(centers, dtype=float).reshape(-1, 3)  # [b]
        self.radii = np.asarray(radii, dtype=float).reshape(-1)         # [b]
        self.n_merged = (np.ones(len(self.radii), dtype=int)
                         if n_merged is None else np.asarray(n_merged, int))

    def __len__(self):
        return len(self.radii)

    @property
    def total_area(self):
        """Sum of loop areas [b^2] -- the conserved quantity under coalescence."""
        return float(np.pi * np.sum(self.radii ** 2))

    @property
    def stored_defects(self):
        """Point defects held in the population: area * |b| / Omega."""
        return self.total_area * self.fam["b_dd"] / OMEGA_B3


def sample_family(nodes, F, weights, box_volume, fam, mask=None, rng=None,
                  b_source="dd", positions="field", box=None, faces=None):
    """Draw the discrete loops of one family from its continuum fields.

    The count is not a parameter: it is ``sum_j n_j V_j`` over the selected
    region, the actual number of loops the field says are there. Positions are
    an inhomogeneous Poisson sample with that intensity, so the boundary
    enrichment survives into the discrete population instead of being averaged
    away.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = F[:, fam["ncol"]].astype(float)
    c = F[:, fam["ccol"]].astype(float)
    ok = np.isfinite(n) & np.isfinite(c) & (n > 0.0) & (c > 0.0)
    if mask is not None:
        ok &= mask
    if not ok.any():
        return LoopPopulation(fam, np.zeros((0, 3)), np.zeros(0))

    # Volume attached to each node, in b^3.
    vol = weights * box_volume

    # Expected number of loops per node, and in total.
    lam = np.where(ok, n * vol, 0.0)
    n_expect = float(lam.sum())
    if n_expect <= 0.0:
        return LoopPopulation(fam, np.zeros((0, 3)), np.zeros(0))

    # Defects per loop, then the radius that stores them. b_source selects which
    # Burgers magnitude closes the volume balance -- see the module docstring.
    bmag = fam["b_dd"] if b_source == "dd" else fam["b_cd"]
    m = np.zeros_like(n)
    m[ok] = c[ok] / (n[ok] * OMEGA_B3)
    r_node = np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * bmag))

    # Deterministic count, stochastic placement: round the expectation rather
    # than drawing Poisson, so repeated calls at the same dose give the same
    # number and a dose sweep is not dominated by counting noise.
    n_draw = int(round(n_expect))
    if n_draw == 0:
        return LoopPopulation(fam, np.zeros((0, 3)), np.zeros(0))

    # Which node each loop belongs to, with probability proportional to its
    # share of the total expected count.
    p = lam / lam.sum()
    idx = rng.choice(len(lam), size=n_draw, p=p)

    if positions == "uniform":
        # A homogeneous cell at the region's mean density. The count was scaled
        # to the FULL domain, so the loops have to fill the full domain too --
        # placing a full-domain count inside the sampling region instead would
        # inflate the density by the inverse of that region's volume fraction.
        # This is the right mode for a bulk DD cell, where the point is to carry
        # the interior state without the boundary layer attached to it.
        #
        # "The full domain" is the CRYSTAL, not its bounding box. With `faces`
        # given, points are rejected until they lie inside the convex body the
        # nodes fill, so on a hexagonal prism no loop is placed in the six empty
        # wedges the bounding box adds -- which is what put loops outside the
        # drawn outline.
        lo, hi = box
        if faces is None:
            centers = lo + (hi - lo) * rng.random((n_draw, 3))
        else:
            N, bb = faces
            centers = np.empty((0, 3))
            while len(centers) < n_draw:
                p = lo + (hi - lo) * rng.random((max(n_draw, 64) * 2, 3))
                p = p[np.all(p @ N.T <= bb[None, :], axis=1)]
                centers = np.vstack([centers, p]) if len(centers) else p
            centers = centers[:n_draw]
    else:
        # Scatter inside each node's cell, keeping the spatial variation of the
        # field. The cell size is not known exactly, so use the local node
        # spacing as its scale -- this only smooths the sample within one cell
        # and does not move loops between regions.
        tree = cKDTree(nodes)
        dnn, _ = tree.query(nodes[idx], k=2)
        jitter = 0.5 * dnn[:, 1][:, None] * (rng.random((n_draw, 3)) - 0.5)
        centers = nodes[idx] + jitter

    return LoopPopulation(fam, centers, r_node[idx])


# ── coalescence ──────────────────────────────────────────────────────────────
def coalesce(pop, box_volume, coplanar_tol="plane", max_passes=20,
             spacing_guard=True, verbose=False):
    """Merge overlapping loops of one family, conserving total area.

    Two loops are taken to overlap when their centre distance is less than the
    sum of their radii. That is the bounding-sphere test, which for coplanar
    discs is exact and for non-coplanar discs is conservative -- it merges some
    pairs that would just miss.

    WHY THIS NEEDS A BOUND, AND WHAT THE BOUND IS
    ---------------------------------------------
    Area-conserving merging cannot repair overlap; it always worsens it. With
    the total area N*pi*r^2 held fixed and the mean spacing d = (V/N)^(1/3),

        r ~ N^(-1/2),   d ~ N^(-1/3),   so   2r/d ~ N^(-1/6),

    and every merge lowers N. Iterating to a fixed point therefore percolates
    the moment the overlap graph connects: run unbounded on the <c> population
    at 1 dpa and 186 loops fuse into a single loop of radius 650 nm, larger than
    the 500 nm box. Two bounds keep it physical.

    ``coplanar_tol`` (in b) requires the two centres to lie within that distance
    of a common habit plane before they may merge. The default ``"plane"`` uses
    the family's interplanar spacing, which is the physical statement that two
    loops on DIFFERENT parallel planes cannot become one loop without climbing.
    ``None`` disables the test -- the fullCD-like behaviour, since fullCD groups
    by finite-element adjacency and never checks coplanarity.

    ``spacing_guard`` enforces consistency with the mean inter-loop distance
    directly: a pass whose result would push 2*<r>/d above 1 -- loops wider than
    their own spacing -- is rejected and the population is reported as
    saturated. Saturation is not a numerical failure. It says the continuum
    state has no non-overlapping discrete representation, which is the regime
    the model's own coalescence channel exists to describe.

    Merging itself is area-conserving,

        r_new = sqrt(sum_i r_i^2),

    the same rule fullCD applies when one discrete loop takes an entire element
    group's defect content. The merged centre is the area-weighted centroid, so
    the first moment of the stored content is preserved too.
    """
    centers = pop.centers.copy()
    radii = pop.radii.copy()
    merged = pop.n_merged.copy()
    area0 = float(np.sum(radii ** 2))
    _, n_hat = family_geometry(pop.fam)
    if coplanar_tol == "plane":
        coplanar_tol = float(pop.fam["d_plane"])
    saturated = False
    n_passes = 0

    for _ in range(max_passes):
        if len(radii) < 2:
            break
        tree = cKDTree(centers)
        rmax = float(radii.max())
        # Union-find over overlapping pairs.
        parent = np.arange(len(radii))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        n_pairs = 0
        for i in range(len(radii)):
            for j in tree.query_ball_point(centers[i], radii[i] + rmax):
                if j <= i:
                    continue
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d >= radii[i] + radii[j]:
                    continue
                if coplanar_tol is not None:
                    if abs(float(n_hat @ (centers[i] - centers[j]))) > coplanar_tol:
                        continue
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
                    n_pairs += 1
        if n_pairs == 0:
            break

        roots = np.array([find(i) for i in range(len(radii))])
        new_c, new_r, new_m = [], [], []
        for root in np.unique(roots):
            sel = roots == root
            a = radii[sel] ** 2                      # area / pi
            new_r.append(float(np.sqrt(a.sum())))
            new_c.append((centers[sel] * a[:, None]).sum(0) / a.sum())
            new_m.append(int(merged[sel].sum()))
        cand_c = np.array(new_c)
        cand_r = np.array(new_r)
        cand_m = np.array(new_m)

        if spacing_guard and len(cand_r):
            d_new = (box_volume / len(cand_r)) ** (1 / 3.)
            if 2.0 * float(cand_r.mean()) / d_new > 1.0:
                saturated = True
                if verbose:
                    print(f"    pass rejected: 2r/d would reach "
                          f"{2*float(cand_r.mean())/d_new:.2f} -- saturated")
                break

        centers, radii, merged = cand_c, cand_r, cand_m
        n_passes += 1
        if verbose:
            print(f"    pass {n_passes}: {n_pairs} merges -> {len(radii)} loops")

    out = LoopPopulation(pop.fam, centers, radii, merged)
    area1 = float(np.sum(radii ** 2))
    if area0 > 0 and abs(area1 - area0) / area0 > 1e-10:
        raise AssertionError(
            f"coalescence lost area: {area0:.12e} -> {area1:.12e}")
    out.saturated = saturated
    out.passes = n_passes
    return out


# ── export ───────────────────────────────────────────────────────────────────
def write_microstructure(pops, out_file, box_shift=None):
    """Write a MoDELib3 ``aLoop`` individual-style microstructure file.

    The format is the one in ``MoDELib3/Library/Microstructures/aLoopsIndividual.txt``
    and is read by ``aLoopIndividualSpecification``. Every field is per-loop, so
    the four families go into a single file: ``planeIDs`` selects basal or
    prismatic and ``isVacancyLoop`` selects the sign of the Burgers vector,
    which ``aLoopGenerator::generateSingle`` then reconciles against the polygon
    winding.

    ``loopRadii_SI`` carries the CIRCUMRADIUS of the area-matched polygon, not
    the disc radius, because the generator puts its vertices on a circle of the
    radius it is handed.
    """
    plane, radii, sides, vac, cen = [], [], [], [], []
    for pop in pops:
        f = pop.fam
        for k in range(len(pop)):
            plane.append(f["plane_id"])
            radii.append(polygon_circumradius(pop.radii[k], f["sides"]) * B_SI)
            sides.append(f["sides"])
            vac.append(f["vacancy"])
            c = pop.centers[k] if box_shift is None else pop.centers[k] + box_shift
            cen.append(c)
    if not plane:
        raise ValueError("no loops to write")

    def row(v):
        return " ".join(f"{x:.10g}" for x in v)

    # NOTE the plural. MicrostructureGenerator dispatches on `type=="aLoops"`;
    # the shipped Library/Microstructures/aLoopsIndividual.txt example writes
    # `type=aLoop;`, which matches no branch and is silently not generated.
    # aLoopsDensity.txt in the same directory has it right.
    lines = ["type=aLoops;", "style=individual;",
             f"planeIDs={row(plane)};",
             f"loopRadii_SI={row(radii)};",
             f"loopSides={row(sides)};",
             "loopCenters=" + "\n".join(f"{c[0]:.10g} {c[1]:.10g} {c[2]:.10g}"
                                        for c in cen) + ";",
             f"isVacancyLoop={row(vac)};"]
    Path(out_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(plane)


def loop_polygon(fam, center, r_area, in_plane_ref=None):
    """Vertices of one loop, closed, in units of b.

    The polygon is drawn in the habit plane with the AREA of the disc it stands
    for, so the number of point defects it holds is exactly the number the
    continuum field assigned to it.

    Both families are drawn as circles: <c> as a 64-gon (see CIRCLE_SIDES) and
    <a> as a 16-gon, neither distinguishable from a disc at these radii. <c>
    was a hexagon until 16 August 2026, on the faceting argument recorded at
    CIRCLE_SIDES.
    """
    n_sides = int(fam["sides"])
    _, n_hat = family_geometry(fam)
    R = polygon_circumradius(r_area, n_sides)

    # In-plane frame. The starting vertex is placed along the a1 direction. It
    # mattered while <c> was a hexagon, whose edges then ran along <10-10> as
    # the faceting required; for a circle the choice is immaterial, and it is
    # kept only so that a given loop draws identically from run to run.
    ref = np.array([1.0, 0.0, 0.0]) if in_plane_ref is None else np.asarray(in_plane_ref)
    if abs(float(n_hat @ ref)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    u = ref - n_hat * float(n_hat @ ref)
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)

    th = np.arange(n_sides) * 2.0 * np.pi / n_sides
    return center[None, :] + R * (np.cos(th)[:, None] * u[None, :]
                                  + np.sin(th)[:, None] * v[None, :])


def _tube_quads(p0, p1, r_tube, n_facets=6):
    """Quads of a cylinder from ``p0`` to ``p1``. Vectorized over all edges.

    ``r_tube`` may be a scalar or one value per edge.
    """
    t = p1 - p0
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    t = t / ln
    # A reference not parallel to t, per edge.
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(t), 1))
    bad = np.abs((t * ref).sum(1)) > 0.9
    ref[bad] = np.array([1.0, 0.0, 0.0])
    u = ref - t * (t * ref).sum(1)[:, None]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(t, u)

    th = np.arange(n_facets) * 2.0 * np.pi / n_facets
    c, s = np.cos(th), np.sin(th)
    rt = np.broadcast_to(np.asarray(r_tube, dtype=float).reshape(-1, 1, 1),
                         (len(t), 1, 1))
    # ring[e, k] = offset of facet k on edge e
    ring = (rt * (c[None, :, None] * u[:, None, :]
                  + s[None, :, None] * v[:, None, :]))
    a0 = p0[:, None, :] + ring
    a1 = p1[:, None, :] + ring
    k1 = (np.arange(n_facets) + 1) % n_facets
    quads = np.stack([a0, a0[:, k1], a1[:, k1], a1], axis=2)
    return quads.reshape(-1, 4, 3)


def _densify_closed(pts, max_step):
    """Resample a closed polygon so no edge is longer than ``max_step``.

    Only used before clipping the drawn line at the crystal surface: the tube is
    built one cylinder per edge and clipped whole cylinders at a time, so the
    cut would otherwise land up to half an edge away from the wall. This matters
    most for the large <c> loops -- at 1 dpa their radius reaches 47 nm, giving
    a 4.6 nm edge even at 64 sides. Adding collinear points along each edge does
    not change the shape and moves the cut onto the surface.
    """
    pts = np.asarray(pts, float)
    closed = np.vstack([pts, pts[:1]])
    out = []
    for a, b in zip(closed[:-1], closed[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / max_step)), 1)
        out.append(a + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * (b - a))
    return np.concatenate(out)


def render(pops, box_lo, box_hi, out_file, title="", domain_pts=None,
           max_loops=4000, elev=22.0, azim=-58.0, dpi=150, verbose=True,
           orientation=True, clip_to_domain=True):
    """Draw the discrete population as tubular dislocation lines.

    Tube width is a drawing choice only -- a dislocation line has no thickness --
    and is set per loop by ``tube_radius_nm``, proportional to the loop's own
    radius so the large <c> loops do not read as wire next to the small <a>
    loops.

    ``domain_pts`` is the CD node cloud. Given it, the outline drawn is the
    actual single crystal -- a hexagonal prism for a `GEOMETRY['type'] =
    'hexagonal'` run -- and the axes take its true proportions. Without it the
    outline falls back to the bounding box, which for a prism is a container the
    crystal does not fill and which shows loops apparently floating in its empty
    corners.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from dislocluster_code.post.fields import _shaded_facecolors
    from matplotlib.colors import to_rgb

    nm = B_SI * 1e9
    fig = plt.figure(figsize=(9.0, 7.2))
    # computed_zorder=False so the crystal outline is drawn OVER the loops
    # rather than depth-sorted as one artist against them; otherwise the edges
    # behind a large <c> loop disappear and the prism stops closing.
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Loop CENTRES are inside the crystal, but a loop is not a point: at 1 dpa
    # the <c> radius is 47 nm against a 100 nm prism half-width, so a loop
    # centred anywhere but the middle overhangs a wall. The drawn line is
    # therefore clipped at the crystal surface -- which is also what a loop
    # meeting a grain boundary does. Only the DRAWING is clipped; the full loop
    # is what goes into loops_*.csv and aLoops_*.txt.
    clip = None
    if domain_pts is not None and clip_to_domain:
        try:
            clip = domain_faces(np.asarray(domain_pts, float) * nm)
        except Exception:
            clip = None

    total = 0
    for pop in pops:
        if not len(pop):
            continue
        fam = pop.fam
        idx = np.arange(len(pop))
        if len(idx) > max_loops:
            idx = np.random.default_rng(0).choice(idx, max_loops, replace=False)
            if verbose:
                print(f"    {fam['key']}: drawing {max_loops}/{len(pop)} loops")
        segs0, segs1, rt = [], [], []
        for k in idx:
            pts = loop_polygon(fam, pop.centers[k], pop.radii[k]) * nm
            if clip is not None:
                pts = _densify_closed(pts, 1.5)      # nm
            segs0.append(pts)
            segs1.append(np.roll(pts, -1, axis=0))
            rt.append(np.full(len(pts), tube_radius_nm(pop.radii[k] * nm)))
        p0 = np.concatenate(segs0)
        p1 = np.concatenate(segs1)
        quads = _tube_quads(p0, p1, np.concatenate(rt))
        if clip is not None:
            N, b = clip
            inside = np.all(quads.mean(axis=1) @ N.T <= b[None, :], axis=1)
            quads = quads[inside]
        if not len(quads):
            continue
        rgb = np.asarray(to_rgb(fam["color"]))
        ax.add_collection3d(Poly3DCollection(
            quads, facecolors=_shaded_facecolors(quads, rgb), linewidths=0,
            zorder=1))
        total += len(idx)

    lo, hi = np.asarray(box_lo) * nm, np.asarray(box_hi) * nm
    if domain_pts is not None:
        # Hidden edges dotted, as in the field figures.
        draw_domain_wireframe(ax, np.asarray(domain_pts, float) * nm, lo, hi,
                              view=(elev, azim), color="0.45", lw=0.9)
    else:
        for s, e in _box_edges(lo, hi):
            ax.plot(*zip(s, e), color="0.6", lw=0.6, zorder=6)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    # True proportions: a 200 x 173 x 320 nm prism is not a cube, and a fixed
    # (1, 1, 1) box aspect draws it as one.
    span = hi - lo
    ax.set_box_aspect(tuple(span / span.max()))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)

    handles = [plt.Line2D([], [], color=f["color"], lw=4,
                          label=f"{f['label']}  b = {f['b_tex']}   ({len(p)})")
               for f, p in ((p.fam, p) for p in pops) if len(p)]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    # After tight_layout, which moves the axes the triad measures itself from.
    if orientation:
        _draw_orientation_triad(ax, (elev, azim), origin=(0.10, 0.12),
                                length_in=0.42, fontsize=10.0)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> {Path(out_file).name} ({total} loops drawn)")
    return out_file


def _box_edges(lo, hi):
    c = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]),
         (lo[0], hi[1], lo[2]), (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
         (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in pairs]


def write_table(pops, out_file, box_shift=None):
    """Per-loop CSV: everything DD needs, in SI and in units of b."""
    rows = ["family,plane_id,is_vacancy,sides,"
            "x_b,y_b,z_b,x_nm,y_nm,z_nm,"
            "r_b,r_nm,r_circum_nm,"
            "bx_b,by_b,bz_b,b_mag_b,nx,ny,nz,n_merged"]
    nm = B_SI * 1e9
    for pop in pops:
        f = pop.fam
        b_vec, n_hat = family_geometry(f)
        for k in range(len(pop)):
            c = pop.centers[k] if box_shift is None else pop.centers[k] + box_shift
            r = float(pop.radii[k])
            rc = polygon_circumradius(r, f["sides"])
            rows.append(
                f"{f['key']},{f['plane_id']},{f['vacancy']},{f['sides']},"
                f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
                f"{c[0]*nm:.4f},{c[1]*nm:.4f},{c[2]*nm:.4f},"
                f"{r:.6f},{r*nm:.4f},{rc*nm:.4f},"
                f"{b_vec[0]:.6f},{b_vec[1]:.6f},{b_vec[2]:.6f},{f['b_dd']:.6f},"
                f"{n_hat[0]:.6f},{n_hat[1]:.6f},{n_hat[2]:.6f},"
                f"{int(pop.n_merged[k])}")
    Path(out_file).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows) - 1


# ── driver helpers ───────────────────────────────────────────────────────────
def populate(nodes, F, weights, faces, region="interior", coalesce_pass=True,
             coplanar_tol=None, seed=0, positions_P=None, return_raw=False):
    """Place the discrete population for ONE CD block. ``([pop], stats, box)``.

    The half of :func:`build` that does the physics, split out so it can be
    driven from a LIVE march state as well as from a finished run's snapshots
    -- the runtime continuum->discrete transition (plan 4b) needs exactly this
    and none of the run-directory loading around it.

    Parameters
    ----------
    nodes : (N,3)   CD node positions, in b. The convex body they fill IS the
                    crystal: the volume that turns a density into a count, the
                    region loops are placed in and the clip all come from it.
    F : (N,12)      one CD block, ``[4 mobile | 4 number | 4 content]`` -- the
                    layout `movies.cd_blocks` returns and `FAMILIES` indexes
                    through ``ncol``/``ccol``. Build it from a marched state as
                    ``np.column_stack([Y[:, :4], immobile_0d_to_modelib(Y, omega)])``.
    weights : (N,)  nodal volume weights, normalized over the whole domain.
    faces           convex-hull faces, or None to fall back to the bounding box.
    positions_P     node cloud used for the box and for placement; defaults to
                    ``nodes``.
    """
    P = nodes if positions_P is None else positions_P
    L = P.max(0) - P.min(0)
    box_volume = float(np.prod(L)) if faces is None else domain_volume(P)

    d_face_nm = gb_distance(nodes) * B_SI * 1e9
    if region == "interior":
        cutoff = 0.25 * float(d_face_nm.max()) * 2.0   # innermost quartile
        mask = d_face_nm > cutoff
        # Renormalize weights over the region so counts scale to the FULL box:
        # the interior density is what a bulk DD cell should carry.
        wr = weights * mask
        wr = wr / wr.sum()
    elif region == "domain":
        mask = np.ones(len(nodes), dtype=bool)
        wr = weights
    else:
        raise ValueError(f"region must be 'interior' or 'domain', got {region!r}")

    rng = np.random.default_rng(seed)
    pops, stats, raws = [], [], []
    for fam in FAMILIES:
        raw = sample_family(nodes, F, wr, box_volume, fam, mask=mask, rng=rng,
                            positions=("uniform" if region == "interior"
                                       else "field"),
                            box=(P.min(0), P.max(0)), faces=faces)
        n0, r0, a0 = len(raw), float(raw.radii.mean()) if len(raw) else 0.0, raw.total_area
        pop = (coalesce(raw, box_volume, coplanar_tol=coplanar_tol)
               if coalesce_pass else raw)
        nm = B_SI * 1e9
        d0 = (box_volume / n0) ** (1 / 3.) * nm if n0 else float("nan")
        d1 = (box_volume / len(pop)) ** (1 / 3.) * nm if len(pop) else float("nan")
        stats.append(dict(
            family=fam["key"], n_before=n0, n_after=len(pop),
            r_before_nm=r0 * nm,
            r_after_nm=float(pop.radii.mean()) * nm if len(pop) else 0.0,
            d_before_nm=d0, d_after_nm=d1,
            ratio_before=2 * r0 * nm / d0 if n0 else float("nan"),
            ratio_after=(2 * float(pop.radii.mean()) * nm / d1) if len(pop) else float("nan"),
            area_b2=pop.total_area,
            area_conserved=bool(abs(pop.total_area - a0) <= 1e-9 * max(a0, 1.0)),
            stored_defects=pop.stored_defects,
            saturated=bool(getattr(pop, "saturated", False)),
            max_merged=int(pop.n_merged.max()) if len(pop) else 0))
        pops.append(pop)
        raws.append(raw)
    if return_raw:
        # The pre-coalescence draw. The transfer ledger needs it because
        # coalescence conserves AREA and not COUNT, so loop number is only
        # comparable with the continuum before that step runs.
        return pops, stats, (P.min(0), P.max(0)), box_volume, L, raws
    return pops, stats, (P.min(0), P.max(0)), box_volume, L


def domain_weights(nodes, mc_samples=2_000_000, verbose=False):
    """``(weights, faces)`` for a node cloud -- the expensive part of `build`."""
    faces = None
    try:
        faces = domain_faces(nodes)
    except Exception:
        pass
    return voronoi_weights(nodes, mc_samples, verbose=verbose,
                           faces=faces), faces


def build(run_dir, dose, region="interior", coalesce_pass=True,
          coplanar_tol=None, seed=0, mc_samples=2_000_000, verbose=True,
          _cache={}):
    """``(dose_actual, [LoopPopulation], stats)`` for one dose."""
    key = str(run_dir)
    if key not in _cache:
        doses, nodes, frames = movies_mod.cd_blocks(run_dir)
        # The domain is whatever convex body the nodes fill: a cube for the
        # reference case, a hexagonal prism for a `GEOMETRY['type']='hexagonal'`
        # run. EVERYTHING below is keyed on that rather than on the bounding
        # box -- the nodal volumes, the total volume that turns a density into
        # a count, and the region the loops are placed in. On a hexagonal prism
        # the bounding box is 4/3 = 1.33x the crystal, so using it
        # over-counted the loops by a third and scattered them through six empty
        # wedges outside the outline.
        w, faces = domain_weights(nodes, mc_samples)
        _cache[key] = (doses, nodes, frames, w, faces)
    doses, nodes, frames, w, faces = _cache[key]

    i = int(np.argmin(np.abs(np.asarray(doses) - dose)))
    P, F = frames[i]
    pops, stats, box, box_volume, L = populate(
        nodes, F, w, faces, region=region, coalesce_pass=coalesce_pass,
        coplanar_tol=coplanar_tol, seed=seed, positions_P=P)

    if verbose:
        print(f"  dose {doses[i]:.4g} dpa, region={region}, box {L[0]:.0f} b "
              f"= {L[0]*B_SI*1e9:.0f} nm")
        print(f"    {'fam':<4} {'N':>6} {'->N':>6} {'r[nm]':>8} {'->r[nm]':>8} "
              f"{'d[nm]':>7} {'2r/d':>7} {'->2r/d':>7} {'mrg':>4}  flag")
        for s in stats:
            flag = "SATURATED" if s["saturated"] else ""
            print(f"    {s['family']:<4} {s['n_before']:>6d} {s['n_after']:>6d} "
                  f"{s['r_before_nm']:>8.2f} {s['r_after_nm']:>8.2f} "
                  f"{s['d_after_nm']:>7.1f} {s['ratio_before']:>7.3f} "
                  f"{s['ratio_after']:>7.3f} {s['max_merged']:>4d}  {flag}")
    # P goes back too: the render needs the node cloud to outline the actual
    # crystal rather than its bounding box.
    return float(doses[i]), pops, stats, (P.min(0), P.max(0)), P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--doses", nargs="+", default=["1e-4", "1e-2", "1", "10"],
                    help="dose values, or the single word 'all' for every "
                         "snapshot in march_state.npz")
    ap.add_argument("--region", default="interior",
                    choices=["interior", "domain"])
    ap.add_argument("--no-coalesce", action="store_true")
    ap.add_argument("--coplanar-tol", type=float, default=None,
                    help="b; require near-coplanarity before merging")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--tube-nm", type=float, default=1.2)
    ap.add_argument("--no-clip", action="store_true",
                    help="draw whole loops even where they cross a domain "
                         "face; the default cuts the drawn line at the crystal "
                         "surface so nothing is shown outside the outline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run / "discrete_loops"
    out.mkdir(parents=True, exist_ok=True)

    if len(args.doses) == 1 and str(args.doses[0]).lower() == "all":
        z = np.load(run / "march_state.npz")
        dose_list = [float(x) for x in z["doses"]]
    else:
        dose_list = [float(x) for x in args.doses]
    verbose_fig = len(dose_list) <= 8

    print(f"{run.name}: discrete loop population, region={args.region}, "
          f"{len(dose_list)} dose steps\n")
    manifest = []
    for n_done, dose in enumerate(dose_list, 1):
        cop = "plane" if args.coplanar_tol is None else args.coplanar_tol
        d, pops, stats, box, P = build(run, dose, region=args.region,
                                    coalesce_pass=not args.no_coalesce,
                                    coplanar_tol=cop, seed=args.seed,
                                    verbose=verbose_fig)
        tag = f"{d:.4g}".replace(".", "p").replace("-", "m") + "dpa"
        if sum(len(p) for p in pops) == 0:
            # The pristine snapshot at dose 0 holds every species at the 1e-20
            # floor, so there is genuinely nothing to discretize. Skip rather
            # than write an empty microstructure file that would fail to parse.
            print(f"  [{n_done:>3}/{len(dose_list)}] {d:>10.4g} dpa  "
                  f"     0 loops  (pristine, skipped)")
            continue
        nloops = write_microstructure(pops, out / f"aLoops_{tag}.txt")
        write_table(pops, out / f"loops_{tag}.csv")
        if not args.no_figures:
            render(pops, box[0], box[1], out / f"loops_{tag}.png",
                   title=f"discrete loop population at {d:.4g} dpa "
                         f"({nloops} loops, {args.region})",
                   domain_pts=P, clip_to_domain=not args.no_clip,
                   verbose=verbose_fig)
            # One figure per family. The combined view is dominated by whichever
            # family is largest -- at 10 dpa the <c> loops are 10x the <a>
            # loops and hide them -- so each population also gets its own panel.
            for pop in pops:
                f = pop.fam
                render([pop], box[0], box[1],
                       out / f"loops_{f['key']}_{tag}.png",
                       title=(f"{f['label']} loops at {d:.4g} dpa   "
                              f"b = {f['b_tex']}   ({len(pop)} loops)"),
                       domain_pts=P, clip_to_domain=not args.no_clip,
                   verbose=verbose_fig)
        manifest.append(dict(dose=d, tag=tag, n_loops=nloops, stats=stats))
        if verbose_fig:
            print(f"    -> aLoops_{tag}.txt, loops_{tag}.csv "
                  f"({nloops} loops)\n")
        else:
            sat = ",".join(s["family"] for s in stats if s["saturated"])
            print(f"  [{n_done:>3}/{len(dose_list)}] {d:>10.4g} dpa  "
                  f"{nloops:>6d} loops  {tag}"
                  + (f"   saturated: {sat}" if sat else ""))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
