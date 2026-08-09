"""Field plots from MoDELib2-NNL (Zr3d_ghoniem) cluster-dynamics output.

Renders the spatially-resolved mobile and immobile fields on a single interior
cut plane, viewed almost frontally, one row per species and one column per dose
with a shared colour bar per row. Loop populations can be overlaid one family at
a time as discrete platelets (visualization only).

Input files, both written by the `zr3d_ghoniem` branch:
    evl/cdNodes.txt   finite-element node coordinates [b], one row per node
    evl/evl_<N>.txt   the configuration at output step N; the trailing
                      (mSize+iSize)-column block is the CD field, in the SAME
                      node order as cdNodes.txt

The node file is essential: the CD trial functions live on second-order elements,
so the field rows do not correspond to the vertices of the .msh file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

__all__ = ["load_cd_fields", "plot_field_panels", "SPECIES", "FAMILIES", "B_SI",
           "OMEGA_SI", "OMEGA_B3"]

B_SI = 3.233e-10          # Burgers vector magnitude [m]
OMEGA_SI = 1.2e-29        # atomic volume [m^3]  (ZrMicro physical_props['Omega'])
OMEGA_B3 = OMEGA_SI / B_SI ** 3

# Column layout of the CD block: 4 mobile, then 4 immobile densities, then 4 contents
SPECIES = {
    "Cv":   (0,  r"$C_v$"),
    "Ci":   (1,  r"$C_i$"),
    "C2i":  (2,  r"$C_{2i}$"),
    "C3i":  (3,  r"$C_{3i}$"),
    "n_vL": (4,  r"$N_{\langle c\rangle}$"),
    "n_a1": (5,  r"$N_{\langle a\rangle 1}$"),
    "n_a2": (6,  r"$N_{\langle a\rangle 2}$"),
    "n_a3": (7,  r"$N_{\langle a\rangle 3}$"),
    "c_vL": (8,  r"$C_{\langle c\rangle}$"),
    "c_a1": (9,  r"$C_{\langle a\rangle 1}$"),
    "c_a2": (10, r"$C_{\langle a\rangle 2}$"),
    "c_a3": (11, r"$C_{\langle a\rangle 3}$"),
}

# Immobile families: (label, density column, content column, |b| in units of b,
#                     habit-plane normal in Cartesian b, colour)
FAMILIES = [
    (r"$\langle c\rangle$",    4, 8,  1.632993, np.array([0.0, 0.0, 1.0]),        "#1f4fbf"),
    (r"$\langle a\rangle_1$",  5, 9,  1.0,      np.array([1.0, 0.0, 0.0]),        "#c62828"),
    (r"$\langle a\rangle_2$",  6, 10, 1.0,      np.array([0.5, 0.8660254, 0.0]),  "#d84315"),
    (r"$\langle a\rangle_3$",  7, 11, 1.0,      np.array([-0.5, 0.8660254, 0.0]), "#ad1457"),
]

_JETISH = LinearSegmentedColormap.from_list(
    "cd_jet",
    ["#00007f", "#0000ff", "#007fff", "#00ffff", "#7fff7f",
     "#ffff00", "#ff7f00", "#ff0000", "#7f0000"],
)


def load_cd_fields(evl_dir, step, n_cd_cols=12):
    """Return (positions [n,3] in b, field [n,n_cd_cols]) for one output step."""
    evl_dir = Path(evl_dir)
    node_file = evl_dir / "cdNodes.txt"
    if not node_file.exists():
        raise FileNotFoundError(
            f"{node_file} not found. It is written by ClusterDynamicsFEM::"
            "writeNodePositions() on the zr3d_ghoniem branch; without it the "
            "field rows cannot be located in space.")
    P = np.loadtxt(node_file)
    rows = [ln.split() for ln in open(evl_dir / f"evl_{step}.txt")]
    F = np.array([r for r in rows if len(r) == n_cd_cols], dtype=float)
    if F.shape[0] != P.shape[0]:
        raise ValueError(
            f"node/field count mismatch: {P.shape[0]} nodes vs {F.shape[0]} field "
            f"rows -- cdNodes.txt is from a different mesh than evl_{step}.txt")
    return P, F


def domain_faces(P, tol=1e-6):
    """Unique outward face planes (normals, offsets) of the convex node cloud.

    Returns `(N, b)` with `N @ x <= b` for every interior point. Coplanar hull
    facets are merged, so a cube gives 6 planes and a hexagonal prism gives 8.
    """
    from scipy.spatial import ConvexHull
    eq = ConvexHull(P).equations              # rows: [n_x, n_y, n_z, d], n.x + d <= 0
    key = np.round(eq / max(np.abs(eq).max(), 1e-30), 6)
    _, keep = np.unique(key, axis=0, return_index=True)
    eq = eq[np.sort(keep)]
    return eq[:, :3], -eq[:, 3]


def gb_distance(P, lo=None, hi=None, faces=None):
    """Distance from the nearest domain face, in b.

    Every outer face carries the Dirichlet condition (the solver builds its node
    list with ExternalAndInternalBoundary), so the whole surface is grain
    boundary and the distance is the minimum over all faces of the domain.

    The faces are taken from the convex hull of the node cloud, so this is
    correct for any convex domain -- a hexagonal prism's slanted prism faces as
    much as a cube's. For a cube the hull reproduces the bounding box exactly,
    so results are unchanged. Passing `lo`/`hi` explicitly forces the old
    axis-aligned-box formula.
    """
    if lo is not None or hi is not None:
        lo = P.min(0) if lo is None else lo
        hi = P.max(0) if hi is None else hi
        return np.minimum(P - lo, hi - P).min(axis=1)
    try:
        N, b = domain_faces(P) if faces is None else faces
    except Exception:                          # degenerate cloud -> box fallback
        lo, hi = P.min(0), P.max(0)
        return np.minimum(P - lo, hi - P).min(axis=1)
    return (b[None, :] - P @ N.T).min(axis=1)


def _plane_grid(lo, hi, plane, n):
    """Corner and in-plane spanning vectors of the mid-cut normal to `plane`."""
    mid = 0.5 * (lo + hi)
    ex, ey, ez = hi - lo
    if plane == "x":
        return (np.array([mid[0], lo[1], lo[2]]),
                np.array([0, ey, 0]), np.array([0, 0, ez]))
    if plane == "y":
        return (np.array([lo[0], mid[1], lo[2]]),
                np.array([ex, 0, 0]), np.array([0, 0, ez]))
    return (np.array([lo[0], lo[1], mid[2]]),
            np.array([ex, 0, 0]), np.array([0, ey, 0]))


def _sample_plane(tree, values, corner, u_vec, v_vec, n, k=4):
    """Inverse-distance sample of `values` on a plane.

    Weighted over k neighbours rather than nearest-node: the mesh is
    unstructured, and plain nearest sampling gives a faceted Voronoi texture
    right across the depletion shell, where the field varies fastest.
    """
    s = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(s, s, indexing="ij")
    pts = (corner[None, None, :]
           + U[..., None] * u_vec[None, None, :]
           + V[..., None] * v_vec[None, None, :])
    dist, idx = tree.query(pts.reshape(-1, 3), k=k)
    w = 1.0 / np.maximum(dist, 1.0e-12)
    sampled = ((values[idx] * w).sum(1) / w.sum(1)).reshape(U.shape)
    return pts[..., 0], pts[..., 1], pts[..., 2], sampled


def _draw_box(ax, lo, hi, color="0.4", lw=0.6):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(c[i], c[j]), color=color, lw=lw)


def domain_edges(P, tol=1e-6):
    """Edges of the convex domain the node cloud fills, as (M, 2, 3) segments.

    An edge of a convex polytope is where two faces meet, so the faces are
    grouped by plane (coplanar hull facets merged) and every pair of planes
    sharing two or more hull vertices contributes the segment between the two
    extreme shared vertices. A cube gives its 12 edges, a hexagonal prism its
    18 — as opposed to the bounding box, which for a prism is not the domain.
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(P)
    V = P[hull.vertices]
    N, b = domain_faces(P, tol)
    scale = max(np.abs(b).max(), 1e-30)
    on = [np.flatnonzero(np.abs(V @ N[k] - b[k]) < 1e-6 * scale)
          for k in range(len(N))]
    segs = []
    for i in range(len(N)):
        for j in range(i + 1, len(N)):
            shared = np.intersect1d(on[i], on[j])
            if shared.size < 2:
                continue
            Q = V[shared]
            d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
            a, c = np.unravel_index(np.argmax(d), d.shape)
            segs.append(np.stack([Q[a], Q[c]]))
    return np.asarray(segs) if segs else np.empty((0, 2, 3))


def _draw_domain(ax, P, lo, hi, color="0.4", lw=0.6):
    """Outline the actual domain; fall back to the bounding box if degenerate."""
    try:
        segs = domain_edges(P)
    except Exception:
        segs = np.empty((0, 2, 3))
    if len(segs) == 0:
        _draw_box(ax, lo, hi, color=color, lw=lw)
        return
    for s in segs:
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=color, lw=lw)


def _inside(pts, faces, tol_frac=1e-9):
    """Boolean mask: which points lie inside the convex domain."""
    N, b = faces
    tol = tol_frac * max(np.abs(b).max(), 1e-30)
    return np.all(pts @ N.T <= b[None, :] + tol, axis=-1)


def _platelet_frame(normal):
    """Orthonormal frame with the third axis along `normal`."""
    nvec = np.asarray(normal, float) / np.linalg.norm(normal)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, nvec)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(nvec, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(nvec, e1)
    return np.column_stack([e1, e2, nvec])


def _platelet_quads(centre, radius, R, n=10, thickness=0.28):
    """Quads (Q,4,3) for one loop: `radius` in-plane, `thickness` along the normal.

    The thickness is deliberately generous. A true platelet has an aspect ratio of
    order 1/100, and the prismatic <a> families have their normals in the basal
    plane, so at any near-horizontal viewing angle at least one family is seen
    almost exactly edge-on and degenerates to a line. Rendering them as thick
    discs keeps every family legible from a single shared viewpoint.

    Returns geometry rather than drawing it: filling the domain means thousands
    of platelets, and one matplotlib artist each (the old plot_surface call) is
    orders of magnitude slower than a single batched Poly3DCollection. `n` is
    kept low for the same reason -- matplotlib z-sorts every polygon in the
    collection, so the quad count per platelet sets the render time.
    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    a, c = radius, radius * thickness
    x = a * np.outer(np.cos(u), np.sin(v))
    y = a * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x, y, z], -1) @ R.T + np.asarray(centre, float)
    return np.stack([pts[:-1, :-1], pts[1:, :-1], pts[1:, 1:], pts[:-1, 1:]],
                    axis=2).reshape(-1, 4, 3)


def _shaded_facecolors(quads, base_rgb, light=(0.3, -0.8, 0.5)):
    """Lambert-ish shading so a batched collection still reads as 3-D."""
    e1 = quads[:, 1] - quads[:, 0]
    e2 = quads[:, 2] - quads[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.where(ln > 0, ln, 1.0))
    lv = np.asarray(light, float)
    lv = lv / np.linalg.norm(lv)
    s = 0.55 + 0.45 * np.abs(nrm @ lv)
    return np.clip(np.asarray(base_rgb)[None, :] * s[:, None], 0.0, 1.0)


def _overlay_family(ax, P, F, lo, hi, rng, fam, n_loops, loop_scale, plane,
                    max_loops=2500, stop_after_misses=1500, faces=None):
    """Draw one loop family as platelets sized by the LOCAL mean loop radius.

    Sites are drawn from the WHOLE domain, not from a slab about the cut plane,
    so the population is represented everywhere including the near-boundary
    region where the size varies most.

    The drawn radius is the true local radius times a single uniform factor, with
    NO upper clamp. An earlier version capped it, which silently flattened every
    platelet to the cap and destroyed exactly the size information the figure is
    meant to carry.

    Placement is a greedy non-overlap fill. Candidates are visited in random
    order and accepted only if the new platelet's drawn radius clears every
    already-accepted one, so no two platelets intersect and the figure can be
    filled to capacity without turning into a solid mass. `n_loops=None` fills
    the domain (up to `max_loops`); an integer caps the accepted count.

    The clearance test compares centre distance against the sum of the two drawn
    radii — the platelets' bounding spheres. That is conservative for discs,
    which may therefore end up slightly further apart than strictly necessary;
    the alternative, exact disc-disc intersection in 3-D, is not worth it here.
    """
    _, ncol, ccol, bmag, normal, color = fam
    # Keep the seed sites off the surface. The margin is a distance from the
    # NEAREST DOMAIN FACE, not from the bounding box: on a hexagonal prism the
    # box test leaves nodes right against the six slanted prism faces, whose
    # platelets then hang outside the crystal.
    d_face = gb_distance(P, faces=faces)
    cand = np.flatnonzero(d_face > 0.02 * d_face.max())
    if cand.size == 0:
        return

    n_k = F[cand, ncol]
    c_k = F[cand, ccol]
    good = (np.isfinite(n_k) & np.isfinite(c_k) & (n_k > 0) & (c_k > 0))
    cand, n_k, c_k = cand[good], n_k[good], c_k[good]
    if cand.size == 0:
        return

    m = c_k / (n_k * OMEGA_B3)                          # defects per loop
    good = m > 1.0
    cand, m = cand[good], m[good]
    if cand.size == 0:
        return

    r_drawn = np.sqrt(m * OMEGA_B3 / (np.pi * bmag)) * loop_scale
    pts = P[cand]
    tree = cKDTree(pts)
    r_max = float(r_drawn.max())
    cap = max_loops if n_loops is None else min(int(n_loops), max_loops)

    accepted = []
    acc_set = np.zeros(len(cand), dtype=bool)
    misses = 0
    for i in rng.permutation(len(cand)):
        if len(accepted) >= cap or misses >= stop_after_misses:
            break
        # Anything that could touch this platelet lies within r_i + r_max.
        near = tree.query_ball_point(pts[i], r_drawn[i] + r_max)
        clash = False
        for j in near:
            if acc_set[j]:
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < r_drawn[i] + r_drawn[j]:
                    clash = True
                    break
        if clash:
            misses += 1
            continue
        misses = 0
        acc_set[i] = True
        accepted.append(i)

    if not accepted:
        return
    R = _platelet_frame(normal)
    quads = np.concatenate([_platelet_quads(pts[i], r_drawn[i], R)
                            for i in accepted])
    rgb = np.asarray(to_rgb(color))
    ax.add_collection3d(Poly3DCollection(
        quads, facecolors=_shaded_facecolors(quads, rgb), linewidths=0))
    return len(accepted)


_DEFAULT_VIEW = {"x": (10.0, -80.0), "y": (10.0, -80.0), "z": (55.0, -70.0)}
# Two orthogonal cuts need a viewpoint that is edge-on to neither. The
# single-cut angles are nearly frontal (elev 10) or nearly overhead (elev 55);
# either reduces one of the two planes to a line.
_MULTI_VIEW = (26.0, -58.0)


def plot_field_panels(evl_dir, steps, doses, species=("Cv", "Ci"),
                      plane="y", view=None, n_slice=200,
                      log=None, floor_decades=5.0, vlims=None,
                      loop_family=None, n_loops=None, loop_scale=45.0, seed=0,
                      figsize_per_panel=(3.3, 3.2), out_file=None, title=None,
                      column_titles=True, fields=None):
    """Panel figure: one row per species, one column per dose.

    Parameters
    ----------
    plane        : "x", "y" or "z" -- ONE interior mid-cut normal to this axis
    view         : (elev, azim), or None to pick one from `plane`. An x/y cut is
                   viewed almost frontally (10, -80), tilted just enough to read
                   as a plane in 3-D rather than flat; a z cut is horizontal and
                   would be edge-on from there, so it is viewed from above
                   (55, -70). On a hexagonal prism the z cut is the informative
                   one -- it shows the hexagonal section and the depletion rim on
                   all six prism faces at once.
    log          : force log/linear colour scale; None picks log when the field
                   spans more than two decades
    floor_decades: on a log scale, how many decades below the maximum to show.
                   This sets how much colour the near-boundary region gets: the
                   interior is a plateau, so all of the structure lives in the
                   depletion shell and a scale anchored at the true minimum
                   (which reaches the Dirichlet value, ~1e-27 for interstitials)
                   would waste the entire colour range on it.
    loop_family  : index into FAMILIES to overlay one loop population, or None
    loop_scale   : exaggeration of the drawn loop radii. NOT to scale -- at ~13 b
                   the mean radius is 0.4% of a 3093 b box. Radii come from the
                   real local field, so relative sizes stay meaningful.
    n_loops      : platelets to draw. None fills the domain with as many
                   non-overlapping platelets as fit (see _overlay_family).
    column_titles: draw the "<dose> dpa" header above each column. Turn this off
                   for a single-dose figure whose `title` already names the dose,
                   otherwise the two headers overlap.
    """
    rng = np.random.default_rng(seed)
    # `plane` may be a single axis or several. Two orthogonal mid-cuts drawn in
    # the SAME axes -- one vertical (normal to y) and one horizontal (normal to
    # z) -- show the boundary layer on four faces at once and reveal whether the
    # field is genuinely isotropic about the centre, which a single cut cannot.
    planes = (plane,) if isinstance(plane, str) else tuple(plane)
    if view is None:
        view = (_MULTI_VIEW if len(planes) > 1
                else _DEFAULT_VIEW.get(planes[0], (10.0, -80.0)))
    # `fields` lets a caller supply {step: (P, F)} directly instead of reading
    # evl files. The movie driver needs this: its frames come from
    # march_state.npz, which stores the state in ZrMicro's 19-variable form
    # under the true dose of each snapshot, and so does not depend on the
    # evl filename convention at all.
    data = (dict(fields) if fields is not None
            else {st: load_cd_fields(evl_dir, st) for st in steps})
    P0 = data[steps[0]][0]
    lo, hi = P0.min(0), P0.max(0)
    trees = {st: cKDTree(data[st][0]) for st in steps}
    grids = [_plane_grid(lo, hi, p, n_slice) for p in planes]

    # The domain is whatever convex body the nodes fill -- a cube for the
    # reference case, a hexagonal prism for the single-crystal case. Both the
    # outline and the cut plane are built from it rather than from the bounding
    # box, which for a prism contains six wedges of empty space.
    try:
        faces = domain_faces(P0)
    except Exception:
        faces = None

    nrow, ncol = len(species), len(steps)
    fig = plt.figure(figsize=(figsize_per_panel[0] * ncol + 1.7,
                              figsize_per_panel[1] * nrow))

    for r, sp in enumerate(species):
        col_idx, label = SPECIES[sp]
        allv = np.concatenate([data[st][1][:, col_idx] for st in steps])
        allv = allv[np.isfinite(allv)]
        vmax = float(np.percentile(allv, 99.9))
        vmin = float(allv[allv > 0].min()) if (allv > 0).any() else 0.0
        if vlims and sp in vlims:
            vmin, vmax = vlims[sp]
        use_log = log if log is not None else (vmin > 0 and vmax / vmin > 100.0)
        if use_log:
            norm = LogNorm(vmin=max(vmin, vmax * 10.0 ** (-floor_decades)), vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        for c, (st, dose) in enumerate(zip(steps, doses)):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            P, F = data[st]
            for corner, uvec, vvec in grids:
                X, Y, Z, S = _sample_plane(trees[st], F[:, col_idx],
                                           corner, uvec, vvec, n_slice)
                fc = _JETISH(norm(S))
                if faces is not None:
                    # The plane grid spans the bounding box, so on a non-box
                    # domain part of it lies outside the crystal. Those cells
                    # are made transparent instead of being painted with an
                    # extrapolated value -- plot_surface colours a cell from its
                    # lower corner, so a cell is dropped if ANY of its four
                    # corners is outside.
                    ins = _inside(np.stack([X, Y, Z], -1), faces)
                    cell = (ins[:-1, :-1] & ins[1:, :-1] &
                            ins[:-1, 1:] & ins[1:, 1:])
                    fc[..., 3] = 0.0
                    fc[:-1, :-1, 3] = cell
                ax.plot_surface(X, Y, Z, facecolors=fc, shade=False,
                                rstride=1, cstride=1, linewidth=0,
                                antialiased=False)
            _draw_domain(ax, P0, lo, hi)
            if loop_family is not None:
                _overlay_family(ax, P, F, lo, hi, rng, FAMILIES[loop_family],
                                n_loops, loop_scale, planes[0], faces=faces)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            # True proportions. A fixed (1,1,1) box renders the 400 x 346 x 653 nm
            # prism as though it were a cube.
            _sp = hi - lo
            ax.set_box_aspect(tuple(_sp / _sp.max()))
            ax.set_axis_off()
            ax.view_init(elev=view[0], azim=view[1])
            if r == 0 and column_titles:
                ax.set_title(f"{dose:g} dpa", fontsize=13, pad=-2)
            if c == 0:
                ax.text2D(-0.02, 0.5, label, transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=13)

        cax = fig.add_axes([0.925, 0.10 + (nrow - 1 - r) / nrow * 0.82,
                            0.013, 0.82 / nrow * 0.70])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=_JETISH), cax=cax)
        cax.tick_params(labelsize=8)
        cax.set_title("log" if use_log else "lin", fontsize=7, pad=3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.90, wspace=0.0, hspace=0.02)
    if out_file:
        fig.savefig(out_file, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig
