"""Field plots from MoDELib2-NNL (Zr3d_ghoniem) cluster-dynamics output.

Renders the spatially-resolved mobile and immobile fields on interior cut
planes, one row per species and one column per dose with a shared color bar per
row. Loop populations can be overlaid one family at a time as discrete
platelets (visualization only).

Two orthogonal cuts are drawn into the same axes by default. They are sampled
into ONE `Poly3DCollection` rather than one `plot_surface` per plane, because
matplotlib depth-sorts 3-D *artists* by a single scalar each: two intersecting
surfaces drawn as separate artists cannot interleave, so whichever sorts in
front hides the other completely -- which is exactly what used to happen, the
vertical cut covering the horizontal one. Within a single collection the
individual cells are depth-sorted, so the two planes occlude each other cell by
cell and both stay visible. The loop platelets go into the same collection for
the same reason, which additionally makes the cuts read as a cutaway: a loop
behind a cut plane is hidden by it.

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
           "OMEGA_SI", "OMEGA_B3", "CRYSTAL_AXES", "domain_faces",
           "domain_edges", "domain_volume"]

from dislocluster_code.coupling.field import (       # noqa: E402
    M_SIZE as _MS, N_FAMILIES as _NF, N_MOMENTS as _NM)

# THE MATERIAL FILE DECIDES THESE, and they used to be a fifth independent copy
# of two constants the project keeps equal in four places. That copy had gone
# stale on both: Omega still read 1.2e-29 (= V_cell/4, four atoms in a cell that
# holds two) and the <c> Burgers vector below still read the full IDEAL c. Both
# reach only the drawn platelet radius in the loop overlays -- an exaggerated,
# illustrative size and not a quantitative one -- but a stale constant is a
# stale constant, and reading the file is what stops it drifting again.
def _material_constants():
    from dislocluster_code.coupling.field import cluster_atomic_volume
    from dislocluster_code import paths as _p
    b = 3.23e-10
    try:
        for line in _p.MODELIB_MATERIAL.read_text(
                encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("b_SI"):
                b = float(line.split("=", 1)[1].split("#")[0].strip()
                          .rstrip(";").strip())
                break
    except Exception:
        pass
    try:
        om_b3 = cluster_atomic_volume(_p.MODELIB_MATERIAL)
    except Exception:
        om_b3 = 2.326553e-29 / b ** 3
    return b, om_b3 * b ** 3, om_b3


B_SI, OMEGA_SI, OMEGA_B3 = _material_constants()

# Column layout of the CD block: M_SIZE mobile, then one group per moment --
# every number, then every content, then every second moment. DERIVED, not
# written out: the literals 4..11 were right only while there were four
# families, and once step 1 widened the block "c_vL" at column 8 was the a1
# NUMBER. A figure drawn from that is well formed and about the wrong quantity.
FAMILY_KEYS = ["vL", "a1", "a2", "a3", "a1v", "a2v", "a3v", "cp", "c0"]
FAMILY_TEX = [r"\langle c\rangle_f", r"\langle a\rangle_1",
              r"\langle a\rangle_2", r"\langle a\rangle_3",
              r"\langle a\rangle_1^{v}", r"\langle a\rangle_2^{v}",
              r"\langle a\rangle_3^{v}", r"\langle c\rangle_p",
              r"\mathrm{pyr}"]


def n_col(k):
    """Column of family ``k``'s number density in the CD block."""
    return _MS + k


def c_col(k):
    """Column of family ``k``'s content."""
    return _MS + _NF + k


def q_col(k):
    """Column of family ``k``'s second content moment (step 5)."""
    return _MS + 2 * _NF + k


SPECIES = {
    "Cv":   (0,  r"$C_v$"),
    "Ci":   (1,  r"$C_i$"),
    "C2i":  (2,  r"$C_{2i}$"),
    "C3i":  (3,  r"$C_{3i}$"),
}
for _k, (_key, _tex) in enumerate(zip(FAMILY_KEYS[:_NF], FAMILY_TEX[:_NF])):
    SPECIES[f"n_{_key}"] = (n_col(_k), rf"$N_{{{_tex}}}$")
    SPECIES[f"c_{_key}"] = (c_col(_k), rf"$C_{{{_tex}}}$")
    if _NM > 2:
        SPECIES[f"q_{_key}"] = (q_col(_k), rf"$q_{{{_tex}}}$")

# |b| in units of b, per family. <c> is 1/2[0001] = c/2, so c/(2a) at the
# PHYSICAL c/a = 1.5944272 -- the value discrete_loops.FAMILIES already carries.
# This read 1.632993, the full IDEAL c: a factor of two out, and 2.4% on top.
_B_C = 0.7972136
_B_A = 1.0
_N_BASAL = np.array([0.0, 0.0, 1.0])
_N_A = [np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.8660254, 0.0]),
        np.array([-0.5, 0.8660254, 0.0])]

# Immobile families: (label, density column, content column, |b| in units of b,
#                     habit-plane normal in Cartesian b, colour). The three
# prismatic VACANCY variants share their interstitial partners' habit plane and
# Burgers vector -- they differ in what they STORE, which no drawn platelet
# shows -- and get their own colours so an overlay can tell them apart.
#
# THE PYRAMID IS DELIBERATELY ABSENT. It is a compact cluster: no habit plane,
# no Burgers vector, and R = (m Omega/sqrt 8)^(1/3) rather than lambda sqrt(m),
# so a platelet drawn for it would be a picture of something that does not
# exist.
FAMILIES = [
    (r"$\langle c\rangle_f$",   n_col(0), c_col(0), _B_C, _N_BASAL, "#1f4fbf"),
    (r"$\langle a\rangle_1$",   n_col(1), c_col(1), _B_A, _N_A[0],  "#c62828"),
    (r"$\langle a\rangle_2$",   n_col(2), c_col(2), _B_A, _N_A[1],  "#d84315"),
    (r"$\langle a\rangle_3$",   n_col(3), c_col(3), _B_A, _N_A[2],  "#ad1457"),
][:min(4, _NF)] + ([
    (r"$\langle a\rangle_1^v$", n_col(4), c_col(4), _B_A, _N_A[0],  "#00838f"),
    (r"$\langle a\rangle_2^v$", n_col(5), c_col(5), _B_A, _N_A[1],  "#00695c"),
    (r"$\langle a\rangle_3^v$", n_col(6), c_col(6), _B_A, _N_A[2],  "#2e7d32"),
    (r"$\langle c\rangle_p$",   n_col(7), c_col(7), _B_C, _N_BASAL, "#4527a0"),
] if _NF >= 8 else [])

# Crystal directions of the Cartesian axes the mesh is built in, drawn as an
# orientation triad on every panel. x is a prismatic-plane normal: the three
# <a> habit-plane normals above are 60 deg apart in the xy plane with the first
# along x, which is the <10-10> set, so y is the <11-20> direction at 90 deg to
# it and z is the c axis. Same convention as the material file, whose <c> loop
# normal is [0001] = z.
#
# This is a property of the MATERIAL, not of the domain: the mesh is built in
# the same crystal frame whether GEOMETRY['type'] is 'cubic' or 'hexagonal', so
# the labels hold for both. Pass a different sequence of (vector, label) pairs
# as `orientation=` to plot_field_panels for a material that is not hcp Zr.
CRYSTAL_AXES = (
    (np.array([1.0, 0.0, 0.0]), r"[10$\bar{1}$0]"),
    (np.array([0.0, 1.0, 0.0]), r"[2$\bar{1}\bar{1}$0]"),
    (np.array([0.0, 0.0, 1.0]), r"[0001]"),
)

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


def domain_volume(P):
    """Volume of the convex body the node cloud fills, in b^3.

    The bounding box is NOT this volume on anything but a box domain: for a
    hexagonal prism it is 4/3 times larger. Anything that turns a density into a
    count needs the body, not the box. For the cubic geometry the hull and the
    box coincide, so this returns exactly `prod(hi - lo)` and nothing changes.
    """
    from scipy.spatial import ConvexHull
    return float(ConvexHull(np.asarray(P, float)).volume)


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


def domain_edges(P, tol=1e-6, return_faces=False):
    """Edges of the convex domain the node cloud fills, as (M, 2, 3) segments.

    An edge of a convex polytope is where two faces meet, so the faces are
    grouped by plane (coplanar hull facets merged) and every pair of planes
    sharing two or more hull vertices contributes the segment between the two
    extreme shared vertices. A cube gives its 12 edges, a hexagonal prism its
    18 — as opposed to the bounding box, which for a prism is not the domain.

    `return_faces` also returns `(N, b)` and, per edge, the indices of the two
    faces that meet there — which is what decides whether an edge is hidden.
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(P)
    V = P[hull.vertices]
    N, b = domain_faces(P, tol)
    scale = max(np.abs(b).max(), 1e-30)
    on = [np.flatnonzero(np.abs(V @ N[k] - b[k]) < 1e-6 * scale)
          for k in range(len(N))]
    segs, pairs = [], []
    for i in range(len(N)):
        for j in range(i + 1, len(N)):
            shared = np.intersect1d(on[i], on[j])
            if shared.size < 2:
                continue
            Q = V[shared]
            d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
            a, c = np.unravel_index(np.argmax(d), d.shape)
            segs.append(np.stack([Q[a], Q[c]]))
            pairs.append((i, j))
    segs = np.asarray(segs) if segs else np.empty((0, 2, 3))
    if return_faces:
        return segs, (N, b), np.asarray(pairs, dtype=int).reshape(-1, 2)
    return segs


def camera_direction(view):
    """Unit vector from the domain toward the camera, for (elev, azim) in deg."""
    el, az = np.radians(view[0]), np.radians(view[1])
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)])


def draw_domain_wireframe(ax, P, lo, hi, view=None, color="0.35", lw=0.8,
                          zorder=6, hidden_ls=(0, (1.2, 2.2)), hidden_lw=0.7):
    """Outline the domain, hidden edges dotted, as a 3-D drawing is conventionally
    dimensioned.

    An edge of a convex body is hidden exactly when BOTH faces meeting there
    point away from the camera; if either faces the viewer the edge is on the
    near surface or on the silhouette. That test needs the face pair per edge,
    which `domain_edges(return_faces=True)` supplies.

    The whole wireframe is drawn at a zorder ABOVE the cut planes, on an axes
    built with `computed_zorder=False`. Matplotlib otherwise depth-sorts each
    edge as a whole artist against the plane collection, so edges behind a cut
    vanish and the crystal stops reading as a container for the cuts — which is
    also why the hidden ones have to be marked as hidden explicitly: nothing in
    the drawing occludes them any more.
    """
    try:
        segs, (N, _), pairs = domain_edges(P, return_faces=True)
    except Exception:
        segs = np.empty((0, 2, 3))
    if len(segs) == 0:
        _draw_box(ax, lo, hi, color=color, lw=lw)
        return
    cam = camera_direction(view) if view is not None else None
    front = None if cam is None else (N @ cam) > 1e-9
    for k, s in enumerate(segs):
        hidden = (front is not None and len(pairs)
                  and not front[pairs[k, 0]] and not front[pairs[k, 1]])
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=color, zorder=zorder,
                lw=(hidden_lw if hidden else lw),
                ls=(hidden_ls if hidden else "-"))


# Kept as the old name; `view` is what turns on the hidden-line style.
_draw_domain = draw_domain_wireframe


def _cut_intersections(planes, lo, hi, faces):
    """Segments where the drawn mid-cuts meet, clipped to the domain.

    Each cut is an axis-aligned plane through the centre, so two of them meet
    along the line through the centre parallel to the third axis. The segment is
    clipped against the convex domain -- on a hexagonal prism the y/z cuts meet
    along x, and the crystal ends at the two prism vertices, not at the bounding
    box.
    """
    axis = {"x": 0, "y": 1, "z": 2}
    mid = 0.5 * (lo + hi)
    segs = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            a, b_ = axis.get(planes[i]), axis.get(planes[j])
            if a is None or b_ is None or a == b_:
                continue
            k = 3 - a - b_                       # the axis both cuts contain
            e = np.zeros(3); e[k] = 1.0
            t0, t1 = lo[k] - mid[k], hi[k] - mid[k]
            if faces is not None:
                N, bb = faces
                for dn, nu in zip(N @ e, bb - N @ mid):
                    if abs(dn) < 1e-12:          # parallel: in or out entirely
                        if nu < 0.0:
                            t0, t1 = 0.0, -1.0
                            break
                        continue
                    t = nu / dn
                    if dn > 0.0:
                        t1 = min(t1, t)
                    else:
                        t0 = max(t0, t)
            if t1 > t0:
                segs.append(np.stack([mid + t0 * e, mid + t1 * e]))
    return segs


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


def _disc_quads(centres, radii, R, n=18, rim=0.84):
    """Flat circular platelets as quads (Q,4,3): `radii` in-plane, no thickness.

    A loop is a *flat* disc, and that is how it is drawn -- the earlier version
    gave every platelet a 0.28 aspect ratio so that a family seen edge-on would
    not collapse to a line, but at the shared two-cut viewpoint no family is
    within 60 deg of edge-on, and the thick version read as a stack of solid
    blobs rather than as loops.

    Each disc is a triangle fan emitted as degenerate quads (centre, p_i,
    p_i+1, centre) so that discs and plane cells share one (Q,4,3) array and can
    go into a single depth-sorted collection. Fan edges are invisible: the
    collection is drawn with no edge line and one flat colour per disc.

    The outer `1 - rim` of each disc is returned as a separate band, which the
    caller shades darker. Without it a field of flat same-coloured discs at
    different depths merges into one blob wherever two overlap in projection,
    and the figure stops reading as a population of loops. A drawn edge line
    would do the same job but matplotlib would then also stroke every internal
    fan edge.

    Returns `(quads, is_rim)`.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n + 1)
    ring = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1) @ R.T
    centres = np.asarray(centres, float)
    rad = np.asarray(radii, float)[:, None, None]
    outer = centres[:, None, :] + rad * ring[None]
    inner = centres[:, None, :] + (rim * rad) * ring[None]
    c = np.repeat(centres[:, None, :], n, axis=1)
    face = np.stack([c, inner[:, :-1], inner[:, 1:], c], axis=2).reshape(-1, 4, 3)
    band = np.stack([inner[:, :-1], outer[:, :-1],
                     outer[:, 1:], inner[:, 1:]], axis=2).reshape(-1, 4, 3)
    is_rim = np.concatenate([np.zeros(len(face), bool), np.ones(len(band), bool)])
    return np.concatenate([face, band]), is_rim


def _shaded_facecolors(quads, base_rgb, light=(0.3, -0.8, 0.5)):
    """Lambert-ish shading so a batched collection still reads as 3-D.

    Not used by the cut-plane platelets, which are flat discs of one colour and
    need no shading; ``discrete_loops.render`` uses it for its tubular
    dislocation lines, which are genuinely three-dimensional.
    """
    e1 = quads[:, 1] - quads[:, 0]
    e2 = quads[:, 2] - quads[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.where(ln > 0, ln, 1.0))
    lv = np.asarray(light, float)
    lv = lv / np.linalg.norm(lv)
    s = 0.55 + 0.45 * np.abs(nrm @ lv)
    return np.clip(np.asarray(base_rgb)[None, :] * s[:, None], 0.0, 1.0)


def _grid_quads(X, Y, Z, S, norm, faces, cmap=None):
    """Cell quads (Q,4,3) and RGBA (Q,4) for one sampled cut plane.

    Cells outside the convex domain are dropped rather than painted
    transparent: the plane grid spans the bounding box, so on a hexagonal prism
    part of it lies outside the crystal. A cell is kept only if all four of its
    corners are inside. Its colour comes from the mean of its four corner
    samples, not from the lower corner as `plot_surface` would use.
    """
    cmap = _JETISH if cmap is None else cmap
    V = np.stack([X, Y, Z], -1)
    quads = np.stack([V[:-1, :-1], V[1:, :-1], V[1:, 1:], V[:-1, 1:]],
                     axis=2).reshape(-1, 4, 3)
    Sc = 0.25 * (S[:-1, :-1] + S[1:, :-1] + S[1:, 1:] + S[:-1, 1:])
    cols = cmap(norm(Sc.ravel()))
    if faces is not None:
        ins = _inside(V, faces)
        keep = (ins[:-1, :-1] & ins[1:, :-1] &
                ins[:-1, 1:] & ins[1:, 1:]).ravel()
        quads, cols = quads[keep], cols[keep]
    return quads, cols


def _axes_fraction(ax, pts):
    """Where `pts` land in the panel, as axes fractions (0..1 is visible).

    Uses matplotlib's own projection matrix, so it reports exactly what will be
    drawn. No prior ``canvas.draw()`` is needed: ``get_proj`` is built from the
    limits, the box aspect and the view angles, all of which are already set.
    """
    from mpl_toolkits.mplot3d import proj3d
    pts = np.asarray(pts, float).reshape(-1, 3)
    u, v, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2],
                                    ax.get_proj())
    disp = ax.transData.transform(np.column_stack([u, v]))
    return ax.transAxes.inverted().transform(disp)


def _fit_zoom(ax, pts, aspect, zoom, margin=0.005, iters=4):
    """Shrink `zoom` until `pts` fit inside the panel, and apply it.

    A 3-D axes clips its artists at the axes rectangle, not at the data cube,
    so a `zoom` that projects the domain taller than the panel silently cuts
    the top and bottom off the wireframe. The fixed 1.22 this defaults to was
    chosen against a CUBE, whose projection it fits (0.007..0.960 of the panel);
    a hexagonal prism of the same box aspect as the 500 nm case projects to
    -0.068..1.054 and loses both end faces. It is the ASPECT RATIO that decides
    this, not the size -- a 200 nm and a 500 nm prism of the same proportions
    overflow identically.

    The projection is very nearly linear in `zoom`, so one correction lands it;
    the loop refines the residual from perspective foreshortening and stops as
    soon as the domain fits. `zoom` is only ever reduced, so a domain that
    already fits -- every cubic case -- renders exactly as it did before.
    """
    for _ in range(iters):
        f = _axes_fraction(ax, pts)
        half = float(np.abs(f - 0.5).max())      # matplotlib centres the box
        need = (0.5 - margin) / max(half, 1e-12)
        if need >= 1.0:
            break
        zoom *= need
        ax.set_box_aspect(aspect, zoom=zoom)
    return zoom


def _draw_orientation_triad(ax, view, axes_dirs=CRYSTAL_AXES,
                            origin=(0.07, 0.13), length_in=0.30, fontsize=8.0):
    """Crystal-direction arrows, projected to match the 3-D viewpoint.

    Drawn in axes coordinates rather than as 3-D arrows inside the domain, so
    the triad keeps a fixed size and a fixed corner of the panel whatever the
    domain aspect ratio is. The screen directions are the analytic projection of
    matplotlib's own camera: for (elev, azim) the screen right and up axes are

        r = (-sin a, cos a, 0)
        u = (-sin e cos a, -sin e sin a, cos e)

    so a crystal direction v lands at (v.r, v.u), which is what `view_init`
    would put it at. Arrow lengths are equalized in INCHES, since a panel's axes
    box is not square and equal axes-fraction lengths would shear the triad.
    """
    el, az = np.radians(view[0]), np.radians(view[1])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az),
                   np.cos(el)])
    fig = ax.figure
    bb = ax.get_position()
    w_in = max(bb.width * fig.get_figwidth(), 1e-6)
    h_in = max(bb.height * fig.get_figheight(), 1e-6)
    for v, label in axes_dirs:
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        d = np.array([v @ right, v @ up])
        n = np.linalg.norm(d)
        if n < 1e-3:                       # pointing at the camera: no arrow
            continue
        d = d / n
        step = np.array([d[0] * length_in / w_in, d[1] * length_in / h_in])
        tip = np.asarray(origin, float) + step
        ax.annotate("", xy=tip, xytext=origin, xycoords="axes fraction",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.16,"
                                              "head_length=0.36",
                                    color="k", lw=1.5,
                                    shrinkA=0.0, shrinkB=0.0))
        lab = tip + 0.18 * step
        ax.annotate(label, xy=lab, xycoords="axes fraction", fontsize=fontsize,
                    ha=("left" if d[0] > 0.25 else
                        "right" if d[0] < -0.25 else "center"),
                    va=("bottom" if d[1] > 0.25 else
                        "top" if d[1] < -0.25 else "center"),
                    annotation_clip=False,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white",
                              ec="none", alpha=0.75))


def _overlay_family(P, F, rng, fam, n_loops, loop_scale, gap=1.4,
                    max_loops=2500, stop_after_misses=1500, faces=None):
    """Geometry for one loop family, platelets sized by the LOCAL mean radius.

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

    The clearance test compares centre distance against `gap` times the sum of
    the two drawn radii — the platelets' bounding spheres. That is conservative
    for discs, which may therefore end up further apart than strictly necessary;
    the alternative, exact disc-disc intersection in 3-D, is not worth it here.
    `gap` above 1 leaves visible space between neighbours: at bare tangency the
    fill is so dense that the discs merge into a mat in projection and hide the
    field they are drawn over.

    Returns `(quads, rgba)` instead of drawing, so the platelets can join the
    cut planes in one depth-sorted collection.
    """
    _, ncol, ccol, bmag, normal, color = fam
    empty = (np.empty((0, 4, 3)), np.empty((0, 4)))
    # Keep the seed sites off the surface. The margin is a distance from the
    # NEAREST DOMAIN FACE, not from the bounding box: on a hexagonal prism the
    # box test leaves nodes right against the six slanted prism faces, whose
    # platelets then hang outside the crystal.
    d_face = gb_distance(P, faces=faces)
    cand = np.flatnonzero(d_face > 0.02 * d_face.max())
    if cand.size == 0:
        return empty

    n_k = F[cand, ncol]
    c_k = F[cand, ccol]
    good = (np.isfinite(n_k) & np.isfinite(c_k) & (n_k > 0) & (c_k > 0))
    cand, n_k, c_k = cand[good], n_k[good], c_k[good]
    if cand.size == 0:
        return empty

    m = c_k / (n_k * OMEGA_B3)                          # defects per loop
    good = m > 1.0
    cand, m = cand[good], m[good]
    if cand.size == 0:
        return empty

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
        near = tree.query_ball_point(pts[i], gap * (r_drawn[i] + r_max))
        clash = False
        for j in near:
            if acc_set[j]:
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < gap * (r_drawn[i] + r_drawn[j]):
                    clash = True
                    break
        if clash:
            misses += 1
            continue
        misses = 0
        acc_set[i] = True
        accepted.append(i)

    if not accepted:
        return empty
    accepted = np.asarray(accepted)
    R = _platelet_frame(normal)
    quads, is_rim = _disc_quads(pts[accepted], r_drawn[accepted], R)
    rgb = np.asarray(to_rgb(color))
    rgba = np.tile(np.append(rgb, 1.0), (len(quads), 1))
    rgba[is_rim, :3] = 0.55 * rgb
    return quads, rgba


def _clip_convex_polygon(poly, N, bb, tol=1e-9):
    """Sutherland-Hodgman: a convex polygon against the half-spaces x.N^T <= b.

    The domain is the convex hull of the CD nodes, so it IS an intersection of
    half-spaces and clipping against them one at a time is exact. The polygon
    stays planar because every cut is by a plane, so the result can be fanned
    back into triangles without leaving the loop's habit plane.
    """
    for k in range(len(bb)):
        if len(poly) == 0:
            return poly
        s = poly @ N[k] - bb[k]                     # <= 0 is inside
        keep = s <= tol
        if keep.all():
            continue
        if not keep.any():
            return np.empty((0, 3))
        out, m = [], len(poly)
        for i in range(m):
            j = (i + 1) % m
            if keep[i]:
                out.append(poly[i])
            if keep[i] != keep[j]:
                t = s[i] / (s[i] - s[j])
                out.append(poly[i] + t * (poly[j] - poly[i]))
        poly = np.asarray(out, dtype=float)
    return poly


def _clip_quads(quads, is_rim, faces):
    """Cut a quad soup at the crystal surface, keeping the (Q,4,3) format.

    A clipped convex polygon has up to `4 + n_faces` vertices, which does not
    fit a quad array -- so it is fanned into triangles and each triangle is
    emitted as a degenerate quad `(a, b, c, c)`. `_disc_quads` already uses
    that trick for the disc fan itself, so nothing downstream has to change.
    """
    N, bb = faces
    V = quads.reshape(-1, 3) @ N.T - bb[None, :]        # (4Q, F)
    V = V.reshape(len(quads), 4, -1)
    inside = V <= 1e-9
    whole = inside.all(axis=(1, 2))                     # every vertex inside
    gone = (~inside).all(axis=1).any(axis=1)            # wholly past one face
    out_q = [quads[whole]]
    out_r = [is_rim[whole]]
    for i in np.flatnonzero(~whole & ~gone):
        poly = _clip_convex_polygon(quads[i], N, bb)
        if len(poly) < 3:
            continue
        tri = np.stack([np.stack([poly[0], poly[j], poly[j + 1], poly[j + 1]])
                        for j in range(1, len(poly) - 1)])
        out_q.append(tri)
        out_r.append(np.full(len(tri), is_rim[i]))
    return np.concatenate(out_q), np.concatenate(out_r)


def _overlay_explicit(pop, fam, loop_scale=1.0, faces=None):
    """Draw a population that was HANDED to us, rather than one packed here.

    `_overlay_family` invents a platelet field: it places one disc per CD node
    and stops when the domain is full, so its count is the packing capacity of
    the figure and not the loop density, and it exaggerates the radii so the
    discs are visible. Useful as a field decoration, useless as a population.

    This draws the discrete population `post.discrete_loops` builds -- the
    actual `sum_j n_j V_j` loops at their sampled positions and their
    polydisperse sampled radii -- so the same loops appear in the field panel
    and in the discrete render, and a reader can put the two side by side.

    NO NON-OVERLAP TEST, deliberately. Real loops do overlap in projection, and
    the packing rule exists only to keep an invented field legible; applying it
    to a real population would silently delete loops.

    `pop` is `(centres, radii)` in b. `loop_scale` still multiplies the radii,
    and the caller passes 1.0 to draw them at true size -- which for a sparse
    family on a 500 nm crystal is genuinely small, and is the honest picture.

    CLIPPED AT THE CRYSTAL SURFACE, like `discrete_loops.render`. `sample_family`
    places loop CENTRES inside the body and never asks whether the disc fits, so
    every loop centred within `r` of a face overhangs it -- 27% of the <c>
    family at 10 dpa on the 500 nm prism, by up to 40 nm. That is a property of
    the sampling and not of the drawing, so the loops in `loops_*.csv` are
    whole and only the picture is cut, which is exactly what the discrete
    render does.
    """
    centres, radii = pop
    centres = np.asarray(centres, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1) * float(loop_scale)
    if centres.size == 0:
        return np.empty((0, 4, 3)), np.empty((0, 4))
    _, _, _, _, normal, color = fam
    quads, is_rim = _disc_quads(centres, radii, _platelet_frame(normal))
    if faces is not None:
        quads, is_rim = _clip_quads(quads, is_rim, faces)
        if len(quads) == 0:
            return np.empty((0, 4, 3)), np.empty((0, 4))
    rgb = np.asarray(to_rgb(color))
    rgba = np.tile(np.append(rgb, 1.0), (len(quads), 1))
    rgba[is_rim, :3] = 0.55 * rgb
    return quads, rgba


_DEFAULT_VIEW = {"x": (10.0, -80.0), "y": (10.0, -80.0), "z": (55.0, -70.0)}
# Two orthogonal cuts need a viewpoint that is edge-on to neither. The
# single-cut angles are nearly frontal (elev 10) or nearly overhead (elev 55);
# either reduces one of the two planes to a line.
_MULTI_VIEW = (26.0, -58.0)


def plot_field_panels(evl_dir, steps, doses, species=("Cv", "Ci"),
                      plane="y", view=None, n_slice=200,
                      log=None, floor_decades=5.0, vlims=None,
                      loop_family=None, n_loops=None, loop_scale=45.0, seed=0,
                      loop_population=None,
                      figsize_per_panel=(3.3, 3.2), out_file=None, title=None,
                      column_titles=True, fields=None, row_labels=None,
                      orientation=True, cbar_fontsize=7.0, zoom=1.22):
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
    loop_population: `{step: (centres_b, radii_b)}` -- draw THESE loops instead
                   of packing an invented field. This is how a panel is made to
                   show the same population `post.discrete_loops` exports, so
                   the field figure and the discrete render agree loop for loop;
                   `loop_scale` then normally wants to be 1.0. Ignored for any
                   step the dict has no entry for, which falls back to packing.
    column_titles: draw the "<dose> dpa" header above each column. Turn this off
                   for a single-dose figure whose `title` already names the dose,
                   otherwise the two headers overlap.
    row_labels   : draw the species label down the left of each row. Default
                   None = only when there is more than one row, since on a
                   one-quantity figure `title` already names the species and the
                   two just repeat each other.
    orientation  : draw the crystal-direction triad in each panel's lower left.
                   True uses CRYSTAL_AXES (hcp Zr, correct for every geometry
                   this repository meshes); pass a sequence of (vector, label)
                   pairs for a different crystal frame, or False for none.
    cbar_fontsize: colour-bar tick size, in points. Small on purpose: the figure
                   is saved with `bbox_inches="tight"`, which crops the empty
                   margin a 3-D axes leaves and so magnifies every font relative
                   to the image. `zoom` works the same problem from the other
                   end, filling more of the panel with the domain.
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

    # The points the panel has to hold: the vertices of the body that is
    # actually drawn, not the bounding box. For a prism the box corners sit out
    # in the six empty wedges, and fitting those would shrink the crystal for
    # nothing; for a cube the two sets coincide.
    try:
        from scipy.spatial import ConvexHull
        fit_pts = P0[ConvexHull(P0).vertices]
    except Exception:
        fit_pts = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    nrow, ncol = len(species), len(steps)
    if row_labels is None:
        row_labels = nrow > 1
    # The extra inch is the colour-bar column. It used to be 1.7, most of which
    # ended up as blank paper between the panel and the bar.
    fig = plt.figure(figsize=(figsize_per_panel[0] * ncol + 1.0,
                              figsize_per_panel[1] * nrow))
    cbar_left = 1.0 - 0.62 / fig.get_figwidth()
    panel_axes = []

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
            # computed_zorder=False: draw order comes from the artists' own
            # zorder instead of from their projected depth. The cut planes are
            # one collection that still sorts its own polygons internally; what
            # this buys is that the domain wireframe and the intersection line
            # stay on top of it, so the crystal is a visible container and the
            # cuts read as sitting inside it.
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d",
                                 computed_zorder=False)
            P, F = data[st]
            quads, colors = [], []
            for corner, uvec, vvec in grids:
                X, Y, Z, S = _sample_plane(trees[st], F[:, col_idx],
                                           corner, uvec, vvec, n_slice)
                q, fc = _grid_quads(X, Y, Z, S, norm, faces)
                quads.append(q); colors.append(fc)
            if loop_family is not None:
                pop = (loop_population or {}).get(st)
                if pop is not None:
                    q, fc = _overlay_explicit(pop, FAMILIES[loop_family],
                                              loop_scale, faces=faces)
                else:
                    q, fc = _overlay_family(P, F, rng, FAMILIES[loop_family],
                                            n_loops, loop_scale, faces=faces)
                quads.append(q); colors.append(fc)
            # ONE collection for the cuts and the platelets together, so that
            # matplotlib depth-sorts them against each other polygon by polygon.
            # Separate artists get one depth each, which is what made the
            # vertical cut hide the horizontal one entirely.
            ax.add_collection3d(Poly3DCollection(
                np.concatenate(quads), facecolors=np.concatenate(colors),
                linewidths=0, antialiased=False, zsort="average", zorder=1))
            _draw_domain(ax, P0, lo, hi, view=view)
            # Where the two cuts meet. Dotted, over the planes, so the viewer
            # can see that they are two orthogonal sections of one body rather
            # than two unrelated images.
            for s in _cut_intersections(planes, lo, hi, faces):
                ax.plot(s[:, 0], s[:, 1], s[:, 2], color="k", lw=1.1,
                        ls=":", zorder=7)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            # True proportions. A fixed (1,1,1) box renders the 400 x 346 x 653 nm
            # prism as though it were a cube.
            _sp = hi - lo
            _aspect = tuple(_sp / _sp.max())
            ax.set_box_aspect(_aspect, zoom=zoom)
            ax.set_axis_off()
            # The view has to be set BEFORE the fit: which way the domain is
            # tallest on screen is a property of the camera, not of the box.
            ax.view_init(elev=view[0], azim=view[1])
            _fit_zoom(ax, fit_pts, _aspect, zoom)
            if r == 0 and column_titles:
                ax.set_title(f"{dose:g} dpa", fontsize=13, pad=-2)
            if c == 0 and row_labels:
                ax.text2D(-0.02, 0.5, label, transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=13)
            panel_axes.append(ax)

        cax = fig.add_axes([cbar_left, 0.12 + (nrow - 1 - r) / nrow * 0.80,
                            0.10 / fig.get_figwidth(), 0.80 / nrow * 0.66])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=_JETISH), cax=cax)
        cax.tick_params(labelsize=cbar_fontsize, pad=1.5, length=2.5)
        # The offset text -- the "1e-12+5.042e-8" a linear scale over a narrow
        # range puts above the bar -- is NOT covered by tick_params, and stays
        # at the 10 pt default. On a panel this size it then comes out larger
        # than the figure title. Size it here and leave the title room above it.
        offset = cax.yaxis.get_offset_text()
        offset.set_fontsize(cbar_fontsize)
        cax.set_title(f"{label}\n{'log' if use_log else 'linear'}",
                      fontsize=cbar_fontsize,
                      pad=(4 if use_log else 4 + 1.9 * cbar_fontsize),
                      linespacing=1.4)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=cbar_left - 0.02,
                        wspace=0.0, hspace=0.02)
    # After the layout, not before: the triad equalizes its arm lengths in
    # inches from the axes box, which subplots_adjust has just moved. On a grid
    # every panel shares one viewpoint, so one triad on the bottom-left panel
    # labels all of them; repeating it in each cell is just clutter.
    if orientation:
        axes_dirs = (CRYSTAL_AXES if orientation is True else orientation)
        for ax in (panel_axes if len(panel_axes) == 1
                   else panel_axes[(nrow - 1) * ncol:(nrow - 1) * ncol + 1]):
            _draw_orientation_triad(ax, view, axes_dirs=axes_dirs)
    if out_file:
        fig.savefig(out_file, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig
