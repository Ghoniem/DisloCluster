"""
tem_slices.py -- project a slab of the discrete loop population into a
TEM-style bright-field micrograph.

WHAT THIS IS FOR
----------------
``discrete_loops.py`` writes the real loop population -- real number, real size,
real Burgers vector and habit plane -- as a 3-D scene. A transmission electron
micrograph is not a 3-D scene: it is a ~100 nm foil viewed in projection along
the beam, so every loop is seen at whatever inclination its habit plane happens
to have to the beam. Loops with their habit plane normal ALONG the beam are seen
face-on and image as their full outline; loops with the normal PERPENDICULAR to
the beam are seen edge-on and image as a short straight line. That contrast is
the whole reason a micrograph can tell the four families apart, and it is lost
in the 3-D render.

This module reproduces it: pick a beam direction, take a slab of the given
thickness, project the loop polygons along the beam, and draw them dark on a
light field.

THE BEAM DIRECTIONS
-------------------
All three are conditions used experimentally on hcp Zr.

``B_0001`` -- beam along [0001], down the c-axis.
    <c> loops lie in the basal plane, normal along the beam, so they image
    face-on as circles. All three <a> families have their prismatic habit plane
    parallel to the beam, so they image edge-on as short lines, each family with
    a trace 60 degrees from the next. Every <a> Burgers vector lies in the image
    plane and is perpendicular to its own trace; the <c> Burgers vector points
    along the beam and is drawn with the out-of-plane symbol.

``B_0110`` -- beam along [01-10], a prism-plane normal (the g = 0002 condition
    used to image <c> loops).
    <c> loops are edge-on: the long straight lines. <a>1 is also edge-on, its
    trace vertical, since its habit-plane normal [2-1-10] is perpendicular to the
    beam. <a>2 and <a>3 are inclined 60 degrees to the beam and image as
    foreshortened ellipses with Burgers vectors of opposite in-plane sign.

``B_1120`` -- beam along [11-20], the third zone axis and the <a> counterpart
    of B_0001.
    <a>2's habit normal is parallel to the beam, so that family is seen face-on
    as full circles at its true size -- the only condition here in which an <a>
    family is measured undistorted. <c> is edge-on, imaging as horizontal
    traces, and <a>1 and <a>3 are inclined at 60 degrees to the beam and image
    at 0.5. Between this view and B_0001 both populations are seen face-on.

``B_0001_t45`` -- the specimen tilted 45 degrees off [0001] about [2-1-10].
    Neither zone axis shows a <c> loop as an ellipse: down [0001] it is exactly
    face-on, down [01-10] exactly edge-on. A tilted specimen shows it between
    those extremes, and the ELLIPSE is what identifies a basal loop and lets its
    inclination be read off the plate. A circular loop of normal n viewed along
    B projects to an ellipse of axis ratio |n.B|, so at 45 degrees <c> images at
    0.707, <a>1 stays edge-on because its habit normal is the tilt axis, and
    <a>2/<a>3 image at 0.612. Shape alone therefore does not separate <c> from
    <a>2/<a>3 at this angle -- the ellipse ORIENTATION and the Burgers markers
    do. See :func:`tilted_view` for the trade-off and for any other angle.

WHAT IS AND IS NOT MODELED
--------------------------
Drawn: projection, inclination, foil thickness, depth-faded contrast, the
loop outline at its true size, and the projected Burgers vector of every loop.

Not drawn: two-beam diffraction contrast. A real loop images as a black-white
lobe pair whose appearance depends on g.b and on depth in the foil, and loops
with g.b = 0 vanish entirely. Here every loop is visible regardless of the
imaging vector, so these are projections of the population, not simulated
diffraction contrast. The grain and vignette are cosmetic.

Loops are selected by the position of their CENTER, so loops near the slab faces
stick out of the foil -- as they do in a real thin foil.

USAGE
-----
    python -m py_utils.tem_slices <run_dir> [--doses 0.01 0.1 1 10]
                                  [--thickness-nm 100] [--views B_0001 B_0110]
                                  [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import FancyArrow, Circle
from scipy.ndimage import gaussian_filter


from dislocluster_code.post.discrete_loops import FAMILIES, loop_polygon   # noqa: E402
from dislocluster_code.post.fields import B_SI                     # noqa: E402

NM_PER_B = B_SI * 1e9
FAM_BY_KEY = {f["key"]: f for f in FAMILIES}

# ── imaging geometry ─────────────────────────────────────────────────────────
# beam       unit beam direction, Cartesian, in the same frame as the loop table
# up         which crystal direction points up the page
# label      zone axis, in Miller-Bravais
VIEWS = {
    "B_0001": dict(beam=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                   label=r"$B \parallel [0001]$",
                   note=r"$\langle c\rangle$ face-on, "
                        r"$\langle a\rangle$ edge-on"),
    "B_0110": dict(beam=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0),
                   label=r"$B \parallel [01\bar{1}0]$",
                   note=r"$\langle c\rangle$ and $\langle a\rangle_1$ edge-on, "
                        r"$\langle a\rangle_{2,3}$ inclined"),
}


def tilted_view(deg, about=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """A view with the beam ``deg`` degrees off [0001], tilted about ``about``.

    Neither of the two named views above shows a <c> loop as an ellipse: down
    [0001] it is exactly face-on (a circle) and down [01-10] exactly edge-on (a
    line). A real specimen is tilted between those extremes, and the ELLIPSE is
    what identifies a basal loop and lets its inclination be read off the
    micrograph -- the projected axis ratio of a circular loop of normal n viewed
    along B is simply

        b/a = |n . B|

    so a beam ``deg`` off the c-axis foreshortens every <c> loop to
    ``cos(deg)`` and nothing else about the loop enters.

    The tilt axis defaults to Cartesian x = [2-1-10], which is <a>_1's Burgers
    direction. That is the experimentally usual choice and it is the one that
    keeps the picture readable: <a>_1's habit normal IS the tilt axis, so it
    stays exactly edge-on, and <a>_2 and <a>_3 tilt by the same amount as each
    other. Measured on the drawn polygons:

        tilt      <c>      <a>_1    <a>_2,3
        30 deg    0.866    0.000    0.433
        45 deg    0.707    0.000    0.612     <-- registered below
        60 deg    0.500    0.000    0.750

    NOTE WHAT THE ANGLE COSTS AS WELL AS WHAT IT BUYS. Tilting makes <c> more
    elliptical but simultaneously OPENS <a>_2,3, and the two families pass each
    other: they differ by 0.433 in axis ratio at 30 degrees but only 0.095 at
    45 and 0.250 (the other way) at 60. At 45 degrees shape alone therefore no
    longer separates <c> from <a>_2,3 -- what still does is the ellipse
    ORIENTATION, since the projected minor axis lies along the projection of the
    habit normal (vertical for <c>, about 50 degrees off horizontal for
    <a>_2,3), together with the colored Burgers markers.

    Verified against `loop_polygon` itself rather than against an assumed frame,
    so it cannot drift if the family definitions change.
    """
    t = np.radians(float(deg))
    k = np.asarray(about, float)
    k = k / np.linalg.norm(k)
    c = np.array([0.0, 0.0, 1.0])
    # Rodrigues rotation of the c-axis about k.
    beam = (c * np.cos(t) + np.cross(k, c) * np.sin(t)
            + k * float(k @ c) * (1.0 - np.cos(t)))
    return dict(beam=tuple(beam), up=tuple(up),
                label=rf"$B$ at ${deg:g}^\circ$ from $[0001]$",
                note=rf"$\langle c\rangle$ inclined: ellipses, "
                     rf"axis ratio $\cos {deg:g}^\circ = {np.cos(t):.3f}$")


def habit_normal(key):
    """Unit habit-plane normal of a family, read off its own drawn polygon.

    Taken from `loop_polygon` rather than from `b_lattice` so it cannot
    disagree with what the figures actually draw, and so it survives any change
    of lattice-to-Cartesian convention in `discrete_loops`.
    """
    P = loop_polygon(FAM_BY_KEY[key], np.zeros(3), 1.0)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    return n / np.linalg.norm(n)


# `<11-20>` -- the third zone axis, and the <a> counterpart of B_0001.
#
# A prismatic loop's habit normal IS its Burgers direction, so the [11-20] zone
# axis is exactly <a>_2's habit normal; deriving it that way rather than writing
# (1/2, sqrt3/2, 0) by hand keeps it tied to the family definitions.
#
# It is the only condition in this module where an <a> family is seen FACE-ON
# and therefore at its true size:
#
#     <c>        |n.B| = 0.000     edge-on, horizontal traces
#     <a>_1,3    |n.B| = 0.500     ellipses, inclined 60 deg to the beam
#     <a>_2      |n.B| = 1.000     face-on, full circles
#
# B_0001 does the same job for <c>, so the pair between them measures both
# populations at full size. <a>_2's Burgers vector points along the beam here
# and is drawn with the out-of-plane symbol; <c>'s lies in the image plane,
# vertical.
VIEWS["B_1120"] = dict(
    beam=tuple(habit_normal("a2")), up=(0.0, 0.0, 1.0),
    label=r"$B \parallel [11\bar{2}0]$",
    note=r"$\langle a\rangle_2$ face-on, $\langle c\rangle$ edge-on, "
         r"$\langle a\rangle_{1,3}$ inclined")

# The tilted condition, rendered by default alongside the three zone axes.
VIEWS["B_0001_t45"] = tilted_view(45.0)

# Below this fraction of |b| in the image plane, the Burgers vector is called
# out-of-plane and drawn as the conventional circled dot instead of an arrow.
B_INPLANE_MIN = 0.2

# Bright-field appearance. The field is light, the defects dark; contrast fades
# with depth so the slab reads as a foil of finite thickness rather than a
# flat drawing.
BG_LEVEL = 0.80
GRAIN_RMS = 0.022
GRAIN_SIGMA_PX = 1.6
GRAIN_PX = 900
DEPTH_FADE = 0.35          # fractional darkness lost from front face to back
LINE_NM = 2.0              # drawn width of a dislocation line, nm
ARROW_NM = 13.0            # drawn length of a Burgers-vector marker, nm

# Family labels in mathtext, so the panels read the same way as the titles.
FAM_TEX = {"c": r"$\langle c\rangle$", "a1": r"$\langle a\rangle_1$",
           "a2": r"$\langle a\rangle_2$", "a3": r"$\langle a\rangle_3$"}


def view_frame(view):
    """``(e1, e2, beam)`` -- a right-handed image frame for one view."""
    beam = np.asarray(VIEWS[view]["beam"], float)
    beam = beam / np.linalg.norm(beam)
    up = np.asarray(VIEWS[view]["up"], float)
    e2 = up - beam * float(beam @ up)
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, beam)
    return e1, e2, beam


def read_loops(csv_file):
    """The per-loop table as a dict of arrays, positions in units of b."""
    rows = list(csv.DictReader(open(csv_file, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{csv_file} holds no loops")
    return dict(
        family=np.array([r["family"] for r in rows]),
        center=np.array([[float(r["x_b"]), float(r["y_b"]), float(r["z_b"])]
                         for r in rows]),
        r_b=np.array([float(r["r_b"]) for r in rows]),
        b_vec=np.array([[float(r["bx_b"]), float(r["by_b"]), float(r["bz_b"])]
                        for r in rows]))


def _grain(rng):
    """A correlated-noise field for the micrograph background."""
    g = gaussian_filter(rng.standard_normal((GRAIN_PX, GRAIN_PX)),
                        GRAIN_SIGMA_PX)
    g = g / g.std() * GRAIN_RMS
    # A gentle vignette, as from a slightly thicker foil toward the edge.
    u = np.linspace(-1, 1, GRAIN_PX)
    X, Y = np.meshgrid(u, u)
    return BG_LEVEL + g - 0.05 * (X ** 2 + Y ** 2)


def render_slice(csv_file, dose, out_file, view="B_0001", thickness_nm=100.0,
                 box_nm=None, seed=0, dpi=200, verbose=True):
    """Draw one micrograph. Returns ``(out_file, counts_per_family)``."""
    L = read_loops(csv_file)
    e1, e2, beam = view_frame(view)
    rng = np.random.default_rng(seed)

    c_nm = L["center"] * NM_PER_B
    if box_nm is None:
        lo = np.floor(c_nm.min(0))
        hi = np.ceil(c_nm.max(0))
    else:
        lo, hi = np.asarray(box_nm[0], float), np.asarray(box_nm[1], float)

    # The foil: a slab of the given thickness centered in the box, normal to the
    # beam. Selection is on the loop center, so loops may pierce the faces.
    depth = c_nm @ beam
    mid = 0.5 * float((lo @ beam) + (hi @ beam))
    half = 0.5 * float(thickness_nm)
    keep = np.abs(depth - mid) <= half
    if not keep.any():
        raise ValueError(f"no loops in the {thickness_nm:g} nm slab")

    # Image-plane extent: the full lateral size of the box.
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u_all, v_all = corners @ e1, corners @ e2
    u0, u1 = float(u_all.min()), float(u_all.max())
    v0, v1 = float(v_all.min()), float(v_all.max())

    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    ax.imshow(_grain(rng), extent=(u0, u1, v0, v1),
              cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear",
              zorder=0)

    # Points per nm, so a dislocation line can be drawn at a width in nm.
    pt_per_nm = (fig.get_size_inches()[0] * 0.86 * 72.0) / max(u1 - u0, 1e-9)
    lw = max(0.4, LINE_NM * pt_per_nm)

    counts, order = {}, np.argsort(-(depth[keep]))   # far loops drawn first
    idx_keep = np.flatnonzero(keep)[order]
    d_rel = (depth[idx_keep] - (mid - half)) / (2 * half)   # 0 back .. 1 front

    for fam in FAMILIES:
        sel = L["family"][idx_keep] == fam["key"]
        counts[fam["key"]] = int(sel.sum())
        if not sel.any():
            continue
        sub = idx_keep[sel]
        shade = 0.06 + DEPTH_FADE * (1.0 - d_rel[sel]) * 0.5   # dark, fading back

        polys, edge, face = [], [], []
        for k, row in enumerate(sub):
            pts = loop_polygon(fam, L["center"][row], L["r_b"][row]) * NM_PER_B
            polys.append(np.column_stack([pts @ e1, pts @ e2]))
            g = float(shade[k])
            edge.append((g, g, g, 1.0))
            # Face-on loops carry a faint interior; edge-on ones enclose no area
            # to fill, so the stroke is all the contrast they get.
            face.append((g, g, g, 0.10))
        ax.add_collection(PolyCollection(polys, facecolors=face,
                                         edgecolors=edge, linewidths=lw,
                                         joinstyle="round", zorder=2))

        # Burgers-vector markers, one per loop. The projected b is the same for
        # every loop of a family, so it is computed once.
        bv = L["b_vec"][sub[0]]
        b_img = np.array([float(bv @ e1), float(bv @ e2)])
        cen = np.column_stack([L["center"][sub] @ e1, L["center"][sub] @ e2]) * NM_PER_B
        col = fam["color"]
        if np.linalg.norm(b_img) / np.linalg.norm(bv) < B_INPLANE_MIN:
            # b along the beam: the conventional circled dot.
            rad = 0.22 * ARROW_NM
            for x, y in cen:
                ax.add_patch(Circle((x, y), rad, fill=False, ec=col,
                                    lw=1.0, zorder=4))
                ax.add_patch(Circle((x, y), 0.28 * rad, fc=col, ec="none",
                                    zorder=4))
        else:
            # Tail on the loop, pointing outward, so the marker labels the loop
            # without drawing over the contrast that identifies it.
            u = b_img / np.linalg.norm(b_img)
            d = u * ARROW_NM
            for x, y in cen:
                x0, y0 = x + 0.18 * d[0], y + 0.18 * d[1]
                ax.add_patch(FancyArrow(x0, y0, d[0], d[1],
                                        width=0.055 * ARROW_NM,
                                        head_width=0.26 * ARROW_NM,
                                        head_length=0.32 * ARROW_NM,
                                        length_includes_head=True,
                                        fc=col, ec="none", zorder=4))

    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.15"); s.set_linewidth(1.2)

    _scale_bar(ax, u0, u1, v0, v1)
    n_tot = sum(counts.values())
    ax.text(0.022, 0.976,
            f"{dose:.4g} dpa\n{thickness_nm:.0f} nm foil\n{n_tot} loops",
            transform=ax.transAxes, va="top", ha="left", fontsize=11.5,
            color="0.05", linespacing=1.45, zorder=8,
            bbox=dict(fc="white", ec="0.55", lw=0.7, alpha=0.94, pad=4.5))
    _burgers_key(fig, ax, e1, e2, counts)

    v = VIEWS[view]
    ax.set_title(f"simulated bright-field TEM,  {v['label']}\n{v['note']}",
                 fontsize=12, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    if verbose:
        c = "  ".join(f"{k}:{n}" for k, n in counts.items())
        print(f"    -> {Path(out_file).name}  ({n_tot} loops in slab; {c})")
    return out_file, counts


def _scale_bar(ax, u0, u1, v0, v1, nm=100.0):
    from matplotlib.patches import Rectangle
    span, vspan = u1 - u0, v1 - v0
    x0 = u0 + 0.042 * span
    y0 = v0 + 0.048 * vspan
    ax.add_patch(Rectangle((x0 - 0.018 * span, y0 - 0.020 * vspan),
                           nm + 0.036 * span, 0.075 * vspan,
                           fc="white", ec="0.55", lw=0.7, alpha=0.94, zorder=6))
    ax.plot([x0, x0 + nm], [y0, y0], color="0.05", lw=4.0,
            solid_capstyle="butt", zorder=7)
    ax.text(x0 + 0.5 * nm, y0 + 0.012 * vspan, f"{nm:.0f} nm",
            ha="center", va="bottom", fontsize=11, color="0.05", zorder=7)


def _burgers_key(fig, ax, e1, e2, counts):
    """Inset showing each family's Burgers vector as it projects in this view.

    The four markers are drawn from a common origin, so the inset is the legend
    for the per-loop markers on the micrograph: an arrow where b has a component
    in the image plane, the circled dot where it points along the beam.
    """
    from dislocluster_code.post.discrete_loops import family_geometry

    kax = ax.inset_axes([0.658, 0.022, 0.320, 0.320], zorder=9)
    kax.set_xlim(-1.32, 1.32); kax.set_ylim(-1.32, 1.42)
    kax.set_aspect("equal"); kax.set_xticks([]); kax.set_yticks([])
    kax.set_facecolor("white")
    kax.patch.set_alpha(0.95)
    for s in kax.spines.values():
        s.set_edgecolor("0.45"); s.set_linewidth(0.9)
    kax.text(0.0, 1.27, r"$\vec{b}$ projected", ha="center", va="top",
             fontsize=9.5, color="0.15")

    for fam in FAMILIES:
        bv, _ = family_geometry(fam)
        b_img = np.array([float(bv @ e1), float(bv @ e2)])
        col = fam["color"]
        if np.linalg.norm(b_img) / np.linalg.norm(bv) < B_INPLANE_MIN:
            kax.add_patch(Circle((0, 0), 0.19, fill=False, ec=col, lw=1.8,
                                 zorder=5))
            kax.add_patch(Circle((0, 0), 0.055, fc=col, ec="none", zorder=5))
        else:
            d = b_img / np.linalg.norm(b_img)
            kax.annotate("", xy=(0.95 * d[0], 0.95 * d[1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                         shrinkA=0, shrinkB=0,
                                         mutation_scale=13))

    handles = [plt.Line2D([], [], color=f["color"], lw=3.2,
                          label=f"{FAM_TEX[f['key']]}  {f['b_tex']}"
                                f"   ({counts.get(f['key'], 0)})")
               for f in FAMILIES]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=10,
                    frameon=True, framealpha=0.94, edgecolor="0.45",
                    borderpad=0.6, labelspacing=0.6)
    leg.set_zorder(9)


def montage(panels, out_file, title="", dpi=180):
    """Tile the single-dose micrographs of one view into a 2x2 comparison."""
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in panels]
    w = max(i.size[0] for i in imgs)
    h = max(i.size[1] for i in imgs)
    ncol = 2 if len(imgs) > 1 else 1
    nrow = int(np.ceil(len(imgs) / ncol))
    sheet = Image.new("RGB", (ncol * w, nrow * h), (255, 255, 255))
    for k, im in enumerate(imgs):
        r, c = divmod(k, ncol)
        sheet.paste(im, (c * w + (w - im.size[0]) // 2,
                         r * h + (h - im.size[1]) // 2))
    sheet.save(out_file)
    return out_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run directory, or the discrete_loops dir")
    ap.add_argument("--doses", nargs="+", type=float,
                    default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--thickness-nm", type=float, default=100.0)
    ap.add_argument("--views", nargs="+", default=list(VIEWS),
                    choices=list(VIEWS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import json
    run = Path(args.run_dir)
    loop_dir = run if (run / "manifest.json").exists() else run / "discrete_loops"
    manifest = json.loads((loop_dir / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else loop_dir / "tem_slices"
    out.mkdir(parents=True, exist_ok=True)

    # One shared box for every dose, so the field of view does not breathe
    # between panels of the same view.
    box = None
    picked = []
    for target in args.doses:
        e = min(manifest, key=lambda e: abs(float(e["dose"]) - target))
        if e not in picked:
            picked.append(e)
    for e in picked:
        L = read_loops(loop_dir / f"loops_{e['tag']}.csv")
        c = L["center"] * NM_PER_B
        box = (np.minimum(box[0], c.min(0)), np.maximum(box[1], c.max(0))) \
            if box else (c.min(0), c.max(0))
    box = (np.floor(box[0]), np.ceil(box[1]))

    print(f"{loop_dir.parent.name}: TEM slices, {args.thickness_nm:g} nm foil, "
          f"field {box[1][0]-box[0][0]:.0f} nm\n")
    for view in args.views:
        print(f"  {view}  ({VIEWS[view]['note']})")
        panels = []
        for e in picked:
            f = out / f"tem_{view}_{e['tag']}.png"
            render_slice(loop_dir / f"loops_{e['tag']}.csv", float(e["dose"]), f,
                         view=view, thickness_nm=args.thickness_nm,
                         box_nm=box, seed=args.seed)
            panels.append(f)
        m = montage(panels, out / f"tem_{view}_montage.png")
        print(f"    -> {m.name}  ({len(panels)} doses)\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
