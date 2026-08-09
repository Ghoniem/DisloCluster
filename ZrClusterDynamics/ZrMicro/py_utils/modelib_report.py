"""modelib_report.py — one figure per quantity from the 3-D MoDELib output.

`modelib_fields` and `modelib_gb` draw multi-panel figures: four species or four
loop families side by side, four doses across. That is the layout of Figures
19-27 of the D1/M1 deliverable. This module renders the same content but writes
**one file per plotted quantity**, which is what the coupling driver needs so
each field can be dropped into a document or compared across runs on its own.

    3d/   Figs 19-23 — one file per quantity per dose
          mobile   Cv, Ci, C2i, C3i                  (Fig 19)
          density  N_c, N_a1, N_a2, N_a3             (Fig 20)
          content  C_c, C_a1, C_a2, C_a3             (Fig 21)
          loops    platelet overlay per family       (Figs 22-23)

    gb/   Figs 24-27 — one file per quantity, all doses overlaid as curves
          mobile   Cv, Ci, C2i, C3i                  (Fig 24)
          density  N_c, N_a1, N_a2, N_a3             (Fig 25)
          content  C_c, C_a1, C_a2, C_a3             (Fig 26)
          size     mean diameter per family          (Fig 27)

All of it reads `evl/cdNodes.txt` + `evl/evl_<N>.txt`; nothing here runs a solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

from py_utils.modelib_fields import (
    load_cd_fields, plot_field_panels, SPECIES, FAMILIES, B_SI, OMEGA_B3,
)
from py_utils.modelib_gb import profile

__all__ = ["MOBILE", "DENSITY", "CONTENT", "FAMILY_SLUG", "LOOP_SCALE",
           "write_3d_figures", "write_gb_figures", "write_mesh_figure",
           "write_size_distribution_figures", "dose_steps"]

# Quantity groups, in the order the deliverable presents them.
MOBILE = ("Cv", "Ci", "C2i", "C3i")
DENSITY = ("n_vL", "n_a1", "n_a2", "n_a3")
CONTENT = ("c_vL", "c_a1", "c_a2", "c_a3")

# Short, filesystem-safe names. n_vL is the basal <c> family, so it is filed as
# N_c rather than N_vL to match how the figures are labelled.
FILE_SLUG = {
    "Cv": "Cv", "Ci": "Ci", "C2i": "C2i", "C3i": "C3i",
    "n_vL": "N_c", "n_a1": "N_a1", "n_a2": "N_a2", "n_a3": "N_a3",
    "c_vL": "C_c", "c_a1": "C_a1", "c_a2": "C_a2", "c_a3": "C_a3",
}
FAMILY_SLUG = {0: "c", 1: "a1", 2: "a2", 3: "a3"}
FAMILY_BG = {0: "c_vL", 1: "c_a1", 2: "c_a2", 3: "c_a3"}
# <c> loops are ~6x larger than <a>, so a single exaggeration factor would leave
# the <a> platelets invisible. Sizes are therefore comparable within a figure
# but NOT between the <c> and <a> figures. Halved relative to the deliverable's
# figures so that the domain can be filled with non-overlapping platelets rather
# than showing a sparse scatter.
LOOP_SCALE = {0: 1.5, 1: 7.0, 2: 7.0, 3: 7.0}


def dose_steps(doses, dose_per_step=1.0, dose_seed=0.1):
    """Dose values -> output step indices.

    MoDELib writes output AFTER solve(), so ``evl_N`` holds the state after
    (N+1) steps taken from the seed:

        dose = dose_seed + (N + 1) * dose_per_step

    ``dose_seed`` defaults to 0.1 because that is what every case in this
    repository used originally, and the default reproduces the previous
    behaviour exactly. It must be passed for a run seeded anywhere else: a run
    seeded at 1 dpa holds 6 dpa in ``evl_4``, not ``evl_5``, so rendering it
    against the default reads one whole dose step off and labels the figure
    with a dose the file does not contain.
    """
    return [int(round((d - dose_seed) / dose_per_step)) - 1 for d in doses]


def _tag(dose):
    """`6.0 -> '06dpa'`, `0.5 -> '00p5dpa'` — zero-padded so a listing sorts."""
    if float(dose).is_integer():
        return f"{int(dose):02d}dpa"
    return f"{dose:04.1f}".replace(".", "p") + "dpa"


# ── 3d/ : one field figure per quantity per dose ─────────────────────────────
def write_3d_figures(evl_dir, doses, out_dir, dose_per_step=1.0,
                     overlays=True, n_loops=None, plane="y", view=None,
                     verbose=True, dose_seed=0.1):
    """Figs 19-23, split one quantity per file. Returns the list of paths.

    `plane` selects the mid-cut. "y" is right for the cube, where every cut is
    equivalent by symmetry. For the hexagonal prism "z" is the informative one:
    it cuts normal to the c-axis and so shows the hexagonal section with the
    depletion rim on all six prism faces, whereas a y cut passes through two
    opposite vertices and is just a rectangle. `view=None` picks the viewpoint
    from the plane.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = dose_steps(doses, dose_per_step, dose_seed)
    written = []

    for dose, step in zip(doses, steps):
        tag = _tag(dose)
        for group in (MOBILE, DENSITY, CONTENT):
            for sp in group:
                out = out_dir / f"{FILE_SLUG[sp]}_{tag}.png"
                plot_field_panels(
                    evl_dir, [step], [dose], species=(sp,), plane=plane,
                    view=view, out_file=out, column_titles=False,
                    title=f"{SPECIES[sp][1]}  at {dose:g} dpa")
                written.append(out)
                if verbose:
                    print(f"  {out.name}")

        if overlays:
            for k, slug in FAMILY_SLUG.items():
                out = out_dir / f"loops_{slug}_{tag}.png"
                plot_field_panels(
                    evl_dir, [step], [dose], species=(FAMILY_BG[k],),
                    loop_family=k, loop_scale=LOOP_SCALE[k], n_loops=n_loops,
                    plane=plane, view=view, out_file=out, column_titles=False,
                    title=(f"{FAMILIES[k][0]} loop population at {dose:g} dpa "
                           f"(platelet radii x{LOOP_SCALE[k]:g}, not to scale)"))
                written.append(out)
                if verbose:
                    print(f"  {out.name}")
    return written


# ── 3d/ : the finite-element mesh ────────────────────────────────────────────
def write_mesh_figure(msh_path, out_file, extents_nm=None, cut="y",
                      view=(10.0, -80.0), title=None, color_by_size=False):
    """Draw the FE mesh: the domain cut in half so the interior is visible.

    Tetrahedra whose centroid lies on the near side of the cut plane are
    discarded and the exposed surface of what remains is drawn. That shows the
    exterior faceting AND the element sizes on the cut plane in one view, which
    is what makes a boundary-layer refinement legible — a plain surface mesh
    would only ever show the smallest elements.
    """
    import sys
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from py_utils import paths

    if str(paths.GMSH_DIR) not in sys.path:
        sys.path.insert(0, str(paths.GMSH_DIR))
    import generate_mesh as gm

    nodes, tets = gm.read_msh(msh_path)
    if len(tets) == 0:
        raise ValueError(f"no tetrahedra in {msh_path}")

    lo, hi = nodes.min(0), nodes.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    if extents_nm is not None:
        nodes = (nodes - lo) / span * np.asarray(extents_nm, float)
        lo, hi = nodes.min(0), nodes.max(0)
    axis = {"x": 0, "y": 1, "z": 2}[cut]

    # Retain the half on the FAR side of the cut from the camera, so the cut
    # surface faces the viewer. Keeping the near half instead shows the outer
    # boundary skin — which on a boundary-layer mesh is uniformly the finest
    # elements in the model, hiding the grading completely behind it.
    el, az = np.radians(view[0]), np.radians(view[1])
    cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    mid = 0.5 * (lo[axis] + hi[axis])
    centroid = nodes[tets].mean(axis=1)
    keep = (tets[centroid[:, axis] >= mid] if cam[axis] < 0
            else tets[centroid[:, axis] <= mid])
    if len(keep) == 0:
        keep = tets

    # Faces on the surface of the retained solid appear exactly once. Track
    # which tet each face came from so the face can be coloured by its size.
    faces = np.concatenate([keep[:, [0, 1, 2]], keep[:, [0, 1, 3]],
                            keep[:, [0, 2, 3]], keep[:, [1, 2, 3]]])
    owner = np.tile(np.arange(len(keep)), 4)
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    single = idx[counts == 1]
    surf, surf_owner = faces[single], owner[single]

    fig = plt.figure(figsize=(6.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    if color_by_size:
        # Mean edge length of the owning tet — the natural measure of local
        # resolution, and the only way a graded mesh reads unambiguously in a
        # projection where near and far elements overlap.
        v = nodes[keep]
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        edge = np.mean([np.linalg.norm(v[:, a] - v[:, b], axis=1)
                        for a, b in pairs], axis=0)
        vals = edge[surf_owner]
        norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
        cmap = plt.cm.viridis
        ax.add_collection3d(Poly3DCollection(
            nodes[surf], facecolors=cmap(norm(vals)), edgecolor="#263238",
            linewidths=0.2))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("element edge length [nm]", fontsize=10)
    else:
        ax.add_collection3d(Poly3DCollection(
            nodes[surf], facecolor="#cfd8e6", edgecolor="#37474f",
            linewidths=0.25, alpha=1.0))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"n_nodes": len(nodes), "n_tets": len(tets),
            "n_shown": len(keep), "n_faces": len(surf)}


# ── gb/ : loop size distribution over the domain ─────────────────────────────
def write_size_distribution_figures(evl_dir, doses, out_dir, dose_per_step=1.0,
                                    n_bins=40, verbose=True, dose_seed=0.1):
    """Size distribution of the loop population over the whole cube, per dose.

    IMPORTANT — what this is and is not. The model carries ONE mean size per
    family per node, not a spectrum, so this is not a true per-loop size
    distribution in the nucleation-and-growth sense. It is the distribution of
    the LOCAL MEAN loop diameter across the domain, weighted by how many loops
    each node represents (its number density). Spatial variation is the only
    source of spread: near the boundary the loops are small and numerous, in the
    interior large and fewer, so the width of this distribution measures the
    denuded-zone structure.

    Nodes are weighted by density alone, i.e. treated as equal-volume. On a
    graded mesh — one with boundary-layer refinement — that over-weights the
    refined region, because no nodal volumes are available: `evl/cdNodes.txt`
    carries positions only, with no connectivity.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = dose_steps(doses, dose_per_step, dose_seed)
    written = []

    for dose, step in zip(doses, steps):
        P, F = load_cd_fields(evl_dir, step)
        tag = _tag(dose)
        for k, (label, ncol, ccol, bmag, _n, _c) in enumerate(FAMILIES):
            n_k, c_k = F[:, ncol], F[:, ccol]
            ok = (np.isfinite(n_k) & np.isfinite(c_k) & (n_k > 0) & (c_k > 0))
            m = np.zeros_like(n_k)
            m[ok] = c_k[ok] / (n_k[ok] * OMEGA_B3)          # defects per loop
            ok &= m > 1.0
            if not ok.any():
                continue
            d_nm = 2.0 * np.sqrt(m[ok] * OMEGA_B3 / (np.pi * bmag)) * B_SI * 1e9
            w = n_k[ok] / B_SI ** 3                          # loops per m^3

            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            bins = np.linspace(d_nm.min(), d_nm.max(), n_bins + 1) \
                if d_nm.max() > d_nm.min() else n_bins
            ax.hist(d_nm, bins=bins, weights=w, color="#3f6fb5",
                    edgecolor="#1b2f4d", linewidth=0.4)
            dbar = float(np.average(d_nm, weights=w))
            ax.axvline(dbar, color="#c62828", lw=1.6, ls="--",
                       label=f"number-weighted mean {dbar:.2f} nm")
            ax.set_xlabel("loop diameter [nm]", fontsize=11)
            ax.set_ylabel(r"loops per m$^{3}$ in bin", fontsize=11)
            ax.set_title(f"{label} loop size distribution at {dose:g} dpa",
                         fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, frameon=False)
            fig.tight_layout()
            out = out_dir / f"size_dist_{FAMILY_SLUG[k]}_{tag}.png"
            fig.savefig(out, dpi=250, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
            if verbose:
                print(f"  {out.name}")
    return written


# ── gb/ : one profile figure per quantity, all doses overlaid ────────────────
def _profile_figure(x_sets, labels, xlabel, ylabel, title, out_file, logy=True):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = colormaps["coolwarm"]
    colors = [cmap(t) for t in np.linspace(0.0, 1.0, max(len(x_sets), 1))]
    for (x, y), lab, c in zip(x_sets, labels, colors):
        ax.plot(x, y, color=c, lw=1.9, label=lab)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_gb_figures(evl_dir, doses, out_dir, dose_per_step=1.0, max_nm=150.0,
                     n_bins=60, verbose=True, dose_seed=0.1):
    """Figs 24-27, split one quantity per file. Returns the list of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = dose_steps(doses, dose_per_step, dose_seed)
    data = {s: load_cd_fields(evl_dir, s) for s in steps}
    labels = [f"{d:g} dpa" for d in doses]
    XL = "distance from grain boundary, $x$ [nm]"
    written = []

    # Fig 24 — mobile concentrations (per atom, as solved)
    for sp in MOBILE:
        col, math_label = SPECIES[sp]
        sets = []
        for s in steps:
            P, F = data[s]
            sets.append(profile(P, np.maximum(F[:, col], 1e-300), n_bins, max_nm))
        out = out_dir / f"gb_{FILE_SLUG[sp]}.png"
        _profile_figure(sets, labels, XL, f"{math_label}  [per atom]",
                        f"{math_label} vs distance from the grain boundary", out)
        written.append(out)
        if verbose:
            print(f"  {out.name}")

    # Figs 25 & 26 — loop density [m^-3] and stored content [defects m^-3]
    for k, (label, ncol, ccol, bmag, _n, _c) in enumerate(FAMILIES):
        slug = FAMILY_SLUG[k]

        sets = [profile(data[s][0], data[s][1][:, ncol] / B_SI ** 3, n_bins, max_nm)
                for s in steps]
        out = out_dir / f"gb_N_{slug}.png"
        _profile_figure(sets, labels, XL, r"number density [m$^{-3}$]",
                        f"{label} loop density vs distance from the GB", out)
        written.append(out)
        if verbose:
            print(f"  {out.name}")

        sets = [profile(data[s][0],
                        data[s][1][:, ccol] / (OMEGA_B3 * B_SI ** 3), n_bins, max_nm)
                for s in steps]
        out = out_dir / f"gb_C_{slug}.png"
        _profile_figure(sets, labels, XL, r"stored defects [m$^{-3}$]",
                        f"{label} stored content vs distance from the GB", out)
        written.append(out)
        if verbose:
            print(f"  {out.name}")

        # Fig 27 — mean diameter. Formed from the BINNED mean content and
        # density, never by averaging nodal diameters: size is a ratio of two
        # fields and the mean of a ratio is not the ratio of the means. Near the
        # boundary the density rises by three orders while the content does not,
        # so averaging nodal ratios there is dominated by the sparsest nodes.
        sets = []
        for s in steps:
            P, F = data[s]
            x, nbar = profile(P, F[:, ncol], n_bins, max_nm)
            _, cbar = profile(P, F[:, ccol], n_bins, max_nm)
            m = cbar / np.maximum(nbar * OMEGA_B3, 1e-300)
            r_b = np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * bmag))
            sets.append((x, 2.0 * r_b * B_SI * 1e9))
        out = out_dir / f"gb_d_{slug}.png"
        _profile_figure(sets, labels, XL, "mean loop diameter [nm]",
                        f"{label} loop diameter vs distance from the GB", out,
                        logy=False)
        written.append(out)
        if verbose:
            print(f"  {out.name}")

    return written
