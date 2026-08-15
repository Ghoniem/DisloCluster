"""
report_march.py — the complete figure set for one 3-D march, from its state file.

Produces, under the run directory:

    3d/               two orthogonal mid-cuts per quantity per selected dose
    gb/               grain-boundary profiles, selected doses overlaid
    movies/           animated GIFs over every snapshot, plus --interp
                      interpolated frames per interval (labelled, not solved)
    volume_average/   the 0-D figure suite, computed from the volume-averaged
                      trajectory of the 3-D march
    report.md         what was produced and the conventions behind it

EVERYTHING COMES FROM march_state.npz
------------------------------------
Not from the written ``evl`` files. Two reasons, both of which have bitten this
project already. The snapshots of a logarithmic dose grid do not lie on the
1 dpa lattice that names ``evl_<N>.txt``, so several would collide on one
index; and ``march_state.npz`` stores the true dose of every snapshot next to
its state, so no filename convention has to be reconstructed to know what a
frame holds.

CONVENTIONS
-----------
* **One prismatic variant.** a1, a2 and a3 are crystallographically equivalent
  at zero applied stress and the coupling bridge splits the lumped 0-D <a>
  population equally across them, so they carry identical fields. Plotting a1
  is complete; plotting all three triples the figure count for nothing.
* **Two cuts per panel**, one vertical (normal to y) and one horizontal (normal
  to z), drawn into the same axes.
* **Fixed colour limits across a movie**, computed once over all frames, so
  that motion in the image is motion in the field rather than in the colour bar.
* **Volume-weighted averages.** A plain nodal mean is wrong on a
  boundary-refined mesh; see ``py_utils/volume_average.py``.

USAGE
-----
    python -m py_utils.report_march <run_dir> [--n-doses 6] [--no-movies]
                                              [--interp 5]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


import matplotlib                                            # noqa: E402
matplotlib.use("Agg")

from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.post.fields import (                        # noqa: E402
    plot_field_panels, SPECIES, FAMILIES, B_SI, OMEGA_B3,
)
from dislocluster_code.post.gb import profile                      # noqa: E402
from dislocluster_code.post.report import (                        # noqa: E402
    FILE_SLUG, DEFAULT_PLANES, MOBILE, DENSITY_A1, CONTENT_A1,
    FAMILY_SLUG_A1, LOOP_SCALE, FAMILY_BG, _profile_figure,
)
from dislocluster_code.post import movies as movies_mod                    # noqa: E402
from dislocluster_code.post import volume_average as va                    # noqa: E402
from dislocluster_code.integration import ADAPTIVE            # noqa: E402


def dose_tag(d):
    """Filesystem-safe, sortable tag for an arbitrary dose."""
    if d <= 0:
        return "0000dpa"
    e = int(np.floor(np.log10(d)))
    return f"{d:.3g}".replace(".", "p").replace("-", "m") + "dpa"


def select_doses(doses, n=6):
    """`n` roughly log-spaced snapshots, always including the first and last."""
    doses = np.asarray(doses, dtype=float)
    pos = doses[doses > 0]
    if pos.size == 0:
        return list(range(len(doses)))
    want = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n)
    idx = sorted({int(np.argmin(np.abs(doses - w))) for w in want})
    if idx[-1] != len(doses) - 1:
        idx.append(len(doses) - 1)
    return idx


# ── 3d/ ──────────────────────────────────────────────────────────────────────
def write_3d(doses, frames, idx, out_dir, planes=DEFAULT_PLANES,
             overlays=True, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in idx:
        d, tag = float(doses[i]), dose_tag(float(doses[i]))
        for group in (MOBILE, DENSITY_A1, CONTENT_A1):
            for sp in group:
                out = out_dir / f"{FILE_SLUG[sp]}_{tag}.png"
                plot_field_panels(None, [i], [d], species=(sp,), plane=planes,
                                  fields={i: frames[i]}, out_file=out,
                                  column_titles=False,
                                  title=f"{SPECIES[sp][1]}   {d:.4g} dpa")
                written.append(out)
        if overlays:
            for k, slug in FAMILY_SLUG_A1.items():
                out = out_dir / f"loops_{slug}_{tag}.png"
                plot_field_panels(
                    None, [i], [d], species=(FAMILY_BG[k],), loop_family=k,
                    loop_scale=LOOP_SCALE[k], plane=planes, fields={i: frames[i]},
                    out_file=out, column_titles=False,
                    title=(f"{FAMILIES[k][0]} loop population at {d:.4g} dpa "
                           f"(platelet radii x{LOOP_SCALE[k]:g}, not to scale)"))
                written.append(out)
        if verbose:
            print(f"  3d/ {tag}: {len(MOBILE) + 4 + (2 if overlays else 0)} panels")
    return written


# ── gb/ ──────────────────────────────────────────────────────────────────────
def write_gb(doses, frames, idx, out_dir, max_nm=150.0, n_bins=60,
             verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{doses[i]:.4g} dpa" for i in idx]
    XL = "distance from grain boundary, $x$ [nm]"
    written = []

    for sp in MOBILE:
        col, math_label = SPECIES[sp]
        sets = [profile(frames[i][0], np.maximum(frames[i][1][:, col], 1e-300),
                        n_bins, max_nm) for i in idx]
        out = out_dir / f"gb_{FILE_SLUG[sp]}.png"
        _profile_figure(sets, labels, XL, f"{math_label}  [per atom]",
                        f"{math_label} vs distance from the grain boundary", out)
        written.append(out)

    for k, slug in FAMILY_SLUG_A1.items():
        label, ncol, ccol = FAMILIES[k][0], FAMILIES[k][1], FAMILIES[k][2]
        sets = [profile(frames[i][0], frames[i][1][:, ncol] / B_SI ** 3,
                        n_bins, max_nm) for i in idx]
        out = out_dir / f"gb_N_{slug}.png"
        _profile_figure(sets, labels, XL, r"number density [m$^{-3}$]",
                        f"{label} loop density vs distance from the GB", out)
        written.append(out)

        sets = [profile(frames[i][0],
                        frames[i][1][:, ccol] / (OMEGA_B3 * B_SI ** 3),
                        n_bins, max_nm) for i in idx]
        out = out_dir / f"gb_C_{slug}.png"
        _profile_figure(sets, labels, XL, r"stored defects [m$^{-3}$]",
                        f"{label} stored content vs distance from the GB", out)
        written.append(out)
    if verbose:
        print(f"  gb/ {len(written)} profiles over {len(idx)} doses")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--n-doses", type=int, default=6,
                    help="snapshots to draw as static 3-D panels and GB curves")
    ap.add_argument("--no-movies", action="store_true")
    ap.add_argument("--no-volume-average", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--interp", type=int, default=movies_mod.DEFAULT_INTERP,
                    help="movie frames per output interval (1 = solved "
                         "snapshots only); interpolated frames are labelled")
    ap.add_argument("--mc-samples", type=int, default=4_000_000)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    t_all = time.perf_counter()
    print(f"{run.name}\n{ADAPTIVE.key} (Option {ADAPTIVE.option}) — "
          f"{ADAPTIVE.integrator}\n")

    doses, nodes, frames = movies_mod.cd_blocks(run)
    idx = select_doses(doses, args.n_doses)
    print(f"{len(doses)} snapshots, {nodes.shape[0]} CD nodes")
    print(f"static figures at: "
          f"{', '.join(f'{doses[i]:.4g}' for i in idx)} dpa\n")

    write_3d(doses, frames, idx, run / "3d")
    write_gb(doses, frames, idx, run / "gb")

    if not args.no_movies:
        print("\nmovies:")
        movies_mod.render(run, run / "movies", fps=args.fps,
                          interp=args.interp)

    if not args.no_volume_average:
        print("\nvolume average:")
        va.main([str(run), "--out", str(run / "volume_average"),
                 "--samples", str(args.mc_samples)])

    (run / "report.md").write_text("\n".join([
        f"# {run.name}", "",
        f"- integration: **{ADAPTIVE.key}** (Option {ADAPTIVE.option}) — "
        f"{ADAPTIVE.integrator}",
        f"- snapshots  : {len(doses)}, {doses[0]:.4g} .. {doses[-1]:.4g} dpa",
        f"- CD nodes   : {nodes.shape[0]}",
        f"- generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- git hash   : {paths.git_hash()}", "",
        "## Directories", "",
        "| directory | contents |",
        "|---|---|",
        "| `3d/` | two orthogonal mid-cuts (one vertical, one horizontal) per "
        "quantity, at selected doses |",
        "| `gb/` | grain-boundary profiles over the first 150 nm, doses "
        "overlaid |",
        "| `movies/` | animated GIFs over every snapshot, fixed colour scale |",
        "| `volume_average/` | the 0-D figure suite from the volume-averaged "
        "trajectory |", "",
        "## Conventions", "",
        "- Only the **a1** prismatic variant is plotted. a1, a2 and a3 are "
        "equivalent at zero applied stress and the coupling bridge makes them "
        "identical, so the other two carry no information.",
        "- Movie colour limits are computed once over all frames and held, so "
        "motion in the image is motion in the field.",
        "- Volume averages are **volume-weighted**, not nodal means: the "
        "boundary-refined mesh has far higher node density near the wall, and "
        "an unweighted mean would report close to the boundary value.",
        "- Near-boundary loop densities are not quantitative — the faces sink "
        "mobile defects but not loops.",
    ]) + "\n", encoding="utf-8")

    print(f"\ntotal {time.perf_counter() - t_all:.0f} s -> {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
