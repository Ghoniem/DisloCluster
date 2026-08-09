"""
movies.py — animated field evolution over the snapshots of a 3-D march.

Renders one frame per snapshot for each requested quantity and assembles them
into an animated GIF. GIF rather than MP4 because no ffmpeg is present in this
environment; matplotlib offers only the ``pillow`` and ``html`` writers, and a
GIF is self-contained and opens anywhere.

TWO THINGS THAT DECIDE WHETHER AN ANIMATION IS READABLE
-------------------------------------------------------
1. **A fixed colour scale.** Per-frame autoscaling makes every frame look the
   same and animates the colour bar instead of the physics -- a population that
   grows by four decades would appear static. The limits here are computed once
   over ALL frames and held, so motion in the image is motion in the field.
2. **Frames taken from the march state, not from the evl filenames.** The
   snapshots of a logarithmic dose grid do not lie on the 1 dpa lattice that
   names ``evl_<N>.txt``, so several would collide on one index.
   ``march_state.npz`` carries the true dose of every snapshot, so it is read
   instead and the CD block rebuilt from it.

USAGE
-----
    python -m py_utils.movies <run_dir> [--species Cv Ci n_vL n_a1] [--fps 12]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")

from py_utils import paths                                   # noqa: E402
from py_utils import modelib_field as mfield                 # noqa: E402
from py_utils.modelib_fields import (                        # noqa: E402
    plot_field_panels, SPECIES,
)
from py_utils.modelib_report import DEFAULT_PLANES           # noqa: E402

# What the request asks for: the two mobile species and the two loop families
# that carry the microstructure.
DEFAULT_SPECIES = ("Cv", "Ci", "n_vL", "n_a1")


def cd_blocks(run_dir, variant_weights=(1 / 3, 1 / 3, 1 / 3)):
    """``(doses, nodes, {i: (P, F)})`` rebuilt from ``march_state.npz``.

    ``F`` is the 12-column CD block the field plotting code expects: four
    mobile species followed by (number, content) for the four loop families.
    """
    z = np.load(Path(run_dir) / "march_state.npz")
    doses, Y, nodes = z["doses"], z["Y"], z["nodes"]
    omega = mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL)
    out = {}
    for i in range(len(doses)):
        F = np.empty((Y.shape[1], mfield.N_CD_COLS))
        F[:, :mfield.M_SIZE] = Y[i][:, :mfield.M_SIZE]
        F[:, mfield.M_SIZE:] = mfield.immobile_0d_to_modelib(
            Y[i], omega, variant_weights)
        out[i] = (nodes, F)
    return np.asarray(doses, dtype=float), nodes, out


def global_limits(frames, species, floor_decades=6.0):
    """One (vmin, vmax) per species over every frame.

    ``vmax`` is the 99.9th percentile over all frames, as the single-frame code
    uses; ``vmin`` is anchored ``floor_decades`` below it rather than at the true
    minimum, which on a Dirichlet face reaches 1e-27 and would spend the whole
    colour range on the boundary nodes.
    """
    lims = {}
    for sp in species:
        col = SPECIES[sp][0]
        allv = np.concatenate([F[:, col] for _, F in frames.values()])
        allv = allv[np.isfinite(allv)]
        vmax = float(np.percentile(allv, 99.9))
        pos = allv[allv > 0]
        vmin = float(pos.min()) if pos.size else 0.0
        vmin = max(vmin, vmax * 10.0 ** (-floor_decades))
        lims[sp] = (vmin, vmax)
    return lims


def render(run_dir, out_dir, species=DEFAULT_SPECIES, planes=DEFAULT_PLANES,
           fps=12, keep_frames=False, n_slice=140, verbose=True):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doses, nodes, frames = cd_blocks(run_dir)
    lims = global_limits(frames, species)
    if verbose:
        print(f"{run_dir.name}: {len(doses)} frames, "
              f"{doses[0]:.4g} .. {doses[-1]:.4g} dpa")
        for sp, (a, b) in lims.items():
            print(f"  {sp:<5} colour scale {a:.3e} .. {b:.3e}")

    from PIL import Image
    written = []
    for sp in species:
        label = SPECIES[sp][1]
        fdir = out_dir / f"_frames_{sp}"
        fdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        paths_png = []
        for i, d in enumerate(doses):
            png = fdir / f"f{i:04d}.png"
            plot_field_panels(
                None, [i], [d], species=(sp,), plane=planes,
                n_slice=n_slice, vlims={sp: lims[sp]}, fields={i: frames[i]},
                out_file=png, column_titles=False,
                title=f"{label}   {d:.4g} dpa")
            paths_png.append(png)
        gif = out_dir / f"{sp}.gif"
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                for p in paths_png]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        for im in imgs:
            im.close()
        if not keep_frames:
            shutil.rmtree(fdir, ignore_errors=True)
        written.append(gif)
        if verbose:
            print(f"  {gif.name}  ({len(doses)} frames, "
                  f"{time.perf_counter() - t0:.0f} s, "
                  f"{gif.stat().st_size / 1e6:.1f} MB)")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None,
                    help="default: <run_dir>/movies")
    ap.add_argument("--species", nargs="+", default=list(DEFAULT_SPECIES))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--n-slice", type=int, default=140)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "movies"
    render(run_dir, out, species=tuple(args.species), fps=args.fps,
           n_slice=args.n_slice, keep_frames=args.keep_frames)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
