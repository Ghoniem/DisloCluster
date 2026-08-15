"""
movies.py — animated field evolution over the snapshots of a 3-D march.

Renders one frame per snapshot for each requested quantity and assembles them
into an animated GIF. GIF rather than MP4 because no ffmpeg is present in this
environment; matplotlib offers only the ``pillow`` and ``html`` writers, and a
GIF is self-contained and opens anywhere.

THREE THINGS THAT DECIDE WHETHER AN ANIMATION IS READABLE
---------------------------------------------------------
1. **A fixed colour scale.** Per-frame autoscaling makes every frame look the
   same and animates the colour bar instead of the physics -- a population that
   grows by four decades would appear static. The limits here are computed once
   over ALL frames and held, so motion in the image is motion in the field.
2. **Frames taken from the march state, not from the evl filenames.** The
   snapshots of a logarithmic dose grid do not lie on the 1 dpa lattice that
   names ``evl_<N>.txt``, so several would collide on one index.
   ``march_state.npz`` carries the true dose of every snapshot, so it is read
   instead and the CD block rebuilt from it.
3. **Enough frames.** The march writes ONE snapshot per dose interval, so a
   run over eight snapshot doses animates as an eight-frame flipbook however
   high the fps. ``interp`` subdivides each output interval into that many
   sub-frames, filling the gaps by interpolation -- see below.

INTERPOLATED FRAMES ARE NOT SOLVED STATES
-----------------------------------------
``subdivide`` synthesizes intermediate frames from the two snapshots that
bracket them. Nothing is re-solved: these frames are an interpolation of the
march, not part of it, and every one is labelled ``(interp)`` in the rendered
title so a reader cannot mistake one for a computed state. Use them to make the
motion legible, never to read a value off.

Both the dose axis and the state are interpolated GEOMETRICALLY wherever both
endpoints are positive, linearly otherwise (the pristine seed sits at dose 0,
and species at the 1e-20 floor can be exactly zero). Geometric is the right
choice twice over: the fields grow by decades through the nucleation transient,
where a linear interpolant would sit at the upper endpoint for the whole
interval; and it commutes with the power-law reductions the figures draw, so a
mean loop diameter d ~ (c/N)^(1/3) formed from interpolated c and N is the same
number as the interpolated d.

The extra frames are free of solver cost but not of render cost: wall clock and
GIF size both scale with the frame count, and the frame buffer holds one CD
block per frame.

USAGE
-----
    python -m py_utils.movies <run_dir> [--species Cv Ci n_vL n_a1] [--fps 12]
                                        [--interp 5 | --no-interp]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np


import matplotlib                                            # noqa: E402
matplotlib.use("Agg")

from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.coupling import field as mfield                 # noqa: E402
from dislocluster_code.post.fields import (                        # noqa: E402
    plot_field_panels, SPECIES,
)
from dislocluster_code.post.report import DEFAULT_PLANES           # noqa: E402

# What the request asks for: the two mobile species and the two loop families
# that carry the microstructure.
DEFAULT_SPECIES = ("Cv", "Ci", "n_vL", "n_a1")

# Species drawn on a LINEAR colour scale from zero.
#
# The mobile fields are quasi-steady by construction -- the fast solve returns
# the steady field for the immobile state currently held -- so their entire
# story is a slow interior drawdown as the loop sinks accumulate: Cv falls 15%
# and Ci 16% over the whole 10 dpa march. Their spatial range, by contrast, is
# enormous, because Dirichlet faces pin them at thermal equilibrium: Cv reaches
# 8.5e-15 and Ci 4.1e-27 at the wall. A log scale wide enough to show the wall
# spends six decades on a layer a few nodes thick and compresses the interior
# drawdown into 1% of the colour range -- the movie then looks perfectly static,
# which is the opposite of what it is for. On a linear scale from zero the same
# drawdown moves the interior from 62% to 53% of the range, and the boundary
# layer still reads as a dark rim of the correct thickness.
#
# The loop densities need the opposite: they grow fifteen decades from the
# pristine floor, so only a log scale shows anything at all.
LINEAR_SPECIES = ("Cv", "Ci")

# Sub-frames per output interval when interpolation is on. 5 turns the eight
# snapshot doses of a typical march into 41 frames, which reads as motion rather
# than as a flipbook. `cd_blocks` deliberately does NOT default to this -- the
# static panels and the discrete-loop population must see solved states only.
DEFAULT_INTERP = 5


def _blend(A, B, s):
    """``A`` and ``B`` blended at fraction ``s``, geometrically where it can be.

    Element-wise: the geometric mean exp((1-s)lnA + s lnB) wherever both
    endpoints are strictly positive, the linear blend elsewhere. Concentrations
    here run over thirty decades and grow as powers of dose, so a linear
    interpolant would pin an interval to its upper endpoint; but the pristine
    seed holds species at exactly zero, which has no logarithm.
    """
    out = A + s * (B - A)
    pos = (A > 0.0) & (B > 0.0)
    if pos.any():
        out[pos] = np.exp((1.0 - s) * np.log(A[pos]) + s * np.log(B[pos]))
    # exp(log(x)) is not exactly x, so an unchanged cell would drift by ~1e-14
    # relative and land just outside the band its two neighbours define. That
    # is most of the domain, not an edge case: every Dirichlet-pinned node and
    # every species still at the 1e-20 floor has identical endpoints.
    same = A == B
    out[same] = A[same]
    return out


def subdivide_doses(doses, n=DEFAULT_INTERP):
    """The interpolated dose axis, and which of its entries are solved states.

    Returns ``(doses_out, src, frac, is_real)``: for each output frame, the
    index of the interval it starts from, the fraction along it, and whether it
    is one of the original snapshots. Splitting this out from the state blend
    keeps the (large) state arrays out of the schedule arithmetic.
    """
    doses = np.asarray(doses, dtype=float)
    n = max(int(n), 1)
    if n == 1 or doses.size < 2:
        k = np.arange(doses.size)
        return doses, k, np.zeros(doses.size), np.ones(doses.size, dtype=bool)

    d_out, src, frac, real = [], [], [], []
    for j in range(doses.size - 1):
        d0, d1 = float(doses[j]), float(doses[j + 1])
        # Geometric subdivision keeps a logarithmic snapshot grid logarithmic,
        # so the frames stay evenly spaced in the variable the physics moves in.
        geometric = d0 > 0.0 and d1 > 0.0
        for k in range(n):
            s = k / n
            d_out.append(d0 * (d1 / d0) ** s if geometric
                         else d0 + s * (d1 - d0))
            src.append(j)
            frac.append(s)
            real.append(k == 0)
    d_out.append(float(doses[-1]))
    src.append(doses.size - 2)
    frac.append(1.0)
    real.append(True)
    return (np.asarray(d_out, dtype=float), np.asarray(src, dtype=int),
            np.asarray(frac, dtype=float), np.asarray(real, dtype=bool))


def cd_blocks(run_dir, variant_weights=(1 / 3, 1 / 3, 1 / 3), interp=1):
    """``(doses, nodes, {i: (P, F)})`` rebuilt from ``march_state.npz``.

    ``F`` is the 12-column CD block the field plotting code expects: four
    mobile species followed by (number, content) for the four loop families.

    ``interp`` subdivides each output interval into that many frames; the
    default of 1 returns the solved snapshots and nothing else, which is what
    every caller other than the movie renderer wants. Use
    :func:`cd_blocks_interpolated` when you also need to know which frames are
    synthetic.
    """
    return cd_blocks_interpolated(run_dir, variant_weights, interp)[:3]


def cd_blocks_interpolated(run_dir, variant_weights=(1 / 3, 1 / 3, 1 / 3),
                           interp=DEFAULT_INTERP):
    """``(doses, nodes, frames, is_real)`` with ``interp`` frames per interval.

    ``is_real[i]`` marks frame ``i`` as a solved snapshot rather than an
    interpolant.

    The blend is taken on the CD BLOCK, not on the 19-column state it is built
    from. ``immobile_0d_to_modelib`` sums species to form a family's number and
    content, and a sum of geometric blends is not the geometric blend of the
    sums (Holder gives an inequality, not equality) -- blending the state first
    therefore produces frames that are NOT bracketed by the two snapshots
    around them, which in a movie reads as a population briefly overshooting
    and falling back. Blending what the figures actually draw keeps every
    interpolant between its neighbours, and is cheaper: the conversion runs
    once per solved snapshot rather than once per frame.
    """
    z = np.load(Path(run_dir) / "march_state.npz")
    doses, Y, nodes = z["doses"], z["Y"], z["nodes"]
    omega = mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL)

    solved = []
    for i in range(len(doses)):
        F = np.empty((Y.shape[1], mfield.N_CD_COLS))
        F[:, :mfield.M_SIZE] = Y[i][:, :mfield.M_SIZE]
        F[:, mfield.M_SIZE:] = mfield.immobile_0d_to_modelib(
            Y[i], omega, variant_weights)
        solved.append(F)

    d_out, src, frac, real = subdivide_doses(doses, interp)
    out = {}
    for i, (j, s, is_real) in enumerate(zip(src, frac, real)):
        if is_real:
            F = solved[j if s == 0.0 else j + 1]
        else:
            F = _blend(solved[j], solved[j + 1], float(s))
        out[i] = (nodes, F)
    return d_out, nodes, out, real


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
        if sp in LINEAR_SPECIES:
            lims[sp] = (0.0, vmax)
            continue
        pos = allv[allv > 0]
        vmin = float(pos.min()) if pos.size else 0.0
        vmin = max(vmin, vmax * 10.0 ** (-floor_decades))
        lims[sp] = (vmin, vmax)
    return lims


def render(run_dir, out_dir, species=DEFAULT_SPECIES, planes=DEFAULT_PLANES,
           fps=12, keep_frames=False, n_slice=90, interp=DEFAULT_INTERP,
           verbose=True):
    """Render one GIF per species.

    ``n_slice`` is the plane sampling grid. 90 rather than the 140 used for a
    single static panel: on a 0.5 um box that is 5.5 nm per sample against a
    ~15 nm CD node spacing, so the field is still oversampled, while the cost
    per frame drops from 13.4 s to 5.7 s -- which over 4 species x 101 frames
    is the difference between 90 minutes and 38.

    ``interp`` sub-frames per output interval fill in the gaps between solved
    snapshots (``1`` disables it). Render time and GIF size scale with it.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doses, nodes, frames, is_real = cd_blocks_interpolated(
        run_dir, interp=interp)
    lims = global_limits(frames, species)
    if verbose:
        n_snap = int(np.count_nonzero(is_real))
        print(f"{run_dir.name}: {len(doses)} frames, "
              f"{doses[0]:.4g} .. {doses[-1]:.4g} dpa")
        if len(doses) != n_snap:
            print(f"  {n_snap} solved snapshots + {len(doses) - n_snap} "
                  f"interpolated ({interp} frames per interval); "
                  f"interpolated frames are labelled in the title")
        for sp, (a, b) in lims.items():
            kind = "linear" if sp in LINEAR_SPECIES else "log"
            print(f"  {sp:<5} {kind:>6} colour scale {a:.3e} .. {b:.3e}")

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
                log=(sp not in LINEAR_SPECIES),
                out_file=png, column_titles=False,
                # An interpolated frame says so. It is a blend of the two
                # snapshots around it, not a state the march ever solved for.
                title=f"{label}   {d:.4g} dpa"
                      + ("" if is_real[i] else "   (interp)"))
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
    ap.add_argument("--n-slice", type=int, default=90)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--interp", type=int, default=DEFAULT_INTERP,
                    help="frames per output interval; interpolated frames are "
                         "labelled and are not solved states (1 = off)")
    ap.add_argument("--no-interp", dest="interp", action="store_const",
                    const=1, help="solved snapshots only")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "movies"
    render(run_dir, out, species=tuple(args.species), fps=args.fps,
           n_slice=args.n_slice, keep_frames=args.keep_frames,
           interp=args.interp)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
