"""One figure, several doses, one loop family -- panels only, no titles.

`march_report.write_3d` writes the loop overlays one file per family per dose,
which is right for a run directory and wrong for a paper: the dose evolution of
a population is one figure, not four, and each panel there carries a title that
becomes redundant the moment they sit side by side.

    python -m dislocluster_code.post.loop_montage <run> --family c \\
        --doses 1e-4 1e-2 1 10

Defaults to the basal <c> family at 1e-4, 1e-2, 1 and 10 dpa in a 2x2 grid.
Panels carry NO titles; label them in the caption or with `--panel-labels`,
which draws a bare (a)/(b)/(c)/(d) instead.

HOW THE PANELS ARE MADE. Each one is `plot_field_panels` with a single dose, so
it is the same drawing as the run's own `loops_<fam>_<dose>.png` -- same
platelet placement, same background field, same wireframe and orientation
triad -- and the montage only arranges them. Rendering each panel separately
and compositing (rather than asking for a 1 x N strip) is what allows a 2 x 2
grid, and it costs nothing here because these figures are raster anyway: the
platelet overlay is ~16 s per panel, which dwarfs any compositing.

THE RADII ARE EXAGGERATED AND NOT COMPARABLE BETWEEN FAMILIES. `loop_scale` is
0.55 for <c> and 2.2 for <a>, because <c> loops are ~6x larger; sizes are
faithful WITHIN a figure and meaningless between a <c> figure and an <a> one.
The montage keeps one family precisely so the comparison it invites -- across
dose -- is the one that is valid.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from dislocluster_code.post import movies as movies_mod
from dislocluster_code.post.march_report import (DEFAULT_PLANES, FAMILY_BG,
                                                 FAMILY_SLUG_A1, LOOP_SCALE)
from dislocluster_code.post.fields import plot_field_panels

#: slug -> family index, the inverse of march_report's map.
BY_SLUG = {v: k for k, v in FAMILY_SLUG_A1.items()}

DEFAULT_DOSES = (1e-4, 1e-2, 1.0, 10.0)


def _nearest(doses, want):
    """Index of the SOLVED snapshot nearest each requested dose, in log space.

    Requesting a dose the march never wrote would otherwise either fail or --
    worse -- silently interpolate, and an interpolated loop population is not a
    population: `cd_blocks` keeps `interp=1` for exactly this reason. The dose
    actually used is returned so a caller can say so.
    """
    ld = np.log10(np.maximum(np.asarray(doses, float), 1e-30))
    out = []
    for w in want:
        i = int(np.argmin(np.abs(ld - np.log10(max(float(w), 1e-30)))))
        out.append(i)
    return out


def _crop_white(img, tol=0.995):
    """Trim the white margin around a rendered panel.

    `plot_field_panels` lays out for a standalone figure, so each panel PNG
    carries its own generous margin. Composited untrimmed, four of those put a
    band of white between the rows wider than the gap between the axes -- the
    montage looked sparse for a reason that had nothing to do with the data.
    Cropping to the ink is what makes a 2 x 2 read as one figure.
    """
    a = img[..., :3] if img.ndim == 3 and img.shape[2] >= 3 else img
    ink = (a.min(axis=2) < tol) if a.ndim == 3 else (a < tol)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return img[r0:r1, c0:c1]


def montage(run, family='c', doses=DEFAULT_DOSES, layout=None, out_file=None,
            planes=DEFAULT_PLANES, panel_labels=False, dpi=200,
            panel_size=(3.3, 3.2), verbose=True, loop_source='packed',
            loop_region='interior'):
    """Assemble the dose evolution of ONE loop family into one figure.

    `loop_source` is `march_report.write_3d`'s, and means the same thing:
    `'packed'` fills each panel with exaggerated non-overlapping platelets,
    which shows where the population is dense and how its size varies;
    `'discrete'` draws the real `sum_j n_j V_j` loops that
    `post.discrete_loops` exports, at true radii and their sampled positions,
    so the montage and the discrete render show the same objects.
    """
    run = Path(run)
    if loop_source not in ('packed', 'discrete'):
        raise SystemExit(f"loop_source must be 'packed' or 'discrete', "
                         f"not {loop_source!r}")
    key = BY_SLUG.get(family)
    if key is None:
        raise SystemExit(f"unknown family {family!r}; "
                         f"choose from {sorted(BY_SLUG)}")

    all_doses, nodes, frames = movies_mod.cd_blocks(run)
    idx = _nearest(all_doses, doses)
    used = [float(all_doses[i]) for i in idx]
    if verbose:
        print(f"{run.name}: {len(all_doses)} snapshots, "
              f"{nodes.shape[0]} CD nodes")
        for w, u in zip(doses, used):
            note = '' if abs(u - float(w)) <= 1e-9 * max(1.0, u) \
                else f"   <- nearest solved snapshot to {float(w):g}"
            print(f"  panel at {u:<10.4g} dpa{note}")

    n = len(idx)
    if layout is None:
        layout = (2, 2) if n == 4 else (1, n)
    nrow, ncol = layout
    if nrow * ncol < n:
        raise SystemExit(f"layout {nrow}x{ncol} cannot hold {n} panels")

    tmp = Path(tempfile.mkdtemp(prefix='loop_montage_'))
    panels = []
    for j, i in enumerate(idx):
        t0 = time.perf_counter()
        f = tmp / f"panel_{j}.png"
        if loop_source == 'discrete':
            from dislocluster_code.post.march_report import _discrete_populations
            pop = _discrete_populations(run, used[j], loop_region,
                                        verbose=False).get(family)
            kw = dict(loop_scale=1.0,
                      loop_population=({i: pop} if pop else {}))
        else:
            kw = dict(loop_scale=LOOP_SCALE[key])
        plot_field_panels(
            None, [i], [used[j]], species=(FAMILY_BG[key],), loop_family=key,
            plane=planes, fields={i: frames[i]},
            out_file=f, column_titles=False, title=None,
            figsize_per_panel=panel_size, **kw)
        panels.append(f)
        if verbose:
            print(f"    panel {j + 1}/{n} rendered "
                  f"({time.perf_counter() - t0:.0f} s)")

    imgs = [_crop_white(mpimg.imread(p)) for p in panels]
    # SIZE THE FIGURE TO THE PANELS, not the other way round. A cropped panel
    # is wide -- drawing plus colorbar -- while a default axes box is nearly
    # square, so `imshow` letterboxes it and the white it adds reappears as a
    # band between the rows. Taking the aspect from the images themselves makes
    # each axes exactly the shape of what goes in it, and the rows close up.
    aspect = float(np.median([im.shape[0] / im.shape[1] for im in imgs]))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(ncol * panel_size[0],
                                      nrow * panel_size[0] * aspect))
    for ax in axes.ravel():
        ax.axis('off')
    for j, im in enumerate(imgs):
        ax = axes.ravel()[j]
        ax.imshow(im)
        if panel_labels:
            # A bare letter, not a dose. The dose belongs in the caption: a
            # per-panel dose label is what the run's own figures already carry
            # and what this figure exists to strip.
            ax.text(0.02, 0.98, f"({chr(97 + j)})", transform=ax.transAxes,
                    ha='left', va='top', fontsize=12, fontweight='bold')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0.01, hspace=0.01)
    out_file = Path(out_file) if out_file else (
        run / '3d' / f"loops_{family}_montage.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    for p in panels:
        p.unlink(missing_ok=True)
    tmp.rmdir()
    if verbose:
        print(f"\nwrote {out_file}")
        print(f"  {nrow}x{ncol}, doses "
              f"{', '.join(f'{u:g}' for u in used)} dpa, no panel titles")
    return out_file, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--family', default='c',
                    help=f"one of {sorted(BY_SLUG)} (default c, basal <c>)")
    ap.add_argument('--doses', nargs='*', type=float, default=list(DEFAULT_DOSES))
    ap.add_argument('--layout', default=None,
                    help='e.g. 2x2 or 1x4; default 2x2 for four panels')
    ap.add_argument('--out', default=None)
    ap.add_argument('--panel-labels', action='store_true',
                    help='draw a bare (a)/(b)/(c)/(d) in each panel')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--loop-source', choices=('packed', 'discrete'),
                    default='packed',
                    help="'packed' fills the panel with exaggerated "
                         "platelets; 'discrete' draws the population "
                         "discrete_loops exports, at true radii")
    ap.add_argument('--loop-region', choices=('interior', 'domain'),
                    default='interior')
    a = ap.parse_args(argv)
    layout = None
    if a.layout:
        r, c = a.layout.lower().split('x')
        layout = (int(r), int(c))
    montage(a.run_dir, family=a.family, doses=a.doses, layout=layout,
            out_file=a.out, panel_labels=a.panel_labels, dpi=a.dpi,
            loop_source=a.loop_source, loop_region=a.loop_region)


if __name__ == '__main__':
    main()
