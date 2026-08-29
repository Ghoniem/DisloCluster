"""The loop-denuded zone: loop density and size against distance to a face.

`gb_absorption` removes a loop when it grows large enough to touch a free
surface, which is what produces the denuded band TEM sees at a grain boundary.
This is the figure that shows it, and the comparison it needs is a run with the
channel against the same run without it -- one difference, so the difference
measures that one thing.

    python -m dislocluster_code.post.denuded_zone <gb-run> [<twin-run>]

WHY THIS IS BINNED BY DISTANCE AND NOT PLOTTED AS A FIELD. The zone is a
boundary layer a few nm wide on a 500 nm crystal, so a cut plane renders it as
one pixel. `gb_distance` is the natural coordinate -- the minimum distance to
any face of the convex body the CD nodes fill -- and it is the same coordinate
the channel itself is driven by (`--x_gb` per node), so the figure is drawn in
the variable the physics is written in.

THE SIZE IS CONTENT-WEIGHTED, NOT A NODE MEAN. `d = 2 l_k sqrt(c/n)` is a
non-linear reduction, so averaging `d` over the nodes of a bin and reducing the
bin's summed `c` and `n` are different numbers. The second is the one that
means something: it is the mean size of the loops in that shell, where the first
weights a node holding one loop the same as a node holding a million.

THE ZERO FLOOR IS A FLOOR. Number and content are clamped at `C_floor = 1e-20`,
so a fully swept node reports 1e-20 and not 0; a bin at 1e-6 of its twin is
swept as completely as this model can express, not partially swept.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import field as mfield
from dislocluster_code.post.fields import gb_distance

#: Family index -> the two state slots holding its number and its content.
#: Families 0..3 sit in the legacy slots, 4..8 in the appended block.
def _slots(k):
    return (4 + k, 8 + k) if k < 4 else (19 + (k - 4), 24 + (k - 4))


#: Basal <c> is c_f, c_p and the pyramid c_0; prismatic <a> is the three
#: interstitial variants and their three vacancy counterparts.
CFAM = (0, 7, 8)
AFAM = (1, 2, 3, 4, 5, 6)


def _lengths():
    Om = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL,
                                           'atomicVolume_SI'))
    b_a, b_c = 3.23e-10, 2.575e-10          # <a> = a, <c> = c/2
    return Om, np.sqrt(Om / (np.pi * b_a)), np.sqrt(Om / (np.pi * b_c))


def profile(run, dose=10.0, edges=None):
    """(bin centres, N_a, N_c, d_a, d_c, counts) at the snapshot near `dose`."""
    run = Path(run)
    z = np.load(run / 'march_state.npz')
    doses, Y, nodes = z['doses'], z['Y'], z['nodes']
    i = int(np.argmin(np.abs(doses - float(dose))))
    y = Y[i]
    b_SI = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL, 'b_SI'))
    x = gb_distance(np.asarray(nodes, float)) * b_SI * 1e9        # nm
    Om, l_a, l_c = _lengths()
    if edges is None:
        edges = np.concatenate([[0.0], np.geomspace(1.0, max(x.max(), 2.0), 18)])
    out = {}
    for tag, fams, l in (('a', AFAM, l_a), ('c', CFAM, l_c)):
        n = sum(y[:, _slots(k)[0]] for k in fams)
        c = sum(y[:, _slots(k)[1]] for k in fams)
        N, d, cnt = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            cnt.append(int(m.sum()))
            if not m.any():
                N.append(np.nan); d.append(np.nan); continue
            N.append(float(n[m].mean()) / Om)
            ns, cs = float(n[m].sum()), float(c[m].sum())
            d.append(2.0 * l * np.sqrt(cs / ns) * 1e9 if ns > 0 else np.nan)
        out[tag] = (np.array(N), np.array(d), np.array(cnt))
    centres = np.sqrt(np.maximum(edges[:-1], 1e-3) * edges[1:])
    return centres, out, float(doses[i]), edges


def render(run, twin=None, dose=10.0, out_file=None, dpi=200):
    xc, A, used, edges = profile(run, dose)
    B = profile(twin, dose)[1] if twin else None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, lab in ((axes[0], 'N', 'number density  [m$^{-3}$]'),
                         (axes[1], 'd', 'mean diameter  [nm]')):
        j = 0 if key == 'N' else 1
        for fam, col, name in (('c', 'C3', r'$\langle c\rangle$ basal'),
                               ('a', 'C0', r'$\langle a\rangle$ prismatic')):
            ax.plot(xc, A[fam][j], color=col, marker='o', ms=3.5, lw=1.6,
                    label=f"{name}, GB absorption on")
            if B is not None:
                ax.plot(xc, B[fam][j], color=col, ls='--', lw=1.2, alpha=0.75,
                        label=f"{name}, no GB channel")
        ax.set_xscale('log')
        ax.set_xlabel('distance to the nearest face  [nm]')
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=7.5, loc='lower right')
    fig.suptitle(f"loop-denuded zone at {used:g} dpa — "
                 f"{Path(run).name}", fontsize=10)
    fig.tight_layout()
    out_file = Path(out_file) if out_file else Path(run) / 'gb' / 'denuded_zone.png'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  {Path(run).name}   {used:g} dpa")
    print(f"  {'x [nm]':>14}{'nodes':>8}{'N_a [m^-3]':>13}{'N_c [m^-3]':>13}"
          f"{'d_a [nm]':>10}{'d_c [nm]':>10}"
          + ("      N_a/twin   N_c/twin" if B is not None else ""))
    for k in range(len(xc)):
        if A['a'][2][k] == 0:
            continue
        row = (f"  {edges[k]:6.1f}-{edges[k+1]:<7.1f}{A['a'][2][k]:>8d}"
               f"{A['a'][0][k]:>13.3e}{A['c'][0][k]:>13.3e}"
               f"{A['a'][1][k]:>10.2f}{A['c'][1][k]:>10.2f}")
        if B is not None:
            row += (f"{A['a'][0][k] / B['a'][0][k]:>14.3g}"
                    f"{A['c'][0][k] / B['c'][0][k]:>11.3g}")
        print(row)
    print(f"\n  wrote {out_file}")
    return out_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('twin', nargs='?', default=None,
                    help='the same case without gb_absorption, for comparison')
    ap.add_argument('--dose', type=float, default=10.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    render(a.run, a.twin, dose=a.dose, out_file=a.out)


if __name__ == '__main__':
    main()
