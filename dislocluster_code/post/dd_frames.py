"""dd_frames.py — draw the dislocation network of a DD run, step by step.

`post/discrete_loops.py` draws the population a continuum field IMPLIES. This
module draws what the solver actually has: the `evl_<N>.txt` configuration
MoDELib writes every `outputFrequency` steps, with the gliding dislocation in
the position it reached and the shape it bowed into. That difference is the
whole point of a hardening run — the obstacle field is an input, and what the
line does in it is the result.

WHAT IS IN AN evl FILE, AND WHAT THIS TAKES OUT OF IT
-----------------------------------------------------
Ten integer header lines (`DDconfigIO::writeTxtStream`), then four record
blocks. Their layouts are the `operator<<` of the four IO classes, and none of
them is self-describing, so they are transcribed here once:

===========  ==============================================================
node         `sID  P(3)  V(3)  climbVelocityScalar(mSize=4)  velRed  meshLoc`
loop         `sID  B(3)  N(3)  P(3)  grainID  loopType  loopLength(4)  area`
loopLink     `loopID  sourceID  sinkID  hasNetworkLink  meshLocation`
loopNode     `loopID  sID  P(3)  networkNodeID  periodicShift(3)  edge0 edge1`
===========  ==============================================================

`loopType` is `DislocationLoopIO::DislocationLoopType`: 0 GLISSILE, 1 SESSILE,
2 VIRTUAL. **That integer is what separates the moving dislocation from the
irradiation loops**, and it is the only thing that does: both are loops, and
`aLoopGenerator`/`FrankLoopsGenerator` insert every irradiation loop as
SESSILE while `PeriodicDipoleGenerator` inserts its two glide arms as GLISSILE.

DRAWING A PERIODIC LOOP
-----------------------
A loop that crosses a cell face is stored as several patches: the loop nodes
carry a `periodicShift`, and the NETWORK node they point at is the position
inside the primary cell. Following the link chain through network positions
therefore produces a segment that jumps across the whole cell wherever the loop
wraps. Those jumps are cut (`_split_wrapped`) rather than drawn, so a wrapped
loop appears as the several arcs it really is inside the cell.

SEGMENTS THAT ARE NOT DISLOCATION
---------------------------------
`hasNetworkLink` is not the whole of the "is there dislocation here" test. A
loop boundary can run out and back along the same node pair -- the loop pinches
to a sliver of zero area -- and both records carry `hasNetworkLink = 1` while
the two contributions have OPPOSITE sense, so the network link between those
two nodes has **zero Burgers vector** and there is no dislocation along it.
`periodicDipoleIndividual` makes exactly this: its glissile arm closes back
onto the anchor node it shares with the sessile prismatic loop, and the tie
runs from the cell centre out to wherever the line has bowed to. Drawn, it is a
straight segment joining the two arms across the cell -- which a reader takes
for dislocation, and which is the same failure the `hasNetworkLink` filter
already exists to prevent, one level up. `_null_pairs` sums the loop Burgers
vectors per network-node pair and `drop_null` cuts the polyline wherever the
sum is zero. On the 200 nm hardening case at 0.1 dpa that is 8 pairs of 1576:
the dipole's own scaffolding, plus one contact where the gliding line has
zipped onto an <a> loop and locally annihilated it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dislocluster_code import paths                                  # noqa: E402
from dislocluster_code.post.fields import B_SI                       # noqa: E402

__all__ = ["read_network", "loop_polylines", "plot_frame", "snapshots",
           "movie", "frame_steps", "GLISSILE", "SESSILE", "VIRTUAL"]

GLISSILE, SESSILE, VIRTUAL = 0, 1, 2

N_HEADER = 10
NODE_COLS = 13          # 1 + 3 + 3 + 4 + 1 + 1
LOOP_COLS = 17          # 1 + 3 + 3 + 3 + 1 + 1 + 4 + 1
LINK_COLS = 5
LOOPNODE_COLS = 11      # 1 + 1 + 3 + 1 + 3 + 2

# Ink and mark colours. The moving dislocation is the subject of these figures,
# so it takes the one saturated colour; the frozen loops are context and are
# drawn muted, by family, in the same hues `discrete_loops` uses for them.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
BOX = "#b9b7b0"
DISLOCATION = "#d03b3b"
OBSTACLE_C = "#2a78d6"
OBSTACLE_A = "#7f8c8d"


def read_network(evl_file):
    """Parse one `evl_<N>.txt` into `{nodes, loops, links, loop_nodes}`.

    Only the four dislocation blocks are read; the CD block and the inclusion
    blocks are skipped by their header counts, which is why the header is
    consumed in full rather than assumed empty.
    """
    lines = Path(evl_file).read_text(encoding="utf-8",
                                     errors="replace").splitlines()
    h = [int(lines[i].strip()) for i in range(N_HEADER)]
    n_nodes, n_loops, n_links, n_loop_nodes = h[0], h[1], h[2], h[3]

    i = N_HEADER

    def block(count, ncols):
        nonlocal i
        rows = np.array([[float(x) for x in lines[i + k].split()[:ncols]]
                         for k in range(count)], dtype=float) if count else \
            np.zeros((0, ncols))
        i += count
        return rows

    nodes = block(n_nodes, NODE_COLS)
    loops = block(n_loops, LOOP_COLS)
    links = block(n_links, LINK_COLS)
    loop_nodes = block(n_loop_nodes, LOOPNODE_COLS)
    return dict(nodes=nodes, loops=loops, links=links, loop_nodes=loop_nodes,
                file=str(evl_file))


def _split_wrapped(pts, box_edges, frac=0.5):
    """Split a polyline wherever it jumps more than `frac` of a box edge."""
    if len(pts) < 2:
        return [pts]
    d = np.abs(np.diff(pts, axis=0))
    cut = np.any(d > frac * np.asarray(box_edges)[None, :], axis=1)
    out, start = [], 0
    for k in np.nonzero(cut)[0]:
        if k + 1 - start >= 2:
            out.append(pts[start:k + 1])
        start = k + 1
    if len(pts) - start >= 2:
        out.append(pts[start:])
    return out


def _null_pairs(net, tol=1e-9):
    """Network-node pairs whose loop contributions cancel: `b_net` = 0.

    One network link can carry several loop links -- that is what a junction
    is -- and the dislocation it holds is their SUM. A pair whose sum is zero
    is a bookkeeping edge (a pinched loop boundary, or a line that has zipped
    onto another loop and annihilated it), not dislocation.
    """
    B = {int(r[0]): np.asarray(r[1:4], float) for r in net["loops"]}
    ln_net = {int(r[1]): int(r[5]) for r in net["loop_nodes"]}
    acc = {}
    for r in net["links"]:
        lid, src, snk, real = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        if not real:
            continue
        a, b = ln_net.get(src), ln_net.get(snk)
        if a is None or b is None or a == b or lid not in B:
            continue
        key, sgn = ((a, b), 1.0) if a < b else ((b, a), -1.0)
        acc[key] = acc.get(key, np.zeros(3)) + sgn * B[lid]
    return {k for k, v in acc.items() if np.all(np.abs(v) < tol)}


def loop_polylines(net, box_edges=None, close=True, real_only=True,
                   drop_null=True):
    """`[(loop_type, burgers, [polyline, ...]), ...]`, one entry per loop.

    The order of a loop's nodes comes from the LINK CHAIN, not from the file
    order: `loopLinks` is a set of directed source->sink pairs per loop, and
    following it is what turns a bag of nodes into a line. A loop whose chain
    is broken (it happens while the network is being remeshed) contributes the
    fragments it does have rather than nothing.

    `drop_null` additionally cuts the chain wherever the network link carries
    no net Burgers vector -- see `_null_pairs` and the module docstring. It is
    on by default: such a segment is not dislocation, and drawn it reads as
    dislocation.
    """
    node_pos = {int(r[0]): r[1:4] for r in net["nodes"]}
    ln_pos = {}
    for r in net["loop_nodes"]:
        lid, sid, nid = int(r[0]), int(r[1]), int(r[5])
        # The network node is the position inside the primary cell; fall back to
        # the loop node's own P + shift if the network node is absent (a
        # boundary loop node with no network link).
        p = node_pos.get(nid)
        if p is None:
            p = r[2:5] + r[6:9]
        ln_pos.setdefault(lid, {})[sid] = np.asarray(p, float)

    # `hasNetworkLink` (column 3) separates real dislocation from bookkeeping.
    # A loop that spans the periodic cell is closed through links that carry no
    # network link -- they exist to make the loop a loop, not because there is a
    # dislocation there -- and drawing them puts straight red segments across
    # the cell that a reader would take for dislocation. 72 of this frame's 194
    # glissile links are of that kind.
    chains, closure = {}, {}
    for r in net["links"]:
        lid, src, snk, real = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        (chains if real else closure).setdefault(lid, {})[src] = snk

    # Segments carrying no net Burgers vector, addressed by loop-node pair so
    # the walk below can cut on them directly.
    ln_net = {int(r[1]): int(r[5]) for r in net["loop_nodes"]}
    null_net = _null_pairs(net) if drop_null else set()

    def is_null(u, v):
        a, b = ln_net.get(u), ln_net.get(v)
        if a is None or b is None:
            return False
        return ((a, b) if a < b else (b, a)) in null_net

    out = []
    for r in net["loops"]:
        # Column 11, not 10: the loop record is
        # `sID B(3) N(3) P(3) grainID loopType ...`, so column 10 is the GRAIN
        # and reading it as the type marks every loop sessile in a single-grain
        # cell -- including the dislocation whose motion is the measurement.
        lid, b, ltype = int(r[0]), r[1:4], int(r[11])
        nxt = dict(chains.get(lid, {}))
        if not real_only:
            nxt.update(closure.get(lid, {}))
        pos = ln_pos.get(lid, {})
        if not nxt or not pos:
            continue

        # Walk EVERY component of the link chain, not just the one containing an
        # arbitrary starting node, and emit each as its own polyline. A loop
        # being remeshed can momentarily present more than one chain, and the
        # earlier version appended the nodes it had not walked in dictionary
        # order -- which drew straight segments between unrelated parts of the
        # line, exactly where a reader would interpret them as dislocation.
        pieces, visited = [], set()
        for seed in list(nxt):
            if seed in visited:
                continue
            walk, node = [], seed
            while node in nxt and node not in visited:
                visited.add(node)
                walk.append(node)
                node = nxt[node]
            closed = (node == seed and len(walk) > 2)
            if close and closed:
                walk.append(seed)
            elif node != seed and node in pos:
                # The walk stops ON the sink of its last link and had not
                # appended it: `while node in nxt and node not in visited`
                # exits with `node` still to be drawn. Every open run was
                # therefore one segment SHORT at its far end, and because a
                # seed landing mid-chain leaves the stretch behind it as its
                # own walk, that lost segment falls in the MIDDLE of a line.
                # On the 200 nm hardening case at 0.1 dpa step 7950 the
                # gliding dislocation came out as four disjoint stubs with
                # three gaps -- every second segment of it missing.
                walk.append(node)

            # One run of nodes per stretch of real dislocation: the walk is
            # cut at a null segment, and at a node whose position is missing
            # (joining its neighbours would draw a chord that is not line).
            runs, cur = [], []
            for k, n in enumerate(walk):
                p = pos.get(n)
                if p is None:
                    runs.append(cur)
                    cur = []
                    continue
                cur.append(p)
                if k + 1 < len(walk) and is_null(n, walk[k + 1]):
                    runs.append(cur)
                    cur = []
            runs.append(cur)

            for run in runs:
                if len(run) < 2:
                    continue
                pts = np.array(run, float)
                pieces.extend(_split_wrapped(pts, box_edges)
                              if box_edges is not None else [pts])
        if pieces:
            out.append((ltype, np.asarray(b, float), pieces))
    return out


def _box_wireframe(ax, edges, color=BOX, lw=0.8):
    lo, hi = -0.5 * np.asarray(edges), 0.5 * np.asarray(edges)
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(c[a], c[b]), color=color, lw=lw, zorder=1)


def plot_frame(evl_file, box_edges, out_file=None, title="", subtitle="",
               elev=22.0, azim=-58.0, dpi=150, show_obstacles=True,
               dislocation_lw=2.6, glide_normal=1, panels=("3d", "glide")):
    """Draw one configuration: the obstacle field and the dislocation in it.

    Two panels, because neither alone answers the question:

    * **3d** — where the dislocation is in the cell.
    * **glide** — everything projected along the glide-plane normal (y for
      slip system 6), which is the plane the line moves in. Pinning, bowing
      between obstacles and the released arc after breakaway are visible here
      and essentially invisible in the 3-D view, where they are a few b of
      curvature seen edge-on.

    The projection is of the WHOLE cell, not a slice: an obstacle is an
    obstacle to this line only if it sits near its glide plane, but which ones
    those are is exactly what the picture is for, so nothing is filtered.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = read_network(evl_file)
    edges = np.asarray(box_edges, float)
    lines = loop_polylines(net, box_edges=edges)
    nm = B_SI * 1e9
    ax_names = [a for a in ("x", "y", "z")]
    ip = [k for k in range(3) if k != glide_normal]      # in-plane axes

    n_panels = len(panels)
    fig = plt.figure(figsize=(6.4 * n_panels, 6.2))
    axes = {}
    for k, kind in enumerate(panels):
        if kind == "3d":
            axes[kind] = fig.add_subplot(1, n_panels, k + 1, projection="3d",
                                         computed_zorder=False)
        else:
            axes[kind] = fig.add_subplot(1, n_panels, k + 1)

    if "3d" in axes:
        _box_wireframe(axes["3d"], edges * nm)
    if "glide" in axes:
        a = axes["glide"]
        e = edges * nm
        a.add_patch(plt.Rectangle((-0.5 * e[ip[0]], -0.5 * e[ip[1]]),
                                  e[ip[0]], e[ip[1]], fill=False,
                                  edgecolor=BOX, lw=0.8, zorder=1))

    n_gl = n_se = 0
    for ltype, b, pieces in lines:
        if ltype == VIRTUAL:
            continue
        glissile = (ltype == GLISSILE)
        if not glissile and not show_obstacles:
            continue
        basal = abs(b[2]) > max(abs(b[0]), abs(b[1]))
        color = (DISLOCATION if glissile
                 else (OBSTACLE_C if basal else OBSTACLE_A))
        lw = dislocation_lw if glissile else 1.3
        alpha = 1.0 if glissile else 0.8
        z = 6 if glissile else 3
        for q in pieces:
            v = q * nm
            if "3d" in axes:
                axes["3d"].plot(v[:, 0], v[:, 1], v[:, 2], color=color, lw=lw,
                                alpha=alpha, zorder=z, solid_capstyle="round")
            if "glide" in axes:
                axes["glide"].plot(v[:, ip[0]], v[:, ip[1]], color=color,
                                   lw=lw, alpha=alpha, zorder=z,
                                   solid_capstyle="round")
        n_gl += glissile
        n_se += (not glissile)

    e = edges * nm
    if "3d" in axes:
        a = axes["3d"]
        a.set_xlim(-0.5 * e[0], 0.5 * e[0])
        a.set_ylim(-0.5 * e[1], 0.5 * e[1])
        a.set_zlim(-0.5 * e[2], 0.5 * e[2])
        a.set_box_aspect(tuple(e / e.max()))
        a.view_init(elev=elev, azim=azim)
        a.set_axis_off()
        a.text2D(0.5, 0.02, "the cell", transform=a.transAxes,
                 color=INK_MUTED, fontsize=9, ha="center")
    if "glide" in axes:
        a = axes["glide"]
        a.set_xlim(-0.55 * e[ip[0]], 0.55 * e[ip[0]])
        a.set_ylim(-0.55 * e[ip[1]], 0.55 * e[ip[1]])
        a.set_aspect("equal")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(BOX)
        a.tick_params(colors=INK_MUTED, labelsize=8, length=3)
        a.set_xlabel(f"{ax_names[ip[0]]} [nm]", color=INK_MUTED, fontsize=9)
        a.set_ylabel(f"{ax_names[ip[1]]} [nm]", color=INK_MUTED, fontsize=9)
        a.text(0.5, -0.16, f"projected along {ax_names[glide_normal]}, "
                           f"the glide-plane normal", transform=a.transAxes,
               color=INK_MUTED, fontsize=9, ha="center")
        # Which way the load drives the line. Without it the projection is a
        # picture of a tangle; with it, it is a picture of a line being pushed
        # into obstacles.
        a.annotate("", xy=(0.30, 1.045), xytext=(0.06, 1.045),
                   xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.4))
        a.text(0.32, 1.045, r"glide direction ($\tau$ on the prismatic system)",
               transform=a.transAxes, color=INK_MUTED, fontsize=9,
               va="center")

    if title:
        fig.suptitle(title, color=INK, fontsize=13, x=0.02, ha="left", y=0.98)
    head = subtitle + (f"        dislocation {n_gl}    obstacles {n_se}"
                       if subtitle else "")
    if head:
        fig.text(0.02, 0.93, head, color=INK_MUTED, fontsize=10, va="top")
    fig.subplots_adjust(top=0.85, left=0.04, right=0.98, bottom=0.16,
                        wspace=0.08)

    if out_file is not None:
        fig.savefig(out_file, dpi=dpi, facecolor="white")
        plt.close(fig)
        return out_file
    return fig


def frame_steps(sim_dir):
    """`[(runID, path), ...]` for every `evl_<N>.txt` in a case, in step order."""
    d = Path(sim_dir) / "evl"
    out = []
    for f in d.glob("evl_*.txt"):
        try:
            out.append((int(f.stem.split("_")[1]), f))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def _history(sim_dir):
    """`{runID: (tau_MPa, gamma_p)}` from `F/F_0.txt`, for frame captions."""
    from dislocluster_code.studies import hardening as H
    try:
        meta = json.loads((Path(sim_dir) / "hardening.json").read_text(
            encoding="utf-8"))
        mat = H.Material.from_file(None, meta.get("temperature_K", 573.0))
        r = H.reduce_run(sim_dir, mat, verbose=False)
        lab, data = H.read_F(sim_dir)
        run_col = lab.index("runID")
        return {int(row[run_col]): (r["tau_MPa"][k], r["gamma_p"][k])
                for k, row in enumerate(data)}
    except Exception:
        return {}


def snapshots(sim_dir, out_dir=None, n=6, steps=None, dose=None, verbose=True,
              **kw):
    """Render `n` configurations spread over the run (or the given `steps`)."""
    sim_dir = Path(sim_dir)
    meta = json.loads((sim_dir / "hardening.json").read_text(encoding="utf-8"))
    edges = np.array(meta["box"]["edges_nm"]) * 1e-9 / B_SI
    out_dir = Path(out_dir or sim_dir / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps = frame_steps(sim_dir)
    if not all_steps:
        raise FileNotFoundError(f"no evl_*.txt under {sim_dir}/evl")
    if steps is None:
        idx = np.unique(np.linspace(0, len(all_steps) - 1, n).round().astype(int))
        chosen = [all_steps[i] for i in idx]
    else:
        want = set(int(s) for s in steps)
        chosen = [(s, f) for s, f in all_steps if s in want]
    hist = _history(sim_dir)
    dose = meta.get("dose") if dose is None else dose

    written = []
    for step, f in chosen:
        tg = hist.get(step)
        sub = (f"step {step}" if tg is None else
               fr"step {step}    $\tau$ = {tg[0]:.1f} MPa    "
               fr"$\gamma_p$ = {tg[1]:.2e}")
        title = ("loop-free control" if not meta.get("loops") else
                 f"{dose:g} dpa" if dose is not None else "")
        title = (f"{meta['box']['edges_nm'][0]:.0f} nm periodic cell"
                 + (f" — {title}" if title else ""))
        png = out_dir / f"frame_{step:06d}.png"
        plot_frame(f, edges, out_file=png, title=title, subtitle=sub, **kw)
        written.append(png)
    if verbose:
        print(f"  {len(written)} snapshots -> {out_dir}")
    return written


def movie(sim_dir, out_file=None, fps=8, every=1, max_frames=200, gif=True,
          mp4=True, keep_frames=False, verbose=True, **kw):
    """One movie of the dislocation moving through the obstacle field.

    Every `evl_<N>.txt` the run wrote is a frame, so the cadence is the run's
    own `outputFrequency` and nothing is interpolated: each frame is a solved
    configuration.
    """
    import shutil
    import subprocess
    import tempfile
    from PIL import Image
    from dislocluster_code.post.loop_movie import ffmpeg_exe

    sim_dir = Path(sim_dir)
    steps = frame_steps(sim_dir)[::max(1, int(every))]
    if len(steps) > max_frames:
        idx = np.linspace(0, len(steps) - 1, max_frames).round().astype(int)
        steps = [steps[i] for i in np.unique(idx)]
    tmp = Path(tempfile.mkdtemp(prefix="ddframes_"))
    try:
        pngs = snapshots(sim_dir, out_dir=tmp, steps=[s for s, _ in steps],
                         verbose=False, **kw)
        out_file = Path(out_file or sim_dir / "dislocation_motion")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        w = max(Image.open(p).size[0] for p in pngs)
        h = max(Image.open(p).size[1] for p in pngs)
        size = (w + w % 2, h + h % 2)

        def padded(p):
            im = Image.open(p).convert("RGB")
            if im.size == size:
                return im
            c = Image.new("RGB", size, (255, 255, 255))
            c.paste(im, ((size[0] - im.size[0]) // 2,
                         (size[1] - im.size[1]) // 2))
            return c

        written = []
        if mp4:
            # The GIF is the fallback that opens anywhere and needs nothing
            # installed; the MP4 needs ffmpeg, which this environment may not
            # have. Missing ffmpeg downgrades the format, it does not fail the
            # movie.
            try:
                ffmpeg_exe()
            except Exception as e:
                if verbose:
                    print(f"  no ffmpeg ({e}); writing the GIF only")
                mp4 = False
        if mp4:
            for k, p in enumerate(pngs):
                padded(p).save(tmp / f"pad_{k:05d}.png")
            target = out_file.with_suffix(".mp4")
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error",
                            "-framerate", str(fps),
                            "-i", str(tmp / "pad_%05d.png"),
                            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                            "-pix_fmt", "yuv420p", str(target)], check=True)
            written.append(target)
        if gif:
            imgs = [padded(p).resize((int(size[0] * 0.6), int(size[1] * 0.6)),
                                     Image.LANCZOS)
                    .convert("P", palette=Image.ADAPTIVE, colors=128)
                    for p in pngs]
            target = out_file.with_suffix(".gif")
            imgs[0].save(target, save_all=True, append_images=imgs[1:],
                         duration=int(round(1000 / fps)), loop=0, optimize=True)
            written.append(target)
        if keep_frames:
            dest = out_file.parent / f"{out_file.name}_frames"
            dest.mkdir(exist_ok=True)
            for p in pngs:
                shutil.copy2(p, dest / p.name)
        if verbose:
            for f in written:
                print(f"  {f.name}  ({len(pngs)} frames, {fps} fps, "
                      f"{f.stat().st_size / 1e6:.1f} MB)")
        return written
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sim_dir")
    ap.add_argument("--snapshots", type=int, default=6)
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    snapshots(a.sim_dir, a.out, n=a.snapshots)
    if a.movie:
        movie(a.sim_dir, None if a.out is None else Path(a.out) / "motion",
              fps=a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
