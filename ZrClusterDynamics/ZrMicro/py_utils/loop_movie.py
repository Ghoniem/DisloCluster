"""
loop_movie.py -- assemble the per-dose discrete-loop renders into a movie.

``discrete_loops.py`` writes one PNG per dose per family (plus a combined view)
into ``<run>/discrete_loops/``. Those are already the frames of a movie; all
this module does is order them by dose and encode them.

WHY THE ORDER COMES FROM THE MANIFEST
-------------------------------------
The filenames carry the dose as a tag with the decimal point replaced by ``p``
(``loops_0p01048dpa.png``, ``loops_1p0975dpa.png``, ``loops_10dpa.png``), so a
lexical sort of the directory puts 10 dpa between 0.001 and 0.1. The dose is
therefore read from ``manifest.json``, which records it as a float alongside the
tag, and the frames are sorted on that.

ENCODING
--------
MP4 (H.264) through the ``imageio-ffmpeg`` bundled binary, and an animated GIF
through Pillow. The GIF is downscaled -- it is the fallback that opens anywhere,
while the MP4 carries the full render resolution. ``bbox_inches="tight"`` lets
the source PNGs differ in size by a pixel or two between doses, and H.264
requires even dimensions, so every frame is padded onto a common even-sized
canvas rather than resampled.

USAGE
-----
    python -m py_utils.loop_movie <run_dir> [--fps 5] [--series loops loops_c]
                                  [--no-gif] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

# The combined view first, then one series per immobile family. These are the
# prefixes discrete_loops.render writes: `loops_<tag>` and `loops_<key>_<tag>`.
DEFAULT_SERIES = ("loops", "loops_c", "loops_a1", "loops_a2", "loops_a3")

GIF_SCALE = 0.55            # GIF is the portable fallback, not the deliverable
GIF_COLORS = 128


def ffmpeg_exe():
    """Path to an ffmpeg binary, preferring the one bundled with imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError(
                "no ffmpeg available -- `pip install imageio-ffmpeg`, or run "
                "with --no-mp4 to get the GIF only")
        return exe


def frame_list(loop_dir, series):
    """``[(dose, Path), ...]`` for one series, in ascending dose order.

    Doses come from ``manifest.json`` rather than from the filename, because the
    tag is not lexically sortable -- see the module docstring.
    """
    loop_dir = Path(loop_dir)
    manifest = json.loads((loop_dir / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for entry in sorted(manifest, key=lambda e: float(e["dose"])):
        png = loop_dir / f"{series}_{entry['tag']}.png"
        if png.exists():
            out.append((float(entry["dose"]), png))
    return out


def _canvas_size(paths):
    """Smallest even-sided canvas holding every frame."""
    w = max(Image.open(p).size[0] for p in paths)
    h = max(Image.open(p).size[1] for p in paths)
    return w + (w % 2), h + (h % 2)


def _padded(path, size):
    """One frame, RGB, centered on a white canvas of ``size``."""
    im = Image.open(path).convert("RGB")
    if im.size == size:
        return im
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2))
    return canvas


def encode(loop_dir, out_dir, series="loops", fps=5, mp4=True, gif=True,
           verbose=True):
    """Encode one series. Returns the list of files written."""
    frames = frame_list(loop_dir, series)
    if not frames:
        if verbose:
            print(f"  {series}: no frames, skipped")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for _, p in frames]
    size = _canvas_size(paths)
    written = []

    if mp4:
        tmp = Path(tempfile.mkdtemp(prefix=f"{series}_frames_"))
        try:
            for k, p in enumerate(paths):
                _padded(p, size).save(tmp / f"f{k:05d}.png")
            target = out_dir / f"{series}_{fps}fps.mp4"
            cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
                   "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
                   "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                   "-pix_fmt", "yuv420p", str(target)]
            subprocess.run(cmd, check=True)
            written.append(target)
            if verbose:
                print(f"  {target.name}  ({len(paths)} frames, {fps} fps, "
                      f"{size[0]}x{size[1]}, "
                      f"{target.stat().st_size / 1e6:.1f} MB)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if gif:
        gsize = (max(2, int(size[0] * GIF_SCALE)), max(2, int(size[1] * GIF_SCALE)))
        imgs = [_padded(p, size).resize(gsize, Image.LANCZOS)
                .convert("P", palette=Image.ADAPTIVE, colors=GIF_COLORS)
                for p in paths]
        target = out_dir / f"{series}_{fps}fps.gif"
        imgs[0].save(target, save_all=True, append_images=imgs[1:],
                     duration=int(round(1000 / fps)), loop=0, optimize=True)
        written.append(target)
        if verbose:
            print(f"  {target.name}  ({len(imgs)} frames, {fps} fps, "
                  f"{gsize[0]}x{gsize[1]}, "
                  f"{target.stat().st_size / 1e6:.1f} MB)")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run directory, or the discrete_loops dir itself")
    ap.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    loop_dir = run if (run / "manifest.json").exists() else run / "discrete_loops"
    if not (loop_dir / "manifest.json").exists():
        raise SystemExit(f"no manifest.json under {loop_dir}")
    out = Path(args.out) if args.out else loop_dir / "movies"

    frames = frame_list(loop_dir, args.series[0])
    print(f"{loop_dir.parent.name}: {len(frames)} dose steps, "
          f"{frames[0][0]:.4g} -> {frames[-1][0]:.4g} dpa, {args.fps} fps\n")
    for series in args.series:
        encode(loop_dir, out, series=series, fps=args.fps,
               mp4=not args.no_mp4, gif=not args.no_gif)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
