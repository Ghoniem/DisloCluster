"""
compare_architectures.py — reference vs optimized fast solve, on the same case.

WHAT IS BEING COMPARED
----------------------
Two marches of the SAME configuration, differing only in the DDomp binary that
takes the fast step:

    reference   MoDELib3/build_dc    as it stood before this work
    optimized   MoDELib3/build_fast  dead assembly removed, Dirichlet selection
                                     cached, A1 built by filtering instead of
                                     T^T*A*T, element assembly parallelized,
                                     loop-invariant weak forms hoisted out of
                                     the Newton iteration

Every one of those removes repeated work. None approximates anything, and each
was verified bit-identical on the 200 nm case before being used here. So this
comparison has two jobs, and the second matters more than the first:

  1. how much faster the optimized fast solve is, in a real march rather than
     on one isolated DDomp call;
  2. whether the two marches agree on the PHYSICS. They should agree to the
     last bit. Anything else means an optimization changed the answer, and the
     speed number is then worthless.

Wall-clock comparisons are only meaningful if the two runs did not overlap --
the fast step is single-threaded but the slow step uses every core, so
concurrent runs would contend and the ratio would measure the scheduler.

USAGE
-----
    python -m dislocluster_code.studies.compare_architectures \\
        <reference_run> <optimized_run> [--out <report.md>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(run):
    run = Path(run)
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    z = np.load(run / "march_state.npz")
    return run, s, z


def _hms(x):
    x = float(x)
    h, r = divmod(x, 3600.0)
    m, sec = divmod(r, 60.0)
    if h >= 1:
        return f"{int(h)} h {int(m):02d} m {int(sec):02d} s"
    if m >= 1:
        return f"{int(m)} m {int(sec):02d} s"
    return f"{sec:.1f} s"


def compare(ref_dir, opt_dir):
    (rp, rs, rz), (op, os_, oz) = _load(ref_dir), _load(opt_dir)
    L = ["# Fast-solve architecture comparison", "",
         f"- reference : `{rp.name}`",
         f"- optimized : `{op.name}`",
         f"- CD nodes  : {rz['nodes'].shape[0]:,}",
         "",
         "## What changed", "",
         "Both runs solve the same case with the same Python driver. They",
         "differ only in the DDomp binary that takes the fast step. Every",
         "change removes repeated work; none approximates anything, which is",
         "why the physics section below is the one that decides whether the",
         "timings mean anything.", "",
         "| # | change | effect |",
         "|---|---|---|",
         "| 1 | `AcIR`, a sparse matrix assembled every Newton iteration and "
         "never read, removed | 1.64x |",
         "| 2 | Dirichlet selection matrix `T` and its index map cached; "
         "`A1 = T'AT` replaced by a filtered copy, `T` being a selection "
         "matrix | ~1.00x alone |",
         "| 3 | element assembly `BilinearWeakForm::globalTriplets` "
         "parallelized over explicit contiguous chunks | 1.33x |",
         "| 4 | the three loop-invariant weak forms and `rSolver` hoisted out "
         "of the Newton iteration | 1.11x |",
         "",
         "Cumulative on one isolated DDomp call at 200 nm: **96.6 s -> 37.8 s, "
         "2.556x**. Change 2 measured as nothing on its own because `rSolver` "
         "was rebuilt each iteration and discarded the cache before it could "
         "pay; change 4 is what made it count.", "",
         "Assembly order is preserved exactly at every step. `setFromTriplets` "
         "sums duplicate entries, so a different triplet order would "
         "re-associate those sums and the result would agree only to rounding "
         "-- which would have made bit-identity unavailable as a check.", ""]

    # ── physics ─────────────────────────────────────────────────────────────
    L += ["## Physics", ""]
    same_grid = np.array_equal(rz["doses"], oz["doses"])
    L.append(f"- dose grid identical: **{same_grid}**")
    if same_grid and rz["Y"].shape == oz["Y"].shape:
        Y1, Y2 = rz["Y"], oz["Y"]
        identical = np.array_equal(Y1, Y2)
        L.append(f"- state array `{Y1.shape}` bit-identical: **{identical}**")
        if not identical:
            d = np.abs(Y1 - Y2)
            ref = np.maximum(np.maximum(np.abs(Y1), np.abs(Y2)), 1e-300)
            rel = d / ref
            L += [f"- max absolute difference: {d.max():.3e}",
                  f"- max relative difference: {rel.max():.3e}", "",
                  "| dose | max rel. diff |", "|---:|---:|"]
            for k, dose in enumerate(rz["doses"]):
                L.append(f"| {dose:.4g} | {rel[k].max():.3e} |")
            L += ["", "**A non-zero difference means an optimization changed "
                  "the answer.** Every change made here removes repeated work "
                  "and must reproduce the reference exactly; investigate "
                  "before trusting the timings below."]
    else:
        L.append("- shapes differ; the two runs are not the same case")
    L.append("")

    # ── cost ────────────────────────────────────────────────────────────────
    L += ["## Wall clock", "", "| quantity | reference | optimized | speedup |",
          "|---|---:|---:|---:|"]

    def row(label, a, b, fmt=_hms):
        sp = (a / b) if b else float("nan")
        L.append(f"| {label} | {fmt(a)} | {fmt(b)} | {sp:.3f}x |")

    rf = (rs.get("diagnostics", {}).get("fast_solver") or {})
    of = (os_.get("diagnostics", {}).get("fast_solver") or {})
    r_slow = sum(float(t.get("slow_s", 0)) for t in rs.get("timing", []))
    o_slow = sum(float(t.get("slow_s", 0)) for t in os_.get("timing", []))
    row("march total", rs.get("wall_s", 0), os_.get("wall_s", 0))
    if rf.get("wall_s") and of.get("wall_s"):
        row(f"fast solves ({rf.get('calls')} x DDomp)",
            rf["wall_s"], of["wall_s"])
        row("one fast solve", rf["wall_s"] / max(rf.get("calls", 1), 1),
            of["wall_s"] / max(of.get("calls", 1), 1))
    row("slow substeps", r_slow, o_slow)
    L.append("")

    # The slow side runs the same code in both, so it is the control: a ratio
    # far from 1.0 there means the two runs were not measured under comparable
    # load, and the fast-side speedup is not trustworthy either.
    ctrl = (r_slow / o_slow) if o_slow else float("nan")
    L += [f"The slow step is untouched by this work and acts as the control: "
          f"its ratio is **{ctrl:.3f}x**. A value far from 1.0 means the two "
          f"runs were not measured under comparable machine load, and the "
          f"fast-side number should not be quoted.", ""]

    L += ["## Per interval", "",
          "| dose from | dose to | ref fast | opt fast | speedup | ref slow | opt slow |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in zip(rs.get("timing", []), os_.get("timing", [])):
        fa, fb = float(a.get("fast_s", 0)), float(b.get("fast_s", 0))
        # An interval with no fast solve carries only timer noise; a ratio of
        # two sub-millisecond numbers is not a speedup and must not be printed
        # as one.
        sp = f"{fa / fb:.3f}x" if (fb > 1.0 and fa > 1.0) else "--"
        L.append(f"| {a['dose_from']:.4g} | {a['dose_to']:.4g} | "
                 f"{_hms(fa)} | {_hms(fb)} | {sp} | "
                 f"{_hms(a.get('slow_s', 0))} | {_hms(b.get('slow_s', 0))} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference_run")
    ap.add_argument("optimized_run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    text = compare(args.reference_run, args.optimized_run)
    print(text)
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
