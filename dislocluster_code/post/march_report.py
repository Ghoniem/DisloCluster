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
import json
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
    FAMILY_SLUG_A1, LOOP_SCALE, FAMILY_BG, MOMENT2_A1, _profile_figure,
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
def _discrete_populations(run_dir, dose, region="interior", verbose=True):
    """`{slug: (centres_b, radii_b)}` -- the loops `discrete_loops` exports.

    Built by calling `discrete_loops.build` with its OWN defaults, not by
    reading the CSV it writes: the draw is seeded (`seed=0`) and the whole
    family sequence is replayed in one `populate` call, so this reproduces the
    export loop for loop, and it works before the export has run. Reading the
    CSV would have tied `3d/` to `discrete_loops/` having been written first,
    which in `driver.report` it has not.
    """
    from dislocluster_code.post import discrete_loops as DL
    # `coplanar_tol="plane"` is what `discrete_loops.main` passes and NOT what
    # `build` defaults to. Without it coalescence merges loops that are merely
    # close rather than on a shared habit plane, and the two figures disagree:
    # 78 <c> loops here against the export's 1031 at 1 dpa. Every argument that
    # changes the draw has to be the export's, not the function's.
    _, pops, _, _, _ = DL.build(run_dir, dose, region=region, seed=0,
                                coalesce_pass=True, coplanar_tol="plane",
                                verbose=False)
    out = {}
    for pop in pops:
        if len(pop):
            out[pop.fam["key"]] = (np.asarray(pop.centers, dtype=float),
                                   np.asarray(pop.radii, dtype=float))
    if verbose:
        got = ", ".join(f"{k} {len(v[1])}" for k, v in sorted(out.items()))
        print(f"    discrete population at {dose:.4g} dpa: {got or 'none'}")
    return out


def write_3d(doses, frames, idx, out_dir, planes=DEFAULT_PLANES,
             overlays=True, verbose=True, run_dir=None,
             loop_source="packed", loop_region="interior", suffix="",
             only_overlays=False):
    """Static 3-D panels for the selected snapshots.

    `loop_source` decides what the `loops_*` overlays draw:

    - ``"packed"`` (the default, and every figure this module has written) is a
      DECORATION. One platelet per CD node, greedily packed until the domain is
      full, radii exaggerated by `LOOP_SCALE`. It shows where the population is
      dense and how its size varies across the domain; its count is the
      figure's packing capacity and not a loop density.
    - ``"discrete"`` draws the population `post.discrete_loops` exports -- the
      real count, at the sampled positions, with the polydisperse sampled radii
      and no exaggeration. The same loops then appear in `3d/loops_<fam>_*.png`
      and in `discrete_loops/loops_<fam>_*.png`.

    THE TWO ANSWER DIFFERENT QUESTIONS and neither is a better version of the
    other. In particular `discrete_loops`' own default is `region="interior"`,
    which places loops UNIFORMLY from interior statistics -- so a discrete
    overlay deliberately carries no boundary structure, which is exactly what
    the packed overlay is drawn to show. Pass `loop_region="domain"` for a
    field-following discrete population instead.

    `suffix` is appended to every filename and `only_overlays` skips the field
    panels, so a second set can be written beside the first rather than over
    it -- the comparison is the point, and overwriting one with the other
    destroys it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if loop_source not in ("packed", "discrete"):
        raise ValueError(f"loop_source must be 'packed' or 'discrete', "
                         f"not {loop_source!r}")
    if loop_source == "discrete" and run_dir is None:
        raise ValueError("loop_source='discrete' needs run_dir")
    written = []
    for i in idx:
        d, tag = float(doses[i]), dose_tag(float(doses[i]))
        groups = ([] if only_overlays else
                  [g for g in (MOBILE, DENSITY_A1, CONTENT_A1,
                               MOMENT2_A1) if g])
        for group in groups:
            for sp in group:
                out = out_dir / f"{FILE_SLUG[sp]}_{tag}{suffix}.png"
                plot_field_panels(None, [i], [d], species=(sp,), plane=planes,
                                  fields={i: frames[i]}, out_file=out,
                                  column_titles=False,
                                  title=f"{SPECIES[sp][1]}   {d:.4g} dpa")
                written.append(out)
        if overlays:
            pops = (_discrete_populations(run_dir, d, loop_region, verbose)
                    if loop_source == "discrete" else {})
            for k, slug in FAMILY_SLUG_A1.items():
                out = out_dir / f"loops_{slug}_{tag}{suffix}.png"
                if loop_source == "discrete":
                    pop = pops.get(slug)
                    kw = dict(loop_scale=1.0,
                              loop_population=({i: pop} if pop else {}))
                    note = (f"{len(pop[1])} loops, true radii"
                            if pop else "no loops at this dose")
                else:
                    kw = dict(loop_scale=LOOP_SCALE[k])
                    note = f"platelet radii x{LOOP_SCALE[k]:g}, not to scale"
                plot_field_panels(
                    None, [i], [d], species=(FAMILY_BG[k],), loop_family=k,
                    plane=planes, fields={i: frames[i]},
                    out_file=out, column_titles=False,
                    title=(f"{FAMILIES[k][0]} loop population at {d:.4g} dpa "
                           f"({note})"), **kw)
                written.append(out)
        if verbose:
            n_panels = (0 if only_overlays else
                        len(MOBILE) + len(DENSITY_A1) + len(CONTENT_A1)
                        + len(MOMENT2_A1))
            n_panels += len(FAMILY_SLUG_A1) if overlays else 0
            print(f"  3d/ {tag}: {n_panels} panels")
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


def _hms(s):
    s = float(s)
    h, rem = divmod(s, 3600.0)
    m, sec = divmod(rem, 60.0)
    if h >= 1:
        return f"{int(h)} h {int(m):02d} m {int(sec):02d} s"
    if m >= 1:
        return f"{int(m)} m {int(sec):02d} s"
    return f"{sec:.1f} s"


def computational_stats(run, nodes):
    """The cost of the run, as markdown lines, from what the march recorded.

    Everything here comes out of ``summary.json``; nothing is re-measured and
    nothing is re-solved, so this is as true of a re-render as of the original
    run. Returns an empty list when there is no summary to read -- a run made
    before the file existed still renders, just without this section.

    The two solves are counted in DIFFERENT units on purpose, because they are
    different kinds of problem:

      * the FAST solve is not an ODE system at all. MoDELib's
        ``solveMobileClusters`` has no time derivative -- it is the steady
        mobile field for the immobile state currently held, so what it solves
        is a nonlinear ALGEBRAIC system of ``M_SIZE * n_cd`` unknowns, by
        Newton iteration;
      * the SLOW solve is a system of coupled ODEs per quadrature point,
        integrated independently, so the count that matters is
        (unique points) x (equations) per substep.

    Reporting both as "number of equations" without that distinction is how a
    reader ends up thinking the fast step integrates something.

    The state vector is 19 long but a substep does not integrate 19 of
    anything. The march runs ``freeze_mobile=1`` and ``acc_mode=2``
    (`coupling.immobile.build_cases`), which is ``ACC_STATE_RELAX`` in
    `ZrMicro/cpp_utils/parameters.h`, so per point:

      * ``N_EQ`` = 19 values are CARRIED;
      * ``N_EQ - M_SIZE`` = 15 are INTEGRATED, and 15 is also the width of the
        implicit block -- ``N_RLX_FROZEN = N_RED_FROZEN + N_ACC``, what
        ``red_dim()`` returns for this mode. The four mobile species are held
        fixed by construction of the operator split; the six accumulators stay
        in the state and are integrated, they are merely given
        ``atol = 1e300`` so their error weights underflow and they stop
        driving the step size.

    It is tempting to subtract the accumulators as well and call it 9. That is
    ``N_RED_FROZEN``, the block under ``acc_mode=1`` (``ACC_QUADRATURE``),
    where they really do leave the ODE system and become CVODES quadrature
    variables. The march does not use that mode: the quadrature right-hand side
    needs its own core evaluation at each accepted step and its memo almost
    never hits (1 in 2565), costing about one extra core sweep per step, which
    measured slower overall despite the smaller Newton block.
    """
    from dislocluster_code.coupling.field import M_SIZE, I_SIZE
    from dislocluster_code.coupling.immobile import (
        N_EQ as N_EQ_NATIVE, ACCUMULATOR_SLICE)

    n_acc = ACCUMULATOR_SLICE.stop - ACCUMULATOR_SLICE.start
    # The state width is the RUN'S, not the module constant. A march carrying
    # nine families and three moments integrates 34 ODEs per point, not 15, and
    # quoting 19 here would understate the whole cost table by a factor of two.
    N_EQ = N_EQ_NATIVE
    try:
        import numpy as _np
        with _np.load(Path(run) / "march_state.npz") as _z:
            if "Y" in _z.files:
                N_EQ = int(_z["Y"].shape[-1])
    except Exception:
        pass
    # acc_mode=2 with freeze_mobile=1: N_RLX_FROZEN = (N_EQ - N_ACC - N_MOB)
    # + N_ACC, which is just N_EQ - N_MOB. Integrated count and Newton block
    # are the same number here.
    n_ode = N_EQ - M_SIZE
    n_quad_block = n_ode - n_acc          # acc_mode=1's block; NOT what runs

    try:
        s = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    n_cd = int(nodes.shape[0])
    timing = s.get("timing") or []
    diag = s.get("diagnostics") or {}
    fast = diag.get("fast_solver") or {}
    ckpt = diag.get("checkpoint") or {}
    cpl = (s.get("config") or {}).get("coupling") or {}
    march_s = float(s.get("wall_s") or 0.0)

    n_sub = sum(int(t.get("substeps", 0)) for t in timing)
    n_int = sum(int(t.get("integrations", 0)) for t in timing)
    slow_s = sum(float(t.get("slow_s", 0.0)) for t in timing)
    fast_s = sum(float(t.get("fast_s", 0.0)) for t in timing)
    # The seed relaxation is a fast solve that happens BEFORE the first
    # interval, so it is in `fast_solver` but in no interval's `fast_s`.
    n_fast = int(fast.get("calls", 0))
    fast_total_s = float(fast.get("wall_s", fast_s))
    dedup = (n_sub * n_cd / n_int) if n_int else 1.0

    L = ["## Computational statistics", "",
         "### Problem size", "",
         "| quantity | value |", "|---|---:|",
         f"| CD nodes (2nd-order trial functions) | {n_cd:,} |",
         f"| mobile field `m` — {M_SIZE} species | {M_SIZE * n_cd:,} dof |",
         f"| immobile field `i` — {I_SIZE} (number, content, 2nd moment) | "
         f"{I_SIZE * n_cd:,} dof |",
         f"| elastic displacement `u` | {3 * n_cd:,} dof |", ""]

    L += ["### What each solve actually solves", "",
          f"**Fast step** — steady mobile field for the immobile state held. "
          f"No time derivative: a nonlinear **algebraic** system of "
          f"**{M_SIZE * n_cd:,} unknowns** ({M_SIZE} species x {n_cd:,} nodes), "
          f"by Newton iteration, one DDomp call each.", "",
          f"**Slow step** — the immobile ODEs at every quadrature point with "
          f"the mobile species frozen: **{n_ode} coupled ODEs per point**, "
          f"which is also the width of the implicit block Newton and the dense "
          f"LU factor. The state vector carries {N_EQ} ({M_SIZE} mobile, "
          f"{N_EQ - M_SIZE - n_acc - 1} immobile, {n_acc} conservation "
          f"accumulators, rho_N); the "
          f"{M_SIZE} mobile are held fixed over the substep. The {n_acc} "
          f"accumulators ARE integrated — `acc_mode=2` only relaxes their "
          f"`atol` so they stop driving the step size. Dropping them from the "
          f"system too would give {n_quad_block}, but that is `acc_mode=1`, "
          f"which costs an extra core sweep per step and is not used.", ""]

    if n_int:
        per_sub = n_int / max(n_sub, 1)
        L += ["| ODE count | per point | per substep | over the march |",
              "|---|---:|---:|---:|",
              f"| state carried | {N_EQ} | {per_sub * N_EQ:,.0f} | "
              f"{n_int * N_EQ:,} |",
              f"| **integrated** (= Newton block) | **{n_ode}** | "
              f"**{per_sub * n_ode:,.0f}** | **{n_int * n_ode:,}** |", "",
              "| | value |", "|---|---:|",
              f"| points integrated per substep (after dedup) | "
              f"{per_sub:,.0f} of {n_cd:,} |",
              f"| substeps | {n_sub} |",
              f"| **point-integrations over the march** | {n_int:,} |",
              f"| dedup factor (identical states solved once) | x{dedup:.2f} |",
              ""]

    L += ["### Wall clock", "", "| stage | time | share |", "|---|---:|---:|"]
    if march_s > 0:
        L += [f"| fast solves ({n_fast} x DDomp) | {_hms(fast_total_s)} | "
              f"{100 * fast_total_s / march_s:.0f}% |",
              f"| slow substeps ({n_sub}) | {_hms(slow_s)} | "
              f"{100 * slow_s / march_s:.0f}% |"]
        if ckpt.get("write_s"):
            L += [f"| checkpoint writes ({ckpt.get('writes', 0)}) | "
                  f"{_hms(ckpt['write_s'])} | "
                  f"{100 * float(ckpt['write_s']) / march_s:.1f}% |"]
        L += [f"| **march total** | **{_hms(march_s)}** | 100% |", ""]

    L += ["| per-call cost | value |", "|---|---:|"]
    if n_fast:
        L += [f"| one fast solve | {_hms(fast_total_s / n_fast)} |"]
    if n_sub:
        L += [f"| one slow substep | {_hms(slow_s / n_sub)} |"]
    if n_int and slow_s:
        L += [f"| one point-integration | {1e6 * slow_s / n_int:,.0f} us |",
              f"| ODE throughput (integrated) | "
              f"{n_int * n_ode / slow_s:,.0f} scalar ODE/s |"]
    if cpl:
        L += [f"| substeps per interval | {cpl.get('substeps_per_interval')} |",
              f"| fast solve every | {cpl.get('fem_every')} substeps |"]
    L += [""]

    if timing:
        L += ["### Per interval", "",
              "| dose from | dose to | wall | fast | slow | FEM | dedup |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for t in timing:
            L += [f"| {t['dose_from']:.4g} | {t['dose_to']:.4g} | "
                  f"{_hms(t['wall_s'])} | {_hms(t['fast_s'])} | "
                  f"{_hms(t['slow_s'])} | {t.get('fast_solves', 0)} | "
                  f"x{t.get('dedup_ratio', 1.0):.2f} |"]
        L += [""]

    if ckpt.get("resumed"):
        L += ["This run **resumed** from a checkpoint, so the wall clock above "
              "is the total across sessions.", ""]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--n-doses", type=int, default=6,
                    help="snapshots to draw as static 3-D panels and GB curves")
    ap.add_argument("--no-movies", action="store_true")
    ap.add_argument("--no-volume-average", action="store_true")
    ap.add_argument("--no-boundary-flux", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--interp", type=int, default=movies_mod.DEFAULT_INTERP,
                    help="movie frames per output interval (1 = solved "
                         "snapshots only); interpolated frames are labelled")
    ap.add_argument("--mc-samples", type=int, default=4_000_000)
    ap.add_argument("--loop-source", choices=("packed", "discrete"),
                    default="packed",
                    help="what the 3d/ loops_* overlays draw. 'packed' is the "
                         "decorative platelet fill (exaggerated radii, count "
                         "set by packing capacity); 'discrete' draws the "
                         "population discrete_loops exports -- same loops, "
                         "same positions, true radii")
    ap.add_argument("--loop-region", choices=("interior", "domain"),
                    default="interior",
                    help="with --loop-source discrete: which state the "
                         "population is drawn from. 'interior' matches the "
                         "export and places loops uniformly; 'domain' follows "
                         "the field and keeps the boundary structure")
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

    write_3d(doses, frames, idx, run / "3d", run_dir=run,
             loop_source=args.loop_source, loop_region=args.loop_region)
    write_gb(doses, frames, idx, run / "gb")

    if not args.no_movies:
        print("\nmovies:")
        movies_mod.render(run, run / "movies", fps=args.fps,
                          interp=args.interp)

    if not args.no_volume_average:
        print("\nvolume average:")
        va.main([str(run), "--out", str(run / "volume_average"),
                 "--samples", str(args.mc_samples)])

    if not args.no_boundary_flux:
        # The INDEPENDENT check on the grain-boundary channel: the accumulator
        # version of it is closure by difference and cannot disagree with
        # itself. Guarded, because a diagnostic must never cost a report that
        # took a march of several hours to earn -- and it can be re-run alone
        # with `python -m dislocluster_code.post.boundary_flux <run>`.
        print("\nboundary flux (independent of the accumulators):")
        try:
            from dislocluster_code.post import boundary_flux
            boundary_flux.main([str(run), "--samples",
                                str(min(args.mc_samples, 1_000_000))])
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}")

    (run / "report.md").write_text("\n".join([
        f"# {run.name}", "",
        f"- integration: **{ADAPTIVE.key}** (Option {ADAPTIVE.option}) — "
        f"{ADAPTIVE.integrator}",
        f"- snapshots  : {len(doses)}, {doses[0]:.4g} .. {doses[-1]:.4g} dpa",
        f"- CD nodes   : {nodes.shape[0]}",
        f"- generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- git hash   : {paths.git_hash()}", "",
    ] + computational_stats(run, nodes) + [
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
