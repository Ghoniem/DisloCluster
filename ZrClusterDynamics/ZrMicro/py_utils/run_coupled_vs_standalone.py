"""
run_coupled_vs_standalone.py — run the 1 um single-crystal reference through
BOTH routes from a shared 0-D seed, render both, and compare them.

Both routes start from the SAME microstructure: the 0-D ZrMicro state at
``DOSE_SEED`` written uniformly onto the CD nodes of the 1 um cube.

  standalone   MoDELib3 does the whole split itself. ``ClusterDynamicsFEM::
               solve()`` is exactly {solveMobileClusters(); solveImmobile-
               Clusters();} -- the steady mobile solve, then its own nodal
               semi-implicit immobile update (nSub = 20 substeps per 1 dpa
               step), first order, no error control. Run by DDomp; this
               module reads its evl snapshots.

  coupled      the same split, with the SLOW step replaced. MoDELib3 still
               solves the fast step -- DDomp is called with
               ``useImmobileSolver=0``, which runs solveMobileClusters() and
               passes the immobile field through untouched -- and the immobile
               state is then advanced at every node by CVODE BDF with
               acc_mode=2, the exact AD Jacobian, per-thread workspace reuse,
               the base+delta case protocol and deduplication.

THE FAST STEP IS IN THE LOOP
----------------------------
At every substep the coupled route re-solves C_M*(x) for the immobile state it
has built, freezes it, and integrates the immobile ODEs over the substep. That
is the operator split. An earlier version of this module did NOT do this: it
lifted the mobile field out of the standalone run's recorded snapshots. Under
that arrangement the mobile field could never respond to the immobile state the
march was producing, so the march had no mechanism to relax a seed that was not
already at quasi-steady state -- and a 0-D seed never is, because the 0-D
carries no boundary layer while the true C_M*(x) is pinned to thermal
equilibrium on all six faces. ``MOBILE_MODE`` retains that behaviour as
``"replay"`` purely so the effect of closing the loop can be measured.

WHAT THE COMPARISON DOES AND DOES NOT ISOLATE
---------------------------------------------
With the fast step closed, both routes now solve the SAME fast problem with the
SAME code, and each solves it against its own immobile state. Two differences
remain, and neither is the integrator alone:

  1. DIFFERENT IMMOBILE EQUATIONS. The coupled route integrates ZrMicro's
     immobile right-hand side; the standalone integrates MoDELib3's. These are
     different models, not two integrations of one model -- see the known
     non-correspondences in CLAUDE.md (DAD bias, the bi-pyramid vacancy family,
     size-dependent vacancy-loop emission).
  2. DIFFERENT INTEGRATORS, and different splitting frequency. CVODE BDF with
     error control against a first-order semi-implicit nodal update with none;
     and ``FEM_EVERY`` sets how often the coupled route refreshes C_M* against
     the standalone's once per internal substep.

Layout mirrors the reference case:
    <stamp>_<hash>_<tag>/
        0d/        the 0-D reference trajectory and the seed state
        3d/        field panels, standalone/ and coupled/ side by side
        gb/        grain-boundary profiles, both routes
        tables/    comparison tables (markdown + csv + json)
        evl_coupled/   the marched CD fields, restartable by MoDELib
        provenance.md
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from py_utils import paths                                   # noqa: E402
from py_utils import modelib_field as mfield                 # noqa: E402
from py_utils import modelib_coupling as mc                  # noqa: E402
from py_utils import modelib_qssa as mqssa                   # noqa: E402
from py_utils import modelib_report as mreport               # noqa: E402
from py_utils.calibration import build_sim                   # noqa: E402
from py_utils.cpp_bridge import collect_solver_args          # noqa: E402

DOSE_SEED = 0.1
DOSE_MAX = 20.1
DOSE_INTERVAL = 5.0
SUBSTEPS_PER_INTERVAL = 20
DEDUP_RTOL = 1e-8          # exact-duplicate removal only; measured error 0.0

# How many immobile substeps run between two fast solves. 1 refreshes C_M*(x)
# at every substep, which is the split as written; larger values trade
# splitting accuracy for DDomp invocations. The standalone refreshes once per
# 1 dpa output step, so FEM_EVERY = SUBSTEPS_PER_INTERVAL * DOSE_INTERVAL
# matches its cadence.
FEM_EVERY = 1

# "qssa"   re-solve C_M*(x) from MoDELib for the current immobile state (the
#          split; the only physically defensible setting)
# "replay" read the mobile field out of the standalone snapshots (what this
#          module used to do -- kept to measure what closing the loop changed)
# "seed"   hold the 0-D seed's mobile field for the whole march (a control:
#          shows what the march does with no fast step at all)
MOBILE_MODE = "qssa"

# 0-D lumped quantities compared between the routes, in 0-D units.
COMPARE = [
    ("N_a", "<a> loop number density", ("CiL", "CaiL")),
    ("N_c", "<c> loop number density", ("CvL", "CavL")),
    ("c_a", "<a> loop content",        ("CiL_i", "CaiL_i")),
    ("c_c", "<c> loop content",        ("CvL_v", "CavL_v")),
]
IMM_ORDER = ["CiL", "CaiL", "CvL", "CavL", "CiL_i", "CaiL_i", "CvL_v", "CavL_v"]


def dose_snapshots():
    n = int(round((DOSE_MAX - DOSE_SEED) / DOSE_INTERVAL))
    return DOSE_SEED + DOSE_INTERVAL * np.arange(n + 1)


def evl_step_for(dose, dose_seed=None):
    """Output-step index holding `dose`, for a 1 dpa step.

    MoDELib writes after solve(), so evl_N is the state after (N+1) steps taken
    from the seed: dose = dose_seed + (N+1) and N = round(dose - dose_seed) - 1.
    This is exactly modelib_report.dose_steps, reused so the coupled snapshots
    are named on the same convention the standalone writes and the figure
    renderer reads.

    The seed dose is taken from ``DOSE_SEED`` rather than assumed to be 0.1.
    Hardcoding 0.1 was correct only while every case used that seed; a march
    seeded at 1 dpa would name its 6 dpa snapshot ``evl_5``, which the renderer
    and the standalone both read as 7 dpa.
    """
    return mreport.dose_steps([dose], 1.0,
                              DOSE_SEED if dose_seed is None else dose_seed)[0]


def standalone_dose_map(sim_dir, G, dose_per_step=1.0):
    """evl index -> the dose that snapshot holds.

    MoDELib keeps ONE appended F file (F_0.txt), not one per output step: each
    row is `runID time dt ...`, so row r carries the time at the START of
    runID r, i.e. r completed dose steps. Output is written after solve(), so
    evl_N holds the state after (N+1) steps.

    `dose_per_step` is 1 dpa by construction of DD.txt (dtMax was chosen as the
    time for exactly 1 dpa at this dose rate). That is asserted against the F
    time increment rather than trusted, since a changed dtMax would silently
    rescale every dose label in the comparison.
    """
    sim_dir = Path(sim_dir)
    dd = (sim_dir / "inputFiles" / "DD.txt").read_text(encoding="utf-8",
                                                       errors="replace")
    dtMax = float(re.search(r"^dtMax=([-\d.eE+]+)", dd, re.M).group(1))

    fpath = sim_dir / "F" / "F_0.txt"
    if not fpath.is_file():
        return {}
    rows = np.loadtxt(fpath, ndmin=2)
    if rows.size == 0:
        return {}
    if rows.shape[0] > 1:
        step = rows[1, 1] - rows[0, 1]
        if not np.isclose(step, dtMax, rtol=1e-9):
            raise ValueError(
                f"F time increment {step:.6e} does not match dtMax {dtMax:.6e}; "
                "the dose-per-step assumption behind every label is invalid.")

    completed = {int(r) for r in rows[:, 0]}
    out = {}
    for p in (sim_dir / "evl").glob("evl_*.txt"):
        m = re.fullmatch(r"evl_(\d+)\.txt", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n in completed or n + 1 in completed:
            out[n] = DOSE_SEED + (n + 1) * dose_per_step
    return out


def lumped_from_cd(cd, omega):
    """CD block -> the 0-D lumped quantities of COMPARE, per node."""
    imm = mfield.modelib_immobile_to_0d(cd[:, mfield.M_SIZE:], omega)
    d = {n: imm[:, i] for i, n in enumerate(IMM_ORDER)}
    return {key: sum(d[c] for c in cols) for key, _, cols in COMPARE}


def metrics(a, b):
    """Compare coupled (a) against standalone (b), per node."""
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    if not m.any():
        return dict(mean_rel=float("nan"), max_rel=float("nan"),
                    ratio=float("nan"), a_mean=float("nan"), b_mean=float("nan"))
    r = np.abs(a[m] - b[m]) / np.abs(b[m])
    return dict(mean_rel=float(r.mean()), max_rel=float(r.max()),
                ratio=float(a[m].mean() / b[m].mean()),
                a_mean=float(a[m].mean()), b_mean=float(b[m].mean()))


# ── setting up a simulation directory for the fast solves ────────────────────

def prepare_qssa_dir(standalone_sim, dest, seed_evl=None):
    """Clone a case into a directory used ONLY for fast solves.

    Every fast solve overwrites ``evl/evl_0.txt``, so it must not run in the
    standalone case's own directory: that would destroy the reference history
    the comparison is against.
    """
    standalone_sim, dest = Path(standalone_sim), Path(dest)
    (dest / "evl").mkdir(parents=True, exist_ok=True)
    (dest / "F").mkdir(exist_ok=True)
    if (dest / "inputFiles").exists():
        shutil.rmtree(dest / "inputFiles")
    shutil.copytree(standalone_sim / "inputFiles", dest / "inputFiles")
    shutil.copy2(standalone_sim / "evl" / "cdNodes.txt", dest / "evl" / "cdNodes.txt")

    src = Path(seed_evl) if seed_evl else (standalone_sim / "evl" /
                                           f"evl_seed_{DOSE_SEED:.3f}dpa.txt")
    if not src.is_file():
        raise FileNotFoundError(f"no seed configuration at {src}")
    shutil.copy2(src, dest / "evl" / "evl_0.txt")
    return dest, src


# ── the coupled march ────────────────────────────────────────────────────────

def run_coupled(sim, qssa_sim, seed_evl, snaps, evl_out, standalone_sim=None,
                dose_map=None, mobile_mode=None, fem_every=None, verbose=True):
    """The operator split: fast FEM mobile solve, then the frozen-mobile march.

    Returns ``(history, timing, bridge, diagnostics)``.
    """
    mobile_mode = MOBILE_MODE if mobile_mode is None else mobile_mode
    fem_every = FEM_EVERY if fem_every is None else int(fem_every)

    br = mfield.FieldBridge(qssa_sim, paths.MODELIB_MATERIAL)
    N = br.n_nodes
    evl_out = Path(evl_out)
    evl_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(qssa_sim) / "evl" / "cdNodes.txt", evl_out / "cdNodes.txt")

    fast = None
    if mobile_mode == "qssa":
        fast = mqssa.MobileQSSASolver(qssa_sim, paths.MODELIB_MATERIAL,
                                      seed_evl=seed_evl, verbose=verbose)
        if verbose:
            print(fast.describe())

    base_cli = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-6, atol=1e-20, stats=True))

    ev0 = mfield.EvlFile(seed_evl)
    Y = np.zeros((N, mc.N_EQ))
    Y[:, 0:4] = ev0.mobile
    Y[:, 4:12] = mfield.modelib_immobile_to_0d(ev0.immobile, br.omega)
    Y[:, mc.IDX_RHO_N] = float(sim.input_data.material_params["rho"])
    Y_seed = Y.copy()

    diagnostics = {}
    # The measurement the split exists to make: is the seeded mobile field the
    # quasi-steady one? Solve for C_M* against the seed's own immobile state
    # BEFORE any immobile marching, and compare.
    if fast is not None:
        if verbose:
            print("\n  relaxing the seed to quasi-steady state:")
        C0 = fast.solve(Y)
        diagnostics["seed_relaxation"] = mqssa.relaxation_report(Y_seed, C0)
        Y[:, 0:4] = C0
        if verbose:
            for nm, d in diagnostics["seed_relaxation"].items():
                print(f"      {nm:<4} seed {d['seed_mean']:.4e} -> QSSA "
                      f"{d['qssa_mean']:.4e}  (x{d['ratio_mean']:.3g}), "
                      f"spatial contrast {d['spatial_contrast']:.3g}")

    G = float(sim.input_data.material_params["G"])
    history = {float(snaps[0]): Y.copy()}
    timing = []
    # Summary of C_M*(x) at every fast solve. If the split is doing its job the
    # first entries move and the later ones settle: the mobile field is being
    # re-established against an immobile state that is itself still changing.
    mobile_trace = []

    def _trace(dose, C_M):
        mobile_trace.append(dict(
            dose=float(dose),
            **{nm: dict(mean=float(C_M[:, j].mean()),
                        min=float(C_M[:, j].min()),
                        max=float(C_M[:, j].max()))
               for j, nm in enumerate(("Cv", "Ci", "C2i", "C3i"))}))

    if fast is not None:
        _trace(snaps[0], Y[:, 0:4])

    for i in range(len(snaps) - 1):
        d0, d1 = float(snaps[i]), float(snaps[i + 1])
        Y[:, 12:18] = 0.0

        edges = np.linspace(d0 / G, d1 / G, SUBSTEPS_PER_INTERVAL + 1)
        t_start = time.perf_counter()
        n_int, n_fast, fast_s = 0, 0, 0.0
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            # ── FAST STEP: steady mobile field for the CURRENT immobile state
            if k % fem_every == 0:
                if fast is not None:
                    t_f = time.perf_counter()
                    if not (i == 0 and k == 0):      # step 0 solved above
                        Y[:, 0:4] = fast.solve(Y)
                        n_fast += 1
                        _trace(d0 + (d1 - d0) * k / SUBSTEPS_PER_INTERVAL,
                               Y[:, 0:4])
                    fast_s += time.perf_counter() - t_f
                elif mobile_mode == "replay" and dose_map:
                    dose_now = d0 + (d1 - d0) * k / SUBSTEPS_PER_INTERVAL
                    n = min(dose_map, key=lambda kk: abs(dose_map[kk] - dose_now))
                    f = Path(standalone_sim) / "evl" / f"evl_{n}.txt"
                    if abs(dose_map[n] - dose_now) < 0.75 and f.is_file():
                        Y[:, 0:4] = mfield.EvlFile(f).mobile
                        n_fast += 1
                # mobile_mode == "seed": nothing to do, the field is held

            # ── SLOW STEP: immobile ODEs with the mobile species frozen
            st = {}
            out = mc.run_immobile_step(base_cli, Y, a, b,
                                       base_dir=paths.ZRMICRO_DIR,
                                       dedup_rtol=DEDUP_RTOL, stats=st)
            nfail = sum(o is None for o in out)
            if nfail and verbose:
                print(f"      warning: {nfail}/{N} points failed")
            Y = np.array([o if o is not None else Y[q] for q, o in enumerate(out)])
            n_int += st.get("n_integrated", N)
            if verbose:
                print(f"      substep {k + 1:2d}/{SUBSTEPS_PER_INTERVAL} "
                      f"[{d0:.1f}->{d1:.1f} dpa]  "
                      f"{time.perf_counter() - t_start:7.1f} s cumulative",
                      flush=True)

        dt = time.perf_counter() - t_start
        timing.append(dict(dose_from=d0, dose_to=d1, wall_s=dt,
                           substeps=SUBSTEPS_PER_INTERVAL, integrations=n_int,
                           dedup_ratio=(SUBSTEPS_PER_INTERVAL * N) / max(n_int, 1),
                           fast_solves=n_fast, fast_s=fast_s,
                           slow_s=dt - fast_s, mobile_mode=mobile_mode))
        history[d1] = Y.copy()
        if verbose:
            print(f"    {d0:5.1f} -> {d1:5.1f} dpa  {dt:8.1f} s "
                  f"({fast_s:.0f} s fast / {dt - fast_s:.0f} s slow, "
                  f"{n_fast} FEM solves)   dedup x{timing[-1]['dedup_ratio']:.2f}")

        # 0-D -> 3-D: a complete, restartable configuration at the reference
        # step index, so both routes render through the same code path.
        br.write_immobile_field(Y, evl_src=seed_evl,
                                dest=evl_out / f"evl_{evl_step_for(d1)}.txt")

    if fast is not None:
        diagnostics["fast_solver"] = dict(calls=fast.n_calls,
                                          wall_s=fast.wall_s,
                                          mean_s=fast.wall_s / max(fast.n_calls, 1))
        diagnostics["mobile_trace"] = mobile_trace
    return history, timing, br, diagnostics


# ── comparison ───────────────────────────────────────────────────────────────

def compare(standalone_sim, evl_coupled, snaps, dose_map, omega, out_dir,
            timing, ddomp_wall_s, diagnostics=None, mobile_mode="qssa",
            fem_every=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_dose = [], {}

    for d in snaps[1:]:
        n = evl_step_for(d)
        fs = Path(standalone_sim) / "evl" / f"evl_{n}.txt"
        fc = Path(evl_coupled) / f"evl_{n}.txt"
        if not (fs.is_file() and fc.is_file()):
            print(f"  skip {d:.1f} dpa (missing "
                  f"{fs.name if not fs.is_file() else fc.name})")
            continue
        cds = mfield.EvlFile(fs).cd
        cdc = mfield.EvlFile(fc).cd
        Ls, Lc = lumped_from_cd(cds, omega), lumped_from_cd(cdc, omega)
        per_dose[float(d)] = {}
        for key, label, _ in COMPARE:
            m = metrics(Lc[key], Ls[key])
            per_dose[float(d)][key] = m
            rows.append(dict(dose=float(d), quantity=key, label=label, **m))

    # ── markdown ────────────────────────────────────────────────────────────
    md = ["# Coupled march versus standalone MoDELib3", "",
          f"1 um single crystal, seeded from the 0-D at {DOSE_SEED} dpa, "
          f"advanced to {DOSE_MAX} dpa in {DOSE_INTERVAL} dpa intervals.", "",
          "Both routes run the SAME operator split and the SAME fast step "
          "(MoDELib3 `solveMobileClusters`, a steady diffusion-reaction solve). "
          "They differ in the slow step: MoDELib3's nodal semi-implicit update "
          "against ZrMicro's CVODE BDF march. The immobile right-hand sides are "
          "themselves different models, so this is a model-plus-method "
          "comparison, not an integrator comparison.", "",
          f"Coupled route: mobile mode `{mobile_mode}`, fast solve every "
          f"{fem_every} immobile substep(s).", ""]

    if diagnostics and "seed_relaxation" in diagnostics:
        md += ["## Does the split repair a seed that is not quasi-steady?", "",
               f"The 0-D seed at {DOSE_SEED} dpa is spatially uniform and carries no "
               "boundary layer. Solving the fast step once against the seed's own "
               "immobile state, before any immobile marching, gives C_M*(x):", "",
               "| species | seeded (uniform) | QSSA mean | ratio | QSSA min | QSSA max | max/min |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for nm, d in diagnostics["seed_relaxation"].items():
            md.append(f"| {nm} | {d['seed_mean']:.4e} | {d['qssa_mean']:.4e} | "
                      f"{d['ratio_mean']:.4g} | {d['qssa_min']:.4e} | "
                      f"{d['qssa_max']:.4e} | {d['spatial_contrast']:.4g} |")
        md += ["", "A ratio far from 1, or a max/min far from 1, means the seeded "
                   "field was not the quasi-steady one and the fast step supplied "
                   "the difference. Under the old replay arrangement no such "
                   "correction could occur.", ""]

    trace = (diagnostics or {}).get("mobile_trace") or []
    if len(trace) > 1:
        md += ["### C_M*(x) over the march", "",
               "Each row is one fast solve: the mobile field re-established "
               "against the immobile state the march had reached.", "",
               "| dose [dpa] | Cv mean | Cv min | Cv max | Ci mean | Ci min | Ci max |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for t in trace:
            md.append(f"| {t['dose']:.2f} | {t['Cv']['mean']:.4e} | "
                      f"{t['Cv']['min']:.4e} | {t['Cv']['max']:.4e} | "
                      f"{t['Ci']['mean']:.4e} | {t['Ci']['min']:.4e} | "
                      f"{t['Ci']['max']:.4e} |")
        md.append("")

    md += ["## Accuracy — coupled (CVODE) against standalone (semi-implicit)", ""]
    md += ["| dose [dpa] | quantity | standalone mean | coupled mean | ratio | mean rel. diff | max rel. diff |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['dose']:.1f} | {r['label']} | {r['b_mean']:.4e} | "
                  f"{r['a_mean']:.4e} | {r['ratio']:.3f} | {r['mean_rel']:.3e} | "
                  f"{r['max_rel']:.3e} |")

    tot_coupled = sum(t["wall_s"] for t in timing)
    tot_fast = sum(t["fast_s"] for t in timing)
    n_int = sum(t["integrations"] for t in timing)
    n_pts = sum(t["substeps"] for t in timing)
    n_fast = sum(t["fast_solves"] for t in timing)
    span = DOSE_MAX - DOSE_SEED
    md += ["", "## Speed", "",
           "| route | fast step | slow step | wall clock [s] | per dpa [s] |",
           "|---|---|---|---:|---:|"]
    md.append(f"| standalone | MoDELib3 FEM, in-process | MoDELib3 semi-implicit, "
              f"20 substeps/dpa | {ddomp_wall_s:.0f} | {ddomp_wall_s / span:.1f} |")
    md.append(f"| coupled | MoDELib3 FEM, {n_fast} DDomp calls, {tot_fast:.0f} s | "
              f"CVODE BDF, acc_mode=2, AD Jacobian, {tot_coupled - tot_fast:.0f} s | "
              f"{tot_coupled:.0f} | {tot_coupled / span:.1f} |")
    md += ["", f"Coupled march: {n_pts} substeps, {n_int} integrations actually "
               f"dispatched after deduplication, {n_fast} fast solves.", "",
           "The coupled route pays a full DDomp start-up per fast solve — mesh "
           "load, FE assembly and factorization are redone every call — where "
           "the standalone amortizes them over the whole run. That overhead is "
           "an artifact of driving MoDELib as a subprocess, not of the split.",
           ""]
    md += ["| dose interval [dpa] | wall [s] | fast [s] | slow [s] | FEM solves | integrations | dedup |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for t in timing:
        md.append(f"| {t['dose_from']:.1f} -> {t['dose_to']:.1f} | {t['wall_s']:.1f} | "
                  f"{t['fast_s']:.1f} | {t['slow_s']:.1f} | {t['fast_solves']} | "
                  f"{t['integrations']} | x{t['dedup_ratio']:.2f} |")

    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import csv
    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "comparison.json").write_text(
        json.dumps(dict(rows=rows, timing=timing, diagnostics=diagnostics or {},
                        mobile_mode=mobile_mode, fem_every=fem_every,
                        ddomp_wall_s=ddomp_wall_s,
                        coupled_wall_s=tot_coupled,
                        coupled_fast_s=tot_fast), indent=2, default=float),
        encoding="utf-8")
    return rows, per_dose, tot_coupled


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-sim",
                    default=str(paths.MODELIB_ROOT / "tutorials" / "zrmicro_seeded"))
    ap.add_argument("--qssa-sim", default=None,
                    help="directory for the fast solves (default: "
                         "<standalone>_qssa; NEVER the standalone dir)")
    ap.add_argument("--seed-evl", default=None)
    ap.add_argument("--tag", default="coupled_vs_standalone")
    ap.add_argument("--mobile-mode", default=MOBILE_MODE,
                    choices=("qssa", "replay", "seed"))
    ap.add_argument("--fem-every", type=int, default=FEM_EVERY)
    ap.add_argument("--dose-interval", type=float, default=None)
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--ddomp-wall", type=float, default=float("nan"),
                    help="measured DDomp wall time [s], for the speed table")
    ap.add_argument("--figures", action="store_true", help="render 3d/ and gb/")
    args = ap.parse_args(argv)

    global DOSE_INTERVAL, SUBSTEPS_PER_INTERVAL
    if args.dose_interval is not None:
        DOSE_INTERVAL = args.dose_interval
    if args.substeps is not None:
        SUBSTEPS_PER_INTERVAL = args.substeps

    standalone_sim = Path(args.standalone_sim)
    qssa_sim = Path(args.qssa_sim) if args.qssa_sim else \
        standalone_sim.with_name(standalone_sim.name + "_qssa")
    if qssa_sim.resolve() == standalone_sim.resolve():
        raise SystemExit("--qssa-sim must differ from --standalone-sim: every "
                         "fast solve overwrites evl_0.txt in it.")
    snaps = dose_snapshots()

    # The workbook alone is NOT the calibrated model -- 28 parameters have
    # drifted and 11 are missing from it entirely. Building straight from it,
    # as this driver used to, integrates a different model: N_a comes out
    # 2.3e-2 against the calibrated 8.1e-8, and the loop content exceeds unit
    # atom fraction. See py_utils/calibration.py.
    sim = build_sim()
    G = float(sim.input_data.material_params["G"])
    T = float(sim.input_data.material_params["T"])
    print(f"calibrated 0-D: {len(sim.overrides_applied)} fitted parameters "
          f"applied on top of {Path(sim.input_file).name}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = paths.OUTPUT_DIR / f"{stamp}_{paths.git_hash()}_{args.tag}"
    for sub in ("0d", "3d", "gb", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"output : {out}")
    print(f"T = {T} K, G = {G} dpa/s")
    print(f"doses  : {', '.join(f'{d:.1f}' for d in snaps)} dpa")
    print(f"mobile : {args.mobile_mode}, fast solve every {args.fem_every} substep(s)")

    dose_map = standalone_dose_map(standalone_sim, G)
    if dose_map:
        print(f"standalone: {len(dose_map)} snapshots, "
              f"{min(dose_map.values()):.2f} .. {max(dose_map.values()):.2f} dpa")

    qssa_sim, seed_evl = prepare_qssa_dir(standalone_sim, qssa_sim, args.seed_evl)
    print(f"fast-solve dir: {qssa_sim}")

    print("\ncoupled march:")
    history, timing, br, diag = run_coupled(
        sim, qssa_sim, seed_evl, snaps, out / "evl_coupled",
        standalone_sim=standalone_sim, dose_map=dose_map,
        mobile_mode=args.mobile_mode, fem_every=args.fem_every)

    np.savez_compressed(out / "march_state.npz",
                        doses=np.array(sorted(history)),
                        Y=np.array([history[d] for d in sorted(history)]),
                        nodes=br.nodes)

    print("\ncomparison:")
    rows, per_dose, tot = compare(standalone_sim, out / "evl_coupled", snaps,
                                  dose_map, br.omega, out / "tables",
                                  timing, args.ddomp_wall, diagnostics=diag,
                                  mobile_mode=args.mobile_mode,
                                  fem_every=args.fem_every)
    for r in rows:
        print(f"  {r['dose']:5.1f} dpa  {r['quantity']:<5} ratio {r['ratio']:7.3f}  "
              f"mean rel {r['mean_rel']:.3e}  max rel {r['max_rel']:.3e}")

    if args.figures:
        doses_fig = [float(d) for d in snaps[1:]]
        for route, evl in (("standalone", standalone_sim / "evl"),
                           ("coupled", out / "evl_coupled")):
            print(f"\nrendering {route} ...")
            try:
                mreport.write_3d_figures(evl, doses_fig, out / "3d" / route,
                                         dose_per_step=1.0, verbose=True)
                mreport.write_gb_figures(evl, doses_fig, out / "gb" / route,
                                         dose_per_step=1.0, verbose=True)
            except Exception as e:                      # noqa: BLE001
                print(f"  {route} rendering failed: {e}")

    n_fast = sum(t["fast_solves"] for t in timing)
    tot_fast = sum(t["fast_s"] for t in timing)
    (out / "provenance.md").write_text("\n".join([
        "# Coupled march versus standalone MoDELib3", "",
        f"- generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- git hash  : {paths.git_hash()}",
        f"- input     : {sim.input_file}",
        f"- T, G      : {T} K, {G} dpa/s",
        f"- doses     : {', '.join(f'{d:.1f}' for d in snaps)} dpa",
        f"- seed      : 0-D state at {DOSE_SEED} dpa, uniform on {br.n_nodes} CD nodes",
        f"- standalone: {standalone_sim}",
        f"- fast solve: {args.mobile_mode}, every {args.fem_every} substep(s), "
        f"{n_fast} DDomp calls, {tot_fast:.1f} s",
        f"- coupled   : CVODE BDF, acc_mode=2, analytic AD Jacobian, "
        f"dedup_rtol={DEDUP_RTOL:g}, {SUBSTEPS_PER_INTERVAL} substeps/interval",
        f"- coupled wall clock: {tot:.1f} s",
        f"- standalone wall clock: {args.ddomp_wall:.1f} s",
    ]) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    main()
