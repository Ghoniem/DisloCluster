"""
compare_seeding.py — judge the coupled march against the standalone MoDELib3
run and against the 0-D, for one or more seed doses, and quantify what the
choice of seed dose does to the answer.

THE TWO QUESTIONS
-----------------
1. COUPLED vs STANDALONE at a fixed seed. Both routes start from the same
   microstructure and run the same operator split with the same fast step
   (MoDELib3 ``solveMobileClusters``). They differ in the slow step: MoDELib3's
   first-order nodal semi-implicit update against ZrMicro's CVODE BDF march,
   and the immobile right-hand sides are themselves different models. So the
   spread is model-plus-method, never the integrator alone.

2. WHAT THE SEED DOSE COSTS. Seeding at 0.1 dpa hands both routes a 0-D state
   taken while the mobile transient is still running; seeding at 1 dpa hands
   them one taken after it has settled. A 0-D seed is spatially uniform either
   way and carries no boundary layer, so the fast step always has a correction
   to make — the question is how large it is and whether the march carries the
   difference all the way to 20 dpa.

WHERE EACH ROUTE'S DATA COMES FROM
----------------------------------
    coupled     ``march_state.npz`` — doses, Y (n_dose, n_node, 19) in 0-D
                variables, and the node coordinates. Read directly rather than
                through the written ``evl`` files, because the coupled route
                names its snapshots on the 0.1 dpa step convention regardless
                of where it was actually seeded, and matching filenames across
                two different seed doses would silently compare different doses.
    standalone  ``<sim>/evl/evl_N.txt``. MoDELib writes after ``solve()``, so
                for a run seeded at dose s, ``evl_N`` holds s + (N+1) dpa.
    0-D         one adaptive integration of the calibrated model, sampled at
                the snapshot doses.

INTERIOR vs WHOLE DOMAIN
------------------------
Both means are reported. The whole-domain mean is dominated by the boundary
shell, where loop densities are known not to be quantitative: cascade
nucleation is spatially uniform but the only loop-removal channel is
coalescence driven by the absorbed mobile flux, which vanishes where Dirichlet
pins the mobile concentrations. The interior mean (innermost quartile by
distance to the nearest face) is the number to compare against the 0-D.

USAGE
-----
    python -m py_utils.compare_seeding \
        --case 1.0:<coupled_out_dir>:<standalone_sim_dir> \
        --case 0.1:<coupled_out_dir>:<standalone_sim_dir> \
        --out <report_dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from py_utils import paths                                   # noqa: E402
from py_utils import modelib_field as mfield                 # noqa: E402
from py_utils.calibration import build_sim                   # noqa: E402
from py_utils.cpp_bridge import collect_solver_args          # noqa: E402

# The lumped loop quantities, in 0-D units, and the 19-state columns they sum.
COMPARE = [
    ("N_a", "<a> loop number density", (4, 5)),
    ("N_c", "<c> loop number density", (6, 7)),
    ("c_a", "<a> loop content",        (8, 9)),
    ("c_c", "<c> loop content",        (10, 11)),
]
IMM_ORDER = ["CiL", "CaiL", "CvL", "CavL", "CiL_i", "CaiL_i", "CvL_v", "CavL_v"]
INTERIOR_FRAC = 0.25


# ── readers ──────────────────────────────────────────────────────────────────

def interior_mask(nodes, frac=INTERIOR_FRAC):
    """Innermost `frac` of the box by distance to the nearest face."""
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    d = np.minimum(nodes - lo, hi - nodes).min(axis=1)
    return d >= np.quantile(d, 1.0 - frac)


def lumped_from_Y(Y):
    """(n_node, 19) 0-D state -> the lumped quantities, per node."""
    return {k: Y[:, a] + Y[:, b] for k, _, (a, b) in COMPARE}


def lumped_from_cd(cd, omega):
    """A MoDELib CD block -> the same lumped quantities, per node."""
    imm = mfield.modelib_immobile_to_0d(cd[:, mfield.M_SIZE:], omega)
    d = {n: imm[:, i] for i, n in enumerate(IMM_ORDER)}
    return {"N_a": d["CiL"] + d["CaiL"], "N_c": d["CvL"] + d["CavL"],
            "c_a": d["CiL_i"] + d["CaiL_i"], "c_c": d["CvL_v"] + d["CavL_v"]}


def read_coupled(out_dir):
    """doses -> (n_node, 19), plus the node coordinates."""
    z = np.load(Path(out_dir) / "march_state.npz")
    doses, Y, nodes = z["doses"], z["Y"], z["nodes"]
    return {float(d): Y[i] for i, d in enumerate(doses)}, nodes


def standalone_step(dose, dose_seed, dose_per_step=1.0):
    """evl index holding `dose` for a run seeded at `dose_seed`.

    Output is written after solve(), so evl_N is the state after (N+1) steps:
    dose = dose_seed + (N+1)*dose_per_step. Derived from the run's own seed
    rather than from the 0.1 dpa convention baked into modelib_report, which
    would be off by one whole interval for a 1 dpa seed.
    """
    return int(round((dose - dose_seed) / dose_per_step)) - 1


def read_standalone(sim_dir, doses, dose_seed, omega, dose_per_step=1.0):
    """`dose -> (lumped, actual_dose)` for the nearest available snapshot.

    The standalone can only report on its own grid: seeded at 0.1 dpa with
    1 dpa steps it has 1.1, 2.1, ... dpa and nothing at 6.0. The dose it
    actually holds is carried alongside rather than rounded away, so a table
    row never claims a dose the run did not produce.
    """
    sim_dir = Path(sim_dir)
    out = {}
    for d in doses:
        n = standalone_step(d, dose_seed, dose_per_step)
        if n < 0:
            continue
        f = sim_dir / "evl" / f"evl_{n}.txt"
        if f.is_file():
            actual = dose_seed + (n + 1) * dose_per_step
            out[float(d)] = (lumped_from_cd(mfield.EvlFile(f).cd, omega),
                             float(actual))
    return out


def zero_d_trajectory(sim, doses, n_points=4000):
    """Calibrated 0-D state at each requested dose, from one adaptive run."""
    G = float(sim.input_data.material_params["G"])
    cli = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=max(doses) / G, n_points=n_points, log_time=True,
        rtol=1e-10, atol=1e-24, analytic_jac=True))
    p = subprocess.run([str(paths.zrmicro_solver_exe())] + cli,
                       capture_output=True, text=True)
    rows = np.array([[float(x) for x in L.split()] for L in p.stdout.splitlines()
                     if L.strip() and L[0] not in "#[="])
    t, Y = rows[:, 0], rows[:, 1:]
    return {float(d): Y[int(np.argmin(np.abs(t * G - d)))].copy() for d in doses}


# ── one seeding case ─────────────────────────────────────────────────────────

def analyze_case(dose_seed, coupled_dir, standalone_sim, sim, omega,
                 dose_per_step=1.0, verbose=True):
    """Assemble every route's answer for one seed dose."""
    coupled, nodes = read_coupled(coupled_dir)
    doses = sorted(coupled)
    interior = interior_mask(nodes)
    standalone = read_standalone(standalone_sim, doses, dose_seed, omega,
                                 dose_per_step)
    zerod = zero_d_trajectory(sim, doses)

    rows = []
    for d in doses:
        Lc = lumped_from_Y(coupled[d])
        Ls, d_std = standalone.get(d, (None, float("nan")))
        z = lumped_from_Y(zerod[d][None, :])
        for key, label, _ in COMPARE:
            r = dict(dose_seed=dose_seed, dose=d, standalone_dose=d_std,
                     quantity=key, label=label,
                     zero_d=float(z[key][0]),
                     coupled_all=float(Lc[key].mean()),
                     coupled_int=float(Lc[key][interior].mean()))
            if Ls is not None:
                r.update(standalone_all=float(Ls[key].mean()),
                         standalone_int=float(Ls[key][interior].mean()))
            else:
                r.update(standalone_all=float("nan"),
                         standalone_int=float("nan"))
            r["coupled_over_0d"] = r["coupled_int"] / max(r["zero_d"], 1e-300)
            r["standalone_over_0d"] = r["standalone_int"] / max(r["zero_d"], 1e-300)
            r["coupled_over_standalone"] = (
                r["coupled_int"] / r["standalone_int"]
                if np.isfinite(r["standalone_int"]) and r["standalone_int"] != 0
                else float("nan"))
            rows.append(r)
        if verbose:
            print(f"  {d:6.1f} dpa  " + "  ".join(
                f"{k}: 0d {rows[-4 + i]['zero_d']:.3e} "
                f"cpl {rows[-4 + i]['coupled_int']:.3e}"
                for i, (k, _, _) in enumerate(COMPARE)))
    return dict(dose_seed=dose_seed, doses=doses, rows=rows,
                n_nodes=int(nodes.shape[0]), n_interior=int(interior.sum()),
                coupled_dir=str(coupled_dir), standalone_sim=str(standalone_sim))


# ── reporting ────────────────────────────────────────────────────────────────

def _fmt(x, w=11):
    return "n/a".rjust(w) if not np.isfinite(x) else f"{x:{w}.4e}"


def write_report(cases, out_dir, sim):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G = float(sim.input_data.material_params["G"])
    T = float(sim.input_data.material_params["T"])

    md = ["# Seeding study — Adaptive (Option B), IMEX (Option A), and the 0-D", "",
          f"1 um single crystal, T = {T} K, G = {G} dpa/s, zero stress. Every "
          f"route uses the calibrated 0-D parameter set "
          f"({len(sim.overrides_applied)} fitted values on top of "
          f"`{Path(sim.input_file).name}`).", "",
          "`Adaptive` and `IMEX` columns are **interior** means — the "
          "innermost quartile of the cube by distance to the nearest face. "
          "The whole-domain mean is in the CSV; it is dominated by the boundary "
          "shell, where loop densities are not quantitative because the "
          "Dirichlet faces sink mobile defects but not loops.", ""]

    for c in cases:
        md += [f"## Seeded at {c['dose_seed']} dpa", "",
               f"{c['n_nodes']} CD nodes, {c['n_interior']} in the interior "
               f"quartile.", "",
               "| dose | quantity | 0-D | Adaptive | IMEX (at dose) | "
               "Adpt/0-D | IMEX/0-D | Adpt/IMEX |",
               "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in c["rows"]:
            sd = ("" if not np.isfinite(r["standalone_dose"])
                  else f" ({r['standalone_dose']:.1f})")
            md.append(
                f"| {r['dose']:.1f} | {r['quantity']} | {r['zero_d']:.4e} | "
                f"{r['coupled_int']:.4e} | {_fmt(r['standalone_int'], 0)}{sd} | "
                f"{r['coupled_over_0d']:.3f} | "
                f"{r['standalone_over_0d']:.3f} | "
                f"{r['coupled_over_standalone']:.3f} |")
        md.append("")

    # ── the seed-dose comparison ────────────────────────────────────────────
    if len(cases) > 1:
        md += ["## What the seed dose changes", "",
               "Doses reached by more than one seeding case, so the same state "
               "is being compared. A ratio near 1 means the march has forgotten "
               "where it was seeded.", ""]
        by_seed = {c["dose_seed"]: {(r["dose"], r["quantity"]): r
                                    for r in c["rows"]} for c in cases}
        seeds = sorted(by_seed)
        # Doses within half an interval of each other count as the same dose.
        ref, other = seeds[0], seeds[-1]
        md += [f"| dose (seed {ref}) | dose (seed {other}) | quantity | "
               f"Adaptive {ref} | Adaptive {other} | ratio | "
               f"IMEX {ref} | IMEX {other} | ratio |",
               "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        d_ref = sorted({k[0] for k in by_seed[ref]})
        d_oth = sorted({k[0] for k in by_seed[other]})
        for da in d_ref:
            db = min(d_oth, key=lambda x: abs(x - da))
            if abs(db - da) > 0.5:
                continue
            for key, _, _ in COMPARE:
                ra, rb = by_seed[ref].get((da, key)), by_seed[other].get((db, key))
                if ra is None or rb is None:
                    continue
                cr = rb["coupled_int"] / max(ra["coupled_int"], 1e-300)
                sr = (rb["standalone_int"] / ra["standalone_int"]
                      if np.isfinite(ra["standalone_int"])
                      and np.isfinite(rb["standalone_int"])
                      and ra["standalone_int"] != 0 else float("nan"))
                md.append(f"| {da:.1f} | {db:.1f} | {key} | "
                          f"{ra['coupled_int']:.4e} | {rb['coupled_int']:.4e} | "
                          f"{cr:.3f} | {_fmt(ra['standalone_int'], 0)} | "
                          f"{_fmt(rb['standalone_int'], 0)} | "
                          f"{'n/a' if not np.isfinite(sr) else f'{sr:.3f}'} |")
        md.append("")

    (out_dir / "seeding_comparison.md").write_text("\n".join(md) + "\n",
                                                   encoding="utf-8")

    import csv
    rows = [r for c in cases for r in c["rows"]]
    if rows:
        with open(out_dir / "seeding_comparison.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "seeding_comparison.json").write_text(
        json.dumps(dict(cases=cases, T=T, G=G,
                        input_file=sim.input_file,
                        n_overrides=len(sim.overrides_applied),
                        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                        git_hash=paths.git_hash()), indent=2, default=float),
        encoding="utf-8")
    return out_dir / "seeding_comparison.md"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    # Three separate values rather than one colon-joined string: a Windows path
    # carries its own colon, so any single-string form splits in the wrong place.
    ap.add_argument("--case", action="append", required=True, nargs=3,
                    metavar=("SEED", "COUPLED_DIR", "STANDALONE_SIM"),
                    help="repeatable; SEED is the seed dose in dpa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dose-per-step", type=float, default=1.0)
    args = ap.parse_args(argv)

    sim = build_sim()
    omega = mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL)
    out = Path(args.out) if args.out else (
        paths.OUTPUT_DIR /
        f"{time.strftime('%Y%m%d_%H%M%S')}_{paths.git_hash()}_seeding_study")

    cases = []
    for seed, coupled_dir, std_sim in args.case:
        print(f"\nseed {seed} dpa")
        cases.append(analyze_case(float(seed), coupled_dir, std_sim, sim, omega,
                                  dose_per_step=args.dose_per_step))
    p = write_report(cases, out, sim)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
