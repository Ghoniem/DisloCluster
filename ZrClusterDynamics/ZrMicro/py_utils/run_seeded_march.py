"""
run_seeded_march.py — the coupled operator-split march from a 0-D seed taken at
an arbitrary dose, on an arbitrary snapshot grid.

WHY AN EXPLICIT DOSE LIST
-------------------------
``run_coupled_vs_standalone`` builds its snapshots as equal intervals from the
seed dose, which is right for a single run and wrong for comparing two seed
doses: a march seeded at 0.1 dpa and one seeded at 1 dpa then report their
answers on grids offset by 0.9 dpa, and every cross-seed ratio silently
compares two different doses. Passing the doses explicitly lets both marches
land on the same grid, so seeding at 0.1 dpa costs one extra short interval
(0.1 -> 1 dpa, the transient) and nothing else.

WHAT IT RUNS
------------
At every substep: solve the steady mobile field C_M*(x) for the immobile state
currently held (MoDELib3 ``solveMobileClusters``, driven with
``useImmobileSolver=0``), freeze it, and integrate the immobile ODEs at every
node with CVODE BDF. ``--fem-every`` sets how many immobile substeps run
between two fast solves; the splitting error is first order in that spacing,
not in the snapshot spacing.

USAGE
-----
    python -m py_utils.run_seeded_march --doses 1 6 11 16 21 --fem-every 5
    python -m py_utils.run_seeded_march --doses 0.1 1 6 11 16 21 --fem-every 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from py_utils import paths                                   # noqa: E402
from py_utils import modelib_field as mfield                 # noqa: E402
from py_utils import run_coupled_vs_standalone as rcs        # noqa: E402
from py_utils import setup_standalone as sstd                # noqa: E402
from py_utils.calibration import build_sim                   # noqa: E402

SCAFFOLD = sstd.SCAFFOLD


def build_seed(sim, dose, dest, scaffold=None):
    """Write the calibrated 0-D state at `dose` uniformly onto the CD nodes."""
    y, hit = sstd.zero_d_state(sim, dose)
    ev = mfield.EvlFile(Path(scaffold or SCAFFOLD))
    N = ev.cd.shape[0]
    omega = mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL)
    ev.cd[:, :mfield.M_SIZE] = y[0:4]
    ev.cd[:, mfield.M_SIZE:] = mfield.immobile_0d_to_modelib(
        np.tile(y[:19] if y.size >= 19 else np.r_[y, np.zeros(19 - y.size)],
                (N, 1)), omega)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    ev.write(dest)
    return y, hit


def lumped(Y):
    Y = np.atleast_2d(Y)
    return dict(N_a=Y[:, 4] + Y[:, 5], N_c=Y[:, 6] + Y[:, 7],
                c_a=Y[:, 8] + Y[:, 9], c_c=Y[:, 10] + Y[:, 11])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doses", type=float, nargs="+", required=True,
                    help="seed dose first, then every snapshot dose [dpa]")
    ap.add_argument("--fem-every", type=int, default=5)
    ap.add_argument("--substeps", type=int, default=20,
                    help="immobile substeps per snapshot interval")
    ap.add_argument("--template", default=None,
                    help="case whose inputFiles/mesh the fast solves use")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    snaps = sorted(args.doses)
    dose_seed = snaps[0]
    template = Path(args.template or sstd.TEMPLATE)

    sim = build_sim()
    G = float(sim.input_data.material_params["G"])
    T = float(sim.input_data.material_params["T"])
    print(f"calibrated 0-D: {len(sim.overrides_applied)} fitted parameters, "
          f"T = {T} K, G = {G} dpa/s")
    print(f"doses: {', '.join(f'{d:g}' for d in snaps)} dpa, "
          f"{args.substeps} substeps per interval, fast solve every "
          f"{args.fem_every} substep(s)")

    # rcs carries these as module globals; the march reads them there.
    # DOSE_SEED must be set as well as the substep count: it is what
    # ``evl_step_for`` names the written snapshots on, so leaving it at the
    # module default would file this run's snapshots under a 0.1 dpa seed
    # convention regardless of where it was actually seeded.
    rcs.SUBSTEPS_PER_INTERVAL = int(args.substeps)
    rcs.DOSE_SEED = float(dose_seed)

    seed_path = template / "evl" / f"evl_seed_{dose_seed:.3f}dpa.txt"
    y_seed, hit = build_seed(sim, dose_seed, seed_path)
    L = lumped(y_seed)
    print(f"seed at {dose_seed} dpa (solver reached {hit:.4f}): "
          f"N_a={L['N_a'][0]:.4e} N_c={L['N_c'][0]:.4e} "
          f"c_a={L['c_a'][0]:.4e} c_c={L['c_c'][0]:.4e}")
    print(f"  Cv={y_seed[0]:.4e} Ci={y_seed[1]:.4e}")

    qdir = template.with_name(template.name + f"_qssa{dose_seed:g}")
    qdir, seed_used = rcs.prepare_qssa_dir(template, qdir, seed_evl=seed_path)
    print(f"fast-solve dir: {qdir}")

    tag = args.tag or f"march_seed{dose_seed:g}dpa"
    out = paths.OUTPUT_DIR / (f"{time.strftime('%Y%m%d_%H%M%S')}_"
                              f"{paths.git_hash()}_{tag}")
    (out / "evl_coupled").mkdir(parents=True, exist_ok=True)
    print(f"output: {out}")

    t0 = time.perf_counter()
    history, timing, br, diag = rcs.run_coupled(
        sim, qdir, seed_used, np.array(snaps), out / "evl_coupled",
        fem_every=args.fem_every)
    wall = time.perf_counter() - t0

    print(f"\n{'dose':>6} {'N_a':>12} {'N_c':>12} {'c_a':>12} {'c_c':>12}")
    for d in sorted(history):
        L = {k: float(v.mean()) for k, v in lumped(history[d]).items()}
        print(f"{d:6.1f} {L['N_a']:12.4e} {L['N_c']:12.4e} "
              f"{L['c_a']:12.4e} {L['c_c']:12.4e}")

    np.savez_compressed(out / "march_state.npz",
                        doses=np.array(sorted(history)),
                        Y=np.array([history[d] for d in sorted(history)]),
                        nodes=br.nodes)
    (out / "summary.json").write_text(json.dumps(dict(
        doses=snaps, dose_seed=dose_seed, fem_every=args.fem_every,
        substeps_per_interval=args.substeps, wall_s=wall, T=T, G=G,
        template=str(template), qssa_dir=str(qdir),
        input_file=sim.input_file, n_overrides=len(sim.overrides_applied),
        timing=timing, diagnostics=diag,
        git_hash=paths.git_hash()), indent=2, default=float), encoding="utf-8")
    print(f"\nwall {wall:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
