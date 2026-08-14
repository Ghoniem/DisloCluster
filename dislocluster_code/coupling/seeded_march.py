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
import time
from pathlib import Path

import numpy as np


from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.coupling import field as mfield                 # noqa: E402
from dislocluster_code.coupling import march as rcs        # noqa: E402
from dislocluster_code.coupling.config import default_config            # noqa: E402
from dislocluster_code.staging import standalone as sstd                # noqa: E402
from dislocluster_code.zerod.calibration import build_sim                   # noqa: E402

SCAFFOLD = sstd.SCAFFOLD


C_FLOOR = 1.0e-20


def pristine_state():
    """Unirradiated material: every species at the concentration floor.

    Not literally zero. Both codes floor their populations --- ZrMicro at
    ``C_floor`` before each rate evaluation, MoDELib at ``nFloor =
    concentrationFloor/omega`` and ``cFloor = concentrationFloor`` --- and the
    loop mean size is carried as ``r = l_a*sqrt(c/N)``, which is 0/0 at exactly
    zero. The floor is what both models already mean by "no loops".

    The mobile entries are immaterial: ``run_coupled`` overwrites them with the
    first fast solve before any immobile marching, so the 0-D seed's mobile
    field never influences the result even when one is supplied.
    """
    y = np.zeros(19)
    y[0:12] = C_FLOOR
    return y


def build_seed(sim, dose, dest, scaffold=None):
    """Write a uniform seed onto the CD nodes.

    ``dose <= 0`` seeds pristine material instead of a 0-D state, which is what
    removing the 0-D seed altogether amounts to.
    """
    if dose <= 0.0:
        y, hit = pristine_state(), 0.0
    else:
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
    ap.add_argument("--scaffold", default=None,
                    help="evl configuration whose CD node count the seed is "
                         "built on; REQUIRED when --template is not the "
                         "reference case, since the node set is mesh-specific "
                         "(see py_utils/setup_domain.py)")
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

    # The march's settings travel as a value. `dose_seed` matters as much as
    # the substep count: it is what ``evl_step_for`` names the written
    # snapshots on, so getting it wrong files this run's snapshots under
    # another seed's convention and the comparison silently drops them.
    cfg = default_config(dose_seed=float(dose_seed),
                         snaps=tuple(float(d) for d in snaps),
                         substeps=int(args.substeps),
                         fem_every=int(args.fem_every))

    seed_path = template / "evl" / (
        "evl_seed_pristine.txt" if dose_seed <= 0.0
        else f"evl_seed_{dose_seed:.3f}dpa.txt")
    scaffold = args.scaffold
    if scaffold is None and template.resolve() != Path(sstd.TEMPLATE).resolve():
        auto = template / "evl" / "evl_scaffold_pristine.txt"
        if not auto.is_file():
            raise SystemExit(
                f"--template is {template.name}, which is not the reference "
                f"case, so the CD node set differs and --scaffold must be "
                f"given. Expected {auto} from py_utils.setup_domain.")
        scaffold = auto
    y_seed, hit = build_seed(sim, dose_seed, seed_path, scaffold=scaffold)
    L = lumped(y_seed)
    what = "pristine" if dose_seed <= 0.0 else f"0-D at {dose_seed} dpa"
    print(f"seed: {what} (solver reached {hit:.4f}): "
          f"N_a={L['N_a'][0]:.4e} N_c={L['N_c'][0]:.4e} "
          f"c_a={L['c_a'][0]:.4e} c_c={L['c_c'][0]:.4e}")
    print(f"  Cv={y_seed[0]:.4e} Ci={y_seed[1]:.4e}")

    qdir = template.with_name(template.name + f"_qssa{dose_seed:g}")
    qdir, seed_used = rcs.prepare_qssa_dir(template, qdir, seed_evl=seed_path,
                                           cfg=cfg)
    print(f"fast-solve dir: {qdir}")

    tag = args.tag or f"march_seed{dose_seed:g}dpa"
    out = paths.run_dir(tag)
    (out / "evl_coupled").mkdir(parents=True, exist_ok=True)
    print(f"output: {out}")

    t0 = time.perf_counter()
    history, timing, br, diag = rcs.run_coupled(
        sim, qdir, seed_used, np.array(snaps), out / "evl_coupled",
        fem_every=args.fem_every, cfg=cfg)
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
