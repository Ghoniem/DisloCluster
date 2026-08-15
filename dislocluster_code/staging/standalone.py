"""
setup_standalone.py — stage a standalone MoDELib3 run seeded from the
*calibrated* 0-D state at an arbitrary dose.

WHY THIS EXISTS
---------------
The standalone route is the reference the coupled march is judged against, so
the two must start from the same microstructure and the same physics. An
earlier scratch version of this script built its seed with ``ZrMicroSimulation``
straight from the workbook, which is NOT the calibrated model — 28 parameters
have drifted and 11 are missing entirely (see ``py_utils/calibration.py``). Any
standalone run produced that way is seeded from a different model than the
coupled march it is compared with, and the comparison measures the parameter
difference rather than the numerics. This module always goes through
``calibration.build_sim``.

WHAT IT PRODUCES
----------------
    <sim>/inputFiles/            copied from the template case, with DD.txt
                                 rewritten for the requested dose span
    <sim>/evl/cdNodes.txt        the CD node set (fixed by mesh + element order)
    <sim>/evl/evl_0.txt          the seeded configuration
    <sim>/evl/evl_seed_<d>dpa.txt   an untouched copy, since DDomp overwrites
                                 evl_0.txt on its first output
    <sim>/seed_state.json        the 0-D seed vector and its provenance

THE SCAFFOLD
------------
``evl_0.txt`` needs the full MoDELib record structure, not just the CD block.
The microstructure generator would emit one with every discrete count zero
(``useDislocations=0``), but it is not built in every build tree, and its output
is invariant for this case anyway. So an existing seed file is used as the
scaffold and only its CD block is replaced — the same thing the generator would
have produced, without the dependency.

USAGE
-----
    python -m py_utils.setup_standalone --dose-seed 1.0 --dose-max 21.0 \
        --sim-dir <path> --run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.coupling import field as mfield                 # noqa: E402
from dislocluster_code.zerod.calibration import build_sim                   # noqa: E402
from dislocluster_code.zerod.cpp_bridge import collect_solver_args          # noqa: E402
from dislocluster_code.coupling.qssa import get_dd_scalar, set_dd_scalar  # noqa: E402

TEMPLATE = paths.MODELIB_ROOT / "tutorials" / "zrmicro_seeded"
SCAFFOLD = TEMPLATE / "evl" / "evl_seed_0.100dpa.txt"

STATE_NAMES = ["Cv", "Ci", "C2i", "C3i", "CiL", "CaiL", "CvL", "CavL",
               "CiL_i", "CaiL_i", "CvL_v", "CavL_v"]


def zero_d_state(sim, dose, n_points=4000):
    """The calibrated 0-D state vector at `dose`, from one adaptive run.

    Integrated to the requested dose in a single call with tight tolerances,
    rather than marched, so the seed carries no accumulated step error.
    """
    G = float(sim.input_data.material_params["G"])
    cli = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=dose / G, n_points=n_points, log_time=True,
        rtol=1e-10, atol=1e-24, analytic_jac=True))
    p = subprocess.run([str(paths.zrmicro_solver_exe())] + cli,
                       capture_output=True, text=True)
    rows = np.array([[float(x) for x in L.split()] for L in p.stdout.splitlines()
                     if L.strip() and L[0] not in "#[="])
    if rows.size == 0:
        raise RuntimeError(f"0-D solver produced no output:\n{p.stdout[-2000:]}"
                           f"\n{p.stderr[-2000:]}")
    return rows[-1, 1:].copy(), float(rows[-1, 0] * G)


# Both names now live with the other input-file writers, in staging.inputs.
# Re-exported because run_seeded_march, setup_domain and the notebooks import
# them from here.
from dislocluster_code.staging.inputs import (  # noqa: E402,F401
    FAST_STEP_SETTINGS, write_dd)


def stage(sim_dir, dose_seed, dose_max, sim=None, scaffold=None, verbose=True):
    """Build a complete, runnable standalone case seeded at `dose_seed`."""
    sim = sim or build_sim()
    sim_dir = Path(sim_dir)
    scaffold = Path(scaffold or SCAFFOLD)
    if not scaffold.is_file():
        raise FileNotFoundError(f"no scaffold configuration at {scaffold}")

    G = float(sim.input_data.material_params["G"])
    T = float(sim.input_data.material_params["T"])
    y, dose_hit = zero_d_state(sim, dose_seed)
    if verbose:
        print(f"calibrated 0-D seed at {dose_seed} dpa "
              f"(solver reached {dose_hit:.4f}):")
        for j, n in enumerate(STATE_NAMES):
            print(f"    {n:<7} {y[j]:.6e}")
        print(f"    N_a = {y[4] + y[5]:.4e}   N_c = {y[6] + y[7]:.4e}")
        print(f"    c_a = {y[8] + y[9]:.4e}   c_c = {y[10] + y[11]:.4e}")

    # ── directory ───────────────────────────────────────────────────────────
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    (sim_dir / "evl").mkdir(parents=True)
    (sim_dir / "F").mkdir()
    shutil.copytree(TEMPLATE / "inputFiles", sim_dir / "inputFiles")

    # The material and mesh are regenerated from the Library with LF endings,
    # exactly as clean_run.sh does: MoDELib's parser treats a trailing CR as
    # part of the value and silently mis-reads the last field on a line.
    lib = paths.MODELIB_ROOT / "Library"
    for src, dst in ((paths.MODELIB_MATERIAL,
                      sim_dir / "inputFiles" / paths.MODELIB_MATERIAL.name),
                     (lib / "Meshes" / "unitCube_15K.msh",
                      sim_dir / "inputFiles" / "unitCube_15K.msh")):
        dst.write_bytes(Path(src).read_bytes().replace(b"\r\n", b"\n"))

    n_steps = int(round(dose_max - dose_seed))
    dtMax = write_dd(sim_dir, n_steps, use_immobile=1)

    # ── the seeded configuration ────────────────────────────────────────────
    shutil.copy2(TEMPLATE / "evl" / "cdNodes.txt", sim_dir / "evl" / "cdNodes.txt")
    omega = mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL)
    ev = mfield.EvlFile(scaffold)
    N = ev.cd.shape[0]
    ev.cd[:, :mfield.M_SIZE] = y[0:4]
    ev.cd[:, mfield.M_SIZE:] = mfield.immobile_0d_to_modelib(
        np.tile(y[:19] if y.size >= 19 else np.r_[y, np.zeros(19 - y.size)],
                (N, 1)), omega)
    ev0 = sim_dir / "evl" / "evl_0.txt"
    ev.write(ev0)
    seed_copy = sim_dir / "evl" / f"evl_seed_{dose_seed:.3f}dpa.txt"
    shutil.copy2(ev0, seed_copy)

    (sim_dir / "seed_state.json").write_text(json.dumps(dict(
        dose_seed=dose_seed, dose_max=dose_max, n_steps=n_steps, dtMax=dtMax,
        fast_step=FAST_STEP_SETTINGS,
        T=T, G=G, n_nodes=int(N), omega=float(omega),
        input_file=sim.input_file,
        n_overrides=len(sim.overrides_applied),
        state={n: float(y[j]) for j, n in enumerate(STATE_NAMES)},
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_hash=paths.git_hash()), indent=2), encoding="utf-8")

    if verbose:
        print(f"\nstaged {sim_dir}")
        print(f"  {N} CD nodes, Nsteps={n_steps} "
              f"({dose_seed} -> {dose_max} dpa in 1 dpa steps)")
        print("  fast step pinned to the coupled route's settings: " +
              ", ".join(f"{k}={v}" for k, v in FAST_STEP_SETTINGS.items()))
        print(f"  seed copy kept as {seed_copy.name}")
    return sim_dir, y, n_steps


def run_ddomp(sim_dir, log=None, verbose=True):
    """Launch DDomp on a staged case. Blocks; the run takes tens of minutes."""
    cmd, cwd = paths.ddomp_cmd(sim_dir)
    if verbose:
        print("running: " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with open(log or Path(sim_dir) / "ddomp.log", "w", encoding="utf-8",
              errors="replace") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                           cwd=cwd)
    wall = time.perf_counter() - t0
    if verbose:
        print(f"DDomp exit={r.returncode}, {wall:.0f} s", flush=True)
    return r.returncode, wall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dose-seed", type=float, default=1.0)
    ap.add_argument("--dose-max", type=float, default=21.0)
    ap.add_argument("--sim-dir", default=None,
                    help="default: <tutorials>/zrmicro_std<dose_seed>")
    ap.add_argument("--scaffold", default=None)
    ap.add_argument("--run", action="store_true", help="launch DDomp after staging")
    args = ap.parse_args(argv)

    sim_dir = Path(args.sim_dir) if args.sim_dir else (
        paths.MODELIB_ROOT / "tutorials" / f"zrmicro_std{args.dose_seed:g}")
    sim = build_sim()
    print(f"calibrated 0-D: {len(sim.overrides_applied)} fitted parameters "
          f"applied on top of {Path(sim.input_file).name}\n")
    stage(sim_dir, args.dose_seed, args.dose_max, sim=sim, scaffold=args.scaffold)
    if args.run:
        rc, wall = run_ddomp(sim_dir)
        (sim_dir / "ddomp_wall_s.txt").write_text(f"{wall:.3f}\n",
                                                  encoding="utf-8")
        return rc
    cmd, _ = paths.ddomp_cmd(sim_dir)
    print("\nrun it with:\n  " + " ".join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
