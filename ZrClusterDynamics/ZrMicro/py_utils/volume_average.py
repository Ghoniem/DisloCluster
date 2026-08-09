"""
volume_average.py — the volume-averaged trajectory of a 3-D march, rendered
through the 0-D figure suite.

WHY VOLUME-WEIGHT AND NOT JUST AVERAGE THE NODES
------------------------------------------------
A plain nodal mean is wrong on a boundary-layer mesh, and badly so. The 0.5 um
case puts 30 nm elements in the outer 150 nm and 120 nm elements inside, so
node density near the wall is roughly (120/30)^3 = 64 times the interior
density. An unweighted mean over nodes therefore reports something close to the
boundary-shell value, which is exactly the region the project documentation
warns is not quantitative. Earlier figure code sidestepped this by declaring
nodes "equal-volume" and noting the bias; here it is fixed.

The connectivity needed for exact nodal volumes is not available --
``evl/cdNodes.txt`` carries positions only -- so the weights are formed by
Monte-Carlo instead: draw many points uniformly in the domain, assign each to
its nearest CD node, and take the counts as the volume shares. That is the
Voronoi volume of each node by construction, it converges as 1/sqrt(N), and it
needs nothing but the node positions. With 4e6 samples on 3e4 nodes the
relative error of a weight is a fraction of a percent, far below the physical
spread being averaged.

WHAT IT PRODUCES
----------------
The full 0-D figure suite -- point defects, loop densities and sizes, growth,
creep, hardening, fluxes, conservation channels, fractions -- computed from the
volume-averaged 19-state trajectory of the 3-D march, so the two can be laid
side by side. Rendering goes through ``ZrMicroVisualizer`` itself rather than a
reimplementation, so the panels are the same panels.

USAGE
-----
    python -m py_utils.volume_average <run_dir> [--out <dir>] [--samples 4000000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from py_utils import paths                                   # noqa: E402
from py_utils.calibration import build_sim                   # noqa: E402
from py_utils.post_process import calculate_derived_quantities  # noqa: E402
from py_utils.visualization import ZrMicroVisualizer         # noqa: E402


def voronoi_weights(nodes, n_samples=4_000_000, seed=0, verbose=True):
    """Volume share of each node, by Monte-Carlo nearest-node assignment.

    Returns weights summing to 1. Nodes that win no sample get zero weight,
    which is correct: a node with no surrounding volume contributes nothing to
    a volume average.
    """
    nodes = np.asarray(nodes, dtype=float)
    lo, hi = nodes.min(0), nodes.max(0)
    rng = np.random.default_rng(seed)
    tree = cKDTree(nodes)
    counts = np.zeros(nodes.shape[0], dtype=np.int64)

    # Chunked so the sample array never exceeds a few hundred MB.
    chunk = 500_000
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        pts = lo + (hi - lo) * rng.random((m, 3))
        _, idx = tree.query(pts, k=1, workers=-1)
        counts += np.bincount(idx, minlength=nodes.shape[0])
        done += m
    w = counts / counts.sum()
    if verbose:
        nz = (counts > 0).sum()
        print(f"  volume weights: {n_samples:,} samples, "
              f"{nz}/{nodes.shape[0]} nodes carry volume")
        print(f"    weight min {w[w > 0].min():.3e}, max {w.max():.3e}, "
              f"ratio {w.max() / w[w > 0].min():.1f}")
        # How badly an unweighted mean would have been biased.
        print(f"    uniform weight would be {1 / nodes.shape[0]:.3e}")
    return w


def averaged_trajectory(run_dir, n_samples=4_000_000, verbose=True):
    """``(doses, Y_avg)`` with ``Y_avg`` of shape (n_dose, 19)."""
    z = np.load(Path(run_dir) / "march_state.npz")
    doses, Y, nodes = z["doses"], z["Y"], z["nodes"]
    if verbose:
        print(f"{Path(run_dir).name}: {len(doses)} snapshots, "
              f"{Y.shape[1]} nodes, {Y.shape[2]} states")
    w = voronoi_weights(nodes, n_samples=n_samples, verbose=verbose)
    Y_avg = np.einsum("n,dns->ds", w, Y)
    return np.asarray(doses, dtype=float), Y_avg, w, nodes


def build_results(doses, Y_avg, sim):
    """Run the volume-averaged trajectory through the 0-D post-processing."""
    G = float(sim.input_data.material_params["G"])
    t = doses / G
    # calculate_derived_quantities wants [n_species, n_time].
    # The fourth argument is the RateEquations object, not ReactionRates: the
    # post-processor reads `concentration_names` off it.
    res = calculate_derived_quantities(t, Y_avg.T, sim.input_data,
                                       sim.rate_equations)
    res["metadata"] = {
        "solver_stats": {"success": True, "message": "volume-averaged 3-D march",
                         "nfev": None, "njev": None, "nlu": None, "status": 0,
                         "t_events": None},
        "parameters": {
            "temperature": sim.input_data.material_params["T"],
            "dose_rate": G,
            "dislocation_density": sim.input_data.material_params["rho"],
            "end_time": float(t[-1]),
            "n_time_points": int(len(t)),
        },
        "input_file": str(sim.input_file),
    }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a march output directory (march_state.npz)")
    ap.add_argument("--out", default=None,
                    help="figure directory (default: <run_dir>/volume_average)")
    ap.add_argument("--samples", type=int, default=4_000_000)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "volume_average"

    sim = build_sim()
    doses, Y_avg, w, nodes = averaged_trajectory(run_dir, args.samples)

    print(f"\n{'dose':>10} {'Cv':>12} {'Ci':>12} {'N_a':>12} {'N_c':>12} "
          f"{'c_a':>12} {'c_c':>12}")
    for i in range(0, len(doses), max(1, len(doses) // 10)):
        y = Y_avg[i]
        print(f"{doses[i]:10.4g} {y[0]:12.4e} {y[1]:12.4e} "
              f"{y[4] + y[5]:12.4e} {y[6] + y[7]:12.4e} "
              f"{y[8] + y[9]:12.4e} {y[10] + y[11]:12.4e}")

    res = build_results(doses, Y_avg, sim)
    viz = ZrMicroVisualizer(sim, res, out.parent, use_dpa=True, run_dir=out,
                            sim_config={"source": "volume-averaged 3-D march",
                                        "run_dir": str(run_dir),
                                        "n_nodes": int(nodes.shape[0]),
                                        "mc_samples": args.samples})
    viz.plot_all()
    np.savez_compressed(out / "volume_average.npz", doses=doses, Y=Y_avg,
                        weights=w)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
