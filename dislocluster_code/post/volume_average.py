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
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.coupling import field as _field                # noqa: E402
from dislocluster_code.config import sim_for_run                           # noqa: E402
from dislocluster_code.zerod.post_process import calculate_derived_quantities  # noqa: E402
from dislocluster_code.post.visualization import ZrMicroVisualizer         # noqa: E402
from dislocluster_code.coupling.immobile import ACCUMULATOR_SLICE as ACC  # noqa: E402


def voronoi_weights(nodes, n_samples=4_000_000, seed=0, verbose=True,
                    faces=None):
    """Volume share of each node, by Monte-Carlo nearest-node assignment.

    Returns weights summing to 1. Nodes that win no sample get zero weight,
    which is correct: a node with no surrounding volume contributes nothing to
    a volume average.

    `faces` is an optional `(N, b)` half-space description of the domain (from
    `fields.domain_faces`). Samples are drawn in the bounding box, so on a
    non-box domain -- a hexagonal prism -- the empty corners are sampled too and
    every one of those samples is charged to whichever boundary node happens to
    be nearest. Passing `faces` rejects them. Left at None the old behaviour is
    kept exactly, which is what `averaged_trajectory` still does; it is exact
    for the cubic domains that function has always been used on.
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
        if faces is not None:
            N, b = faces
            pts = pts[np.all(pts @ N.T <= b[None, :], axis=1)]
            if not len(pts):
                done += m
                continue
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


def averaged_trajectory(run_dir, n_samples=4_000_000, verbose=True,
                        region="domain"):
    """``(doses, Y_avg)`` with ``Y_avg`` of shape (n_dose, 19).

    `region` selects WHAT IS AVERAGED, and on a Dirichlet domain the two
    answers differ by orders of magnitude for any loop quantity:

    ``"domain"``   every node, bounding-box Voronoi weights. The default, and
                   bit-identical to what this module always did, so figures
                   already published still reproduce.
    ``"interior"`` the innermost quartile by distance to the nearest face --
                   the same rule `post.coarsening.interior_mask` and
                   `discrete_loops.populate(region="interior")` use -- with the
                   Voronoi weights built against the CRYSTAL rather than its
                   bounding box.

    WHY THE DOMAIN MEAN IS NOT COMPARABLE TO EXPERIMENT, on this geometry.
    The Dirichlet shell is a sink for mobile defects but NOT for loops:
    cascade nucleation keeps making loops there while C_i -> 0 means nothing
    absorbs into them, so they accumulate as fresh nuclei that never grow.
    Measured on the 500 nm refit march at 10 dpa, the boundary nodes hold
    100.0% of the total <a> loop number; the domain mean is 210x the interior
    for N_a and 845x for N_c, and -- worse for the size figures -- the domain
    mean <c> DIAMETER falls from 46 nm to 5.7 nm between 0.1 and 10 dpa while
    the interior grows to 69.5 nm, because a population of fresh nuclei drags
    the mean down to the nucleation size. A 0-D fit has no boundary at all, so
    the interior mean is the quantity a fitted parameter set describes.

    The interior path also passes `faces`, so Monte-Carlo samples landing in a
    hexagonal prism's six empty wedges are rejected instead of being charged
    to whichever boundary node is nearest. The domain path deliberately does
    NOT, because changing it would move already-published figures.
    """
    z = np.load(Path(run_dir) / "march_state.npz")
    doses, Y, nodes = z["doses"], z["Y"], z["nodes"]
    if region not in ("domain", "interior"):
        raise ValueError(f"region must be 'domain' or 'interior', not {region!r}")
    if verbose:
        print(f"{Path(run_dir).name}: {len(doses)} snapshots, "
              f"{Y.shape[1]} nodes, {Y.shape[2]} states")
    if region == "interior":
        from dislocluster_code.post.coarsening import interior_mask
        from dislocluster_code.post.fields import domain_faces
        m = np.asarray(interior_mask(nodes), dtype=bool)
        w_all = voronoi_weights(nodes, n_samples=n_samples, verbose=verbose,
                                faces=domain_faces(nodes))
        w = np.where(m, w_all, 0.0)
        tot = w.sum()
        if tot <= 0.0:
            raise RuntimeError("interior nodes carry no Voronoi volume")
        w = w / tot
        if verbose:
            print(f"  interior region: {int(m.sum())} of {len(nodes)} nodes "
                  f"({100 * m.sum() / len(nodes):.0f}%), "
                  f"{100 * w_all[m].sum():.1f}% of the crystal volume")
    else:
        w = voronoi_weights(nodes, n_samples=n_samples, verbose=verbose)
    Y_avg = np.einsum("n,dns->ds", w, Y)
    # A self-consistent march stores its nine families in the immobile slots.
    # The 0-D figure suite reads them by the legacy names, so they are moved
    # into the slots those names expect -- lossless for every aggregate any
    # figure plots, since the aligned / non-aligned split the legacy layout
    # carries does not exist in the other model. A legacy march is untouched.
    #
    # THE SLOTS KEEP THEIR LEGACY NAMES AND NO LONGER MEAN THEM. The partner
    # slots are not empty: they hold the prismatic VACANCY variants and the
    # perfect basal state. Anything that LABELS these must ask `loop_model`.
    lm = _field.run_loop_model(run_dir)
    if lm:
        Y_avg = _field.to_legacy_layout(Y_avg, lm)
        if verbose:
            print("  self-consistent march: <a> i / <a> v into the "
                  "interstitial pair, c_f / c_p into the vacancy pair")
    return np.asarray(doses, dtype=float), Y_avg, w, nodes


def cumulate_accumulators(Y):
    """Turn per-interval conservation accumulators into cumulative ones.

    THE MARCH RESETS y[12:18] AT EVERY INTERVAL START
    (`coupling.march`, "the accumulator reset is scoped to an interval"), and
    `march_state.npz` stores a snapshot at every interval END. Each stored
    accumulator therefore holds ONE interval's production, recombination and
    sink absorption -- not the running total since dose zero.

    `zerod.post_process._calculate_conservation` was written for a standalone
    0-D run, where a single integration makes them genuinely cumulative, and
    differences them as `prod - prod[0]`. Applied to a march that reads the
    last interval only.

    The error hid on a logarithmic dose grid. With a constant ratio of 10 each
    interval is 90% of the dose accumulated so far, so every channel came out a
    uniform 10% low -- inside the noise of everything else. Appending 1, 2, 5,
    10 dpa to the grid drops that fraction to 0.5-0.6 and the same code is 40-50%
    wrong. The check that caught it is `post.boundary_flux`, which measures the
    boundary channel independently and stopped agreeing.

    Verified against the stored data: `accumulator / interval width` is exactly
    1.0 at every snapshot of both the 200 nm and 500 nm marches.
    """
    Y = np.array(Y, dtype=float, copy=True)
    Y[:, ..., ACC] = np.cumsum(Y[:, ..., ACC], axis=0)
    return Y


def build_results(doses, Y_avg, sim, per_interval_accumulators=True):
    """Run the volume-averaged trajectory through the 0-D post-processing.

    ``per_interval_accumulators`` reflects how the march stores them; see
    :func:`cumulate_accumulators`. Pass False only for a trajectory whose
    accumulators are already cumulative.
    """
    if per_interval_accumulators:
        Y_avg = cumulate_accumulators(Y_avg)
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
    ap.add_argument("--region", choices=("domain", "interior"), default="domain",
                    help="what to average. 'domain' (default) is every node and "
                         "reproduces every figure this module has ever written; "
                         "'interior' is the innermost quartile, which is the "
                         "quantity a 0-D fit describes and the only one "
                         "comparable to experiment on a Dirichlet domain")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else (
        run_dir / ("volume_average" if args.region == "domain"
                   else "volume_average_interior"))

    # The run's OWN model. `build_sim()` with no arguments takes the applied
    # load from the workbook, where `sigma_n` is 1.0e8 Pa -- so the conservation
    # channels for a zero-stress run were formed at f_a = 0.4015 while the march
    # integrated 1/3, and the two loop families differ in capture efficiency.
    sim = sim_for_run(run_dir)
    doses, Y_avg, w, nodes = averaged_trajectory(
        run_dir, args.samples, region=args.region)

    print(f"\n{'dose':>10} {'Cv':>12} {'Ci':>12} {'N_a':>12} {'N_c':>12} "
          f"{'c_a':>12} {'c_c':>12}")
    for i in range(0, len(doses), max(1, len(doses) // 10)):
        y = Y_avg[i]
        print(f"{doses[i]:10.4g} {y[0]:12.4e} {y[1]:12.4e} "
              f"{y[4] + y[5]:12.4e} {y[6] + y[7]:12.4e} "
              f"{y[8] + y[9]:12.4e} {y[10] + y[11]:12.4e}")

    res = build_results(doses, Y_avg, sim)

    # Atom fractions become COUNTS once the number of lattice atoms in the
    # simulated crystal is known: N = V / Omega, with V the convex body the CD
    # nodes fill (the prism, not its bounding box) and Omega the atomic volume
    # from the same material file MoDELib reads. Both are in b^3, so the ratio
    # is dimensionless and exact.
    n_atoms, open_system = None, False
    try:
        from dislocluster_code import paths
        from dislocluster_code.coupling import field as mfield
        from dislocluster_code.post.fields import domain_volume
        v_b3 = float(domain_volume(nodes))
        omega = float(mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL))
        n_atoms = v_b3 / omega
        # The balance residual is boundary absorption only if there IS an
        # absorbing boundary. Faces made periodic in BOUNDARY are not sinks, so
        # a fully periodic case is closed and the residual means what it always
        # meant. config.json records what was staged.
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        open_system = not cfg["boundary"]["periodic_face_ids"]
        print(f"  volume {v_b3:.4e} b^3 / Omega {omega:.6g} b^3 "
              f"= {n_atoms:.4e} atoms"
              + ("  (Dirichlet surface: an open system)" if open_system
                 else "  (periodic: closed)"))
    except Exception as exc:                      # pragma: no cover - diagnostics
        print(f"  note: defect counts unavailable ({exc}); plotting fractions")

    # THE FORMULATION HAS TO REACH THE LEGEND. `to_legacy_layout` moved the
    # self-consistent families into the legacy slots, which is lossless for
    # every aggregate plotted -- but the slots keep their legacy NAMES, and
    # "aligned interstitial" then labels a curve that is the prismatic VACANCY
    # population. A reader cannot detect that from the figure.
    lm = _field.run_loop_model(run_dir)
    viz = ZrMicroVisualizer(sim, res, out.parent, use_dpa=True, run_dir=out,
                            n_atoms=n_atoms, open_system=open_system,
                            loop_model=lm,
                            sim_config={"source": "volume-averaged 3-D march",
                                        "loop_model": int(lm),
                                        "run_dir": str(run_dir),
                                        "n_nodes": int(nodes.shape[0]),
                                        "n_atoms": n_atoms,
                                        "open_system": open_system,
                                        "mc_samples": args.samples})
    viz.plot_all()
    np.savez_compressed(out / "volume_average.npz", doses=doses, Y=Y_avg,
                        weights=w)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
