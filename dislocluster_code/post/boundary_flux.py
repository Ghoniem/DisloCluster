"""
boundary_flux.py — how many defects leave through the grain boundary, MEASURED.

WHY THIS EXISTS
---------------
`zerod.post_process._calculate_conservation` reports a `grain_boundary` channel
obtained by closure: production minus recombination minus sink absorption minus
what is stored, everything left over having gone out through the Dirichlet
surface. That is almost certainly right -- the residual is -71% of production
on the 200 nm march, and the same balance closes to 3e-12 on a standalone 0-D
run, so the discrepancy is the open boundary and not the solver -- but it is an
IDENTITY. It cannot disagree with itself, so it can neither confirm the physics
nor detect a channel that has been dropped.

This module measures the same quantity a second, independent way, and never
touches the accumulators.

THE MEASUREMENT
---------------
The fast solve returns a STEADY mobile field: for each mobile species m,

    0 = div(D_m grad C_m) + R_m(C)

with R_m the net reaction rate -- production minus recombination, minus
absorption at network dislocations and loops. Integrating over the domain and
applying the divergence theorem,

    integral_Omega R_m dV  =  -integral_Omega div(D_m grad C_m) dV
                           =  boundary_integral (-D_m grad C_m) . n dA
                           =  net OUTWARD flux of species m.

So the flux through the surface is the volume integral of the reaction rate,
evaluated on the converged field. Whatever the reactions make and do not
consume has to leave, because nothing is accumulating.

`R_m` is exactly what the 0-D right-hand side returns for the mobile
components: the 0-D model has no diffusion, so its mobile equations ARE the
reaction terms, and evaluating them at the 3-D field gives the reaction rate at
that point. The grain boundary appears nowhere in the 0-D model, which is
precisely why the volume integral of its mobile RHS measures the flux to it.

Interstitial ATOMS carry weights (1, 2, 3) for (Ci, C2i, C3i) and vacancies (1)
for Cv, matching `I_stored` and `V_stored` in the balance being checked.

WHAT IT IS AND IS NOT
---------------------
Independent of the accumulators: yes, entirely. A surface integral: no -- it is
a volume integral that EQUALS the surface integral at steady state. The
remaining approximations are that the mobile field is converged (the fast solve
reports its own convergence error, ~7e-8) and that the rate is integrated over
dose by the trapezoid rule between snapshots, which is as fine as the snapshot
grid.

The gold standard would be MoDELib evaluating the surface integral directly at
each fast solve, where the FE gradients already exist. This needs no C++ change
and applies to runs that already exist, which is why it comes first.

USAGE
-----
    python -m dislocluster_code.post.boundary_flux <run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import field as mfield
from dislocluster_code.post.fields import domain_volume, domain_faces
from dislocluster_code.post.volume_average import voronoi_weights
from dislocluster_code.config import sim_for_run

# Weights that turn a mobile concentration into ATOMS of each species, matching
# I_stored = Ci + 2*C2i + 3*C3i and V_stored = Cv in the balance being checked.
I_WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.0}     # Ci, C2i, C3i
V_WEIGHTS = {0: 1.0}                     # Cv


def reaction_rates_at_nodes(Y, sim):
    """``(dI/dt, dV/dt)`` in atoms per second per atom, per node.

    The 0-D right-hand side evaluated at each node's state. With no diffusion
    in that model these are the reaction terms alone, which is what the flux
    identity needs.
    """
    req = sim.rate_equations
    n = Y.shape[0]
    rI = np.zeros(n)
    rV = np.zeros(n)
    for q in range(n):
        dydt = np.asarray(req.ode_system(0.0, Y[q]), dtype=float)
        rI[q] = sum(w * dydt[k] for k, w in I_WEIGHTS.items())
        rV[q] = sum(w * dydt[k] for k, w in V_WEIGHTS.items())
    return rI, rV


def measure(run_dir, n_samples=1_000_000, verbose=True):
    """Cumulative boundary absorption over a march, measured and compared.

    Returns a dict with, per species, the measured cumulative outward flux in
    defect counts and the closure value from the accumulators.
    """
    run_dir = Path(run_dir)
    z = np.load(run_dir / "march_state.npz")
    doses, Y, nodes = z["doses"], z["Y"], z["nodes"]
    # `reaction_rates_at_nodes` evaluates the LEGACY 0-D mobile right-hand side,
    # which addresses the loop populations by the legacy slot names. On a
    # self-consistent march the families are moved into those slots first --
    # correct for the totals, which is all the mobile sink terms use, but note
    # that the sink strengths are then the legacy phenomenological ones rather
    # than the tensor efficiencies the march itself integrated. This is a
    # cross-check on the boundary channel, not a measurement of the new model.
    lm = mfield.run_loop_model(run_dir)
    if lm:
        Y = np.stack([mfield.to_legacy_layout(Y[d], lm)
                      for d in range(len(doses))])

    # The run's OWN model, not the workbook's: `sigma_n` stands at 1.0e8 Pa in
    # the workbook, so a zero-stress run rebuilt with `build_sim()` alone gets
    # f_a = 0.4015 where the march itself ran at exactly 1/3.
    sim = sim_for_run(run_dir, verbose=False)
    G = float(sim.input_data.material_params["G"])
    omega = float(mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL))
    n_atoms = float(domain_volume(nodes)) / omega

    # Volume weights on the CRYSTAL, rejecting samples in the empty wedges of a
    # prism -- the flux is a volume integral and a bounding-box weight would
    # inflate it by 4/3 on a hexagonal domain.
    try:
        faces = domain_faces(nodes)
    except Exception:
        faces = None
    w = voronoi_weights(nodes, n_samples=n_samples, faces=faces,
                        verbose=verbose)

    if verbose:
        print(f"{run_dir.name}: {len(doses)} snapshots, {nodes.shape[0]} nodes, "
              f"{n_atoms:.4e} atoms")

    rate_I, rate_V = [], []
    for d in range(len(doses)):
        rI, rV = reaction_rates_at_nodes(Y[d], sim)
        rate_I.append(float(w @ rI))          # volume-averaged, per s per atom
        rate_V.append(float(w @ rV))
        if verbose:
            print(f"  {doses[d]:10.4g} dpa   dI/dt {rate_I[-1]:+.4e}   "
                  f"dV/dt {rate_V[-1]:+.4e}  [1/s per atom]")

    # Integrate the rate over TIME (t = dose / G) and convert to counts.
    t = np.asarray(doses, dtype=float) / G
    cum_I = np.concatenate([[0.0], np.cumsum(
        0.5 * (np.asarray(rate_I)[1:] + np.asarray(rate_I)[:-1]) * np.diff(t))])
    cum_V = np.concatenate([[0.0], np.cumsum(
        0.5 * (np.asarray(rate_V)[1:] + np.asarray(rate_V)[:-1]) * np.diff(t))])

    out = {"doses": np.asarray(doses, dtype=float), "n_atoms": n_atoms,
           "measured": {"interstitial": cum_I * n_atoms,
                        "vacancy": cum_V * n_atoms}}

    # The closure value, for comparison only.
    from dislocluster_code.post.volume_average import (averaged_trajectory,
                                                       build_results)
    _, Y_avg, _, _ = averaged_trajectory(run_dir, n_samples, verbose=False)
    res = build_results(np.asarray(doses, dtype=float), Y_avg, sim)
    cons = res.get("conservation", {})
    out["closure"] = {sp: np.asarray(cons[sp]["grain_boundary"]) * n_atoms
                      for sp in ("interstitial", "vacancy") if sp in cons}
    return out


def report(run_dir, n_samples=1_000_000, verbose=True):
    m = measure(run_dir, n_samples, verbose=verbose)
    lines = ["", "grain-boundary absorption: MEASURED vs CLOSURE", ""]
    lines.append(f"{'dose':>10}  {'species':<13} {'measured':>16} "
                 f"{'closure':>16} {'ratio':>8}")
    for sp in ("interstitial", "vacancy"):
        if sp not in m["closure"]:
            continue
        for k in range(len(m["doses"])):
            a = m["measured"][sp][k]
            b = m["closure"][sp][k]
            r = (a / b) if b else float("nan")
            lines.append(f"{m['doses'][k]:10.4g}  {sp:<13} {a:16,.0f} "
                         f"{b:16,.0f} {r:8.3f}")
    text = "\n".join(lines)
    if verbose:
        print(text)
    return m, text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--out", default=None,
                    help="write the comparison here (default: "
                         "<run_dir>/boundary_flux.md)")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    m, text = report(run, args.samples)
    dest = Path(args.out) if args.out else run / "boundary_flux.md"
    dest.write_text(f"# Grain-boundary flux — {run.name}\n\n"
                    "Measured as the volume integral of the mobile reaction\n"
                    "rate on the converged steady field, which equals the\n"
                    "outward surface flux by the divergence theorem. This is\n"
                    "INDEPENDENT of the conservation accumulators.\n"
                    f"\n```{text}\n```\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
