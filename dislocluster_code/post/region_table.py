"""The domain-vs-interior loop table of the manuscript's results section.

    python -m dislocluster_code.post.region_table <run-with-channel> \
        --twin <run-without-channel> --dose 10 [--latex]

WHY THIS MODULE EXISTS. The table it writes was previously assembled by hand,
and the numbers in it could not afterwards be reproduced from the run
directories by any reduction in this package -- four were tried, and the
domain-vs-interior comparison came out with the opposite sign to the one the
surrounding prose claimed. A table that carries a scientific claim has to be
regenerable from the runs it describes, so it is generated here, by one
function, with the reduction stated below and repeated in the caption.

THE REDUCTION, in full:

  region    `domain` is every CD node; `interior` is the innermost quartile by
            distance to the nearest face (`coarsening.interior_mask`), which is
            the same definition `discrete_loops.populate` and the hardening
            study use. The Dirichlet shell is a sink for mobile defects but not
            for loops, so a domain mean over it measures the boundary layer.

  average   volume-weighted over the CD nodes, the weights being Monte-Carlo
            Voronoi volumes clipped to the CRYSTAL (`fields.domain_faces`), not
            to its bounding box. For the hexagonal prism the box is 4/3 the
            crystal.

  N_k       the region-averaged number fraction divided by the atomic volume
            of the material file, so m^-3. The families are summed over the
            legacy pair AFTER `field.to_legacy_layout`, i.e. <a> is every
            prismatic variant and <c> is the basal faulted plus perfect state.

  d_k       NUMBER-WEIGHTED MEAN DIAMETER, 2*(sum_j C_j r_j)/(sum_j C_j), which
            is the observation operator the calibration used and exactly what
            `plot_loop_sizes_vs_experiment` draws. It is a diameter because the
            experimental `Mean Size` column is one; the state carries radii.

So the table and the dose-axis figures are the same numbers at one dose, which
is the property the previous table did not have.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dislocluster_code.post.volume_average import (averaged_trajectory,
                                                   build_results)
from dislocluster_code.config import sim_for_run


def region_state(run_dir, dose, region, n_samples=4_000_000, verbose=False):
    """``(N_a, d_a, N_c, d_c)`` for one run, one dose, one region."""
    doses, Y_avg, _w, _nodes = averaged_trajectory(
        run_dir, n_samples, verbose=verbose, region=region)
    j = int(np.argmin(np.abs(np.asarray(doses, float) - float(dose))))
    if not np.isclose(doses[j], dose, rtol=1e-6):
        raise ValueError(f"{Path(run_dir).name}: no snapshot at {dose} dpa; "
                         f"nearest is {doses[j]:g}")
    sim = sim_for_run(run_dir)
    res = build_results(np.asarray(doses, float), Y_avg, sim)
    conc, loops = res["concentrations"], res["loop_sizes"]
    # Omega in m^3, from the run's OWN model rather than the material file,
    # so the table converts with exactly the Omega the 0-D figures convert with.
    omega = float(sim.input_data.physical_props["Omega"])

    def pair(n0, n1, r0, r1):
        C0, C1 = conc[n0][j], conc[n1][j]
        N = C0 + C1
        d = 2e9 * (C0 * loops[r0][j] + C1 * loops[r1][j]) / max(N, 1e-30)
        return N / omega, d

    N_a, d_a = pair("CiL", "CaiL", "r_iL", "r_aiL")
    N_c, d_c = pair("CvL", "CavL", "r_vL", "r_avL")
    return float(N_a), float(d_a), float(N_c), float(d_c)


def table(run_on, run_off, dose=10.0, n_samples=4_000_000, verbose=False):
    """``{label: {region: (N_a, d_a, N_c, d_c)}}`` for the two runs."""
    out = {}
    for label, run in (("loop channel on", run_on),
                       ("loop channel off", run_off)):
        if run is None:
            continue
        out[label] = {r: region_state(run, dose, r, n_samples, verbose)
                      for r in ("domain", "interior")}
    return out


def as_latex(tab, dose):
    """The `tabular` body, in the manuscript's own column order."""
    def num(x):
        e = int(np.floor(np.log10(abs(x)))) if x else 0
        return rf"${x / 10 ** e:.2f}\times10^{{{e}}}$"
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"& Region & $N_a$ (m$^{-3}$) & $d_a$ (nm) "
             r"& $N_c$ (m$^{-3}$) & $d_c$ (nm) \\", r"\midrule"]
    for i, (label, regions) in enumerate(tab.items()):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{2}}{{*}}{{{label}}}")
        for r in ("domain", "interior"):
            N_a, d_a, N_c, d_c = regions[r]
            lines.append(f" & {r:9s} & {num(N_a)} & {d_a:.2f} & "
                         f"{num(N_c)} & {d_c:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ratios(tab):
    """Domain/interior ratio per quantity -- what the prose quotes."""
    out = {}
    for label, regions in tab.items():
        dm, it = regions["domain"], regions["interior"]
        out[label] = {"N_a": dm[0] / it[0], "N_c": dm[2] / it[2],
                      "d_a": dm[1] / it[1], "d_c": dm[3] / it[3]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run WITH grain-boundary loop absorption")
    ap.add_argument("--twin", default=None,
                    help="the same case without it")
    ap.add_argument("--dose", type=float, default=10.0)
    ap.add_argument("--samples", type=int, default=4_000_000)
    ap.add_argument("--latex", action="store_true", help="emit the tabular")
    a = ap.parse_args(argv)
    tab = table(a.run, a.twin, a.dose, a.samples)
    print(f"\n  at {a.dose:g} dpa\n")
    print(f"  {'':18s}{'region':10s}{'N_a [m^-3]':>13s}{'d_a [nm]':>10s}"
          f"{'N_c [m^-3]':>13s}{'d_c [nm]':>10s}")
    for label, regions in tab.items():
        for r in ("domain", "interior"):
            N_a, d_a, N_c, d_c = regions[r]
            print(f"  {label:18s}{r:10s}{N_a:13.3e}{d_a:10.2f}"
                  f"{N_c:13.3e}{d_c:10.2f}")
            label = ""
    print("\n  domain / interior")
    for label, rr in ratios(tab).items():
        print(f"  {label:18s}" + "  ".join(f"{k} {v:8.3g}" for k, v in rr.items()))
    if a.latex:
        print("\n" + as_latex(tab, a.dose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
