"""The 500 nm hexagonal single crystal, run with the REFITTED parameter set.

This is the fit's first test outside the 0-D. The refit was done against
experimental loop densities and diameters in the 0-D, which IS the march's slow
step -- one templated function in `rate_equations_core.h`, the same binary, the
same switches, differing only by `freeze_mobile`. So the parameters transfer
term for term. What does NOT transfer is everything the 0-D cannot see: the
spatial structure, the Dirichlet boundary layer, and the mobile field the fast
solve supplies in place of the 0-D's free pool.

    python -m dislocluster_code.studies.run_500nm_hex_refit <refit-dir>
    python -m dislocluster_code.studies.run_500nm_hex_refit <refit-dir> --dry-run

The configuration is the reference `500nmHex_Matched_SelfConsistent` leg --
same prism, same mesh, same dose grid, same solver settings -- so that the only
difference from the recorded run is the parameter set. That is deliberate: a
comparison in which two things moved measures neither.

WHAT TO EXPECT, AND WHAT WOULD BE A SURPRISE.

  - The interior mean is the number to read, never the domain mean. On this
    geometry the Dirichlet shell carries almost all of the <a> density and is a
    sink for mobile defects but not for loops, so a domain mean measures the
    boundary, not the material.
  - The refit moved the ZEROTH moment hard (the small-end leak takes the <c>
    density down ~8x on its own), so the loop densities here should sit well
    below the recorded matched leg. That is the fit, not a regression.
  - The 0-D and the march need not agree on the SIGN of a change: in the 0-D
    the vacancy pool is free and re-equilibrates, while the march pins the
    mobile species to the fast solve. That difference has already been measured
    to move `c_c` in opposite directions between the two.

THE BASAL CHAIN IS OFF, matching the refit. `n_fam = 9` still allocates `c_0`,
which stays empty because nothing nucleates into it -- the chain's rates are
not calibrated and were deliberately excluded from the fit, so switching it on
here would run a channel the parameters know nothing about.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dislocluster_code import config as C
from dislocluster_code import driver, paths
from dislocluster_code.fitting import apply_refit


def build_config(workbook, model_params, tag="500nmHex_Refit"):
    material = dict(C.MATERIAL, temperature_K=573.0, dose_rate_dpa_s=1e-7,
                    overrides=dict(workbook))
    geometry = dict(C.GEOMETRY, type="hexagonal", size_nm=500.0,
                    height_nm=800.0)
    mesh = dict(C.MESH, target_elements=0, lc_nm=60.0, element_order=1,
                boundary_layer=True, boundary_thickness_nm=100.0,
                boundary_lc_nm=20.0, boundary_layers_across=None)
    boundary = dict(C.BOUNDARY)
    coupling = dict(C.COUPLING, route="Adaptive", dose_seed=0.0,
                    doses=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0],
                    substeps_per_interval=3, fem_every=3,
                    variant_weights=(1 / 3, 1 / 3, 1 / 3))
    solver = dict(C.SOLVER, analytic_jac=True, loop_model=1, n_fam=9,
                  emission_model=1, basal_chain=0, moments=1,
                  model_params=dict(model_params))
    output = dict(C.OUTPUT, tag=tag, figures=True, movies=True,
                  movie_interp=5, discrete_loops=True, checkpoint=True,
                  resume="auto")
    return C.SimulationConfig.from_dicts(
        MATERIAL=material, GEOMETRY=geometry, MESH=mesh, BOUNDARY=boundary,
        COUPLING=coupling, SOLVER=solver, OUTPUT=output)


def _provenance(run_dir, refit_dir, workbook, model_params, bias_edits):
    """Say, in the run, exactly which fit produced it and where each half went."""
    lines = [
        "# Parameter provenance for this run", "",
        f"Refit source: `{refit_dir}`", "",
        "The fitted set does not live in one place. This run carries it in "
        "three, and all three had to agree for the run to mean anything:", "",
        "| where | what | count |", "|---|---|---:|",
        f"| `MATERIAL['overrides']` | workbook parameters | {len(workbook)} |",
        f"| `SOLVER['model_params']` | solver switches | {len(model_params)} |",
        f"| `{Path(bias_edits['file']).name}` | the bias | "
        f"{len(bias_edits['after'])} vectors |", "",
        "## The bias, as this run reads it", "", "```",
    ]
    a = bias_edits['after']
    for k, v in a.items():
        lines.append(f"{k:<16}{' '.join(f'{x:.6g}' for x in v)}")
    pv, pi = a['dadAnisotropy'][0], a['dadAnisotropy'][1]
    z0v, z0i = a['dadZ0'][0], a['dadZ0'][1]
    lines += ["```", "", "which gives the Woo capture efficiencies", "", "```",
              f"Z_basal(v)      {z0v * pv:.6f}   <c> gain",
              f"Z_basal(i)      {z0i * pi:.6f}   <c> loss",
              f"Z_prismatic(i)  {z0i * (pi + pi ** -2) / 2:.6f}   <a> gain",
              f"Z_prismatic(v)  {z0v * (pv + pv ** -2) / 2:.6f}   <a> loss",
              f"co-growth  p_i < p_v : {pi:.4f} < {pv:.4f}  "
              f"{'satisfied' if pi < pv else 'VIOLATED'}", "```", "",
              "## Solver switches", "", "```",
              "loop_model = 1   n_fam = 9   emission_model = 1   moments = 1",
              "basal_chain = 0  -- matching the refit; its rates are not "
              "calibrated and were excluded from the fit", "```", ""]
    if model_params:
        lines += ["```"]
        for k, v in sorted(model_params.items()):
            lines.append(f"{k:<14}{v:.6g}")
        lines += ["```", ""]
    lines += ["## Workbook overrides", "", "| parameter | value |", "|---|---:|"]
    for k in sorted(workbook):
        lines.append(f"| `{k}` | {workbook[k]:.6g} |")
    lines += ["", "## Reading this run", "",
              "- Quote the **interior mean**, never the domain mean. The "
              "Dirichlet shell is a sink for mobile defects but not for loops, "
              "so a domain mean measures the boundary layer.",
              "- The refit is a 0-D fit. The 0-D is the march's slow step term "
              "for term, but the march pins the mobile species to the fast "
              "solve where the 0-D leaves them free, so the two need not move "
              "a given quantity in the same direction.",
              "- `m_vanish` and `nu_vanish` are fitted here against loop "
              "densities and pinned by no independent measurement.", ""]
    (Path(run_dir) / 'parameter_provenance.md').write_text(
        "\n".join(lines), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('refit', help='refit directory written by refit_production')
    ap.add_argument('--tag', default='500nmHex_Refit')
    ap.add_argument('--dry-run', action='store_true',
                    help='stage and report, do not march')
    ap.add_argument('--no-apply-bias', action='store_true',
                    help='assume the material file already carries the bias')
    a = ap.parse_args(argv)

    from dislocluster_code.fitting.refit_report import load
    params, meta = load(a.refit)
    workbook, model_params, bias = apply_refit.split(params)

    edits = apply_refit.material_edits(
        bias, dry_run=a.no_apply_bias or not bias, T=573.0)
    apply_refit.show(params, edits)
    if bias and not a.no_apply_bias:
        print(f"\n  material file written; backup {edits['backup'].name}")

    cfg = build_config(workbook, model_params, tag=a.tag)
    print(f"\n  staging {a.tag} ...")
    t0 = time.time()
    prep = driver.prepare(cfg)
    run_dir = Path(prep.out_dir)
    _provenance(run_dir, a.refit, workbook, model_params, edits)
    print(f"  run directory {run_dir}")
    if a.dry_run:
        print("  --dry-run: staged only.")
        return run_dir
    print("  marching ... (the reference leg took ~0.7 h)")
    result = driver.march(prep)
    driver.report(prep, result)
    print(f"\n  done in {(time.time() - t0) / 3600:.2f} h -> {run_dir}")
    return run_dir


if __name__ == '__main__':
    main()
