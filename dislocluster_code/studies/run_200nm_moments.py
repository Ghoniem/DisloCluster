"""A 200 nm cube carrying all nine families and three moments per family.

This is the plan's step-5 demonstration case: the first coupled march in which
the immobile state is the full ``[n_k, c_k, q_k]`` for nine families -- 38
components per quadrature point on the 0-D side, 27 in MoDELib's CD block -- and
the first in which the size distribution exists rather than being assumed
monodisperse.

What is switched on, and why each one::

    loop_model     = 1   the self-consistent formulation; every switch below
                         describes a family or a moment the legacy slots do not
                         name, so none of them means anything without it
    n_fam          = 9   c_f, a1-a3, a1v-a3v, c_p, c_0
    emission_model = 1   step 3: peripheral emission against each family's own
                         c^{v,eq}, replacing the two fitted annealing lifetimes
    basal_chain    = 1   step 4: c_0 -> c_f -> c_p
    moments        = 1   step 5: the second content moment and the log-normal
                         closure built on it

THE BASAL-CHAIN RATES ARE PLACEHOLDERS AND ARE NOT CALIBRATED. Nothing in
``Zr3d_ghoniem.txt`` or the workbook supplies eps_sfp, tau_sfp, nu_col or nu_uf,
and the formulation gives their Arrhenius form without values for the barriers.
They are chosen here on ONE criterion -- that the transfers resolve on the run's
own timescale (10 dpa at 1e-7 dpa/s is 1e8 s, so a rate near 1e-6 1/s converts an
appreciable fraction of the population within the run and neither instantly nor
never) -- and on nothing else. Every number below is stated in the run's
provenance as a placeholder. Do not read a basal loop density off this run.

The rest of the configuration is the reference 500 nm case's, reduced to a
200 nm cube so the march finishes in minutes rather than hours: same material,
same temperature and dose rate, same dose grid, same solver settings.

    python -m dislocluster_code.studies.run_200nm_moments
    python -m dislocluster_code.studies.run_200nm_moments --no-report
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from dislocluster_code import config as C
from dislocluster_code import driver, paths

#: Rates and thresholds the formulation needs and the material file does not
#: supply. See the module docstring: chosen for timescale, fitted to nothing.
PLACEHOLDER_CHAIN = {
    # Cascade yield into the pyramid, as a rate [1/s], and vacancies per
    # cascade-nucleated pyramid. 1e-10 1/s at 1e-7 dpa/s is one pyramid per
    # 1e-3 dpa per atom-equivalent -- a visible but not dominant channel.
    "eps_sfp": 1.0e-10,
    "n_sfp_nuc": 100.0,
    # Pyramid dissolution. Equal to 1/nu_col, so the branching ratio
    # f_col = nu_col/(1/tau + nu_col) is exactly 1/2: half the pyramids convert
    # and half dissolve back. A round number, chosen to be obviously arbitrary.
    "tau_sfp": 1.0e6,
    "nu_col": 1.0e-6,
    # Unfaulting, three times slower, so c_f accumulates rather than passing
    # straight through to c_p.
    "nu_uf": 3.0e-7,
    "alpha_sfp": 1.0,
    # The size floor the dissolution current sits at, and the two transfer
    # barriers, all in defects. m_col above the pyramid's nucleation size so the
    # gate is a genuine fraction of the distribution rather than ~1.
    "m_min": 10.0,
    "m_col": 150.0,
    "m_uf": 400.0,
}


def build_config(tag="200nmCube_Moments", doses=None, lc_nm=25.0):
    material = dict(C.MATERIAL)
    geometry = dict(C.GEOMETRY, type="cubic", size_nm=200.0)
    mesh = dict(C.MESH, lc_nm=lc_nm, element_order=1,
                boundary_layer=True, boundary_thickness_nm=40.0,
                boundary_lc_nm=10.0)
    boundary = dict(C.BOUNDARY)
    coupling = dict(
        C.COUPLING,
        dose_seed=0.0,
        doses=list(doses if doses is not None
                   else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0]),
        substeps_per_interval=3, fem_every=3,
        variant_weights=(1 / 3, 1 / 3, 1 / 3))
    solver = dict(
        C.SOLVER, analytic_jac=True, loop_model=1,
        n_fam=9, emission_model=1, basal_chain=1, moments=1,
        model_params=dict(PLACEHOLDER_CHAIN))
    output = dict(C.OUTPUT, tag=tag, figures=True, movies=True,
                  movie_interp=5, discrete_loops=True)
    return C.SimulationConfig.from_dicts(
        MATERIAL=material, GEOMETRY=geometry, MESH=mesh, BOUNDARY=boundary,
        COUPLING=coupling, SOLVER=solver, OUTPUT=output)


def _placeholder_note(run_dir):
    """Say in the run itself what is not calibrated in it."""
    lines = [
        "# Uncalibrated inputs in this run", "",
        "The basal chain's rates and the two transfer thresholds are NOT",
        "fitted to anything. Neither `Zr3d_ghoniem.txt` nor the workbook",
        "supplies them, and the formulation gives their Arrhenius form without",
        "values for the barriers. They were chosen on one criterion: that the",
        "transfers resolve on this run's own timescale (10 dpa at 1e-7 dpa/s is",
        "1e8 s, so a rate near 1e-6 1/s converts an appreciable fraction of the",
        "population within the run, and neither instantly nor never).", "",
        "| parameter | value | unit |", "|---|---:|---|",
    ]
    units = {"eps_sfp": "1/s", "n_sfp_nuc": "vacancies", "tau_sfp": "s",
             "nu_col": "1/s", "nu_uf": "1/s", "alpha_sfp": "-",
             "m_min": "defects", "m_col": "defects", "m_uf": "defects"}
    for k, v in PLACEHOLDER_CHAIN.items():
        lines.append(f"| `{k}` | {v:g} | {units.get(k, '-')} |")
    lines += [
        "", "**Do not read a basal loop density off this run.** What it does",
        "demonstrate is the mechanism and the plumbing: nine populated",
        "families, three moments each, the chain conserving vacancies across",
        "two transfers, and the gates moving a subpopulation larger than the",
        "family mean.", "",
        "The 28-parameter set is in any case stale for `loop_model = 1` --",
        "it was fitted against the legacy formulation, and against the",
        "pre-correction Omega and <c> Burgers vector. See the project's",
        "CLAUDE.md.", "",
    ]
    (run_dir / "uncalibrated_inputs.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="200nmCube_Moments")
    ap.add_argument("--lc-nm", type=float, default=25.0)
    ap.add_argument("--doses", type=float, nargs="*", default=None)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--force-stage", action="store_true")
    a = ap.parse_args(argv)

    cfg = build_config(tag=a.tag, doses=a.doses, lc_nm=a.lc_nm)
    t0 = time.perf_counter()
    run = driver.prepare(cfg, force_stage=a.force_stage, verbose=True)
    _placeholder_note(run.out_dir)
    print(f"\nrun directory: {run.out_dir}\n")

    result = driver.march(run, verbose=True)
    print(f"\nmarch wall time: {(time.perf_counter() - t0) / 60:.1f} min")

    if not a.no_report:
        driver.report(run, result)
    print(f"\ndone: {run.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
