"""
calibration.py — the fitted 0-D parameter set, as a single source of truth.

WHY THIS MODULE EXISTS
----------------------
The input workbook is NOT the calibrated model. It has drifted away from the
parameter set the 0-D reference run was produced with: 28 parameters differ and
11 are absent from it entirely. Section 6c of
``code/coupled_0d_3d_ZrMicro.ipynb`` therefore pins every fitted value
explicitly and applies it on top of the workbook before building the model
chain.

Any driver that builds ``InputData`` straight from the workbook and skips that
step is running a different model. The difference is not subtle. Integrated to
20 dpa at 573 K and 1e-7 dpa/s:

    workbook only   N_a = 2.32e-2,  c_a = 8.77     <- c_a is an atom fraction,
                                                      so this is impossible
    calibrated      N_a = 8.05e-8,  c_a = 1.34e-4  <- 6.7e21 m^-3, r = 4.4 nm

Five orders of magnitude in loop density, and only the calibrated set is
physical. ``run_coupled_vs_standalone.py`` and the seed builder both skipped the
overrides, so every coupled-march result produced before this module existed was
computed with the uncalibrated set and has to be discarded.

USAGE
-----
    from dislocluster_code.zerod.calibration import build_sim
    sim = build_sim()                       # workbook + fitted overrides
    sim = build_sim(T=573.0, G=1.0e-7)      # ... with T and G pinned too

The returned object exposes ``input_data``, ``reaction_rates``,
``rate_equations`` and ``input_file``, which is the interface
``collect_solver_args`` and the coupling drivers expect.
"""
from __future__ import annotations

import types

from dislocluster_code import paths
from dislocluster_code.zerod.input_data import InputData
from dislocluster_code.zerod.rate_equations import RateEquations
from dislocluster_code.zerod.reaction_rates import ReactionRates

# The fitted parameter set of the 0-D reference run
# (output/20260622_144021_7959445), transcribed from Section 6c of
# coupled_0d_3d_ZrMicro.ipynb at full precision. provenance.md stores only
# %.4e, so re-reading the rounded text would NOT reproduce the run.
#
# `rho` and `rho_600` are deliberately absent: the workbook holds
# 1.88004e14 / 2.00116e14 while provenance shows the rounded values, and
# pinning the rounded text would inject a 2e-5 relative error.
REFERENCE_OVERRIDES = {
    "delta_DAD_i_300": 0.153343,
    "delta_DAD_i_600": 0.0640797,
    "delta_DAD":       0.107886,
    "Z_N":             1.01,
    "Z_iL":            1.34408,
    "Z_vL":            1.02,
    "Q":               0.287931,
    "c_LL_a":          120.577,
    "c_LN_a":          1130.85,
    "c_LL_c":          0.162483,
    "c_LN_c":          2.95829,
    "kappa_LL":        2.0,
    "kappa_LN":        0.4,
    "epsilon_iL":      0.00211286,
    "epsilon_vL":      0.005,
    "epsilon_2i":      0.01,
    "epsilon_3i":      0.01,
    "n_iL_nuc":        300,
    "n_vL_nuc":        400,
    "E_m_i":           0.759101,
    "E_m_2i":          1.36292,
    "E_b_2i":          0.957033,
    "E_a_vL":          0.470079,
    "tau_vL0":         1.05145,
    "c_rhoN":          0.001,
    "k_rhoN_rec":      3.47726e-05,
}


def default_input_file():
    """The workbook the 0-D reads, preferring ``input_parameters.xlsx``."""
    p = paths.INPUT_DIR / "input_parameters.xlsx"
    return p if p.exists() else paths.INPUT_DIR / "Zr_input_parameters.xlsx"


def apply_overrides(idata, overrides=None, verbose=False):
    """Route each override to the sheet that already defines it.

    A key the workbook does not carry at all is added to ``model_params``,
    which is what the notebook does and what the 11 missing parameters need.
    Returns the list of (key, sheet, old, new) actually applied.
    """
    overrides = REFERENCE_OVERRIDES if overrides is None else overrides
    sheets = {"Material_Environment": idata.material_params,
              "Physical_Properties":  idata.physical_props,
              "Model_Parameters":     idata.model_params}
    applied = []
    for k, v in overrides.items():
        if v is None:
            continue
        tgt = next((n for n, d in sheets.items() if k in d), None)
        if tgt is None:
            idata.model_params[k] = v
            applied.append((k, "Model_Parameters (new)", None, v))
        else:
            applied.append((k, tgt, sheets[tgt][k], v))
            sheets[tgt][k] = v
    idata.calculate_derived_parameters()
    if verbose:
        for k, s, old, new in applied:
            o = "n/a" if old is None else f"{old:.6g}"
            print(f"  {k:<16} {s:<26} {o:>14} -> {new:g}")
    return applied


# The atomic volume this model carried before the hcp correction: V_cell/4,
# i.e. four atoms in a primitive cell that holds two. Kept so a pre-correction
# run can be reproduced exactly -- build_sim(extra={"Omega": LEGACY_OMEGA}) --
# and so the wrong number appears in exactly one place, named.
LEGACY_OMEGA = 1.2e-29   # [m^3]


def build_sim(input_file=None, T=None, G=None, extra=None, verbose=False,
              physical_omega=False, check_omega=True):
    """The calibrated 0-D model chain.

    ``T`` and ``G`` are pinned last so a caller can set the irradiation
    condition without editing the fitted set.

    THE ATOMIC VOLUME WAS CORRECTED. The workbook and ``atomicVolume_SI`` in
    the MoDELib material file both now carry the hcp Zr value,
    ``(sqrt(3)/2) a^2 c / 2 = 2.326553e-29 m^3`` with a = b_a and c = b_c and
    TWO atoms per primitive cell, cross-checked against ``M_Zr/(rho_SI*N_A) =
    2.3233e-29`` to 0.14%. It previously read ``1.2e-29`` -- ``V_cell/4``,
    counting four atoms in a cell that holds two, 0.516x the physical value --
    and ``input_data``'s fallback default read 1.4e-29, a third number for one
    lattice constant.

    ``physical_omega`` is therefore a no-op now, kept so callers that set it do
    not break. To reproduce a run made BEFORE the correction, pass the old
    value explicitly:

        build_sim(extra={"Omega": LEGACY_OMEGA})

    and set ``atomicVolume_SI`` back on the 3-D side too -- the two must always
    match, because the 0-D <-> 3-D bridge converts through that key.

    **THE CALIBRATION HAS NOT BEEN REFITTED.** Omega is not a reporting factor:
    every number density is C/Omega, line density is 2*pi*r*C/Omega, the
    radius prefactors are sqrt(Omega/(pi*b)) and emission carries
    exp(sigma*Omega/kT). Measured on the fitted set at 10 dpa, the correction
    gives N_a 0.483x, N_c 0.516x, c_a 0.522x, Cv 1.047x, Ci 1.059x -- the loop
    densities HALVE. Those 28 parameters were fitted against experimental loop
    densities at the old Omega, so the model now carries the right lattice
    constant and a stale fit. Refit before comparing to experiment.

    ``check_omega`` warns when the workbook disagrees with the lattice-derived
    value. It is silent now, and is what would catch a regression.
    """
    input_file = str(input_file or default_input_file())
    idata = InputData(input_file)

    ov = dict(REFERENCE_OVERRIDES)
    if T is not None:
        ov["T"] = float(T)
    if G is not None:
        ov["G"] = float(G)
    if physical_omega:
        _, ov["Omega"], _ = idata.check_atomic_volume(verbose=False)
    if extra:
        ov.update(extra)
    applied = apply_overrides(idata, ov, verbose=verbose)
    if check_omega and not physical_omega:
        idata.check_atomic_volume(verbose=verbose)

    rr = ReactionRates(idata)
    req = RateEquations(idata, rr)
    sim = types.SimpleNamespace(input_data=idata, reaction_rates=rr,
                                rate_equations=req, input_file=input_file,
                                overrides_applied=applied)
    return sim
