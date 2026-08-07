"""
pre_process.py – Input resolution, parameter validation, and initial conditions.

Handles everything that must happen *before* the ODE solver is invoked:
  - locating the Excel input file
  - validating parameter ranges
  - building and checking the initial condition vector
"""

import warnings
from pathlib import Path

import numpy as np


def find_input_file(input_file=None):
    """
    Resolve and return a Path to the input Excel file.

    Parameters
    ----------
    input_file : str, Path, or None
        If provided, that path is used directly (must exist).
        If None, common locations are searched automatically.

    Returns
    -------
    Path – resolved path to input_parameters.xlsx
    """
    if input_file is not None:
        p = Path(input_file)
        if p.exists():
            return p
        raise FileNotFoundError(f"Input file not found: {p}")

    current_dir = Path.cwd()
    candidates = [
        current_dir / 'input' / 'input_parameters.xlsx',
        current_dir / 'input_parameters.xlsx',
        current_dir.parent / 'input' / 'input_parameters.xlsx',
        Path(__file__).parent.parent / 'input' / 'input_parameters.xlsx',
    ]
    for path in candidates:
        if path.exists():
            print(f"✓ Found input file at: {path}")
            return path

    raise FileNotFoundError(
        "Could not find input_parameters.xlsx in:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def validate_setup(input_data):
    """
    Warn about parameter values outside typical physical ranges.

    Parameters
    ----------
    input_data : InputData
    """
    T   = input_data.material_params['T']
    G   = input_data.material_params['G']
    rho = input_data.material_params['rho']

    if not (200 <= T <= 1000):
        warnings.warn(f"Temperature {T} K outside typical range [200–1000 K]")
    if not (1e-8 <= G <= 1e-3):
        warnings.warn(f"Dose rate {G} dpa/s outside typical range [1e-8 – 1e-3]")
    if not (1e12 <= rho <= 1e16):
        warnings.warn(f"Dislocation density {rho} m⁻² outside typical range [1e12 – 1e16]")

    C_v_eq = input_data.derived['C_v_eq']
    if C_v_eq > 1e-6:
        warnings.warn(f"Equilibrium vacancy concentration {C_v_eq:.2e} seems high")

    # Warn when physically-meaningful parameters are absent from the input file
    # and a hard-coded default will be substituted (see cpp_bridge.collect_solver_args
    # and the .get(...) fallbacks there). Silent defaults can mask an incomplete sheet.
    _warn_missing(input_data.physical_props, 'recom', 1.0, 'Physical_Properties')
    for _bias, _default in (('Z_N', 1.05), ('Z_iL', 1.05), ('Z_vL', 1.05)):
        _warn_missing(input_data.model_params, _bias, _default, 'Model_Parameters')
    # The aligned vacancy-loop lifetime is now DERIVED (tau_avL0/E_a_avL,
    # defaulting to the fitted tau_vL0/E_a_vL); the legacy constant 'tau_avL'
    # override has been removed, so no warning is needed here.

    print("✓ Setup validation complete.")


def _warn_missing(params, key, default, sheet):
    """Emit a warning when an optional parameter is absent and a default is used."""
    if key not in params:
        warnings.warn(
            f"Parameter '{key}' not found in {sheet}; using default {default}"
        )


def get_initial_conditions(rate_equations, custom_values=None):
    """
    Build and validate the initial condition vector.

    Parameters
    ----------
    rate_equations : RateEquations
    custom_values  : dict, optional – overrides for specific species

    Returns
    -------
    y0 : numpy.ndarray
    """
    y0 = rate_equations.get_initial_conditions(custom_values)

    # The evolving rho_N row carries a dislocation density [m^-2] (~1e14), not an
    # atom fraction, so exclude it from the concentration sanity checks.
    idx_rhoN = getattr(rate_equations, 'idx_rhoN', None)
    y_conc = np.delete(y0, idx_rhoN) if idx_rhoN is not None else y0

    if np.any(y_conc < 0):
        raise ValueError("Negative initial concentrations detected")
    if np.any(np.isnan(y0)) or np.any(np.isinf(y0)):
        raise ValueError("Invalid initial concentrations (NaN or Inf)")
    if np.any(y_conc > 1.0):
        warnings.warn("Some initial concentrations > 1 (100%)")

    return y0
