"""
modelib_export.py — Export a ZrMicro 0-D cluster-dynamics run as a dose-indexed
closure table for the spatially-resolved SR-CD solver MoDELib2-NNL
(https://github.com/mlm335/MoDELib2-NNL).

This is the "Level 1" deliverable of the two-time-scale coupling documented in
    Docs/Formulation/ZrMicro_MoDELib2_two_time_scale_coupling.tex
It serializes, versus dose gamma = G * t:

  * the fast (mobile) point-defect manifold      C_M = [Cv, Ci, C2i, C3i]
  * the slow (immobile) loop manifold, resolved into the THREE crystallographic
    prism <a> interstitial-loop variants (a1, a2, a3) and the single basal <c>
    vacancy-loop family — i.e. "Option A" of the coupling document, WITHOUT the
    aligned/non-aligned doubling carried internally by ZrMicro.
  * the evolving network dislocation density rho_N and the <c>-cluster
    morphology variable n_v (vacancies per c-loop).

ZrMicro stores interstitial <a> loops as a non-aligned + stress-aligned pair
(CiL/CaiL, CiL_i/CaiL_i) and vacancy <c> loops likewise (CvL/CavL, CvL_v/CavL_v).
For the spatial coupling we (a) collapse each pair into a single per-species
total and (b) re-partition the interstitial total among the three prism variants
by the resolved-stress Boltzmann weights w_k. With no resolved stress the weights
are 1/3 each (isotropic) and all three variants share the 0-D mean radius; under
a resolved normal stress sigma_k on prism-plane normal a_k the weights tilt
toward the more-stressed variants. MoDELib then evolves the variants with their
own habit-plane orientation, using these tables for initialization, Newton
initial guesses, and calibration of the shared scalar kinetic coefficients.

Two files are written:
  <out>/modelib_cd_table.txt   whitespace-delimited columns, '#'-commented header
  <out>/modelib_cd_meta.json   scalar material/geometry metadata + column schema

Both are plain text and trivially parsed by MoDELib's C++ I/O, numpy, or pandas.

Typical use (from the ZrMicro notebook, after process_solution):

    from dislocluster_code.coupling.export import export_for_modelib
    export_for_modelib(results, input_data, run_dir)                 # zero-stress
    export_for_modelib(results, input_data, run_dir,                 # uniaxial
                       sigma_a_resolved=(sx, sy, sz))                #   along a_k
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# eV -> J
_EV_J = 1.602176634e-19
# Boltzmann constant in eV/K (matches input_data.py)
_KB_EV = 8.617e-5

# Default prism <a> habit normals (loop Burgers directions) and basal <c> normal,
# expressed in an orthonormal crystal frame with c along z. a1/a2/a3 are the three
# <11-20>-type a-directions at 120 deg in the basal plane.
_DEFAULT_BASIS = {
    "a1": [1.0, 0.0, 0.0],
    "a2": [-0.5, 0.8660254037844386, 0.0],
    "a3": [-0.5, -0.8660254037844386, 0.0],
    "c":  [0.0, 0.0, 1.0],
}


def _variant_weights(sigma_a_resolved, Omega, T):
    """Resolved-stress Boltzmann weights w_k for the three prism <a> variants.

    w_k = exp(sigma_k * Omega / kT) / sum_j exp(sigma_j * Omega / kT)

    sigma_a_resolved : 3-tuple of resolved normal stresses [Pa] on the a1/a2/a3
                       habit normals, or None for the isotropic 1/3 split.
    """
    if sigma_a_resolved is None:
        return np.array([1.0, 1.0, 1.0]) / 3.0
    s = np.asarray(sigma_a_resolved, dtype=float)
    kT_J = _KB_EV * float(T) * _EV_J
    x = s * float(Omega) / kT_J
    x = x - np.max(x)            # stabilize the exponential
    w = np.exp(x)
    return w / np.sum(w)


def _safe_radius(content, number, length_scale, r_floor=5e-9):
    """Mean loop radius r = length_scale * sqrt(content/number), guarded.

    Mirrors post_process.calculate_derived_quantities (Eqs. 54-57 of the model):
    falls back to r_floor where the loop number density is below the floor.
    """
    number = np.asarray(number, dtype=float)
    content = np.asarray(content, dtype=float)
    ok = number > 1e-20
    r = np.where(
        ok,
        length_scale * np.sqrt(np.maximum(content, 0.0) / np.maximum(number, 1e-20)),
        r_floor,
    )
    return r


def build_modelib_fields(results, input_data, sigma_a_resolved=None):
    """Assemble the dose-indexed, crystallographically-resolved field dictionary.

    Returns a dict of 1-D numpy arrays (all the same length as the dose axis)
    plus a scalar 'meta' sub-dict. This is the in-memory form; write_* functions
    serialize it. Kept separate so callers can post-process before writing.
    """
    conc = results["concentrations"]
    time = np.asarray(results["time"], dtype=float)

    G = float(input_data.material_params["G"])
    T = float(input_data.material_params["T"])
    Omega = float(input_data.physical_props["Omega"])
    b_a = float(input_data.physical_props["b_a"])
    b_c = float(input_data.physical_props["b_c"])
    l_a = float(input_data.derived["l_a"])
    l_c = float(input_data.derived["l_c"])

    # Dose [dpa] = G [dpa/s] * t [s]
    gamma = G * time

    # ----- collapse ZrMicro's aligned/non-aligned pairs into per-species totals
    # (Option A: we do NOT carry the aligned/non-aligned split downstream.)
    N_aL = np.asarray(conc["CiL"], float) + np.asarray(conc["CaiL"], float)
    c_aL = np.asarray(conc["CiL_i"], float) + np.asarray(conc["CaiL_i"], float)
    N_cL = np.asarray(conc["CvL"], float) + np.asarray(conc["CavL"], float)
    c_cL = np.asarray(conc["CvL_v"], float) + np.asarray(conc["CavL_v"], float)

    # mean radii (variant-independent in this lumped partition) and morphology
    r_aL = _safe_radius(c_aL, N_aL, l_a)
    r_cL = _safe_radius(c_cL, N_cL, l_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_v = np.where(N_cL > 1e-20, np.abs(c_cL) / np.maximum(N_cL, 1e-20), 0.0)

    # ----- resolve the interstitial <a> total into a1/a2/a3 by stress weights
    w = _variant_weights(sigma_a_resolved, Omega, T)

    # number densities and contents -> physical units [m^-3]
    inv_Om = 1.0 / Omega
    fields = {
        "gamma_dpa": gamma,
        "time_s": time,
        # fast mobile manifold (atom fraction)
        "Cv": np.asarray(conc["Cv"], float),
        "Ci": np.asarray(conc["Ci"], float),
        "C2i": np.asarray(conc["C2i"], float),
        "C3i": np.asarray(conc["C3i"], float),
    }

    # per-variant interstitial a-loops (number m^-3, content m^-3, radius m)
    for k, name in enumerate(("a1", "a2", "a3")):
        fields[f"N_{name}_m3"] = w[k] * N_aL * inv_Om
        fields[f"c_{name}_m3"] = w[k] * c_aL * inv_Om
        fields[f"r_{name}_m"] = r_aL  # shared mean radius (see module docstring)

    # single basal c-loop family
    fields["N_c_m3"] = N_cL * inv_Om
    fields["c_c_m3"] = c_cL * inv_Om
    fields["r_c_m"] = r_cL
    fields["n_v"] = n_v

    # network dislocation density [m^-2]
    fields["rho_N_m2"] = np.asarray(results.get("rho_N", np.zeros_like(time)), float)

    # geometric sink strength S_geo = 2*pi*r*N [m^-2] per family (MoDELib applies
    # the species bias Z_* from the metadata: <k^2>_{X,m} = Z_{X,m} * S_geo_X)
    for name in ("a1", "a2", "a3"):
        fields[f"Sgeo_{name}_m2"] = 2.0 * np.pi * fields[f"r_{name}_m"] * fields[f"N_{name}_m3"]
    fields["Sgeo_c_m2"] = 2.0 * np.pi * fields["r_c_m"] * fields["N_c_m3"]

    # macroscopic strains for cross-check against the spatial solve
    mech = results.get("mechanical_properties", {})
    fields["strain_a"] = np.asarray(mech.get("strain_a", np.zeros_like(time)), float)
    fields["strain_c"] = np.asarray(mech.get("strain_c", np.zeros_like(time)), float)

    meta = {
        "material": {
            "T_K": T,
            "G_dpa_per_s": G,
            "sigma_n_Pa": float(input_data.material_params.get("sigma_n", 0.0)),
            "rho_N_seed_m2": float(input_data.material_params.get("rho", 0.0)),
        },
        "geometry": {
            "Omega_m3": Omega,
            "b_a_m": b_a,
            "b_c_m": b_c,
            "l_a_m": l_a,
            "l_c_m": l_c,
            "activation_volume_m3": Omega,
        },
        "bias_factors": {
            "Z_i_a": float(input_data.derived.get("Z_i_a", float("nan"))),
            "Z_v_a": float(input_data.derived.get("Z_v_a", float("nan"))),
            "Z_i_c": float(input_data.derived.get("Z_i_c", float("nan"))),
            "Z_v_c": float(input_data.derived.get("Z_v_c", float("nan"))),
            "Z_N": float(input_data.model_params.get("Z_N", 1.1)),
        },
        "variant_weights": {
            "a1": float(w[0]), "a2": float(w[1]), "a3": float(w[2]),
            "sigma_a_resolved_Pa": (list(map(float, sigma_a_resolved))
                                    if sigma_a_resolved is not None else None),
        },
        "habit_normals": dict(_DEFAULT_BASIS),
        "units": {
            "gamma_dpa": "dpa", "time_s": "s",
            "Cv,Ci,C2i,C3i": "atom_fraction",
            "N_*_m3": "m^-3", "c_*_m3": "m^-3 (defect content)",
            "r_*_m": "m", "Sgeo_*_m2": "m^-2 (geometric sink strength 2*pi*r*N)",
            "rho_N_m2": "m^-2", "n_v": "vacancies_per_c_loop",
            "strain_a,strain_c": "dimensionless",
        },
        "notes": (
            "Option-A export: aligned/non-aligned pairs collapsed, interstitial "
            "<a> total split into a1/a2/a3 by resolved-stress Boltzmann weights "
            "(1/3 each at zero stress). a-variants share the 0-D mean radius. "
            "Loop sink strength = Z_{X,m} * Sgeo_X; build R_1 from these locally."
        ),
    }
    fields["meta"] = meta
    return fields


def export_for_modelib(results, input_data, output_dir, *,
                       sigma_a_resolved=None, git_hash=None, stride=1,
                       table_name="modelib_cd_table.txt",
                       meta_name="modelib_cd_meta.json"):
    """Write the MoDELib2-NNL closure table + metadata for a ZrMicro run.

    Parameters
    ----------
    results          : dict from post_process.process_solution / calculate_derived_quantities
    input_data       : InputData
    output_dir       : str | Path — destination directory (created if needed)
    sigma_a_resolved : 3-tuple of resolved normal stresses [Pa] on the a1/a2/a3
                       habit normals (default None -> isotropic 1/3 split)
    git_hash         : optional provenance string recorded in the metadata
    stride           : keep every `stride`-th dose row (thin large solutions)

    Returns
    -------
    (table_path, meta_path) : tuple[Path, Path]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = build_modelib_fields(results, input_data, sigma_a_resolved)
    meta = fields.pop("meta")
    if git_hash is not None:
        meta["git_hash"] = str(git_hash)

    # Fixed, documented column order.
    columns = [
        "gamma_dpa", "time_s",
        "Cv", "Ci", "C2i", "C3i",
        "N_a1_m3", "N_a2_m3", "N_a3_m3", "N_c_m3",
        "c_a1_m3", "c_a2_m3", "c_a3_m3", "c_c_m3",
        "r_a1_m", "r_a2_m", "r_a3_m", "r_c_m",
        "Sgeo_a1_m2", "Sgeo_a2_m2", "Sgeo_a3_m2", "Sgeo_c_m2",
        "rho_N_m2", "n_v", "strain_a", "strain_c",
    ]
    meta["columns"] = columns

    data = np.column_stack([fields[c][::stride] for c in columns])

    header_lines = [
        "ZrMicro -> MoDELib2-NNL cluster-dynamics closure table",
        "Two-time-scale coupling, Option A (a1/a2/a3 prism + basal c loops).",
        f"T = {meta['material']['T_K']} K,  G = {meta['material']['G_dpa_per_s']} dpa/s",
        f"variant weights (a1,a2,a3) = "
        f"{meta['variant_weights']['a1']:.4f}, "
        f"{meta['variant_weights']['a2']:.4f}, "
        f"{meta['variant_weights']['a3']:.4f}",
        "See companion modelib_cd_meta.json for units, bias factors, habit normals.",
        "columns: " + " ".join(columns),
    ]
    header = "\n".join(header_lines)

    table_path = out / table_name
    meta_path = out / meta_name
    np.savetxt(table_path, data, header=header, fmt="%.10e")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ MoDELib export: {table_path}  ({data.shape[0]} dose rows, "
          f"{data.shape[1]} columns)")
    print(f"✓ MoDELib metadata: {meta_path}")
    return table_path, meta_path
