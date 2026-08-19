"""inputs.py — write MoDELib's ``inputFiles/`` from a SimulationConfig.

Three files carry everything a run's setup can vary:

===========================  ==================================================
``DD.txt``                   the traits: step count, output cadence, which
                             solvers run, the fast-step settings
``polycrystal.txt``          material, mesh, temperature, ``F`` (the physical
                             size), and the periodic face list -- i.e. the
                             boundary conditions on the cluster-dynamics field
``ElasticDeformation.txt``   the mechanical load, in Voigt order 11 22 33 12 23 13
===========================  ==================================================

Editing them was previously spread over ``setup_domain.stage``,
``setup_standalone.write_dd``, ``modelib_qssa.set_dd_scalar`` and a block of
inline regex in the coupling notebook, with only the notebook touching
``absoluteTemperature`` and nothing at all touching the load or the periodic
faces. This module is the one place that writes them.

Two traps are handled here rather than left to callers:

* **Line endings.** MoDELib's parser treats a trailing CR as part of the value,
  so a CRLF file silently mis-reads the last field on every line. Everything
  copied out of ``Library/`` is LF-converted.
* **Stress units.** MoDELib normalizes stress by ``mu_SI``; the numbers in
  ``ElasticDeformation.txt`` are multiples of the shear modulus, not Pa. For Zr
  (``mu0_SI = 33e9``) writing MPa directly would overstate the load 33 000x.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["set_dd_scalar", "get_dd_scalar", "write_dd", "write_polycrystal",
           "write_elastic_deformation", "read_material_scalar", "material_mu_SI",
           "copy_lf", "clear_empty_F", "FAST_STEP_SETTINGS", "VOIGT",
           "elastic_is_trivial"]

VOIGT = ("11", "22", "33", "12", "23", "13")

# The fast (mobile) step must be configured identically on every route, or a
# route-to-route comparison measures a solver defect instead of the physics.
# MoDELib's default `mobileSolverClampInLoop=1` makes the mobile solve a
# projected Newton iteration that does not converge: it caps out at 30
# iterations with 53.6% error in Ci, while deferring the clamp converges in 5
# with a quadratic tail.
FAST_STEP_SETTINGS = {
    "mobileSolverClampInLoop": "0",
    "mobileSolverMaxIterations": "50",
}


def elastic_is_trivial(sim_dir):
    """True when the elastic solve of this case can only return zero.

    MoDELib solves ElasticDeformation on every DDomp call whether or not there
    is anything to solve. On the coupled fast step that is pure waste whenever

      * ``useDislocations=0`` -- no dislocation eigenstrain to correct, which
        is the case for every CD-only run, loops being continuum fields here;
      * every entry of ElasticDeformation.txt is zero -- no applied stress,
        stress rate, strain or strain rate.

    Both together mean the displacement field is identically zero, so the
    solve is a ~3n_cd-dof sparse direct factorization of a problem whose answer
    is known. It measured 55.6 s of a 149.5 s fast solve on the 200 nm case:
    37% of the step, and it scales the same way the mobile solve does.

    Returns ``(trivial, reason)`` so a caller can say why it did or did not
    switch the solve off, rather than silently changing the physics.
    """
    sim_dir = Path(sim_dir)
    dd = sim_dir / "inputFiles" / "DD.txt"
    ed = sim_dir / "inputFiles" / "ElasticDeformation.txt"
    if not dd.is_file():
        return False, f"no {dd}"

    if (get_dd_scalar(dd, "useDislocations") or "1").strip() not in ("0", "0.0"):
        return False, "useDislocations != 0: there is an eigenstrain to correct"
    if not ed.is_file():
        return False, f"no {ed}"

    txt = ed.read_text(encoding="utf-8", errors="replace")
    nonzero = []
    for key in ("ExternalStress0", "ExternalStressRate", "ExternalStrain0",
                "ExternalStrainRate"):
        m = re.search(rf"^\s*{re.escape(key)}\s*=([^;#\n]*)", txt, re.M)
        if not m:
            continue
        vals = [float(v) for v in m.group(1).split()]
        if any(v != 0.0 for v in vals):
            nonzero.append(key)
    if nonzero:
        return False, f"nonzero load: {', '.join(nonzero)}"
    return True, "useDislocations=0 and every applied load is zero"


def clear_empty_F(sim_dir):
    """Remove zero-byte `F/F_0.txt` and `F/F_labels.txt`. Returns what it removed.

    A DDomp run that dies before writing anything leaves both files at zero
    length, and the NEXT DDomp segfaults reading them: the loader reports
    "(0 entries)" and the deformation-gradient row is then indexed anyway. So a
    first failure -- for any reason at all -- turns every later attempt into a
    different and much more confusing failure. Call this before launching.
    """
    gone = []
    for stub in ("F_0.txt", "F_labels.txt"):
        f = Path(sim_dir) / "F" / stub
        if f.is_file() and f.stat().st_size == 0:
            f.unlink()
            gone.append(stub)
    return gone


def copy_lf(src, dst):
    """Copy a text input file, normalizing CRLF to LF."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(src).read_bytes().replace(b"\r\n", b"\n"))
    return dst


# ── DD.txt ───────────────────────────────────────────────────────────────────

def set_dd_scalar(dd_file, key, value):
    """Set ``key=value;`` in a MoDELib DD.txt, appending the key if absent.

    MoDELib's own ``modlibUtils.setInputVariable`` requires the key to exist
    already; ``useImmobileSolver`` will not exist in a DD.txt written before
    the flag did, so this appends instead of failing.
    """
    dd_file = Path(dd_file)
    txt = dd_file.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=)([^;]*)(;.*)$", re.M)
    if pat.search(txt):
        txt = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", txt, count=1)
    else:
        txt = txt.rstrip("\n") + f"\n{key}={value}; # set by dislocluster_code\n"
    dd_file.write_text(txt, encoding="utf-8")


def get_dd_scalar(dd_file, key, default=None):
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^;#\n]+)",
                  Path(dd_file).read_text(encoding="utf-8", errors="replace"),
                  re.M)
    return m.group(1).strip() if m else default


def write_dd(sim_dir, n_steps, use_immobile=1, fast_step=None,
             output_frequency=1):
    """Set the dose schedule, start from evl_0, and pin the mobile solver.

    ``startAtTimeStep`` is pinned to 0 rather than left at -1: -1 restarts from
    whatever the highest surviving ``evl_N`` is, which in a re-staged directory
    silently double-counts dose.

    Returns ``dtMax``.
    """
    dd = Path(sim_dir) / "inputFiles" / "DD.txt"
    for key, val in (("Nsteps", n_steps),
                     ("outputFrequency", output_frequency),
                     ("startAtTimeStep", 0),
                     ("useImmobileSolver", int(use_immobile))):
        set_dd_scalar(dd, key, val)

    for k, v in (FAST_STEP_SETTINGS if fast_step is None else fast_step).items():
        set_dd_scalar(dd, k, v)
        if get_dd_scalar(dd, k) != v:
            raise RuntimeError(f"failed to pin {k}={v} in {dd}")

    txt = dd.read_text(encoding="utf-8", errors="replace")
    return float(re.search(r"^dtMax\s*=\s*([-\d.eE+]+)", txt, re.M).group(1))


# ── polycrystal.txt ──────────────────────────────────────────────────────────

def _f_block(F):
    # 17 significant digits rather than 10. On a PERIODIC case MoDELib maps each
    # periodic shift into lattice coordinates and requires an integer, so a
    # truncated F is not a rounding difference but a startup failure: ten digits
    # turn a 971*c/a = 1548.1888112 edge into 1548.188811, short of the lattice
    # vector by 1.25e-7, and the run dies with "Input vector is not a lattice
    # vector". Non-periodic cases never noticed, which is why this stood.
    F = np.asarray(F, dtype=float)
    return (f"F={F[0, 0]:.17g} {F[0, 1]:.17g} {F[0, 2]:.17g}\n"
            f"  {F[1, 0]:.17g} {F[1, 1]:.17g} {F[1, 2]:.17g}\n"
            f"  {F[2, 0]:.17g} {F[2, 1]:.17g} {F[2, 2]:.17g}; "
            f"# mesh deformation gradient, x = F*(X-X0)")


def write_polycrystal(sim_dir, mesh_name=None, F=None, temperature_K=None,
                      material_file=None, periodic_face_ids=None):
    """Patch polycrystal.txt in place. Every argument is optional.

    The mesh is written normalized to the unit box, so ``F`` is the only place
    the physical dimension of the domain appears.
    """
    pc = Path(sim_dir) / "inputFiles" / "polycrystal.txt"
    txt = pc.read_text(encoding="utf-8", errors="replace")

    if mesh_name is not None:
        txt = re.sub(r"^meshFile\s*=.*$", f"meshFile={mesh_name}; # mesh file",
                     txt, count=1, flags=re.M)
    if material_file is not None:
        txt = re.sub(r"^materialFile\s*=.*$", f"materialFile={material_file};",
                     txt, count=1, flags=re.M)
    if temperature_K is not None:
        txt = re.sub(r"^absoluteTemperature\s*=[^;]*;.*$",
                     f"absoluteTemperature={temperature_K:g}; "
                     f"# [K] simulation temperature",
                     txt, count=1, flags=re.M)
    if F is not None:
        # `.*?` spans the three lines of the F block up to its first ';'; the
        # trailing comment must then be matched with [^\n]* rather than `.*`,
        # or DOTALL lets it run greedily to the last ';' in the file and
        # delete periodicFaceIDs and C2G1 along with it.
        txt, n = re.subn(r"^F=.*?;[^\n]*$", _f_block(F), txt, count=1,
                         flags=re.M | re.S)
        if n != 1:
            raise KeyError("polycrystal.txt has no F=...; block to replace")
    if periodic_face_ids is not None:
        ids = " ".join(str(int(i)) for i in periodic_face_ids)
        note = ("# non-periodic: Dirichlet on the whole surface"
                if not ids else "# periodic face IDs")
        txt = re.sub(r"^periodicFaceIDs\s*=[^;]*;.*$",
                     f"periodicFaceIDs={ids}; {note}", txt, count=1, flags=re.M)

    for key in ("periodicFaceIDs", "C2G1", "X0", "meshFile", "materialFile"):
        if not re.search(rf"^{key}\s*=", txt, re.M):
            raise KeyError(f"polycrystal.txt lost its {key} line")
    pc.write_text(txt, encoding="utf-8")
    return pc


# ── the material file ────────────────────────────────────────────────────────

def read_material_scalar(material_file, key, default=None):
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([-\d.eE+]+)",
                  Path(material_file).read_text(encoding="utf-8",
                                                errors="replace"), re.M)
    return float(m.group(1)) if m else default


def material_mu_SI(material_file, temperature_K):
    """``mu = mu0_SI + mu1_SI * T`` [Pa] — the stress normalization."""
    mu0 = read_material_scalar(material_file, "mu0_SI")
    mu1 = read_material_scalar(material_file, "mu1_SI", 0.0) or 0.0
    if mu0 is None:
        raise KeyError(f"{material_file} carries no mu0_SI")
    return mu0 + mu1 * float(temperature_K)


# ── ElasticDeformation.txt ───────────────────────────────────────────────────

_ED_TEMPLATE = """\
ExternalStress0={s0}; # applied stress, in units of mu
ExternalStressRate={sr}; # [1/(b/cs)]
ExternalStrain0={g0};
ExternalStrainRate={gr};
stiffnessRatio={sk};

# Voigt order is 11,22,33,12,23,13
# stiffnessRatio =0 means pure stress control,
# stiffnessRatio =infinity (such as 1e20) means pure strain control
#
# Stress is normalized by mu_SI (MoDELib carries a dimensionless `mu`), so
# these are multiples of the shear modulus, NOT Pa. Written by
# dislocluster_code.staging.inputs from BOUNDARY['applied_stress_MPa'] at
# mu_SI = {mu:.6g} Pa.
"""


def _row(v):
    return " ".join(f"{float(x):.10g}" for x in v)


def write_elastic_deformation(sim_dir, boundary, mu_SI):
    """Write ElasticDeformation.txt from a :class:`config.Boundary`.

    ``boundary`` states the load in MPa; MoDELib wants multiples of ``mu``.
    """
    ed = Path(sim_dir) / "inputFiles" / "ElasticDeformation.txt"
    ed.parent.mkdir(parents=True, exist_ok=True)
    ed.write_text(_ED_TEMPLATE.format(
        s0=_row(boundary.stress_in_mu(mu_SI)),
        sr=_row(boundary.stress_rate_in_mu(mu_SI)),
        g0=_row(boundary.applied_strain),
        gr=_row(boundary.applied_strain_rate),
        sk=_row(boundary.stiffness_ratio),
        mu=mu_SI), encoding="utf-8")
    return ed
