"""
anisotropy.py -- write the anisotropic diffusion tensor from one anisotropy
factor per mobile species.

WHY THIS EXISTS RATHER THAN EDITING THE MATERIAL FILE BY HAND
-------------------------------------------------------------
The diffusional anisotropy difference appears in the material file TWICE, in
two unrelated forms, and the two must agree:

  mobileSpeciesEnergyMigration_eV   the tensor the FEM fast solve diffuses with
                                    (FluxMatrix takes the whole 3x3 block) and
                                    that the Green's function would use
  dadAnisotropy                     p_m in loopDADbias(), the closed-form
                                    capture efficiencies the CONTINUUM immobile
                                    families absorb through

Before this module they were independent: the migration energies were isotropic
while dadAnisotropy carried fitted values, so the code simultaneously believed
that diffusion is isotropic and that the sinks are biased by its anisotropy.
Those two statements cannot both be true. Generating both keys from one p_m per
species makes them consistent by construction and removes the possibility of
drift -- which is the same reason `paths.workbook_drift()` exists.

THE RECIPE
----------
The anisotropy factor is Woo's

    p_m = ( D<c> / D<a> )^(1/6)

with D<c> along the basal pole and D<a> in the basal plane. Varying p_m at
FIXED effective diffusivity D_eff = (D<a>^2 D<c>)^(1/3) -- which is what keeps
the overall mobility, and hence the existing calibration, unchanged -- fixes
both components uniquely:

    E_m<11,22> = E_m_eff + 2 kT ln(p_m)
    E_m<33>    = E_m_eff - 4 kT ln(p_m)

This reproduces Table 2 of Li et al., JMPS 206 (2026) 106366 to the last quoted
digit at their T = 553 K, and matches the di-interstitial row of the paper's own
`greatWhitePlots/sims/simA/inputFiles/Zr3.txt` exactly.

NOTE THE TEMPERATURE DEPENDENCE. Because the anisotropy is stored as ENERGIES
rather than as a ratio, p_m(T) = exp(-dE_m/6kT) is temperature-dependent -- a
pair fixed to give 0.7 at 573 K gives 0.649 at 473 K and 0.738 at 673 K. That is
physically right, anisotropy weakening as temperature rises, and it repairs the
previous fitted DAD, which was frozen at one temperature.

USAGE
-----
    python -m dislocluster_code.staging.anisotropy --show
    python -m dislocluster_code.staging.anisotropy --p-m 1.178808 0.913720 \\
        0.913720 0.913720 --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling.field import read_material_scalar

KB_EV = 8.617333262e-5          # eV/K

# The anisotropy this repository runs with by default.
#
# These are the values `dadAnisotropy` already carried, fitted so that
# loopDADbias() reproduces the 0-D symmetric bias forms. Adopting them as the
# TENSOR's anisotropy makes the two representations agree for the first time
# WITHOUT moving the calibration: the capture efficiencies are unchanged, and
# what changes is that the fast solve now diffuses with the tensor those
# efficiencies were always supposed to describe.
#
# It is NOT the paper's choice. Li et al. give the DAD to di-interstitials only,
# with vacancies and single interstitials isotropic, and find p_m < 0.8 is needed
# for simultaneous <a> expansion and <c> contraction. Moving to that requires
# refitting the 28-parameter set, which is deferred pending experimental data.
DEFAULT_PM = (1.178808, 0.913720, 0.913720, 0.913720)   # v, i, 2i, 3i


def split_migration(e_eff, p_m, T):
    """``(E_m<11,22>, E_m<33>)`` in eV for one species."""
    kT = KB_EV * float(T)
    ln_p = np.log(float(p_m))
    return e_eff + 2.0 * kT * ln_p, e_eff - 4.0 * kT * ln_p


def anisotropy_of(e11, e33, T):
    """Inverse of :func:`split_migration` -- the p_m a pair of energies gives."""
    return float(np.exp(-(e33 - e11) / (6.0 * KB_EV * float(T))))


def read_migration(material_file):
    """``(n_species, 6)`` array of the migration-energy components, in eV."""
    txt = Path(material_file).read_text(encoding="utf-8")
    m = re.search(r"^mobileSpeciesEnergyMigration_eV\s*=(.*?);",
                  txt, re.S | re.M)
    if not m:
        raise KeyError("mobileSpeciesEnergyMigration_eV not found")
    vals = [float(x) for x in m.group(1).split()]
    if len(vals) % 6:
        raise ValueError(f"expected a multiple of 6 components, got {len(vals)}")
    return np.array(vals, float).reshape(-1, 6)


def effective_energies(material_file, T):
    """``E_m_eff`` per species -- the isotropic energy giving the same D_eff.

    ``D_eff = (D<a>^2 D<c>)^(1/3)`` means ``E_m_eff = (2 E11 + E33)/3``, so this
    is exact whether the file is currently isotropic or not, and applying
    :func:`split_migration` to it is idempotent: re-deriving at the same p_m
    reproduces the same numbers.
    """
    E = read_migration(material_file)
    return (2.0 * E[:, 0] + E[:, 5]) / 3.0


def current_anisotropy(material_file, T):
    """``p_m`` per species as the file's migration energies currently imply."""
    E = read_migration(material_file)
    return np.array([anisotropy_of(e[0], e[5], T) for e in E])


def _fmt_rows(rows, key, comment):
    body = "\n       ".join(" ".join(f"{v:.10g}" for v in r) for r in rows)
    return f"{key}={body};{comment}"


def apply(p_m=DEFAULT_PM, material_file=None, T=None, dry_run=False):
    """Rewrite the migration energies and `dadAnisotropy` from ``p_m``.

    Returns ``(old_text, new_text, table)``. ``table`` is one row per species:
    ``(p_m, E_eff, E11, E33)``.
    """
    mf = Path(material_file or paths.MODELIB_MATERIAL)
    txt = mf.read_text(encoding="utf-8")
    T = float(T if T is not None
              else read_material_scalar(mf, "absoluteTemperature")
              if "absoluteTemperature" in txt else 573.0)

    e_eff = effective_energies(mf, T)
    p_m = np.asarray(p_m, float)
    if p_m.shape != e_eff.shape:
        raise ValueError(f"p_m has {p_m.shape[0]} entries, the material file "
                         f"has {e_eff.shape[0]} mobile species")
    if np.any(p_m <= 0.0):
        raise ValueError("p_m must be positive")

    rows, table = [], []
    for e, p in zip(e_eff, p_m):
        e11, e33 = split_migration(e, p, T)
        rows.append([e11, 0.0, 0.0, e11, 0.0, e33])
        table.append((float(p), float(e), float(e11), float(e33)))

    new = re.sub(
        r"^mobileSpeciesEnergyMigration_eV\s*=.*?;",
        _fmt_rows(rows, "mobileSpeciesEnergyMigration_eV",
                  f"  # [eV] components [11 12 13 22 23 33]. GENERATED by "
                  f"staging/anisotropy.py from p_m at T={T:g} K -- edit p_m, "
                  f"not these."),
        txt, count=1, flags=re.S | re.M)

    new = re.sub(
        r"^dadAnisotropy\s*=.*?;",
        "dadAnisotropy=" + " ".join(f"{p:.10g}" for p in p_m)
        + ";  # p_m per mobile species. GENERATED with the migration energies "
          "above from one source, so the tensor and the capture efficiencies "
          "cannot disagree.",
        new, count=1, flags=re.S | re.M)

    if not dry_run:
        mf.write_text(new, encoding="utf-8")
    return txt, new, table


def describe(material_file=None, T=None):
    mf = Path(material_file or paths.MODELIB_MATERIAL)
    T = float(T if T is not None else 573.0)
    E = read_migration(mf)
    p_now = current_anisotropy(mf, T)
    try:
        dad = [float(x) for x in
               re.search(r"^dadAnisotropy\s*=(.*?);",
                         mf.read_text(encoding="utf-8"),
                         re.S | re.M).group(1).split()]
    except Exception:
        dad = [float("nan")] * len(p_now)
    names = ["v", "i", "2i", "3i"][:len(p_now)]
    L = [f"{mf.name}  at T = {T:g} K", "",
         f"{'species':>8} {'E11=E22':>10} {'E33':>10} {'E_eff':>10} "
         f"{'p_m(tensor)':>12} {'dadAnisotropy':>14} {'agree':>6}"]
    for nm, e, p, d in zip(names, E, p_now, dad):
        eff = (2 * e[0] + e[5]) / 3
        ok = "yes" if abs(p - d) < 1e-6 else "NO"
        L.append(f"{nm:>8} {e[0]:10.6f} {e[5]:10.6f} {eff:10.6f} "
                 f"{p:12.6f} {d:14.6f} {ok:>6}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-m", nargs="*", type=float, default=None,
                    help="anisotropy factor per mobile species (v i 2i 3i)")
    ap.add_argument("--material", default=None)
    ap.add_argument("--temperature", type=float, default=573.0)
    ap.add_argument("--apply", action="store_true",
                    help="write the file; without it, only the diff is shown")
    ap.add_argument("--show", action="store_true",
                    help="report the file's current state and exit")
    args = ap.parse_args(argv)

    if args.show or args.p_m is None:
        print(describe(args.material, args.temperature))
        return 0

    old, new, table = apply(args.p_m, args.material, args.temperature,
                            dry_run=not args.apply)
    print(f"{'species':>8} {'p_m':>10} {'E_eff':>10} {'E11=E22':>10} "
          f"{'E33':>10}")
    for nm, (p, e, e11, e33) in zip(["v", "i", "2i", "3i"], table):
        print(f"{nm:>8} {p:10.6f} {e:10.6f} {e11:10.6f} {e33:10.6f}")
    print()
    print("WRITTEN" if args.apply else "dry run -- pass --apply to write")
    if old == new and args.apply:
        print("(file unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
