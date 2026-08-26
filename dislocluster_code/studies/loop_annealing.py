"""loop_annealing.py -- thermal annealing of the immobile loop families.

WHAT THIS IS
------------
The helper behind Sec. "Thermal Annealing Simulations" of
``Docs/Formulation/self-consistent/self_consistent_SRCD.tex``. It evaluates, for
each of the eight loop families of Eq. (families):

  * the work to detach one point defect, dE_k/dm  -- fault + ANISOTROPIC elastic
    self-energy, Eq. (dEdm), plus the work the applied stress TENSOR does;
  * the vacancy binding energy E_b^{v,k}, Eq. (Eb);
  * the vacancy concentration the loop core is in equilibrium with, Eq. (cveq);
  * the thermal emission rate per loop, Eq. (loopemission);
  * the radius trajectory R(t) of an ISOLATED loop under a pure anneal, from the
    growth law alone -- no Green's functions, no pair assembly.

It does NOT run a march. Everything here is a closed-form evaluation of the
manuscript equations, so it can be checked by hand; ``verify`` does exactly that.

THE SIGN OF THE STRESS TERM
---------------------------
Emitting a vacancy from a loop ALWAYS adds Omega of material back to the crystal
along the habit normal -- an interstitial loop grows, a vacancy loop shrinks, and
both insert material. So the work the applied stress does per emitted vacancy is
+Sigma_k * Omega with the SAME sign for both characters, and

    c_eq = c_inf * exp{ [ sigma_s(v) * dE_k/dm  +  Sigma_k * Omega ] / kT }.

An earlier revision of the manuscript carried ``sigma_s(v) * (dE/dm - sigma_kk
Omega)``, which is the same thing for the interstitial families and the opposite
sign for the vacancy ones. See :func:`c_eq_vacancy`.

Sigma_k is the resolved stress the *tensor* supplies,

    Sigma_k = (b_k . sigma . n_k) / (b.n)_k,

which equals the normal stress sigma_kk for every PURE EDGE family and does not
for ``c_f``, whose Burgers vector 1/6<20-23> carries a basal partial. That family
therefore responds to basal SHEAR, with weight 2a/(sqrt(3) c) = 0.724 relative to
a c-axis normal stress; ``c_p`` does not respond to it at all.

UNITS
-----
SI throughout, except that energies are reported in eV and radii in nm at the
CLI. Material constants come from ``paths.MODELIB_MATERIAL`` -- never hard-coded
-- with the exception of the per-family crystallography (gamma_k, <K>_k), which
is a property of the formulation's Table (Kfactors) and is not in the material
file. Those are declared once in :data:`CRYSTALLOGRAPHY` and every derived
length is computed from them.

USAGE
-----
    python -m dislocluster_code.studies.loop_annealing verify
    python -m dislocluster_code.studies.loop_annealing binding  [--T 773]
    python -m dislocluster_code.studies.loop_annealing emission [--T 773]
    python -m dislocluster_code.studies.loop_annealing anneal --R0 25 --T 873
    python -m dislocluster_code.studies.loop_annealing figure --out <path.png>
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling.field import (read_material_scalar,
                                              read_material_vector)

KB_EV = 8.617333262e-5          # [eV/K]
EV_J = 1.602176634e-19          # [J/eV]
ALPHA_CORE = 8.0 / math.e ** 2  # so that <K>b^2 R/2 * ln(alpha R/b) is the
                                # standard circular-loop energy with r0 = |b|:
                                # ln(alpha R/b) = ln(8R/b) - 2.


# ── the crystallography, Table (Kfactors) + Table (cstates) ──────────────────
#
# gamma_k   [J/m^2]  the family's own fault energy, 0 for the unfaulted states.
#                    c_f bounds the I1 intrinsic basal fault; ab initio values
#                    for alpha-Zr scatter over ~0.10-0.15 J/m^2 and the fault
#                    term is strictly linear in it, so GAMMA_CF is the one
#                    number in this module worth a sensitivity check.
# Kbar_k    [Pa]     perimeter-averaged anisotropic energy factor, Eq. (Kfactor),
#                    quadrature of the Stroh/Barnett-Lothe form at Fisher &
#                    Renken's 300 K constants. NOT mu/(1-nu).
# bmag_k    [m]      |b_k|, which sets the logarithm's argument.
# bdotn_k            the EDGE component b.n, which converts area into defects.
#                    Expressed as a multiple of (a, c) so it follows the
#                    material file's own lattice constants.
GAMMA_CF = 0.124        # [J/m^2] I1 basal fault
GAMMA_CF_RANGE = (0.100, 0.150)

CRYSTALLOGRAPHY = {
    #  name       character  gamma     Kbar[GPa]  |b|          b.n
    "a_i": dict(character="i", gamma=0.0, Kbar=53.1e9, bmag=("a", 1.0),
                bdotn=("a", 1.0),
                label=r"<a> prismatic, interstitial"),
    "a_v": dict(character="v", gamma=0.0, Kbar=53.1e9, bmag=("a", 1.0),
                bdotn=("a", 1.0),
                label=r"<a> prismatic, vacancy"),
    "c_f": dict(character="v", gamma=GAMMA_CF, Kbar=53.0e9, bmag=("mixed", 1.0),
                bdotn=("c", 0.5),
                label=r"<c> basal, faulted (c/2+p)"),
    "c_p": dict(character="v", gamma=0.0, Kbar=57.8e9, bmag=("c", 1.0),
                bdotn=("c", 1.0),
                label=r"<c> basal, perfect"),
}


@dataclass
class Family:
    """One loop family, with every derived length computed rather than quoted."""
    name: str
    label: str
    character: str          # 'i' or 'v'
    gamma: float            # [J/m^2]
    Kbar: float             # [Pa]
    bmag: float             # [m]  |b|
    bdotn: float            # [m]  b.n, the edge component
    lam: float              # [m]  lambda_k = sqrt(Omega / (pi b.n))
    omega: float            # [m^3]

    @property
    def zeta_v(self) -> float:
        """varsigma_s(v): +1 for a vacancy family, -1 for an interstitial one."""
        return +1.0 if self.character == "v" else -1.0

    def radius(self, m):
        """R = lambda_k sqrt(m), Eq. (radius)."""
        return self.lam * np.sqrt(m)

    def defects(self, R):
        """m = pi R^2 (b.n) / Omega, the inverse of :meth:`radius`."""
        return np.pi * R ** 2 * self.bdotn / self.omega


@dataclass
class Material:
    b: float          # [m]   |b| of 1/3<11-20> = a
    c: float          # [m]   axial lattice parameter
    omega: float      # [m^3]
    Ef_v: float       # [eV]
    D0_v: float       # [m^2/s]  (det D0)^(1/3)
    Em_v: float       # [eV]     (E11+E22+E33)/3, so Dbar = D0 exp(-Em/kT)
    mu: float         # [Pa]
    nu: float
    r_min: float      # [m]  the model's own lower radius floor

    def Dbar_v(self, T):
        """Dbar^v = (det D_v)^(1/3), Eq. (growth)'s orientation average."""
        return self.D0_v * np.exp(-self.Em_v / (KB_EV * T))

    def c_v_inf(self, T):
        """c^{v,eq}_inf = exp(-E^f_v / kT), Eq. (cveq)."""
        return np.exp(-self.Ef_v / (KB_EV * T))


def load_material(material_file=None) -> Material:
    """Every constant from ``Zr3d_ghoniem.txt``; nothing hard-coded."""
    mf = material_file or paths.MODELIB_MATERIAL
    b = read_material_scalar(mf, "b_SI")
    Em = read_material_vector(mf, "mobileSpeciesEnergyMigration_eV")
    D0 = read_material_vector(mf, "mobileSpeciesD0_SI")
    Ef = read_material_vector(mf, "mobileSpeciesEnergyFormation_eV")
    rmin = read_material_vector(mf, "r_min")
    # rows are per species, components [11 12 13 22 23 33]; the vacancy is row 0
    # and the diagonal sits at offsets 0, 3, 5.
    diag = (0, 3, 5)
    Em_v = float(np.mean([Em[i] for i in diag]))
    D0_v = float(np.prod([D0[i] for i in diag]) ** (1.0 / 3.0))
    return Material(
        b=b,
        c=read_material_scalar(mf, "c_SI"),
        omega=read_material_scalar(mf, "atomicVolume_SI"),
        Ef_v=float(Ef[0]),
        D0_v=D0_v,
        Em_v=Em_v,
        mu=read_material_scalar(mf, "mu0_SI"),
        nu=read_material_scalar(mf, "nu"),
        r_min=float(rmin[0]),
    )


def families(mat: Material = None) -> dict:
    """The four distinct loop geometries, built from ``mat``'s own constants."""
    mat = mat or load_material()
    out = {}
    for name, spec in CRYSTALLOGRAPHY.items():
        unit, fac = spec["bdotn"]
        bdotn = fac * (mat.b if unit == "a" else mat.c)
        unit, fac = spec["bmag"]
        if unit == "a":
            bmag = fac * mat.b
        elif unit == "c":
            bmag = fac * mat.c
        else:                      # 1/6<20-23> = (c/2) c_hat + (a/sqrt3) p_hat
            bmag = math.sqrt((mat.c / 2) ** 2 + mat.b ** 2 / 3.0)
        out[name] = Family(
            name=name, label=spec["label"], character=spec["character"],
            gamma=spec["gamma"], Kbar=spec["Kbar"], bmag=bmag, bdotn=bdotn,
            lam=math.sqrt(mat.omega / (math.pi * bdotn)), omega=mat.omega)
    return out


# ── Sec. 4.1: the work to detach one defect, and the binding energy ──────────

def dEdm(R, fam: Family, split=False):
    """dE_k/dm, Eq. (dEdm), in J. Fault term + anisotropic capillarity.

    ``split=True`` returns the two contributions separately, which is what makes
    the c_f / c_p contrast in the manuscript legible: the fault term is
    size-INDEPENDENT and the capillary term falls as ln(R)/R.
    """
    fault = fam.gamma * fam.omega / fam.bdotn
    cap = (fam.Kbar * fam.bmag ** 2 * fam.lam ** 2 / (4.0 * R)
           * (1.0 + np.log(ALPHA_CORE * R / fam.bmag)))
    return (fault, cap) if split else fault + cap


def sigma_resolved(fam: Family, stress=None, mat: Material = None):
    """Sigma_k = (b_k . sigma . n_k) / (b.n)_k, in Pa.

    ``stress`` is the 3x3 Cauchy tensor in the crystal frame (x || [10-10],
    y || [-2-1-10], z || [0001]). For every pure-edge family this returns the
    normal stress on the habit plane. For ``c_f`` it additionally picks up the
    basal shear through the partial, with weight 2a/(sqrt(3) c).
    """
    if stress is None:
        return 0.0
    s = np.asarray(stress, dtype=float).reshape(3, 3)
    mat = mat or load_material()
    if fam.name.startswith("a"):
        n = np.array([1.0, 0.0, 0.0])          # a prismatic habit normal
        bvec = fam.bdotn * n                   # pure edge, b || n
    else:
        n = np.array([0.0, 0.0, 1.0])          # basal habit normal
        bvec = np.array([0.0, 0.0, fam.bdotn])
        if fam.name == "c_f":                  # + the basal partial 1/3<1-100>
            bvec = bvec + np.array([mat.b / math.sqrt(3.0), 0.0, 0.0])
    return float(bvec @ s @ n) / fam.bdotn


def binding_energy(R, fam: Family, mat: Material = None, Sigma=0.0):
    """E_b^{v,k}(m, sigma), Eq. (Eb) generalized to the stress tensor, in eV.

        E_b = E^f_v - varsigma_s(v) dE_k/dm - Sigma_k Omega
    """
    mat = mat or load_material()
    return (mat.Ef_v
            - fam.zeta_v * dEdm(R, fam) / EV_J
            - Sigma * fam.omega / EV_J)


def c_eq_vacancy(R, fam: Family, T, mat: Material = None, Sigma=0.0):
    """c^{v,eq}_{sk}, Eq. (cveq) with the corrected stress sign."""
    mat = mat or load_material()
    return np.exp(-binding_energy(R, fam, mat, Sigma) / (KB_EV * T))


# ── Sec. 4.2: the thermal emission rate ──────────────────────────────────────

def Z_isolated(R, fam: Family):
    """The isolated-loop capture efficiency, Z = 2 pi / ln(8R/r0) with r0 = |b|.

    This is Eq. (Zsk) in its geometric limit: the toroidal-sink capacitance
    pi R / ln(8R/r0) times 4 pi D, divided by the perimeter 2 pi R. It replaces
    the fitted Z of the mean-field model when the loop is genuinely isolated,
    which is what Sec. 4.3 assumes and Sec. 4.4 gives up.
    """
    return 2.0 * np.pi / np.log(8.0 * R / fam.bmag)


def emission_rate(R, fam: Family, T, mat: Material = None, Sigma=0.0):
    """Vacancies emitted per loop per second, Eq. (loopemission) per loop.

        eps_k = Z * Dbar^v * c^{v,eq}_{sk} * 2 pi R / Omega       [1/s]
    """
    mat = mat or load_material()
    return (Z_isolated(R, fam) * mat.Dbar_v(T)
            * c_eq_vacancy(R, fam, T, mat, Sigma)
            * 2.0 * np.pi * R / mat.omega)


def absorption_rate(R, fam: Family, T, mat: Material = None, c_far=None):
    """Vacancies absorbed per loop per second, the same coefficient times c^v."""
    mat = mat or load_material()
    c_far = mat.c_v_inf(T) if c_far is None else c_far
    return (Z_isolated(R, fam) * mat.Dbar_v(T) * c_far
            * 2.0 * np.pi * R / mat.omega)


# ── Sec. 4.3: the isolated-loop trajectory ───────────────────────────────────

def dRdt(R, fam: Family, T, mat: Material = None, c_far=None, Sigma=0.0):
    """dR/dt for an ISOLATED loop under a pure anneal, in m/s.

        dR/dt = varsigma_s(v) Z Dbar^v [ c^v - c^{v,eq}_{sk} ] / (b.n)_k

    which is Eq. (vn-common) with the arrival rate per unit length taken from
    the toroidal sink. It is NEGATIVE for both characters at c^v = c^v_inf:
    a vacancy loop emits, an interstitial loop absorbs, and both shrink.
    """
    mat = mat or load_material()
    c_far = mat.c_v_inf(T) if c_far is None else c_far
    return (fam.zeta_v * Z_isolated(R, fam) * mat.Dbar_v(T)
            * (c_far - c_eq_vacancy(R, fam, T, mat, Sigma)) / fam.bdotn)


def anneal(R0, fam: Family, T, mat: Material = None, c_far=None, Sigma=0.0,
           R_end=None, n=4001):
    """t(R) for a loop shrinking from ``R0`` to ``R_end``.

    Integrates dt/dR = 1/|dR/dt| rather than dR/dt = f(R), because the rate
    diverges as R falls and the radius is the monotone variable. Returns
    ``(t, R)`` with t[0] = 0 at R0, so the last entry is the anneal-out time.
    """
    mat = mat or load_material()
    R_end = mat.r_min if R_end is None else R_end
    R = np.geomspace(R0, R_end, n)
    rate = dRdt(R, fam, T, mat, c_far, Sigma)
    if np.any(rate >= 0):
        raise ValueError(f"{fam.name} is not shrinking over [{R_end}, {R0}] "
                         f"at T={T} K: the anneal has a stable size")
    inv = 1.0 / np.abs(rate)
    # cumulative trapezoid over a descending grid
    dR = np.abs(np.diff(R))
    t = np.concatenate([[0.0], np.cumsum(0.5 * (inv[:-1] + inv[1:]) * dR)])
    return t, R


def anneal_time(R0, fam: Family, T, **kw):
    """Just the anneal-out time, in seconds."""
    t, _ = anneal(R0, fam, T, **kw)
    return float(t[-1])


# ── reporting ────────────────────────────────────────────────────────────────

def _fmt_time(s):
    for unit, f in (("s", 1.0), ("min", 60.0), ("h", 3600.0),
                    ("d", 86400.0), ("yr", 3.15576e7)):
        if s < 100 * f or unit == "yr":
            return f"{s / f:.3g} {unit}"
    return f"{s:.3g} s"


REFERENCE_RADII = {"a_i": 3.6e-9, "a_v": 3.6e-9, "c_f": 57.6e-9, "c_p": 57.6e-9}


def report_binding(T=773.0, radii=None, material_file=None):
    mat = load_material(material_file)
    fams = families(mat)
    radii = radii or REFERENCE_RADII
    print(f"# Binding energy per vacancy, Eq. (Eb).  T = {T:g} K, "
          f"E^f_v = {mat.Ef_v:g} eV, Omega = {mat.omega:.6e} m^3")
    print(f"# {'family':<8}{'R [nm]':>9}{'m':>12}{'fault':>10}{'capil':>10}"
          f"{'dE/dm':>10}{'E_b [eV]':>10}{'c_eq/c_inf':>12}")
    for name, fam in fams.items():
        R = radii[name]
        fault, cap = dEdm(R, fam, split=True)
        Eb = binding_energy(R, fam, mat)
        ratio = c_eq_vacancy(R, fam, T, mat) / mat.c_v_inf(T)
        print(f"  {name:<8}{R * 1e9:>9.1f}{fam.defects(R):>12.4g}"
              f"{fault / EV_J:>10.4f}{cap / EV_J:>10.4f}"
              f"{(fault + cap) / EV_J:>10.4f}{Eb:>10.4f}{ratio:>12.4g}")
    return fams, mat


def report_emission(T=773.0, radii=None, material_file=None):
    mat = load_material(material_file)
    fams = families(mat)
    radii = radii or REFERENCE_RADII
    print(f"# Thermal emission, Eq. (loopemission).  T = {T:g} K, "
          f"Dbar^v = {mat.Dbar_v(T):.4e} m^2/s, "
          f"c^v_inf = {mat.c_v_inf(T):.4e}")
    print(f"# {'family':<8}{'R [nm]':>9}{'Z':>8}{'emit [1/s]':>13}"
          f"{'absorb [1/s]':>14}{'net [1/s]':>13}{'dR/dt [m/s]':>14}")
    for name, fam in fams.items():
        R = radii[name]
        e = emission_rate(R, fam, T, mat)
        a = absorption_rate(R, fam, T, mat)
        print(f"  {name:<8}{R * 1e9:>9.1f}{Z_isolated(R, fam):>8.4f}"
              f"{e:>13.4e}{a:>14.4e}{a - e:>13.4e}"
              f"{dRdt(R, fam, T, mat):>14.4e}")
    return fams, mat


def report_anneal(R0=25e-9, temperatures=(773, 823, 873, 923, 973, 1023),
                  names=("c_f", "c_p"), material_file=None):
    mat = load_material(material_file)
    fams = families(mat)
    print(f"# Anneal-out time of an isolated loop, R0 = {R0 * 1e9:g} nm "
          f"down to r_min = {mat.r_min * 1e9:g} nm")
    head = "".join(f"{n:>16}" for n in names)
    print(f"# {'T [K]':<8}{head}")
    for T in temperatures:
        row = ""
        for n in names:
            row += f"{_fmt_time(anneal_time(R0, fams[n], T, mat=mat)):>16}"
        print(f"  {T:<8g}{row}")
    return fams, mat


def verify(material_file=None):
    """Checks that can be done by hand, so the tables in the paper are testable."""
    mat = load_material(material_file)
    fams = families(mat)
    ok = True

    def chk(what, got, want, tol):
        nonlocal ok
        rel = abs(got - want) / abs(want)
        good = rel <= tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {what:<52}"
              f"{got:>14.6g}  vs {want:<12.6g} ({rel:.1e})")

    # (1) lambda_k against Table (cd-families)
    chk("lambda_a  [nm]", fams["a_i"].lam * 1e9, 0.151, 5e-3)
    chk("lambda_cf [nm]", fams["c_f"].lam * 1e9, 0.170, 5e-3)
    chk("lambda_cp [nm]", fams["c_p"].lam * 1e9, 0.120, 5e-3)
    # lambda_cp = lambda_cf / sqrt(2) exactly
    chk("lambda_cf / lambda_cp", fams["c_f"].lam / fams["c_p"].lam,
        math.sqrt(2.0), 1e-12)
    # (2) |b| of the faulted state, Eq. (c2p)
    chk("|b_c/2+p| [nm]", fams["c_f"].bmag * 1e9, 0.318, 5e-3)
    # (3) the elastic weight ordering of Table (Kfactors)
    chk("<K>|b|^2 c_p / c_f  [-]",
        (fams["c_p"].Kbar * fams["c_p"].bmag ** 2)
        / (fams["c_f"].Kbar * fams["c_f"].bmag ** 2), 15.34 / 5.36, 5e-3)
    # (4) the far-field limit: E_b -> E^f_v as R -> infinity for an UNFAULTED
    #     family, and -> E^f_v - gamma Omega/(b.n) for a faulted one
    chk("E_b(c_p, R=10 um) [eV]", binding_energy(1e-5, fams["c_p"], mat),
        mat.Ef_v, 2e-3)
    chk("E_b(c_f, R=10 um) [eV]", binding_energy(1e-5, fams["c_f"], mat),
        mat.Ef_v - fams["c_f"].gamma * mat.omega / fams["c_f"].bdotn / EV_J,
        2e-3)
    # (5) both characters shrink at c^v = c^v_inf and zero stress
    for n in fams:
        r = dRdt(REFERENCE_RADII[n], fams[n], 873.0, mat)
        print(f"  [{'ok ' if r < 0 else 'FAIL'}] {n} shrinks at 873 K"
              f"{'':<30}{r:>14.4e} m/s")
        ok &= r < 0
    # (6) the climb threshold, Eq. (climbthreshold): an interstitial family
    #     GROWS once Sigma exceeds (1/Omega) dE/dm
    fam = fams["a_i"]
    R = REFERENCE_RADII["a_i"]
    sig_star = dEdm(R, fam) / mat.omega
    below = dRdt(R, fam, 873.0, mat, Sigma=0.99 * sig_star)
    above = dRdt(R, fam, 873.0, mat, Sigma=1.01 * sig_star)
    good = below < 0 < above
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] climb threshold at "
          f"{sig_star / 1e6:.1f} MPa: dR/dt {below:.2e} -> {above:.2e}")
    # (7) the c_f basal-shear weight, 2a/(sqrt3 c), and that c_p ignores it
    tau = 100e6
    s = np.zeros((3, 3))
    s[0, 2] = s[2, 0] = tau
    chk("Sigma_cf / tau under pure basal shear",
        sigma_resolved(fams["c_f"], s, mat) / tau,
        2 * mat.b / (math.sqrt(3.0) * mat.c), 1e-12)
    good = abs(sigma_resolved(fams["c_p"], s, mat)) < 1e-6
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] Sigma_cp = 0 under pure basal shear")
    print("VERIFY:", "all checks passed" if ok else "FAILURES ABOVE")
    return ok


# ── the figure ───────────────────────────────────────────────────────────────

CF = "#1f5fa8"
CP = "#d1690c"


def figure(out, R0=25e-9, T_show=873.0, material_file=None):
    """The three statements of Sec. 4.3, one panel each.

    (a) the SHAPE of R(t): the faulted loop shrinks at nearly constant velocity
        because its driving force is the fault and does not depend on size; the
        perfect loop is capillarity-driven and only runs away at the end.
    (b) the lifetime is Arrhenius with Q close to E^f_v + E^m_v.
    (c) the size scaling those two shapes imply, t ~ R0 against t ~ R0^2.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_material(material_file)
    fams = families(mat)
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.85))

    # (a) R(t) on a LINEAR time axis, at one temperature
    for n, col, lab in (("c_f", CF, r"$c_f$ faulted"),
                        ("c_p", CP, r"$c_p$ perfect")):
        t, R = anneal(R0, fams[n], T_show, mat=mat)
        ax[0].plot(t / 3600.0, R * 1e9, color=col, lw=2.0, label=lab)
        ax[0].plot([t[-1] / 3600.0], [R[-1] * 1e9], "o", ms=4.5, color=col)
        ax[0].annotate(_fmt_time(t[-1]), (t[-1] / 3600.0, R[-1] * 1e9),
                       textcoords="offset points", xytext=(-6, 10),
                       ha="right", fontsize=8.5, color=col)
    ax[0].set_xlabel("annealing time  [h]")
    ax[0].set_ylabel(r"loop radius  $R$  [nm]")
    ax[0].set_ylim(0, R0 * 1e9 * 1.06)
    ax[0].set_xlim(0, None)
    ax[0].set_title(r"(a)  $R(t)$ at $T=%g$ K,  $R_0=%g$ nm"
                    % (T_show, R0 * 1e9), fontsize=10, pad=20)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper right")

    # (b) Arrhenius
    Ts = np.linspace(723.0, 1073.0, 40)
    for n, lab, mk, col in (
            ("c_f", r"$c_f$ faulted, $R_0=25$ nm", "-", CF),
            ("c_p", r"$c_p$ perfect, $R_0=25$ nm", "-", CP),
            ("a_i", r"$a_i$ prismatic, $R_0=3.6$ nm", "--", "#2e7d32"),
            ("a_v", r"$a_v$ prismatic, $R_0=3.6$ nm", "--", "#b3243c")):
        R = R0 if n.startswith("c") else 3.6e-9
        y = [anneal_time(R, fams[n], T, mat=mat) for T in Ts]
        ax[1].semilogy(1000.0 / Ts, y, mk, lw=1.8, color=col, label=lab)
    for hrs, lab in ((1.0, "1 h"), (24.0, "1 d"), (24 * 365.0, "1 yr")):
        ax[1].axhline(hrs * 3600, color="0.78", lw=0.7, zorder=0)
        ax[1].text(1000 / 1068.0, hrs * 3600 * 1.5, lab, fontsize=7.5,
                   color="0.45")
    ax[1].set_xlabel(r"$1000/T$  [K$^{-1}$]")
    ax[1].set_ylabel("anneal-out time  [s]")
    ax[1].set_title("(b)  lifetime, Arrhenius form", fontsize=10, pad=20)
    ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    sec = ax[1].secondary_xaxis(
        "top", functions=(lambda x: 1000.0 / np.maximum(x, 1e-9),
                          lambda T: 1000.0 / np.maximum(T, 1e-9)))
    sec.set_xlabel(r"$T$  [K]", fontsize=9)

    # (c) size scaling
    Rs = np.geomspace(3e-9, 300e-9, 30)
    for n, col, lab in (("c_f", CF, r"$c_f$ faulted"),
                        ("c_p", CP, r"$c_p$ perfect")):
        y = np.array([anneal_time(R, fams[n], T_show, mat=mat) for R in Rs])
        ax[2].loglog(Rs * 1e9, y, color=col, lw=2.0, label=lab)
    Rg = Rs[Rs >= 40e-9]
    for p, name, col, lab in ((1.0, "c_f", CF, r"$\propto R_0$"),
                              (2.0, "c_p", CP, r"$\propto R_0^{2}$")):
        y0 = anneal_time(Rg[-1], fams[name], T_show, mat=mat)
        g = y0 * (Rg / Rg[-1]) ** p
        ax[2].loglog(Rg * 1e9, g, ":", color=col, lw=1.2)
        ax[2].text(Rg[0] * 1e9 * 1.05, g[0] * (0.35 if p == 1 else 1.7), lab,
                   fontsize=9, color=col)
    ax[2].set_xlabel(r"initial radius  $R_0$  [nm]")
    ax[2].set_ylabel("anneal-out time  [s]")
    ax[2].set_title(r"(c)  size scaling at $T=%g$ K" % T_show,
                    fontsize=10, pad=20)
    ax[2].legend(fontsize=8.5, frameon=False, loc="upper left")

    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "binding", "emission", "anneal", "figure"):
        q = sub.add_parser(name)
        q.add_argument("--material", default=None)
        if name in ("binding", "emission"):
            q.add_argument("--T", type=float, default=773.0)
        if name in ("anneal", "figure"):
            q.add_argument("--R0", type=float, default=25.0, help="nm")
        if name == "anneal":
            q.add_argument("--T", type=float, nargs="*", default=None)
        if name == "figure":
            q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "verify":
        return 0 if verify(a.material) else 1
    if a.cmd == "binding":
        report_binding(a.T, material_file=a.material)
    elif a.cmd == "emission":
        report_emission(a.T, material_file=a.material)
    elif a.cmd == "anneal":
        Ts = a.T or (773, 823, 873, 923, 973, 1023)
        report_anneal(a.R0 * 1e-9, tuple(Ts), material_file=a.material)
    elif a.cmd == "figure":
        figure(a.out, a.R0 * 1e-9, material_file=a.material)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
