"""hardening.py — irradiation hardening from a coupled march, by dislocation dynamics.

Implements `Docs/DisloCluster Manual/architecture/irradiation_hardening_dd_plan.md`:
take the continuum immobile field a coupled march produced, rebuild it as a
discrete loop population inside a periodic cube, load that cube, and measure the
increase in flow stress the irradiation microstructure causes.

THE THREE ROUTES, and which one answers what
--------------------------------------------
=========  ================================================  ==================
Route A    stress ramp on a frozen obstacle field, periodic   `stage(route="A")`
           cube: the CRSS increment DELTA-TAU directly
Route C    dispersed-barrier law, alpha fitted on route A     `dispersed_barrier`
           and then applied at every dose for free            `fit_alpha`
Route D    the same cell seeded by DENSITY rather than by     `stage(route="D")`
           per-loop export -- no export pipeline, so it is
           the smoke test that sizes the campaign
=========  ================================================  ==================

Route B (strain-rate control, the full stress-strain curve) is not staged here.
It is the same cell with `ExternalStrainRate` non-zero and a large
`stiffnessRatio`, and the plan's §6 arithmetic puts it at days per dose against
minutes to hours for route A -- so it is a deliberate omission, not an oversight.
`stage(strain_rate_s=...)` writes that deck if you want to pay for it.

WHAT IS MEASURED AND WHAT IS ASSUMED
------------------------------------
Everything this module computes from a run -- densities, radii, loop counts, the
commensurate box -- is measured. Two numbers are NOT: the Taylor factor `M` that
turns DELTA-TAU into DELTA-SIGMA_y, and the obstacle strengths `alpha_k` before
route A has fitted them. They are arguments with no default that pretends
otherwise, and `dispersed_barrier` reports the alpha it used.

UNITS, which is where this kind of code goes wrong
--------------------------------------------------
MoDELib normalizes length by `b_SI`, stress by `mu_SI` and TIME by `b_SI/cs`
with `cs = sqrt(mu/rho)`. So a stress ramp written as MPa/s is wrong by
`mu_SI/1e6 / (b_SI/cs)` -- 3e-2 for Zr, i.e. it looks plausible and is out by a
factor of thirty. `Material.stress_rate_modelib` does that conversion in one
place. Note `config.Boundary.stress_rate_in_mu` does NOT: it converts the stress
and leaves the time unit alone, which is harmless only because every coupled run
so far has had a zero rate.

The atomic volume and `|b|` are read from THE MATERIAL FILE, not from
`post.fields`, whose module-level `B_SI = 3.233e-10` and `OMEGA_SI = 1.2e-29`
predate both corrections recorded in CLAUDE.md. It cancels in the loop radius
(`r = sqrt(c/(n*pi*b))` once `m = c/(n*Omega)` is substituted) but not in a
number density or a defect count.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import field as mfield
from dislocluster_code.post import movies as movies_mod
from dislocluster_code.post.discrete_loops import (
    FAMILIES, LoopPopulation, coalesce, domain_weights, write_microstructure)
from dislocluster_code.post.gb import gb_distance
from dislocluster_code.staging import inputs as minputs

__all__ = [
    "Material", "BoxSpec", "PlacementRejected", "interior_mask", "interior_state", "state_table",
    "commensurate_box", "cube_population", "stage", "run_case", "reduce_run",
    "dispersed_barrier", "fit_alpha", "SLIP_PRISMATIC_A", "schmid_voigt",
    "campaign", "verify", "figures", "plot_flow_curves", "plot_delta_tau",
    "obstacle_count",
]


class PlacementRejected(RuntimeError):
    """MoDELib refused a loop realization; another seed will do."""


# ── material constants ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Material:
    """The handful of constants every part of this module needs, from one file.

    `cs` is the shear wave speed `sqrt(mu/rho)`, which is MoDELib's velocity
    scale; `t_unit_SI = b/cs` is therefore one unit of DD time. For Zr3d_ghoniem
    at 573 K that is 1.4356e-13 s, and the plan's §6 step-count arithmetic is
    built on it.
    """
    file: Path
    temperature_K: float
    b_SI: float
    c_SI: float
    mu_SI: float
    rho_SI: float
    omega_SI: float

    @classmethod
    def from_file(cls, material_file=None, temperature_K=573.0):
        # A BARE NAME resolves against the Library, the way the notebook's
        # MATERIAL["file"] gives it ("Zr3d_ghoniem.txt"); a path is used as
        # given. Without this the notebook's own default raises
        # FileNotFoundError on the first cell that touches the material.
        f = Path(material_file) if material_file else Path(paths.MODELIB_MATERIAL)
        if not f.is_file() and f.parent == Path("."):
            cand = paths.MODELIB_ROOT / "Library" / "Materials" / f.name
            if cand.is_file():
                f = cand
        get = minputs.read_material_scalar
        b = get(f, "b_SI")
        c = get(f, "c_SI", 1.632993 * b)      # absent -> ideal c/a, as HEXlattice does
        return cls(file=f, temperature_K=float(temperature_K), b_SI=b, c_SI=c,
                   mu_SI=minputs.material_mu_SI(f, temperature_K),
                   rho_SI=get(f, "rho_SI"),
                   omega_SI=get(f, "atomicVolume_SI"))

    @property
    def c_over_a(self):
        return self.c_SI / self.b_SI

    @property
    def omega_b3(self):
        return self.omega_SI / self.b_SI ** 3

    @property
    def cs_SI(self):
        return float(np.sqrt(self.mu_SI / self.rho_SI))

    @property
    def t_unit_SI(self):
        return self.b_SI / self.cs_SI

    def stress_modelib(self, sigma_Pa):
        """Pa -> MoDELib's dimensionless stress (multiples of mu)."""
        return np.asarray(sigma_Pa, dtype=float) / self.mu_SI

    def stress_rate_modelib(self, rate_Pa_s):
        """Pa/s -> MoDELib stress rate, i.e. multiples of mu per unit of b/cs.

        BOTH normalizations apply. Converting only the stress -- the obvious
        half -- leaves the rate out by 1/t_unit = 7e12.
        """
        return np.asarray(rate_Pa_s, dtype=float) * self.t_unit_SI / self.mu_SI

    def summary(self):
        return (f"{self.file.name} at {self.temperature_K:g} K: "
                f"b={self.b_SI*1e9:.4g} nm, c/a={self.c_over_a:.7f}, "
                f"mu={self.mu_SI/1e9:.4g} GPa, cs={self.cs_SI:.0f} m/s, "
                f"b/cs={self.t_unit_SI:.4g} s, Omega={self.omega_SI:.6e} m^3 "
                f"({self.omega_b3:.4f} b^3)")


# ── the slip system the load is resolved on ──────────────────────────────────
# MoDELib's slip system 6, TAKEN FROM THE CODE'S OWN LIST rather than assumed:
# with `enabledSlipSystems = fullBasal fullPrismatic` it prints 0-5 basal (all
# with n = [0,0,1]) and 6-11 prismatic, of which 6 is s = [1,0,0], n = [0,-1,0].
# `grain1globalX1 = [1,0,0]` and `grain1globalX3 = [0,0,1]` in polycrystal.txt
# make the crystal and global frames coincide, so these are also the crystal
# directions.
#
# The SIGN of n matters here, and only for bookkeeping: resolving the load on
# (s, +y) rather than (s, -y) leaves tau positive while the dislocation glides
# the other way, so gamma_p comes out negative and the offset criterion never
# fires. Both are taken from the same system, so both agree.
#
# Keep the load on prismatic <a>: `Zr3d_ghoniem.txt` enables only fullBasal and
# fullPrismatic, and pyramidal drag is 50e20 Pa*s, so a load with a large c-axis
# component has no accommodating slip system at all.
SOURCE_SLIP_SYSTEM = 6
SLIP_PRISMATIC_A = (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))

# Mesh face IDs of unitCube24.msh, from the mesh report MoDELib prints:
#   0 = -z, 1 = -y, 2 = -x, 3 = +x, 4 = +y, 5 = +z
# A periodic dipole must exit through a face whose periodic SHIFT lies in its
# glide plane (`PeriodicDipoleGenerator.cpp:151` requires n.dot(shift) == 0).
# For system 6 (n along y) that is the x or z pair; z gives a line along z with
# b along x, i.e. an EDGE dipole gliding in x, which is what a resolved shear
# on this system should drive.
DIPOLE_EXIT_FACE_Z = 5

# The default stress ramp, stated the way the plan states it: 1 MPa per 1e4
# units of DD time. That is quasi-static in the only sense that matters here --
# a pinned line relaxes in ~1e2 b/cs, over which the stress moves by 0.01 MPa.
#
# In SI it is 6.97e14 Pa/s, which is fast for a laboratory and irrelevant to
# one: nothing in this cell has a rate-dependent constitutive law except the
# drag coefficient. NOTE the plan's §3 quotes "~7e11 Pa/s" for this same ramp,
# which is the conversion out by 1e3; the DD-unit form is the correct one, and
# it is the one implemented here.
RAMP_STEPS_PER_MPa = 1.0e4        # units of b/cs per MPa of applied shear


def default_tau_dot_MPa_s(material):
    """The default ramp, 1 MPa per 1e4 b/cs, expressed in MPa/s."""
    return 1.0 / (RAMP_STEPS_PER_MPa * material.t_unit_SI)


def schmid_voigt(s, n):
    """Voigt components (11 22 33 12 23 13) of the symmetric tensor s(x)n+n(x)s.

    A stress sigma = tau*(s(x)n + n(x)s) resolves to exactly tau on that system:
    s.sigma.n = tau[(s.s)(n.n) + (s.n)(n.s)] = tau, since s and n are orthogonal
    unit vectors. So this IS the stress for unit resolved shear -- no factor of
    two, in either direction. `resolve(schmid_voigt(s, n), s, n) == 1` is the
    check, and it is worth keeping: the factor-of-two error here would be a
    clean factor of two in every CRSS this module reports.
    """
    s = np.asarray(s, float) / np.linalg.norm(s)
    n = np.asarray(n, float) / np.linalg.norm(n)
    m = np.outer(s, n) + np.outer(n, s)
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[1, 2], m[0, 2]])


def ramp_voigt(s, n):
    """Voigt stress giving unit RESOLVED SHEAR on the system (s, n)."""
    return schmid_voigt(s, n)


def resolve(sigma_voigt, s, n):
    """Resolved shear s.sigma.n from a Voigt stress row (11 22 33 12 23 13)."""
    v = np.asarray(sigma_voigt, float)
    m = np.array([[v[0], v[3], v[5]],
                  [v[3], v[1], v[4]],
                  [v[5], v[4], v[2]]])
    s = np.asarray(s, float) / np.linalg.norm(s)
    n = np.asarray(n, float) / np.linalg.norm(n)
    return float(s @ m @ n)


# ── §1.1 / §2  the interior state ────────────────────────────────────────────
_WEIGHT_CACHE = {}


def interior_mask(nodes, frac=0.5):
    """Nodes far enough from the surface to stand for bulk material.

    The rule is `d_face > frac * max(d_face)`, which is exactly what
    `discrete_loops.populate(region="interior")` applies, so a population built
    here and one built there select the same nodes. It is not literally the
    innermost quartile BY COUNT -- on the 500 nm prism it keeps 8% of the nodes
    -- but it is the same innermost region CLAUDE.md and the plan mean by the
    word, and consistency between the two code paths matters more than the name.

    Why any of this is needed: the marched domains are Dirichlet over their
    whole surface, so the boundary shell is a sink for mobile defects but NOT
    for loops. <a> density climbs steeply toward the wall and a domain mean
    measures that shell, not the material -- 474x the 0-D as a domain mean and
    0.75x it in the interior, on the case CLAUDE.md records.
    """
    d = gb_distance(np.asarray(nodes, float))
    return d > frac * float(d.max())


def _run_frame(run_dir, dose=None, variant_weights=(1 / 3, 1 / 3, 1 / 3)):
    """`(dose_actual, nodes, F, weights, faces)` for one snapshot of a march."""
    key = (str(run_dir), tuple(variant_weights))
    if key not in _WEIGHT_CACHE:
        doses, nodes, frames = movies_mod.cd_blocks(run_dir, variant_weights)
        w, faces = domain_weights(nodes)
        _WEIGHT_CACHE[key] = (doses, nodes, frames, w, faces)
    doses, nodes, frames, w, faces = _WEIGHT_CACHE[key]
    doses = np.asarray(doses, float)
    i = len(doses) - 1 if dose is None else int(np.argmin(np.abs(doses - dose)))
    P, F = frames[i]
    return float(doses[i]), P, F, w, faces


def interior_state(run_dir, dose=None, material=None, frac=0.5,
                   variant_weights=(1 / 3, 1 / 3, 1 / 3)):
    """Per-family interior state at one dose: `N` [m^-3], `r` [nm], `rho` [m^-2].

    The reduction is the plan's §2, term for term:

        n_k   volume-weighted mean of the CD block's number density [1/b^3]
        c_k   volume-weighted mean of its stored content [defects per atom]
        m_k   = c_k / (n_k * Omega_b3)            defects in one loop
        r_k   = sqrt(m_k * Omega / (pi * b_k))    the disc that stores them
        rho_k = 2*pi*r_k*N_k                      loop LINE density

    Volume weights, not a plain mean over nodes: the mesh is refined in the
    boundary layer, so counting nodes equally would over-weight whatever part
    of the refined shell survives the interior cut.
    """
    mat = material or Material.from_file()
    dose_actual, nodes, F, w, faces = _run_frame(run_dir, dose, variant_weights)
    mask = interior_mask(nodes, frac)
    if not mask.any():
        raise ValueError(f"no interior nodes in {run_dir} at frac={frac}")
    wi = w * mask
    wi_sum = float(wi.sum())

    fams = []
    for fam in FAMILIES:
        n_b3 = float((wi * F[:, fam["ncol"]]).sum() / wi_sum)
        c_at = float((wi * F[:, fam["ccol"]]).sum() / wi_sum)
        N_m3 = n_b3 / mat.b_SI ** 3
        if n_b3 > 0.0 and c_at > 0.0:
            m = c_at / (n_b3 * mat.omega_b3)                       # defects/loop
            r_b = float(np.sqrt(m * mat.omega_b3 / (np.pi * fam["b_dd"])))
        else:
            m, r_b = 0.0, 0.0
        r_nm = r_b * mat.b_SI * 1e9
        fams.append(dict(key=fam["key"], label=fam["label"], n_b3=n_b3,
                         c_at=c_at, N_m3=N_m3, m_defects=m, r_b=r_b, r_nm=r_nm,
                         rho_m2=2.0 * np.pi * r_nm * 1e-9 * N_m3,
                         b_dd=fam["b_dd"]))

    a = [f for f in fams if f["key"] != "c"]
    c = [f for f in fams if f["key"] == "c"][0]
    N_a = sum(f["N_m3"] for f in a)
    return dict(
        run=str(run_dir), dose=dose_actual, n_nodes=int(len(nodes)),
        n_interior=int(mask.sum()), frac=frac, families=fams,
        N_c_m3=c["N_m3"], r_c_nm=c["r_nm"], rho_c_m2=c["rho_m2"],
        N_a_m3=N_a, rho_a_m2=sum(f["rho_m2"] for f in a),
        # The three <a> variants are equal at zero applied stress, so their
        # mean radius is the family radius; weighting by N keeps that true if a
        # stressed run ever tilts the split.
        r_a_nm=(sum(f["r_nm"] * f["N_m3"] for f in a) / N_a) if N_a else 0.0,
    )


def state_table(run_dir, material=None, frac=0.5,
                variant_weights=(1 / 3, 1 / 3, 1 / 3)):
    """`interior_state` at every snapshot the run holds."""
    doses, _, _ = movies_mod.cd_blocks(run_dir, variant_weights)
    return [interior_state(run_dir, d, material, frac, variant_weights)
            for d in doses]


def format_state_table(rows):
    out = [f"{'dpa':>8} {'N_c [m^-3]':>11} {'r_c [nm]':>9} {'N_a [m^-3]':>11} "
           f"{'r_a [nm]':>9} {'rho_c [m^-2]':>13} {'rho_a [m^-2]':>13}"]
    for r in rows:
        out.append(f"{r['dose']:>8.4g} {r['N_c_m3']:>11.3g} {r['r_c_nm']:>9.3g} "
                   f"{r['N_a_m3']:>11.3g} {r['r_a_nm']:>9.3g} "
                   f"{r['rho_c_m2']:>13.3g} {r['rho_a_m2']:>13.3g}")
    return "\n".join(out)


# ── §6  the commensurate periodic box ────────────────────────────────────────
@dataclass(frozen=True)
class BoxSpec:
    """An orthogonal cube whose edges are real lattice vectors of hcp Zr.

    The periodic shifts have to be lattice vectors or they do not map glide
    planes onto themselves, and a cube that is merely 500 nm on a side is not.
    With `a1 = (1,0,0)`, `a2 = (1/2, sqrt3/2, 0)`, `a3 = (0,0,c/a)` in units of
    b, the orthogonal commensurate choice is

        e1 = n*a1          = (n, 0, 0)
        e2 = m*(2*a2 - a1) = (0, m*sqrt3, 0)
        e3 = p*a3          = (0, 0, p*c/a)

    and `c/a` is the PHYSICAL 1.5944272 this material carries through `c_SI`,
    not the ideal 1.6329932. MoDELib's own `modlibUtils.PolyCrystalFile` gets
    this wrong -- its HEX basis hard-codes the ideal ratio, so its `compute()`
    picks p = 948 for a 500 nm target, a 488 nm edge against the real basis and
    a shift vector that is not a lattice vector at all. Hence this class, and
    hence `stage` writing polycrystal.txt itself.
    """
    n: int
    m: int
    p: int
    b_SI: float
    c_over_a: float

    @property
    def edges_b(self):
        return np.array([float(self.n),
                         self.m * np.sqrt(3.0),
                         self.p * self.c_over_a])

    @property
    def F(self):
        return np.diag(self.edges_b)

    @property
    def edges_nm(self):
        return self.edges_b * self.b_SI * 1e9

    @property
    def volume_b3(self):
        return float(np.prod(self.edges_b))

    @property
    def volume_m3(self):
        return self.volume_b3 * self.b_SI ** 3

    def describe(self):
        e, nm = self.edges_b, self.edges_nm
        return (f"(n,m,p) = ({self.n}, {self.m}, {self.p})  "
                f"F = diag({e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}) b  "
                f"= {nm[0]:.2f} x {nm[1]:.2f} x {nm[2]:.2f} nm  "
                f"V = {self.volume_m3:.4g} m^3")


def commensurate_box(L_nm, material=None):
    """The BoxSpec closest to a cube of side `L_nm`."""
    mat = material or Material.from_file()
    L_b = float(L_nm) * 1e-9 / mat.b_SI
    return BoxSpec(n=max(1, int(round(L_b))),
                   m=max(1, int(round(L_b / np.sqrt(3.0)))),
                   p=max(1, int(round(L_b / mat.c_over_a))),
                   b_SI=mat.b_SI, c_over_a=mat.c_over_a)


# ── §7  the cube population ──────────────────────────────────────────────────
def cube_population(box, material=None, run_dir=None, dose=None, state=None,
                    seed=0, coalesce_pass=True, frac=0.5,
                    variant_weights=(1 / 3, 1 / 3, 1 / 3), verbose=False):
    """Discrete loops of every family, uniformly filling `box`. `(pops, stats)`.

    Two sources, one output:

    * `run_dir` (+ optional `dose`) samples the marched field directly. The
      count per family is the interior MEAN density times the cube volume, and
      each loop's radius is drawn from the interior nodes with probability
      proportional to their share of the loop count -- so the radius
      DISTRIBUTION the field carries survives, while the boundary layer does
      not.
    * `state`, an `interior_state` dict, uses the tabulated `(N_k, r_k)` and is
      monodisperse. This is what to use when the march that produced the state
      is not on this machine.

    Positions are uniform in the cube and NOT rejected against a crystal hull:
    the cell is periodic, every loop wraps through the faces, and
    `MicrostructureGenerator::insertJunctionLoop` reads
    `ddBase.isPeriodicDomain ? true : allPointsInGrain(...)` -- so the
    `fits_in_crystal` refusal that costs the free-surface transfer 99% of its
    <c> defects at 1 dpa simply does not arise here. Centres are drawn in
    `[-L/2, +L/2]` because `unitCube24.msh` spans `[-0.5, 0.5]^3` and
    `polycrystal.txt` maps `x = F*(X - X0)` with `X0 = 0`; a centre outside the
    mesh makes `aLoopGenerator::generateSingle` call `exit(EXIT_FAILURE)`.
    """
    mat = material or Material.from_file()
    rng = np.random.default_rng(seed)
    V_b3 = box.volume_b3
    lo = -0.5 * box.edges_b

    if state is None:
        if run_dir is None:
            raise ValueError("cube_population needs either run_dir or state")
        dose_actual, nodes, F, w, _ = _run_frame(run_dir, dose, variant_weights)
        mask = interior_mask(nodes, frac)
        wi = w * mask
        source = "field"
    else:
        dose_actual = state["dose"]
        source = "state"

    pops, stats = [], []
    for i, fam in enumerate(FAMILIES):
        if state is None:
            n_i = F[:, fam["ncol"]].astype(float)
            c_i = F[:, fam["ccol"]].astype(float)
            ok = mask & np.isfinite(n_i) & np.isfinite(c_i) & (n_i > 0) & (c_i > 0)
            n_mean = float((wi * n_i).sum() / wi.sum()) if wi.sum() else 0.0
            n_draw = int(round(n_mean * V_b3))
            if n_draw <= 0 or not ok.any():
                pops.append(LoopPopulation(fam, np.zeros((0, 3)), np.zeros(0)))
                stats.append(dict(family=fam["key"], n=0, r_nm=0.0))
                continue
            lam = np.where(ok, n_i * w, 0.0)
            idx = rng.choice(len(lam), size=n_draw, p=lam / lam.sum())
            m = np.zeros_like(n_i)
            m[ok] = c_i[ok] / (n_i[ok] * mat.omega_b3)
            r_b = np.sqrt(np.maximum(m, 0.0) * mat.omega_b3
                          / (np.pi * fam["b_dd"]))[idx]
        else:
            fs = state["families"][i]
            n_draw = int(round(fs["n_b3"] * V_b3))
            if n_draw <= 0 or fs["r_b"] <= 0.0:
                pops.append(LoopPopulation(fam, np.zeros((0, 3)), np.zeros(0)))
                stats.append(dict(family=fam["key"], n=0, r_nm=0.0))
                continue
            r_b = np.full(n_draw, float(fs["r_b"]))

        centers = lo + box.edges_b * rng.random((n_draw, 3))
        raw = LoopPopulation(fam, centers, r_b)
        pop = coalesce(raw, V_b3) if coalesce_pass else raw
        nm = mat.b_SI * 1e9
        d_nm = (V_b3 / len(pop)) ** (1 / 3.) * nm if len(pop) else float("nan")
        r_mean_nm = float(pop.radii.mean()) * nm if len(pop) else 0.0
        stats.append(dict(
            family=fam["key"], n_before=len(raw), n=len(pop),
            r_nm=r_mean_nm, d_nm=d_nm,
            ratio=2.0 * r_mean_nm / d_nm if len(pop) else float("nan"),
            N_m3=len(pop) / box.volume_m3,
            rho_m2=2.0 * np.pi * r_mean_nm * 1e-9 * len(pop) / box.volume_m3,
            area_b2=pop.total_area,
            # Stored defects are the conserved quantity across coalescence, and
            # the one number that says the cube carries the same material the
            # interior does. Reported, never asserted on: the draw rounds to a
            # whole number of loops.
            stored_defects=pop.total_area * fam["b_dd"] / mat.omega_b3,
            saturated=bool(getattr(pop, "saturated", False)),
            segments=len(pop) * int(fam.get("dd_sides", fam["sides"]))))
        pops.append(pop)

    if verbose:
        print(f"  cube population from {source} at {dose_actual:g} dpa, "
              f"{box.edges_nm[0]:.0f} nm cube")
        print(f"    {'fam':<4} {'N':>6} {'r[nm]':>8} {'d[nm]':>8} {'2r/d':>7} "
              f"{'seg':>7}  flag")
        for s in stats:
            print(f"    {s['family']:<4} {s.get('n', 0):>6d} {s['r_nm']:>8.2f} "
                  f"{s.get('d_nm', float('nan')):>8.1f} "
                  f"{s.get('ratio', float('nan')):>7.3f} "
                  f"{s.get('segments', 0):>7d}  "
                  f"{'SATURATED' if s.get('saturated') else ''}")
    return pops, stats


# ── §7 step 3  the input deck ────────────────────────────────────────────────
_POLYCRYSTAL = """\
materialFile={material};
absoluteTemperature={T:g}; # [K] simulation temperature
meshFile={mesh}; # mesh file
F={F00:.17g} 0 0
  0 {F11:.17g} 0
  0 0 {F22:.17g}; # mesh deformation gradient, x = F*(X-X0)
# 17 significant digits, NOT the 10 that are plenty for a length: the box edges
# have to be LATTICE vectors, and MoDELib checks that by mapping each periodic
# shift into lattice coordinates and demanding an integer. Ten digits truncates
# 971*c/a = 1548.1888112 to 1548.188811, which is 1.25e-7 lattice units short of
# 971 -- and the run dies at startup with "Input vector is not a lattice vector".
X0=0 0 0; # unitCube24.msh spans [-0.5,0.5]^3, so no shift is needed to centre it
periodicFaceIDs=-1; # -1 = every parallel face pair periodic (SimplicialMesh.cpp:350)
C2G1 =
1.000000000000000 0.000000000000000 0.000000000000000
0.000000000000000 1.000000000000000 0.000000000000000
0.000000000000000 0.000000000000000 1.000000000000000;
"""

_ELASTIC = """\
ExternalStress0={s0}; # [mu] applied stress
ExternalStressRate={sr}; # [mu per b/cs] -- {tau_dot_MPa_s:g} MPa/s at mu={mu:.6g} Pa, b/cs={t:.6g} s
ExternalStrain0={g0};
ExternalStrainRate={gr}; # [1/(b/cs)]
stiffnessRatio={sk};

# Voigt order is 11,22,33,12,23,13.
# stiffnessRatio=0 is pure stress control, 1e20 is pure strain control.
#
# Written by dislocluster_code.studies.hardening. BOTH normalizations are
# applied to the rate: stress by mu_SI and time by b_SI/cs. Converting only the
# stress leaves it out by 1/(b/cs) = 7e12.
"""

# What route A changes from Library/DislocationDynamics/DD.txt, and why the ones
# that are not obvious are there:
#
#   useFEM=0            a fully periodic domain builds no FE space anyway
#                       (DislocationDynamicsBase.cpp:107), and with `fe` null
#                       ElasticDeformation falls back to the uniform load
#                       controller -- which is the one that writes stress and
#                       strain into F/F_0.txt at all.
#   climbSolverType=none  the loops must not move. They are SESSILELOOPs, and
#                       DislocationNode::projectVelocity zeroes the velocity of
#                       every node on one during a non-climbing step.
#   useSubCycling=1     the whole affordability argument. A segment whose
#                       average nodal velocity is below FLT_EPSILON lands in the
#                       largest subcycling bin (DislocationSegment.cpp:496), so
#                       the frozen loops are re-evaluated once per 100 steps
#                       while the moving line is evaluated every step. They
#                       still contribute stress every step.
#   EwaldLengthFactor=1 non-zero is what the periodic tutorial uses; 0 leaves
#                       the image sum unscreened.
_DD_ROUTE_A = {
    "useFEM": 0, "useElasticDeformation": 1, "useElasticDeformationFEM": 0,
    "useDislocations": 1, "useClusterDynamics": 0, "useClusterDynamicsFEM": 0,
    "useInclusions": 0, "useCracks": 0,
    "glideSolverType": "Galerkin", "climbSolverType": "none",
    "timeSteppingMethod": "adaptive", "dxMax": 5, "dtMax": "1e25",
    "periodicImageSize": "1 1 1", "EwaldLengthFactor": 1,
    "useSubCycling": 1, "subcyclingBins": "1 2 5 10 50 100",
    "quadPerLength": 0.1, "alphaLineTension": 0.1,
    "remeshFrequency": 10, "Lmin": 25, "Lmax": 100,
    "maxJunctionIterations": 1, "crossSlipModel": 1,
    "use_velocityFilter": 0, "use_stochasticForce": 0,
    "outputQuadraturePoints": 0, "outputBinary": 0,
    "outputPlasticDistortionPerSlipSystem": 1,
    "startAtTimeStep": 0,
}


def stage(sim_dir, box, material=None, *, route="A", pops=None, state=None,
          tau_dot_MPa_s=None, slip=SLIP_PRISMATIC_A, n_steps=100000,
          output_frequency=100, subcycling=True, source_density_m2=None,
          source_slip_system=SOURCE_SLIP_SYSTEM, dipole_height=200,
          dipole_nodes=10, dipole_exit_face=DIPOLE_EXIT_FACE_Z,
          dipole_glide_step=1.0,
          strain_rate_s=None, dose=None, seed=0, loops=True, verbose=True):
    """Write a complete, runnable MoDELib case into `sim_dir`. Returns a dict.

    `route="A"` exports the population loop by loop (`pops`, from
    `cube_population`); `route="D"` seeds both families by DENSITY from `state`,
    which needs no export at all and is the plan's §10 smoke test.

    `loops=False` writes THE CONTROL: the same cell, the same sources, the same
    seed, and no irradiation loops. DELTA-TAU is a difference, so without this
    run there is no measurement -- the absolute stress carries the Peierls law
    and the source configuration, neither of which is what is being asked.
    """
    mat = material or Material.from_file()
    if tau_dot_MPa_s is None:
        tau_dot_MPa_s = default_tau_dot_MPa_s(mat)
    sim_dir = Path(sim_dir)
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    (sim_dir / "evl").mkdir(parents=True)
    (sim_dir / "F").mkdir()
    inp = sim_dir / "inputFiles"
    inp.mkdir()

    # DD.txt, from the Library rather than from a staged CD case: this deck has
    # nothing to do with cluster dynamics and should not inherit its settings.
    minputs.copy_lf(paths.MODELIB_ROOT / "Library" / "DislocationDynamics" / "DD.txt",
                    inp / "DD.txt")
    dd = dict(_DD_ROUTE_A)
    dd["Nsteps"] = int(n_steps)
    dd["outputFrequency"] = int(output_frequency)
    dd["useSubCycling"] = 1 if subcycling else 0
    for k, v in dd.items():
        minputs.set_dd_scalar(inp / "DD.txt", k, v)

    minputs.copy_lf(mat.file, inp / mat.file.name)
    mesh = paths.MODELIB_ROOT / "Library" / "Meshes" / "unitCube24.msh"
    minputs.copy_lf(mesh, inp / mesh.name)

    F = box.F
    (inp / "polycrystal.txt").write_text(_POLYCRYSTAL.format(
        material=mat.file.name, T=mat.temperature_K, mesh=mesh.name,
        F00=F[0, 0], F11=F[1, 1], F22=F[2, 2]), encoding="utf-8")

    # The load: a pure resolved-shear ramp on the target system, stress control
    # on every component (stiffnessRatio = 0), zero at t = 0.
    s, n = slip
    rate = mat.stress_rate_modelib(float(tau_dot_MPa_s) * 1e6) * ramp_voigt(s, n)
    gr = np.zeros(6)
    sk = np.zeros(6)
    if strain_rate_s is not None:                     # route B, if ever wanted
        gr = float(strain_rate_s) * mat.t_unit_SI * ramp_voigt(s, n)
        sk = 1e20 * np.abs(ramp_voigt(s, n))
        rate = np.zeros(6)

    def row(v):
        return " ".join(f"{float(x):.10g}" for x in v)

    (inp / "ElasticDeformation.txt").write_text(_ELASTIC.format(
        s0=row(np.zeros(6)), sr=row(rate), g0=row(np.zeros(6)), gr=row(gr),
        sk=row(sk), tau_dot_MPa_s=float(tau_dot_MPa_s), mu=mat.mu_SI,
        t=mat.t_unit_SI), encoding="utf-8")

    micro_files, counts = [], {}
    if loops:
        if route.upper() == "A":
            if pops is None:
                raise ValueError("route A needs pops from cube_population")
            n_loops = write_microstructure(pops, inp / "aLoops_hardening.txt")
            micro_files.append("aLoops_hardening.txt")
            counts = {p.fam["key"]: len(p) for p in pops}
            counts["total"] = n_loops
        elif route.upper() == "D":
            if state is None:
                raise ValueError("route D needs a state from interior_state")
            micro_files += _write_density_microstructures(inp, state)
            counts = {"style": "density"}
        else:
            raise ValueError(f"route must be A, D (or B via strain_rate_s), "
                             f"got {route!r}")

    # The mobile population: ONE periodic dipole, on the slip system the load is
    # resolved on, in the same place at every dose. Individual style rather than
    # `periodicDipolesDensity`, which the plan suggests, because the density
    # generator picks the slip system itself -- and it picked a BASAL one, whose
    # normal is [0,0,1] and whose resolved shear under a prismatic load is
    # exactly zero. Measured: 3000 steps to 41 MPa with gamma_p = 8e-22, i.e. a
    # source that cannot move however hard it is pulled.
    if source_density_m2 is not None:
        (inp / "periodicDipolesDensity.txt").write_text(
            "type=PeriodicDipole;\nstyle=density;\n"
            f"targetDensity={float(source_density_m2):g}; # [m^-2]\n",
            encoding="utf-8")
        micro_files.append("periodicDipolesDensity.txt")
    else:
        (inp / "periodicDipoleIndividual.txt").write_text(
            "type=PeriodicDipole;\nstyle=individual;\n"
            f"slipSystemIDs={int(source_slip_system)};\n"
            f"exitFaceIDs={int(dipole_exit_face)};\n"
            "dipoleCenters=0 0 0;\n"
            f"dipoleHeights={int(dipole_height)}; # [slip planes] dipole arm separation\n"
            f"nodesPerLine={int(dipole_nodes)};\n"
            # NOT ZERO. `PeriodicDipoleGenerator::generateSingle` builds the
            # sessile prismatic loop unconditionally but the two GLISSILE arms
            # only `if(std::fabs(glideStep)>FLT_EPSILON)`. With glideSteps=0 the
            # case therefore contains one sessile loop and nothing that can
            # move: measured 3000 steps with vMax=0, glissile density 0 and
            # gamma_p exactly 0, which reads exactly like a pinned source.
            f"glideSteps={float(dipole_glide_step):g}; # [b] non-zero, or no glissile arms\n",
            encoding="utf-8")
        micro_files.append("periodicDipoleIndividual.txt")

    (inp / "initialMicrostructure.txt").write_text(
        "".join(f"microstructureFile={f};\n" for f in micro_files),
        encoding="utf-8")

    meta = dict(route=route.upper(), loops=bool(loops), dose=dose, seed=seed,
                box=dict(n=box.n, m=box.m, p=box.p,
                         edges_nm=list(box.edges_nm),
                         volume_m3=box.volume_m3),
                tau_dot_MPa_s=float(tau_dot_MPa_s),
                tau_dot_modelib=float(np.linalg.norm(rate)),
                slip_s=list(map(float, s)), slip_n=list(map(float, n)),
                n_steps=int(n_steps), output_frequency=int(output_frequency),
                subcycling=bool(subcycling),
                source=("density" if source_density_m2 is not None
                        else f"individual dipole on slip system "
                             f"{int(source_slip_system)}"),
                source_density_m2=(float(source_density_m2)
                                   if source_density_m2 is not None else None),
                source_slip_system=int(source_slip_system),
                loop_counts=counts, material=mat.file.name,
                temperature_K=mat.temperature_K,
                git_hash=paths.git_hash())
    (sim_dir / "hardening.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")
    if verbose:
        print(f"staged {sim_dir}")
        print(f"  route {meta['route']}, {box.describe()}")
        print(f"  loops: {counts if loops else 'NONE (control run)'}")
        print(f"  ramp {tau_dot_MPa_s:.4g} MPa/s = {np.linalg.norm(rate):.4g} "
              f"mu per b/cs = 1 MPa per "
              f"{1.0 / (tau_dot_MPa_s * mat.t_unit_SI):.4g} b/cs")
        print(f"  resolved on s={[float(x) for x in s]} n={[float(x) for x in n]}")
    return dict(sim_dir=sim_dir, meta=meta)


def _write_density_microstructures(inp, state):
    """Route D: seed both families by density, with no per-loop export.

    Two different generators, because they mean different things by a loop:

    * <a> -> `aLoopsDensity`, whose `targetDensity` is a NUMBER density in m^-3
      (`aLoopGenerator.cpp:127` divides the loop count by the volume). The
      shipped template's `ellipticityFactor=1.5` is overridden to 1: the
      continuum field carries circular loops, and an ellipse of the same
      semi-minor axis holds 1.5x the defects.
    * <c> -> `frankLoopsDensity`, whose `targetDensity` is a LINE density in
      m^-2 (`FrankLoopsGenerator.cpp:89` accumulates 2*pi*r per loop). It is
      the right generator for the basal family because `burgersFactor=0.5`
      against the basal plane spacing gives |b| = c/2 -- the physical <c> loop
      Burgers vector -- while `aLoopsDensity` would take b from the slip
      system's own direction, which for a basal system is an <a> vector lying
      IN the plane, i.e. not a <c> loop at all.
    """
    fams = {f["key"]: f for f in state["families"]}
    a_keys = ("a1", "a2", "a3")
    a_ss = {"a1": 6, "a2": 8, "a3": 10}     # prismatic pairs, as FAMILIES has it

    def row(vals, fmt="{:.6g}"):
        return " ".join(fmt.format(v) for v in vals)

    ss = [a_ss[k] for k in a_keys]
    n_a = [fams[k]["N_m3"] for k in a_keys]              # [m^-3], number density
    r_a = [fams[k]["r_nm"] * 1e-9 for k in a_keys]       # [m]
    (inp / "aLoopsDensity.txt").write_text(
        "type=aLoops;\nstyle=Density;\n"
        f"slipSystemIDs={row(ss, '{:d}')};\n"
        f"targetDensity={row(n_a)}; # [m^-3] NUMBER density\n"
        f"loopRadiusMean={row(r_a)};\n"
        f"loopRadiusStd={row([0.0] * 3)};\n"
        f"numberOfSides={row([16] * 3, '{:d}')};\n"
        f"areVacancyLoops={row([0] * 3, '{:d}')}; # <a> loops are interstitial\n"
        f"ellipticityFactor={row([1.0] * 3)}; # circular, NOT the template's 1.5\n",
        encoding="utf-8")

    c = fams["c"]
    (inp / "frankLoopsDensity.txt").write_text(
        "type=FrankLoops;\nstyle=density;\n"
        f"targetDensity={c['rho_m2']:.6g}; # [m^-2] LINE density\n"
        "planeIDs=0; # basal\n"
        f"radiusDistributionMean={c['r_nm'] * 1e-9:.6g};\n"
        "radiusDistributionStd=0;\n"
        "numberOfSides=12;\n"
        "burgersFactor=0.5; # |b| = c/2, the physical <c> loop Burgers vector\n"
        "areVacancyLoops=1;\n", encoding="utf-8")
    return ["frankLoopsDensity.txt", "aLoopsDensity.txt"]


# ── running ──────────────────────────────────────────────────────────────────
def run_case(sim_dir, generate=True, verbose=True, timeout=None):
    """`microstructureGenerator` then `DDomp`, on this machine. Returns timings.

    Both commands come from `paths`, which decides between a native binary and
    a WSL invocation, and both run WITH THE CASE AS THEIR WORKING DIRECTORY --
    each writes `evl/` and `F/` to a relative path, so from anywhere else the
    output lands outside the case.
    """
    sim_dir = Path(sim_dir)
    out = dict(sim_dir=str(sim_dir))
    if generate:
        # microstructureGenerator is NOT idempotent: run it on a case that
        # already carries an `evl/evl_0.txt` and it segfaults (exit -11). Since
        # it rebuilds evl_0 from the microstructure files anyway, the previous
        # output is worth nothing -- clear it rather than leaving a rerun to
        # fail in a way that looks like a generator bug.
        for f in list((sim_dir / "evl").glob("*.txt")) + \
                 list((sim_dir / "F").glob("*.txt")):
            f.unlink()
        cmd, cwd = paths.generator_cmd(sim_dir)
        if verbose:
            print(f"  generating microstructure in {sim_dir.name}", flush=True)
        t0 = time.perf_counter()
        log = sim_dir / "generator.log"
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            r = subprocess.run(cmd, cwd=cwd, stdout=fh,
                               stderr=subprocess.STDOUT, text=True)
        out["generator_s"] = time.perf_counter() - t0
        txt = log.read_text(encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"microstructureGenerator exited {r.returncode}; "
                               f"tail:\n{txt[-3000:]}")
        out["loops_generated"] = txt.count("Generating")
        evl0 = sim_dir / "evl" / "evl_0.txt"
        if not evl0.is_file():
            raise RuntimeError(f"the generator wrote no evl_0 in {sim_dir}")
        out["n_loops"] = int(mfield.EvlFile(evl0).n_loops)
        # `n_loops` is every loop row in evl_0, which is NOT the obstacle
        # population: the periodic dipole source adds three of its own -- one
        # sessile prismatic loop and the two glissile arms -- so reporting it
        # as "loops" against a dose overstates the obstacle field by exactly
        # three at every dose. `stage` recorded what it actually wrote, so take
        # the count from there and keep the difference visible rather than
        # subtracting a hard-coded 3.
        meta = sim_dir / "hardening.json"
        if meta.is_file():
            counts = json.loads(meta.read_text(encoding="utf-8")).get(
                "loop_counts") or {}
            if "total" in counts:
                out["n_obstacles"] = int(counts["total"])
                out["n_source_loops"] = out["n_loops"] - out["n_obstacles"]

    # A previous attempt that died leaves zero-byte F files behind, and the next
    # DDomp segfaults reading them -- so without this, the second run of a case
    # that failed for any reason fails differently and unrecognizably.
    minputs.clear_empty_F(sim_dir)

    cmd, cwd = paths.ddomp_cmd(sim_dir, in_case_dir=True)
    if verbose:
        print(f"  DDomp in {sim_dir.name}", flush=True)
    t0 = time.perf_counter()
    log = sim_dir / "ddomp.log"
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
    out["wall_s"] = time.perf_counter() - t0
    txt = log.read_text(encoding="utf-8", errors="replace")
    out["returncode"] = r.returncode
    # runID lines are the reliable step counter: Nsteps may not be reached if
    # the run is stopped, and F_0.txt only carries every outputFrequency-th step.
    out["steps"] = txt.count("runID=")
    out["s_per_step"] = (out["wall_s"] / out["steps"]) if out["steps"] else float("nan")

    # A REALIZATION THAT MoDELib WILL NOT ACCEPT, and it is worth naming rather
    # than leaving as a zero-step run with exit 0. Some loop placements produce
    # a network node a few b outside the primary cell -- the periodic patch
    # construction in MicrostructureGenerator::insertJunctionLoop maps polygon
    # vertices back into the cell, and for a few straddling loops per thousand
    # the result lands just outside -- and DislocationNode refuses it at
    # startup. Measured on the 500 nm cube at 1e-2 dpa: seeds 0 and 1 fail,
    # seeds 2 and 3 run, same population statistics. It is a property of the
    # DRAW, so the remedy is another draw, which is free: the plan asks for
    # three seeds per dose anyway.
    if "DISLOCATION NODE OUTSIDE MESH" in txt:
        out["rejected_placement"] = True
        node = next((l for l in txt.splitlines()
                     if l.startswith("PlanarDislocationNode")), "")
        raise PlacementRejected(
            f"MoDELib refused this realization: a network node fell outside the "
            f"primary cell ({node.strip()}).\n"
            f"  This is the draw, not the configuration -- restage with a "
            f"different --seed and it runs.\n"
            f"  Log: {log}")
    if verbose:
        print(f"    {out['steps']} steps in {out['wall_s']:.0f} s "
              f"= {out['s_per_step']:.3g} s/step (exit {r.returncode})")
    return out


# ── §3  the reduction ────────────────────────────────────────────────────────
def read_F(sim_dir):
    """`(labels, array)` from `F/F_0.txt` + `F/F_labels.txt`.

    Label-driven rather than positional: what is in that file depends on which
    microstructures are enabled and on four separate output flags, so counting
    columns is how a reduction silently reads the wrong quantity.
    """
    sim_dir = Path(sim_dir)
    lab = [l.strip() for l in
           (sim_dir / "F" / "F_labels.txt").read_text(
               encoding="utf-8", errors="replace").splitlines() if l.strip()]

    # F_0.txt IS RAGGED, and np.loadtxt refuses it outright ("the number of
    # columns changed from 42 to 43 at row 2"). With
    # outputPlasticDistortionPerSlipSystem=1, DislocationNetwork writes one
    # column per slip system that has ACQUIRED plastic distortion, and that set
    # grows during the run -- while F_labels.txt is written once, at runID 0.
    # Every column this reduction uses (runID, time, betaP, the uniform
    # controller's strain and stress) is in the fixed prefix that precedes the
    # per-slip-system block, so truncating to the shortest row is lossless here
    # and the alternative -- turning the flag off -- would discard the one
    # diagnostic that says which slip system carried the strain.
    rows = [r.split() for r in (sim_dir / "F" / "F_0.txt").read_text(
        encoding="utf-8", errors="replace").splitlines() if r.strip()]
    if not rows:
        raise ValueError(f"F/F_0.txt is empty in {sim_dir} -- the run wrote no "
                         f"output step")
    n = min(len(r) for r in rows)
    data = np.array([[float(x) for x in r[:n]] for r in rows], ndmin=2)
    if n < len(lab):
        lab = lab[:n]
    return lab, data


OFFSETS = (1e-5, 1e-4, 1e-3, 1e-2)


def _tau_at(tau_MPa, gamma_p, offset):
    """Stress at which `gamma_p` first reaches `offset`, linearly interpolated."""
    hit = np.nonzero(gamma_p >= offset)[0]
    if not len(hit):
        return float("nan"), False
    i = int(hit[0])
    # Linear interpolation between the last sub-offset point and the first over
    # it: with outputFrequency=100 the sampling is coarse enough that taking the
    # crossing point outright biases the answer high.
    if i and gamma_p[i] > gamma_p[i - 1]:
        f = (offset - gamma_p[i - 1]) / (gamma_p[i] - gamma_p[i - 1])
        return float(tau_MPa[i - 1] + f * (tau_MPa[i] - tau_MPa[i - 1])), True
    return float(tau_MPa[i]), True


def reduce_run(sim_dir, material=None, slip=None, offset=1e-3, verbose=True):
    """The tau-gamma_p curve and DELTA-TAU for one staged case.

        tau(t)     = s . sigma(t) . n        from the uniform controller's `s_ij`
        gamma_p(t) = s . betaP(t) . n        from DefectiveCrystal's plastic
                                             distortion, written every output step
        DELTA-TAU  = tau where gamma_p first exceeds `offset`

    The offset criterion is the plan's, and the number it returns is the CRSS of
    THIS cell, not the increment: the absolute value carries the Peierls law and
    whatever the source population does. Subtract the loop-free control run.
    """
    sim_dir = Path(sim_dir)
    mat = material or Material.from_file()
    meta = json.loads((sim_dir / "hardening.json").read_text(encoding="utf-8"))
    if slip is None:
        slip = (np.array(meta["slip_s"], float), np.array(meta["slip_n"], float))
    s, n = slip
    lab, data = read_F(sim_dir)
    col = {name: i for i, name in enumerate(lab)}

    def take(names):
        return np.column_stack([data[:, col[nm]] for nm in names])

    sigma = take([f"s_{i}" for i in ("11", "22", "33", "12", "23", "13")])
    beta = take([f"betaP_{i}{j}" for i in "123" for j in "123"])
    t_dd = data[:, col["time [b/cs]"]]

    tau_MPa = np.array([resolve(row, s, n) for row in sigma]) * mat.mu_SI / 1e6
    gamma_p = np.array([float(s @ b.reshape(3, 3) @ n) for b in beta])

    # BASELINE THE PLASTIC SHEAR AT t=0. The generated configuration is not in
    # equilibrium -- the dipole arms move as soon as the solve starts, before
    # any load is applied -- so gamma_p is already 5e-6 at zero stress on the
    # 200 nm cell. An offset criterion applied to the raw curve measures that
    # relaxation as if it were flow, and does it differently in every cell.
    gamma_0 = float(gamma_p[0]) if len(gamma_p) else 0.0
    gamma_p = gamma_p - gamma_0

    # THE OFFSET IS A CHOICE, AND ON THIS CURVE IT MATTERS. An irradiated cell
    # creeps well below its breakaway stress -- the line bows and unpins one
    # obstacle at a time -- so gamma_p crosses 1e-4 in the creep tail and 1e-2
    # only after general flow. Measured on the 200 nm cell at 10 dpa, the knee
    # is near 25-30 MPa while the 1e-4 crossing is at 3. Every offset is
    # reported and the primary one is stated with the result, rather than a
    # single number whose criterion has to be looked up.
    tau_c, reached = _tau_at(tau_MPa, gamma_p, offset)
    at_offsets = {f"{o:g}": _tau_at(tau_MPa, gamma_p, o)[0] for o in OFFSETS}

    out = dict(sim_dir=str(sim_dir), dose=meta.get("dose"),
               route=meta.get("route"), loops=meta.get("loops"),
               offset=offset, tau_offset_MPa=tau_c, reached_offset=reached,
               gamma_p_at_t0=gamma_0, tau_at_offsets=at_offsets,
               tau_max_MPa=float(tau_MPa.max()) if len(tau_MPa) else float("nan"),
               gamma_p_max=float(gamma_p.max()) if len(gamma_p) else 0.0,
               n_output=len(tau_MPa),
               time_s=float(t_dd[-1] * mat.t_unit_SI) if len(t_dd) else 0.0,
               tau_MPa=tau_MPa.tolist(), gamma_p=gamma_p.tolist(),
               time_dd=t_dd.tolist())
    if verbose:
        state = (f"tau({offset:g}) = {tau_c:.3f} MPa "
                 + "[" + " ".join(f"{k}:{v:.2f}" for k, v in at_offsets.items())
                 + "]" if reached else
                 f"gamma_p never reached {offset:g} "
                 f"(max {out['gamma_p_max']:.3g}); ramp further or longer")
        print(f"  {sim_dir.name}: {state}, tau_max = {out['tau_max_MPa']:.3f} MPa, "
              f"{out['n_output']} output steps")
    return out


def obstacle_count(row):
    """The number of OBSTACLE loops behind one campaign row.

    `n_loops` is every loop row in `evl_0`, and that is NOT the obstacle field:
    the periodic dipole source contributes three of its own -- one sessile
    prismatic loop and the two glissile arms -- so quoting it against a dose
    overstates the obstacle population by exactly three at every dose. `stage`
    recorded what it actually wrote into `loop_counts`, so campaigns run before
    `n_obstacles` existed are still recoverable from the case directory rather
    than by subtracting a constant that would be wrong the moment the source
    changes.
    """
    n = row.get("n_obstacles")
    if n is not None:
        return int(n)
    sim = row.get("sim_dir")
    if sim:
        f = Path(sim) / "hardening.json"
        if f.is_file():
            counts = json.loads(f.read_text(encoding="utf-8")).get(
                "loop_counts") or {}
            if "total" in counts:
                return int(counts["total"])
    return int(row.get("n_loops") or 0)


def delta_tau(irradiated, control):
    """DELTA-TAU = tau_offset(loops) - tau_offset(no loops), in MPa."""
    return float(irradiated["tau_offset_MPa"] - control["tau_offset_MPa"])


# ── §5  route C: the dispersed-barrier law ───────────────────────────────────
def dispersed_barrier(state, alpha, material=None, taylor_M=None):
    """`DELTA-TAU_k = alpha_k mu b sqrt(N_k d_k)`, root-sum-square over families.

    `alpha` is a dict keyed by family (`c`, `a1`, `a2`, `a3`) or by group
    (`c`, `a`). THERE IS NO DEFAULT: the alphas are the unknowns route A exists
    to fit, and a plausible-looking default would be indistinguishable in the
    output from a fitted one. The plan's §5 estimates (0.4 for the sessile
    basal loops, 0.2 for the prismatic ones that share the glide Burgers
    vector) are a starting point, not a result.

    `d_k = 2*r_k` is the obstacle DIAMETER, which is what the barrier spacing
    `1/sqrt(N d)` is built from.
    """
    mat = material or Material.from_file()
    per = []
    for f in state["families"]:
        a = alpha.get(f["key"], alpha.get("a" if f["key"] != "c" else "c"))
        if a is None:
            raise KeyError(f"no alpha for family {f['key']}")
        d_m = 2.0 * f["r_nm"] * 1e-9
        dtau = a * mat.mu_SI * mat.b_SI * np.sqrt(max(f["N_m3"], 0.0) * d_m)
        per.append(dict(family=f["key"], alpha=float(a),
                        N_m3=f["N_m3"], d_nm=2.0 * f["r_nm"],
                        dtau_MPa=float(dtau / 1e6)))
    total = float(np.sqrt(sum(p["dtau_MPa"] ** 2 for p in per)))
    out = dict(dose=state["dose"], per_family=per, dtau_MPa=total,
               dtau_c_MPa=float(np.sqrt(sum(p["dtau_MPa"] ** 2 for p in per
                                            if p["family"] == "c"))),
               dtau_a_MPa=float(np.sqrt(sum(p["dtau_MPa"] ** 2 for p in per
                                            if p["family"] != "c"))))
    if taylor_M is not None:
        out["taylor_M"] = float(taylor_M)
        out["dsigma_y_MPa"] = total * float(taylor_M)
    return out


def fit_alpha(states, dd_delta_tau, material=None, alpha0=(0.4, 0.2)):
    """Fit `(alpha_c, alpha_a)` to DD points. `states` and `dd_delta_tau` align.

    Two parameters, so two DD points determine them and three over-determine
    them; the fit is a plain least squares on DELTA-TAU in MPa. Solved on a log
    grid rather than by a gradient method because with two parameters and three
    points that is exact enough and cannot fail to converge.
    """
    mat = material or Material.from_file()
    obs = np.asarray(dd_delta_tau, float)
    if len(obs) != len(states):
        raise ValueError("states and dd_delta_tau must have the same length")

    def predict(ac, aa):
        return np.array([dispersed_barrier(s, {"c": ac, "a": aa}, mat)["dtau_MPa"]
                         for s in states])

    grid = np.geomspace(0.02, 2.0, 121)
    best, err = None, np.inf
    for ac in grid:
        for aa in grid:
            e = float(np.sum((predict(ac, aa) - obs) ** 2))
            if e < err:
                best, err = (float(ac), float(aa)), e
    return dict(alpha_c=best[0], alpha_a=best[1], rms_MPa=float(np.sqrt(err / len(obs))),
                predicted=predict(*best).tolist(), observed=obs.tolist())


def format_routec(rows, taylor_M=None):
    head = f"{'dpa':>8} {'dtau_c':>9} {'dtau_a':>9} {'dtau':>9}"
    if taylor_M:
        head += f" {'dsigma_y':>10}"
    out = [head, f"{'':>8} {'[MPa]':>9} {'[MPa]':>9} {'[MPa]':>9}"
           + (f" {'[MPa]':>10}" if taylor_M else "")]
    for r in rows:
        line = (f"{r['dose']:>8.4g} {r['dtau_c_MPa']:>9.1f} "
                f"{r['dtau_a_MPa']:>9.1f} {r['dtau_MPa']:>9.1f}")
        if taylor_M:
            line += f" {r['dtau_MPa'] * taylor_M:>10.1f}"
        out.append(line)
    return "\n".join(out)


# ── §7 step 5 / §10 step 4: the campaign ─────────────────────────────────────
def campaign(out_root, doses, L_nm=200.0, run_dir=None, material=None,
             tau_dot_MPa_s=None, n_steps=20000, output_frequency=100,
             seed=0, max_seeds=6, offset=1e-3, alpha_fit=True, verbose=True,
             tag="camp", **stage_kw):
    """Route A at every dose in `doses`, plus ONE control, then fit alpha.

    The control is the same cell with the same source and the same seed and no
    irradiation loops, run once: DELTA-TAU is a difference, and the absolute
    CRSS carries the Peierls law and the source geometry, neither of which is
    what is being measured.

    A realization MoDELib refuses (`PlacementRejected`) is redrawn with the next
    seed rather than abandoned, up to `max_seeds`. That is not papering over a
    failure: the rejection is a property of the draw, the plan asks for several
    seeds per dose anyway, and every seed actually used is recorded.
    """
    mat = material or Material.from_file()
    out_root = Path(out_root)
    box = commensurate_box(L_nm, mat)
    states = (state_table(run_dir, mat) if run_dir else plan_states(mat))
    by_dose = {s["dose"]: s for s in states}

    def nearest(d):
        return by_dose[min(by_dose, key=lambda x: abs(x - d))]

    if verbose:
        print(f"campaign: {box.describe()}")
        print(f"  {len(doses)} doses + 1 control, {n_steps} steps each")

    ctrl_dir = out_root.parent / f"{out_root.name}_control" \
        if out_root.name else out_root / "control"
    stage(ctrl_dir, box, mat, route="A", loops=False,
          tau_dot_MPa_s=tau_dot_MPa_s, n_steps=n_steps,
          output_frequency=output_frequency, seed=seed, verbose=verbose,
          **stage_kw)
    run_case(ctrl_dir, verbose=verbose)
    control = reduce_run(ctrl_dir, mat, offset=offset, verbose=verbose)

    rows = []
    for d in doses:
        st = nearest(d)
        for k in range(max_seeds):
            sd = seed + k
            sim = out_root.parent / f"{out_root.name}_{tag}_{st['dose']:g}dpa_s{sd}"
            pops, pstats = cube_population(
                box, mat, run_dir=run_dir,
                dose=(st["dose"] if run_dir else None),
                state=(None if run_dir else st), seed=sd, verbose=verbose)
            stage(sim, box, mat, route="A", pops=pops, tau_dot_MPa_s=tau_dot_MPa_s,
                  n_steps=n_steps, output_frequency=output_frequency,
                  dose=st["dose"], seed=sd, verbose=verbose, **stage_kw)
            try:
                timing = run_case(sim, verbose=verbose)
            except PlacementRejected as e:
                if verbose:
                    print(f"    seed {sd} refused, redrawing: {str(e).splitlines()[0]}")
                continue
            r = reduce_run(sim, mat, offset=offset, verbose=verbose)
            r["seed"] = sd
            r["s_per_step"] = timing["s_per_step"]
            r["n_loops"] = timing.get("n_loops")
            r["n_obstacles"] = timing.get("n_obstacles")
            r["n_source_loops"] = timing.get("n_source_loops")
            r["delta_tau_MPa"] = delta_tau(r, control)
            r["population"] = pstats
            rows.append(r)
            break
        else:
            raise PlacementRejected(
                f"no accepted realization at {st['dose']:g} dpa in {max_seeds} seeds")

    out = dict(box=box.describe(), control=control, runs=rows,
               doses=[r["dose"] for r in rows],
               delta_tau_MPa=[r["delta_tau_MPa"] for r in rows])
    if alpha_fit and len(rows) >= 2:
        fit_states = [nearest(r["dose"]) for r in rows]
        out["alpha"] = fit_alpha(fit_states, out["delta_tau_MPa"], mat)
        if verbose:
            a = out["alpha"]
            print(f"\n  fitted alpha_c = {a['alpha_c']:.3f}, "
                  f"alpha_a = {a['alpha_a']:.3f}  (rms {a['rms_MPa']:.2f} MPa)")
    (out_root.parent / f"{out_root.name}_campaign.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


# ── the plan's §2 table, for when the source run is not on this machine ──────
# Transcribed from `Docs/DisloCluster Manual/architecture/irradiation_hardening_dd_plan.md`
# §2, measured there on
#   Simulations/output/20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent
# (500 nm hexagonal prism, route Adaptive, loop_model=1, 573 K, 1e-7 dpa/s).
#
# THIS IS A CACHE, NOT A SOURCE. Run directories are git-ignored, so the march
# behind these numbers lives only on the machine that produced it; with it
# present, `interior_state` recomputes every column from `march_state.npz` and
# `--run` should be preferred. Columns: dose, N_c [m^-3], r_c [nm], N_a [m^-3]
# summed over the three variants, r_a [nm].
PLAN_RUN = "20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent"
PLAN_STATE_TABLE = [
    (1e-4, 5.18e19,  3.41, 4.76e19, 2.10),
    (1e-3, 3.86e20,  3.53, 4.68e20, 2.14),
    (1e-2, 7.67e20,  5.41, 3.77e21, 2.51),
    (0.1,  7.66e20, 25.37, 1.18e22, 2.74),
    (1.0,  7.60e20, 56.78, 6.68e21, 3.32),
    (2.0,  7.61e20, 55.56, 4.18e21, 3.76),
    (5.0,  7.61e20, 55.52, 4.19e21, 3.77),
    (10.0, 7.61e20, 55.55, 4.20e21, 3.77),
]


def state_from_row(dose, N_c, r_c_nm, N_a, r_a_nm, material=None):
    """An `interior_state`-shaped dict from tabulated `(N, r)` per family.

    The <a> total is split equally over the three prism variants, which is what
    the march carries at zero applied stress (`variant_weights = 1/3` each, and
    the matched run measured each variant at exactly 1/3).
    """
    mat = material or Material.from_file()
    fams = []
    for fam in FAMILIES:
        is_c = fam["key"] == "c"
        N = float(N_c) if is_c else float(N_a) / 3.0
        r_nm = float(r_c_nm) if is_c else float(r_a_nm)
        r_b = r_nm * 1e-9 / mat.b_SI
        n_b3 = N * mat.b_SI ** 3
        # Invert r = sqrt(m*Omega/(pi*b)) for the content the field would carry.
        m = np.pi * fam["b_dd"] * r_b ** 2 / mat.omega_b3
        fams.append(dict(key=fam["key"], label=fam["label"], n_b3=n_b3,
                         c_at=m * n_b3 * mat.omega_b3, N_m3=N, m_defects=m,
                         r_b=r_b, r_nm=r_nm,
                         rho_m2=2.0 * np.pi * r_nm * 1e-9 * N,
                         b_dd=fam["b_dd"]))
    a = [f for f in fams if f["key"] != "c"]
    return dict(run=f"table:{PLAN_RUN}", dose=float(dose), n_nodes=None,
                n_interior=None, frac=None, families=fams,
                N_c_m3=float(N_c), r_c_nm=float(r_c_nm),
                rho_c_m2=fams[0]["rho_m2"], N_a_m3=float(N_a),
                r_a_nm=float(r_a_nm),
                rho_a_m2=sum(f["rho_m2"] for f in a))


def plan_states(material=None):
    """Every row of `PLAN_STATE_TABLE`, as `interior_state` dicts."""
    return [state_from_row(*row, material=material) for row in PLAN_STATE_TABLE]


def states_for(run_dir=None, material=None, frac=0.5):
    """The state table from a run if one is given, else the plan's §2 cache."""
    if run_dir:
        return state_table(run_dir, material, frac)
    return plan_states(material)


# ── regression against the plan's own tables ─────────────────────────────────
# Two of the plan's tables are pure arithmetic on measured quantities, so they
# are checkable without a run and worth keeping checkable: they are what says
# this module implements THAT plan and not a similar one.
PLAN_BOX_TABLE = {          # §6: L_nm -> (n, m, p)
    200: (619, 357, 388), 300: (929, 536, 583),
    500: (1548, 894, 971), 800: (2477, 1430, 1553),
}
PLAN_ROUTEC_TABLE = {       # §5 at alpha_c=0.4, alpha_a=0.2: dpa -> (dtau_c, dtau_a, dtau)
    1e-4: (2.5, 1.0, 2.7), 1e-3: (7.0, 3.0, 7.7), 1e-2: (12.3, 9.3, 15.4),
    0.1: (26.6, 17.2, 31.6), 1.0: (39.6, 14.2, 42.1), 10.0: (39.2, 12.0, 41.0),
}


def verify(material=None, tol=0.15, verbose=True):
    """Check this module against the plan's §5 and §6 tables. `(ok, lines)`."""
    mat = material or Material.from_file()
    lines, ok = [], True

    for L, want in PLAN_BOX_TABLE.items():
        got = commensurate_box(L, mat)
        good = (got.n, got.m, got.p) == want
        ok &= good
        lines.append(f"  {'OK ' if good else 'FAIL'}  box {L} nm: "
                     f"(n,m,p) = {(got.n, got.m, got.p)}, plan {want}")

    for st in plan_states(mat):
        want = PLAN_ROUTEC_TABLE.get(st["dose"])
        if want is None:
            continue
        got = dispersed_barrier(st, {"c": 0.4, "a": 0.2}, mat)
        vals = (got["dtau_c_MPa"], got["dtau_a_MPa"], got["dtau_MPa"])
        good = all(abs(g - w) <= tol for g, w in zip(vals, want))
        ok &= good
        lines.append(f"  {'OK ' if good else 'FAIL'}  route C {st['dose']:>6g} dpa: "
                     + " ".join(f"{g:.1f}" for g in vals)
                     + "  plan " + " ".join(f"{w:.1f}" for w in want))

    v = resolve(ramp_voigt(*SLIP_PRISMATIC_A), *SLIP_PRISMATIC_A)
    good = abs(v - 1.0) < 1e-12
    ok &= good
    lines.append(f"  {'OK ' if good else 'FAIL'}  unit resolved shear: {v:.12g}")

    if verbose:
        print(mat.summary())
        print("\n".join(lines))
        print("  ALL CHECKS PASS" if ok else "  SOME CHECKS FAILED")
    return ok, lines



# ── figures ──────────────────────────────────────────────────────────────────
# Colour is assigned by the job it does, and dose is a MAGNITUDE, so the dose
# series take one hue light->dark (an ordinal blue ramp) rather than arbitrary
# categorical hues. The steps are the validated ones: on a light surface an
# ordinal ramp starts no lighter than step 250, so 250 / 450 / 700 for three
# doses. The control is not a dose -- it is the reference the increment is
# measured against -- so it is neutral grey and dashed. The two dispersed-barrier
# laws in the second figure ARE categorical (fitted vs assumed) and take
# categorical slots 1 and 2.
DOSE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"
CONTROL_GREY = "#8a8a85"
SLOT_1 = "#2a78d6"
SLOT_2 = "#eb6834"


def _dose_colors(n):
    """`n` steps of the ordinal ramp, spread over its validated range."""
    if n <= 1:
        return [DOSE_RAMP[2]]
    idx = np.linspace(0, len(DOSE_RAMP) - 1, n)
    return [DOSE_RAMP[int(round(i))] for i in idx]


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_color(INK_MUTED)


def plot_flow_curves(campaign, out_file, offset=None, title=None, dpi=160):
    """The tau-gamma_p curves: one line per dose, plus the loop-free control.

    This is the measurement itself, and the reason to draw it rather than quote
    DELTA-TAU alone is that the shape is what says whether the number means
    anything: an obstacle-limited cell creeps, then breaks away, and where the
    offset crosses that curve decides the answer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = campaign["runs"]
    ctrl = campaign["control"]
    offset = offset or ctrl.get("offset", 1e-3)
    colors = _dose_colors(len(runs))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _style_axes(ax)

    def crossing(r):
        y = r["tau_at_offsets"].get(f"{offset:g}")
        return float(y) if y is not None and np.isfinite(y) else None

    # The crossing value goes in the LEGEND ENTRY, not in a label beside the
    # marker. Two of these sit 0.3 MPa apart (13.40 and 13.06), so annotations
    # at the marker either overlap or have to be displaced until it is no longer
    # clear which marker each belongs to. The legend already carries identity;
    # the value rides along with it and nothing collides.
    yc = crossing(ctrl)
    ax.plot(ctrl["gamma_p"], ctrl["tau_MPa"], color=CONTROL_GREY, lw=2.0,
            ls=(0, (5, 3)), zorder=3,
            label="no loops (control)"
                  + (fr"   $\tau$ = {yc:.1f} MPa" if yc is not None else ""))
    for r, c in zip(runs, colors):
        y = crossing(r)
        ax.plot(r["gamma_p"], r["tau_MPa"], color=c, lw=2.0, zorder=4,
                label=f"{r['dose']:g} dpa"
                      + (fr"   $\tau$ = {y:.1f} MPa" if y is not None else ""))

    # The offset criterion, drawn: a vertical rule and one marker per curve at
    # its crossing.
    ax.axvline(offset, color=INK_MUTED, lw=1.0, ls=":", zorder=2)
    ax.annotate(fr"offset $\gamma_p$ = {offset:g}",
                xy=(offset, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=INK_MUTED, fontsize=9,
                va="top")
    for r, c in [(ctrl, CONTROL_GREY)] + list(zip(runs, colors)):
        y = crossing(r)
        if y is not None:
            ax.plot([offset], [y], "o", ms=8, color=c, mec="white", mew=1.5,
                    zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel(r"resolved plastic shear  $\gamma_p$  [-]", color=INK,
                  fontsize=10)
    ax.set_ylabel(r"resolved shear stress  $\tau$  [MPa]", color=INK,
                  fontsize=10)
    ax.set_title(title or "Flow curves on the irradiated cell", color=INK,
                 fontsize=12, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_file


def plot_delta_tau(campaign, states, out_file, alpha_assumed=(0.4, 0.2),
                   material=None, taylor_M=None, title=None, dpi=160):
    """DELTA-TAU against dose: the DD anchors, and route C through them.

    Two laws are drawn, and the gap between them is the point of the figure --
    the plan's ASSUMED alpha against the one fitted to these DD points. One
    axis: `taylor_M` adds a second tick set on the right, which is the same
    quantity rescaled, not a second measure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = material or Material.from_file()
    fit = campaign.get("alpha")
    doses = np.array([s["dose"] for s in states], float)

    def law(ac, aa):
        return np.array([dispersed_barrier(s, {"c": ac, "a": aa}, mat)["dtau_MPa"]
                         for s in states])

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _style_axes(ax)

    ax.plot(doses, law(*alpha_assumed), color=SLOT_2, lw=2.0, zorder=3,
            label=fr"route C, assumed $\alpha_c$={alpha_assumed[0]:g}, "
                  fr"$\alpha_a$={alpha_assumed[1]:g}")
    if fit:
        ax.plot(doses, law(fit["alpha_c"], fit["alpha_a"]), color=SLOT_1,
                lw=2.0, zorder=4,
                label=fr"route C, fitted $\alpha_c$={fit['alpha_c']:.3f}, "
                      fr"$\alpha_a$={fit['alpha_a']:.3f}")

    dd_x = [r["dose"] for r in campaign["runs"]]
    dd_y = [r["delta_tau_MPa"] for r in campaign["runs"]]
    ax.plot(dd_x, dd_y, "o", ms=9, color=INK, mec="white", mew=1.5, zorder=6,
            label="DD (route A), one realization")
    for x, y in zip(dd_x, dd_y):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, color=INK)

    ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("dose [dpa]", color=INK, fontsize=10)
    ax.set_ylabel(r"CRSS increment  $\Delta\tau$  [MPa]", color=INK, fontsize=10)
    ax.set_title(title or "Irradiation hardening against dose", color=INK,
                 fontsize=12, loc="left", pad=12)
    if taylor_M:
        r = ax.secondary_yaxis("right", functions=(lambda v: v * taylor_M,
                                                   lambda v: v / taylor_M))
        r.set_ylabel(fr"$\Delta\sigma_y = M\,\Delta\tau$  [MPa],  M = {taylor_M:g}",
                     color=INK_MUTED, fontsize=10)
        r.tick_params(colors=INK_MUTED, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_file


def figures(campaign_json, out_dir=None, material=None, run_dir=None,
            taylor_M=3.0, render_loops=True, verbose=True):
    """Every figure for one campaign. Returns the paths written.

    The loop renders reuse `discrete_loops.render`, so the obstacle field is
    drawn by the same code that draws it for a march -- same tube widths, same
    wireframe, same orientation triad.
    """
    from dislocluster_code.post.discrete_loops import render

    mat = material or Material.from_file()
    campaign_json = Path(campaign_json)
    camp = json.loads(campaign_json.read_text(encoding="utf-8"))
    out_dir = Path(out_dir or campaign_json.with_suffix("")) 
    out_dir.mkdir(parents=True, exist_ok=True)

    states = states_for(run_dir, mat)
    written = [plot_flow_curves(camp, out_dir / "flow_curves.png"),
               plot_delta_tau(camp, states, out_dir / "delta_tau_vs_dose.png",
                              material=mat, taylor_M=taylor_M)]

    if render_loops:
        for r in camp["runs"]:
            meta = json.loads((Path(r["sim_dir"]) / "hardening.json").read_text(
                encoding="utf-8"))
            box = BoxSpec(n=meta["box"]["n"], m=meta["box"]["m"],
                          p=meta["box"]["p"], b_SI=mat.b_SI,
                          c_over_a=mat.c_over_a)
            st = min(states, key=lambda s: abs(s["dose"] - r["dose"]))
            pops, _ = cube_population(box, mat, state=st, seed=r["seed"])
            lo = -0.5 * box.edges_b
            f = out_dir / f"loops_{r['dose']:g}dpa.png"
            render(pops, lo, lo + box.edges_b, f,
                   title=f"{box.edges_nm[0]:.0f} nm periodic cell, "
                         f"{r['dose']:g} dpa", verbose=False)
            written.append(f)

    if verbose:
        for f in written:
            print(f"  wrote {f}")
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────
def _add_common(ap):
    ap.add_argument("--material", default=None)
    ap.add_argument("--temperature", type=float, default=573.0)
    return ap


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Irradiation hardening from a coupled march (see "
                    "Docs/DisloCluster Manual/architecture/"
                    "irradiation_hardening_dd_plan.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = _add_common(sub.add_parser("state", help="interior state, dose by dose"))
    p.add_argument("run", nargs="?", default=None,
                   help="a march run directory; omitted, the plan's cached table")
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--json", default=None)

    p = _add_common(sub.add_parser("box", help="the commensurate periodic cube"))
    p.add_argument("--L", type=float, nargs="+", default=[200, 300, 500, 800])

    p = _add_common(sub.add_parser("routec", help="dispersed-barrier hardening"))
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("--alpha-c", type=float, required=True)
    p.add_argument("--alpha-a", type=float, required=True)
    p.add_argument("--taylor-M", type=float, default=None)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--json", default=None)

    p = _add_common(sub.add_parser("stage", help="write one DD case"))
    p.add_argument("sim_dir")
    p.add_argument("--run", default=None)
    p.add_argument("--dose", type=float, default=None)
    p.add_argument("--L", type=float, default=500.0)
    p.add_argument("--route", default="A", choices=["A", "D"])
    p.add_argument("--no-loops", action="store_true", help="the control run")
    p.add_argument("--tau-dot", type=float, default=None,
                   help="[MPa/s] stress ramp rate; default is 1 MPa per 1e4 b/cs "
                        "(6.97e8 MPa/s on Zr3d_ghoniem)")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--output-every", type=int, default=100)
    p.add_argument("--no-subcycling", action="store_true")
    p.add_argument("--source-density", type=float, default=None,
                   help="seed sources by density instead of one dipole on the "
                        "loaded slip system (NOT recommended: the density "
                        "generator chooses the system, and a basal choice has "
                        "zero resolved shear under this load)")
    p.add_argument("--dipole-height", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-it", action="store_true", help="generate and solve now")

    _add_common(sub.add_parser("verify", help="check against the plan's tables"))

    p = _add_common(sub.add_parser("campaign", help="route A at several doses + control"))
    p.add_argument("out_root")
    p.add_argument("--doses", type=float, nargs="+", default=[1e-2, 0.1, 10.0])
    p.add_argument("--run", default=None)
    p.add_argument("--L", type=float, default=200.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--output-every", type=int, default=100)
    p.add_argument("--tau-dot", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset", type=float, default=1e-3)

    p = _add_common(sub.add_parser("plot", help="figures for a finished campaign"))
    p.add_argument("campaign_json")
    p.add_argument("--out", default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--taylor-M", type=float, default=3.0)
    p.add_argument("--no-loop-renders", action="store_true")

    p = _add_common(sub.add_parser("run", help="generator + DDomp on a staged case"))
    p.add_argument("sim_dir")
    p.add_argument("--no-generate", action="store_true")

    p = _add_common(sub.add_parser("reduce", help="tau-gamma_p and DELTA-TAU"))
    p.add_argument("sim_dir")
    p.add_argument("--control", default=None)
    p.add_argument("--offset", type=float, default=1e-3)
    p.add_argument("--json", default=None)

    a = ap.parse_args(argv)
    mat = Material.from_file(a.material, a.temperature)

    if a.cmd == "state":
        rows = states_for(a.run, mat, a.frac)
        print(mat.summary())
        print(f"source: {rows[0]['run']}")
        print(format_state_table(rows))
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    if a.cmd == "box":
        print(mat.summary())
        for L in a.L:
            print(f"  {L:>6.0f} nm target -> {commensurate_box(L, mat).describe()}")
        return 0

    if a.cmd == "routec":
        rows = states_for(a.run, mat, a.frac)
        out = [dispersed_barrier(s, {"c": a.alpha_c, "a": a.alpha_a}, mat,
                                 taylor_M=a.taylor_M) for s in rows]
        print(f"alpha_c = {a.alpha_c:g}, alpha_a = {a.alpha_a:g}"
              + (f", M = {a.taylor_M:g}" if a.taylor_M else "")
              + f"   [{rows[0]['run']}]")
        print(format_routec(out, a.taylor_M))
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0

    if a.cmd == "stage":
        box = commensurate_box(a.L, mat)
        state = None
        pops = None
        if a.run:
            rows = state_table(a.run, mat)
            state = min(rows, key=lambda r: abs(r["dose"] - (a.dose if a.dose
                                                             is not None
                                                             else r["dose"])))
        else:
            states = plan_states(mat)
            state = (states[-1] if a.dose is None else
                     min(states, key=lambda r: abs(r["dose"] - a.dose)))
        if not a.no_loops and a.route == "A":
            pops, _ = cube_population(box, mat, run_dir=a.run, dose=a.dose,
                                      state=None if a.run else state,
                                      seed=a.seed, verbose=True)
        res = stage(a.sim_dir, box, mat, route=a.route, pops=pops, state=state,
                    tau_dot_MPa_s=a.tau_dot, n_steps=a.steps,
                    output_frequency=a.output_every,
                    subcycling=not a.no_subcycling,
                    source_density_m2=a.source_density,
                    dipole_height=a.dipole_height, dose=state["dose"], seed=a.seed, loops=not a.no_loops)
        if a.run_it:
            print(json.dumps(run_case(res["sim_dir"]), indent=2, default=str))
        return 0

    if a.cmd == "verify":
        return 0 if verify(mat)[0] else 1

    if a.cmd == "campaign":
        out = campaign(a.out_root, a.doses, L_nm=a.L, run_dir=a.run,
                       material=mat, tau_dot_MPa_s=a.tau_dot, n_steps=a.steps,
                       output_frequency=a.output_every, seed=a.seed,
                       offset=a.offset)
        print()
        print(f"{'dpa':>8} {'seed':>5} {'obst':>6} {'tau_off':>9} {'dtau':>9} "
              f"{'s/step':>8}")
        for r in out["runs"]:
            n_ob = obstacle_count(r)
            print(f"{r['dose']:>8.4g} {r['seed']:>5d} {n_ob:>6d} "
                  f"{r['tau_offset_MPa']:>9.3f} {r['delta_tau_MPa']:>9.3f} "
                  f"{r['s_per_step']:>8.4g}")
        return 0

    if a.cmd == "plot":
        figures(a.campaign_json, a.out, mat, run_dir=a.run,
                taylor_M=a.taylor_M, render_loops=not a.no_loop_renders)
        return 0

    if a.cmd == "run":
        print(json.dumps(run_case(a.sim_dir, generate=not a.no_generate),
                         indent=2, default=str))
        return 0

    if a.cmd == "reduce":
        r = reduce_run(a.sim_dir, mat, offset=a.offset)
        if a.control:
            c = reduce_run(a.control, mat, offset=a.offset)
            r["control"] = c["sim_dir"]
            r["delta_tau_MPa"] = delta_tau(r, c)
            print(f"  DELTA-TAU = {r['delta_tau_MPa']:.3f} MPa "
                  f"({r['tau_offset_MPa']:.3f} - {c['tau_offset_MPa']:.3f})")
        if a.json:
            Path(a.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
