"""config.py — the notebook's control dicts, validated once.

A simulation is described by six dicts edited at the top of a notebook under
``Simulations/``. This module turns them into a checked, hashable
``SimulationConfig`` and is the single place that knows

  * which 0-D parameters a run may set, and that they go through
    :func:`dislocluster_code.zerod.calibration.build_sim` rather than straight into
    the workbook (the workbook is NOT the calibrated model);
  * how ``GEOMETRY`` + ``MESH`` map onto ``generate_mesh.MeshSpec``;
  * how ``BOUNDARY`` maps onto ``polycrystal.txt`` and
    ``ElasticDeformation.txt``, including the stress normalization;
  * what makes two configurations the same case, so a staged mesh can be reused
    instead of rebuilt.

Validation happens BEFORE anything expensive runs. Every check here exists
because getting it wrong costs hours: a one-element dose grid produced an empty
timing table and a division by zero at the end of a march, and a decreasing
grid ran CVODE backwards.

USAGE
-----
    from dislocluster_code.config import SimulationConfig
    cfg = SimulationConfig.from_dicts(MATERIAL=MATERIAL, GEOMETRY=GEOMETRY,
                                      MESH=MESH, BOUNDARY=BOUNDARY,
                                      COUPLING=COUPLING, SOLVER=SOLVER,
                                      OUTPUT=OUTPUT)
    print(cfg.summary())
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dislocluster_code import paths
from dislocluster_code.integration import resolve as resolve_option

__all__ = ["ConfigError", "Material", "Geometry", "Mesh", "Boundary",
           "Coupling", "Solver", "Output", "SimulationConfig",
           "MATERIAL", "GEOMETRY", "MESH", "BOUNDARY", "COUPLING", "SOLVER",
           "OUTPUT", "sim_for_run"]

VOIGT = ("11", "22", "33", "12", "23", "13")


class ConfigError(ValueError):
    """A control dict the run cannot proceed with."""


# ── the defaults, in the shape the notebook edits ────────────────────────────
# Kept as plain dicts so a notebook cell can start from a copy and the diff
# against a run's provenance is readable.

MATERIAL = {
    "file":            "Zr3d_ghoniem.txt",   # under MoDELib3/Library/Materials
    "workbook":        None,                 # None -> the calibrated default
    "temperature_K":   573.0,
    "dose_rate_dpa_s": 1.0e-7,
    "overrides":       {},                   # extra 0-D parameters, on top of
                                             # calibration.REFERENCE_OVERRIDES
    # Derive the 0-D `sigma_n` / `sigma_h` from BOUNDARY instead of reading them
    # from the workbook. ON by default, because the workbook and BOUNDARY had
    # disagreed: `Material_Environment!sigma_n` stands at 1.0e8 Pa, so every
    # run was at 100 MPa in the 0-D loop alignment and at 0 MPa in the 3-D
    # elastic solve. Set False to reproduce a run made before this existed.
    "stress_from_boundary": True,
}

GEOMETRY = {
    "type":      "cubic",        # "cubic" | "hexagonal"
    "size_nm":   500.0,          # cube side, or hexagon corner-to-corner
    "height_nm": None,           # hexagonal only; None -> size_nm
    "aspect":    (1.0, 1.0, 1.0),  # cubic only, per-axis multipliers
}

MESH = {
    "target_elements":        15000,
    "lc_nm":                  None,    # explicit interior size; wins if set
    "element_order":          1,       # 1 = linear tet, 2 = quadratic
    "boundary_layer":         True,
    "boundary_thickness_nm":  150.0,
    # layer size, first one set wins
    "boundary_lc_nm":         None,
    "boundary_layers_across": 5,
    "boundary_size_ratio":    0.35,
    "boundary_expansion":     None,
    "optimize":               True,
    "regenerate":             False,
}

BOUNDARY = {
    # [] -> no periodic faces, i.e. Dirichlet on the whole surface. MoDELib
    # takes these as face IDs in polycrystal.txt.
    "periodic_face_ids":   [],
    # Voigt order 11, 22, 33, 12, 23, 13.
    "applied_stress_MPa":  (0.0,) * 6,
    "applied_stress_rate_MPa_s": (0.0,) * 6,
    "applied_strain":      (0.0,) * 6,
    "applied_strain_rate": (0.0,) * 6,
    # 0 = pure stress control, 1e20 = pure strain control, per component
    "stiffness_ratio":     (0.0,) * 6,
}

COUPLING = {
    "route":                 "Adaptive",   # "Adaptive" (B) | "IMEX" (A)
    "dose_seed":             0.0,          # <= 0 seeds pristine material
    "doses":                 [1.0, 6.0, 11.0, 16.0, 21.0],
    "substeps_per_interval": 20,
    "fem_every":             5,
    "dedup_rtol":            1e-8,
    "variant_weights":       (1 / 3, 1 / 3, 1 / 3),
    "max_failed_nodes":      0,
    "on_unconverged":        "warn",       # "warn" | "raise"
    # Coarsening detector. phi is the model's own Avrami overlap fraction; when
    # it is large the mean-field estimate of coalescence is no longer reliable
    # and the population wants a discrete treatment. Reporting only: the march
    # records the crossing dose, it does not act on it.
    "phi_star":              0.15,         # threshold on phi
    "frac_star":             0.10,         # share of interior nodes over it
    "coarsen_hold":          2,            # consecutive substeps required
    # Act on d_coarsen instead of only reporting it: at the first fast-solve
    # boundary after a family crosses phi*, convert it to discrete loops, hand
    # them to DD and keep in the continuum only what DD cannot take. OFF by
    # default -- every run made before this existed is unaffected.
    "discrete_transition":   False,
    "transition_units":      ("c",),      # which of transition.UNITS may switch
    "climb_cutoff_nL":       4.0,         # R_c = nL * L_s; see neighbors.py
}

SOLVER = {
    "rtol":         1e-6,
    "atol":         1e-20,
    "backend":      "cvode",
    "lmm":          "bdf",
    "linsol":       "dense",
    "analytic_jac": True,
    # Which immobile-loop formulation the SLOW step integrates.
    #   0  legacy      four families split aligned/non-aligned, with the
    #                  phenomenological Z_i_a / Z_v_c fitted from delta_DAD and
    #                  one lumped interstitial flux. This is what the 28
    #                  calibrated parameters were fitted against, so it is the
    #                  default and the C++ reproduces it BIT-FOR-BIT.
    #   1  self-consistent
    #                  one basal <c> and three prismatic <a> variants, matching
    #                  the 3-D family structure exactly, with Woo capture
    #                  efficiencies built from the same p_m that sets the
    #                  diffusion tensor. Every mobile species then carries its
    #                  own efficiency instead of sharing one.
    # Keep 0 for anything being fitted; use 1 for a run that must be consistent
    # with the anisotropic fast solve.
    "loop_model":   0,
}

OUTPUT = {
    "tag":            None,      # None -> derived from the case name
    "figures":        True,
    "movies":         False,
    # Movie frames per output interval. The march writes one snapshot per dose
    # interval, so a short `doses` list animates as a flipbook; the extra frames
    # are interpolated between solved snapshots and labelled as such. 1 = off.
    "movie_interp":   5,
    "discrete_loops": False,
    "checkpoint":     True,
    "resume":         "auto",    # "auto" | "never" | "require"
}


def _take(name, defaults, given):
    """Merge `given` over `defaults`, rejecting keys that are not settings."""
    given = dict(given or {})
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ConfigError(
            f"{name} has no setting(s) {unknown}. Known settings: "
            f"{sorted(defaults)}")
    out = dict(defaults)
    out.update(given)
    return out


def _voigt(name, v):
    v = tuple(float(x) for x in v)
    if len(v) != 6:
        raise ConfigError(f"{name} needs 6 Voigt components "
                          f"({', '.join(VOIGT)}), got {len(v)}")
    return v


# ── the pieces ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Material:
    file: str
    workbook: str | None
    temperature_K: float
    dose_rate_dpa_s: float
    overrides: dict
    stress_from_boundary: bool = True

    @property
    def path(self):
        """The material file MoDELib reads, under Library/Materials."""
        return paths.MODELIB_ROOT / "Library" / "Materials" / self.file

    def validate(self):
        if not self.path.is_file():
            raise ConfigError(f"no material file {self.path}")
        if not 1.0 < self.temperature_K < 3000.0:
            raise ConfigError(f"temperature_K={self.temperature_K} is not a "
                              f"plausible absolute temperature")
        if self.dose_rate_dpa_s <= 0.0:
            raise ConfigError("dose_rate_dpa_s must be positive")
        if self.workbook is not None and not Path(self.workbook).is_file():
            raise ConfigError(f"no workbook {self.workbook}")

    def build_sim(self, verbose=False, boundary=None):
        """The calibrated 0-D chain at this run's T and G.

        Goes through `calibration.build_sim`, never straight to the workbook:
        28 workbook parameters have drifted from the fitted set and 11 are
        absent, which moves N_a by five orders of magnitude.

        THE APPLIED LOAD COMES FROM `boundary`, NOT FROM THE WORKBOOK.
        `Material_Environment!sigma_n` stands at 1.0e8 Pa, and the 0-D used it
        for the aligned/non-aligned loop split and the stress-dependent vacancy
        emission while `BOUNDARY` -- which drives the 3-D elastic solve -- was
        zero. Every run in the 500 nm series was therefore at 100 MPa in the
        0-D and 0 MPa in the 3-D at the same time, and the tell was that the
        aligned fraction came out 0.4015 where a zero-stress run must give
        exactly 1/3 (`f_a = (1 + 2f)/3` with `f = 0` at `sigma_n = 0`).

        An explicit `overrides` entry still wins, so a deliberate value can be
        pinned; `stress_from_boundary=False` restores the old behaviour for
        reproducing a run made before this existed.
        """
        from dislocluster_code.zerod.calibration import build_sim
        extra = dict(self.overrides or {})
        if self.stress_from_boundary and boundary is not None:
            from dislocluster_code.staging import stress as _stress
            derived = _stress.overrides(boundary.applied_stress_MPa)
            for k, v in derived.items():
                extra.setdefault(k, v)          # explicit override wins
        return build_sim(input_file=self.workbook, T=self.temperature_K,
                         G=self.dose_rate_dpa_s, extra=extra or None,
                         verbose=verbose)


@dataclass(frozen=True)
class Geometry:
    type: str
    size_nm: float
    height_nm: float | None
    aspect: tuple

    def validate(self):
        if self.type not in ("cubic", "hexagonal"):
            raise ConfigError(f"geometry type {self.type!r} is not "
                              f"'cubic' or 'hexagonal'")
        if self.size_nm <= 0.0:
            raise ConfigError("size_nm must be positive")
        if len(self.aspect) != 3 or any(a <= 0 for a in self.aspect):
            raise ConfigError(f"aspect must be three positive numbers, "
                              f"got {self.aspect}")
        if self.type == "hexagonal" and self.aspect != (1.0, 1.0, 1.0):
            raise ConfigError("aspect applies to the cubic geometry only")


@dataclass(frozen=True)
class Mesh:
    target_elements: int
    lc_nm: float | None
    element_order: int
    boundary_layer: bool
    boundary_thickness_nm: float
    boundary_lc_nm: float | None
    boundary_layers_across: int
    boundary_size_ratio: float
    boundary_expansion: float | None
    optimize: bool
    regenerate: bool

    def validate(self, geometry):
        if self.element_order not in (1, 2):
            raise ConfigError(f"element_order must be 1 or 2, got "
                              f"{self.element_order}; MoDELib's SimplexReader "
                              f"consumes only msh element types 4 and 11")
        if self.target_elements < 1 and self.lc_nm is None:
            raise ConfigError("set either target_elements or lc_nm")
        if self.boundary_layer:
            if self.boundary_thickness_nm <= 0:
                raise ConfigError("boundary_thickness_nm must be positive "
                                  "when boundary_layer is on")
            if self.boundary_thickness_nm >= 0.5 * geometry.size_nm:
                raise ConfigError(
                    f"boundary_thickness_nm={self.boundary_thickness_nm:g} "
                    f"leaves no interior in a {geometry.size_nm:g} nm domain")
            if self.boundary_layers_across is not None \
                    and self.boundary_layers_across < 1:
                raise ConfigError("boundary_layers_across must be >= 1")

    def spec(self, geometry):
        """The ``generate_mesh.MeshSpec`` for this geometry.

        Imported lazily: `generate_mesh` pulls in the Gmsh SDK, which a
        post-processing-only session has no reason to require.
        """
        if str(paths.GMSH_DIR) not in sys.path:
            sys.path.insert(0, str(paths.GMSH_DIR))
        import generate_mesh as gm
        return gm.MeshSpec(
            geometry=geometry.type,
            size_nm=geometry.size_nm,
            height_nm=geometry.height_nm,
            aspect=tuple(geometry.aspect),
            target_elements=self.target_elements,
            lc_nm=self.lc_nm,
            element_order=self.element_order,
            boundary_layer=self.boundary_layer,
            boundary_thickness_nm=self.boundary_thickness_nm,
            boundary_lc_nm=self.boundary_lc_nm,
            boundary_layers_across=self.boundary_layers_across,
            boundary_size_ratio=self.boundary_size_ratio,
            boundary_expansion=self.boundary_expansion,
            optimize=self.optimize,
        )


@dataclass(frozen=True)
class Boundary:
    periodic_face_ids: tuple
    applied_stress_MPa: tuple
    applied_stress_rate_MPa_s: tuple
    applied_strain: tuple
    applied_strain_rate: tuple
    stiffness_ratio: tuple

    def validate(self):
        for n in ("applied_stress_MPa", "applied_stress_rate_MPa_s",
                  "applied_strain", "applied_strain_rate", "stiffness_ratio"):
            _voigt(n, getattr(self, n))
        if any(int(i) < 0 for i in self.periodic_face_ids):
            raise ConfigError("periodic_face_ids must be non-negative")

    @property
    def all_dirichlet(self):
        return not self.periodic_face_ids

    def stress_in_mu(self, mu_SI):
        """Applied stress in MoDELib's units, which are multiples of mu.

        MoDELib normalizes lengths by b_SI and stresses by mu_SI --
        `PolycrystallineMaterialBase` carries `mu_SI` [Pa] alongside a
        dimensionless `mu`, and the VTK writer converts back with
        `stress * mu_SI`. Writing MPa straight into ElasticDeformation.txt
        would therefore overstate the load by mu_SI/1e6, about 33 000x for Zr.
        """
        return tuple(s * 1.0e6 / mu_SI for s in self.applied_stress_MPa)

    def stress_rate_in_mu(self, mu_SI):
        return tuple(s * 1.0e6 / mu_SI for s in self.applied_stress_rate_MPa_s)


@dataclass(frozen=True)
class Coupling:
    route: str
    dose_seed: float
    doses: tuple
    substeps_per_interval: int
    fem_every: int
    dedup_rtol: float
    variant_weights: tuple
    max_failed_nodes: int
    on_unconverged: str
    phi_star: float = 0.15
    frac_star: float = 0.10
    coarsen_hold: int = 2
    discrete_transition: bool = False
    transition_units: tuple = ("c",)
    climb_cutoff_nL: float = 4.0

    def validate(self):
        resolve_option(self.route)          # raises KeyError on an unknown name
        d = list(self.doses)
        if len(d) < 1:
            raise ConfigError("doses is empty")
        if any(b <= a for a, b in zip(d, d[1:])):
            raise ConfigError(f"doses must increase strictly, got {d}")
        if len(self.snaps) < 2:
            raise ConfigError(
                f"the march needs at least one interval, i.e. two points "
                f"counting the seed; dose_seed={self.dose_seed:g} with "
                f"doses={d} gives {self.snaps}")
        if self.dose_seed > d[0]:
            raise ConfigError(f"dose_seed={self.dose_seed:g} is past the first "
                              f"snapshot {d[0]:g}")
        if self.substeps_per_interval < 1:
            raise ConfigError("substeps_per_interval must be >= 1")
        if self.fem_every < 1:
            raise ConfigError("fem_every must be >= 1")
        if self.dedup_rtol < 0:
            raise ConfigError("dedup_rtol must be >= 0")
        w = tuple(float(x) for x in self.variant_weights)
        if len(w) != 3 or abs(sum(w) - 1.0) > 1e-9:
            raise ConfigError(f"variant_weights must be three numbers summing "
                              f"to 1, got {w}")
        if self.on_unconverged not in ("warn", "raise"):
            raise ConfigError("on_unconverged must be 'warn' or 'raise'")
        # phi is a probability, so a threshold outside (0,1) can never be
        # crossed in one direction or is crossed at dose zero in the other.
        if not 0.0 < self.phi_star < 1.0:
            raise ConfigError(f"phi_star must lie in (0,1), got {self.phi_star}")
        if not 0.0 < self.frac_star <= 1.0:
            raise ConfigError(f"frac_star must lie in (0,1], got {self.frac_star}")
        if self.discrete_transition:
            from dislocluster_code.coupling.transition import UNITS
            bad = [u for u in self.transition_units if u not in UNITS]
            if bad:
                raise ConfigError(
                    f"transition_units {bad} are not transferable; the 0-D "
                    f"state lumps the <a> variants, so the units are "
                    f"{sorted(UNITS)}")
            if self.climb_cutoff_nL <= 0.0:
                raise ConfigError("climb_cutoff_nL must be positive")
        if self.coarsen_hold < 1:
            raise ConfigError("coarsen_hold must be >= 1")

    @property
    def snaps(self):
        """The full march grid: the seed dose followed by every snapshot."""
        d = [float(x) for x in self.doses]
        s = float(self.dose_seed)
        return tuple([s] + d) if s < d[0] else tuple(d)

    @property
    def option(self):
        return resolve_option(self.route)

    @property
    def n_substeps(self):
        return (len(self.snaps) - 1) * self.substeps_per_interval


@dataclass(frozen=True)
class Solver:
    rtol: float
    atol: float
    backend: str
    lmm: str
    linsol: str
    analytic_jac: bool
    loop_model: int = 0

    def validate(self):
        if self.loop_model not in (0, 1):
            raise ConfigError(
                f"loop_model must be 0 (legacy, the fitted formulation) or 1 "
                f"(self-consistent with the anisotropic fast solve), got "
                f"{self.loop_model}")
        if self.backend not in ("cvode", "arkode"):
            raise ConfigError(f"solver backend {self.backend!r} is not "
                              f"'cvode' or 'arkode'")
        if self.linsol not in ("dense", "band", "gmres"):
            raise ConfigError(f"linsol {self.linsol!r} is not dense/band/gmres")
        if not 0 < self.rtol < 1:
            raise ConfigError("rtol must be in (0, 1)")
        if self.atol <= 0:
            raise ConfigError("atol must be positive")

    def as_solver_config(self, t_begin=1e-1, t_end=1e9, n_points=2,
                         log_time=False):
        """The dict shape `cpp_bridge.collect_solver_args` expects."""
        return dict(t_begin=t_begin, t_end=t_end, n_points=n_points,
                    log_time=log_time, rtol=self.rtol, atol=self.atol,
                    analytic_jac=self.analytic_jac, stats=True,
                    loop_model=self.loop_model,
                    solver_method=dict(backend=self.backend, lmm=self.lmm,
                                       linsol=self.linsol))


@dataclass(frozen=True)
class Output:
    tag: str | None
    figures: bool
    movies: bool
    movie_interp: int
    discrete_loops: bool
    checkpoint: bool
    resume: str

    def validate(self):
        if self.resume not in ("auto", "never", "require"):
            raise ConfigError("resume must be 'auto', 'never' or 'require'")
        if self.movie_interp < 1:
            raise ConfigError("movie_interp must be >= 1 (1 disables "
                              "interpolated movie frames)")


# ── the whole thing ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimulationConfig:
    material: Material
    geometry: Geometry
    mesh: Mesh
    boundary: Boundary
    coupling: Coupling
    solver: Solver
    output: Output

    @classmethod
    def from_dicts(cls, MATERIAL=None, GEOMETRY=None, MESH=None, BOUNDARY=None,
                   COUPLING=None, SOLVER=None, OUTPUT=None, validate=True):
        g = globals()
        m = _take("MATERIAL", g["MATERIAL"], MATERIAL)
        geo = _take("GEOMETRY", g["GEOMETRY"], GEOMETRY)
        me = _take("MESH", g["MESH"], MESH)
        b = _take("BOUNDARY", g["BOUNDARY"], BOUNDARY)
        c = _take("COUPLING", g["COUPLING"], COUPLING)
        s = _take("SOLVER", g["SOLVER"], SOLVER)
        o = _take("OUTPUT", g["OUTPUT"], OUTPUT)

        cfg = cls(
            material=Material(file=m["file"], workbook=m["workbook"],
                              temperature_K=float(m["temperature_K"]),
                              dose_rate_dpa_s=float(m["dose_rate_dpa_s"]),
                              overrides=dict(m["overrides"] or {}),
                              stress_from_boundary=bool(
                                  m["stress_from_boundary"])),
            geometry=Geometry(type=geo["type"], size_nm=float(geo["size_nm"]),
                              height_nm=(None if geo["height_nm"] is None
                                         else float(geo["height_nm"])),
                              aspect=tuple(float(a) for a in geo["aspect"])),
            mesh=Mesh(target_elements=int(me["target_elements"]),
                      lc_nm=(None if me["lc_nm"] is None
                             else float(me["lc_nm"])),
                      element_order=int(me["element_order"]),
                      boundary_layer=bool(me["boundary_layer"]),
                      boundary_thickness_nm=float(me["boundary_thickness_nm"]),
                      boundary_lc_nm=(None if me["boundary_lc_nm"] is None
                                      else float(me["boundary_lc_nm"])),
                      boundary_layers_across=me["boundary_layers_across"],
                      boundary_size_ratio=float(me["boundary_size_ratio"]),
                      boundary_expansion=me["boundary_expansion"],
                      optimize=bool(me["optimize"]),
                      regenerate=bool(me["regenerate"])),
            boundary=Boundary(
                periodic_face_ids=tuple(int(i) for i in b["periodic_face_ids"]),
                applied_stress_MPa=_voigt("applied_stress_MPa",
                                          b["applied_stress_MPa"]),
                applied_stress_rate_MPa_s=_voigt(
                    "applied_stress_rate_MPa_s", b["applied_stress_rate_MPa_s"]),
                applied_strain=_voigt("applied_strain", b["applied_strain"]),
                applied_strain_rate=_voigt("applied_strain_rate",
                                           b["applied_strain_rate"]),
                stiffness_ratio=_voigt("stiffness_ratio",
                                       b["stiffness_ratio"])),
            coupling=Coupling(
                route=c["route"], dose_seed=float(c["dose_seed"]),
                doses=tuple(float(x) for x in c["doses"]),
                substeps_per_interval=int(c["substeps_per_interval"]),
                fem_every=int(c["fem_every"]),
                dedup_rtol=float(c["dedup_rtol"]),
                variant_weights=tuple(float(x) for x in c["variant_weights"]),
                max_failed_nodes=int(c["max_failed_nodes"]),
                on_unconverged=c["on_unconverged"],
                # These were declared on the dataclass and accepted by _take,
                # but never forwarded here -- so COUPLING['phi_star'] and its
                # neighbours silently did nothing and every run used the
                # dataclass default. A key that validates and is then dropped
                # is worse than an unknown one, which _take rejects loudly.
                phi_star=float(c["phi_star"]),
                frac_star=float(c["frac_star"]),
                coarsen_hold=int(c["coarsen_hold"]),
                discrete_transition=bool(c["discrete_transition"]),
                transition_units=tuple(c["transition_units"]),
                climb_cutoff_nL=float(c["climb_cutoff_nL"])),
            solver=Solver(rtol=float(s["rtol"]), atol=float(s["atol"]),
                          backend=s["backend"], lmm=s["lmm"],
                          linsol=s["linsol"],
                          analytic_jac=bool(s["analytic_jac"]),
                          loop_model=int(s["loop_model"])),
            output=Output(tag=o["tag"], figures=bool(o["figures"]),
                          movies=bool(o["movies"]),
                          movie_interp=int(o["movie_interp"]),
                          discrete_loops=bool(o["discrete_loops"]),
                          checkpoint=bool(o["checkpoint"]),
                          resume=o["resume"]),
        )
        if validate:
            cfg.validate()
        return cfg

    def validate(self):
        self.material.validate()
        self.geometry.validate()
        self.mesh.validate(self.geometry)
        self.boundary.validate()
        self.coupling.validate()
        self.solver.validate()
        self.output.validate()
        # The continuum -> discrete handoff has not been ported to the
        # self-consistent formulation. `transition.cd_block` and the
        # scale/zero helpers address the immobile state by the LEGACY slot
        # names (CiL, CaiL, CvL, CavL, ...), which under loop_model=1 name a
        # different family in every slot. The failure would be silent -- a
        # well-formed set of discrete loops drawn from the wrong population --
        # so the combination is refused rather than approximated.
        if self.solver.loop_model and self.coupling.discrete_transition:
            raise ValueError(
                "COUPLING['discrete_transition'] is not supported with "
                "SOLVER['loop_model'] = 1: the transfer addresses the immobile "
                "state by the legacy aligned/non-aligned slot names. Run the "
                "handoff with loop_model = 0, or the self-consistent model "
                "without the handoff.")
        return self

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def domain_key(self):
        """Hash of everything that decides the MESH and the staged case.

        The dose grid is deliberately excluded: two runs over different doses
        on the same domain share a mesh, a CD node set and a scaffold, and
        re-bootstrapping that costs a DDomp call.
        """
        blob = json.dumps(dict(
            geometry=dataclasses.asdict(self.geometry),
            mesh={k: v for k, v in dataclasses.asdict(self.mesh).items()
                  if k != "regenerate"},
            boundary=dataclasses.asdict(self.boundary),
            material=self.material.file,
            # The material file's CONTENT, not just its name. The staged case
            # holds a COPY of it, and the bootstrap is a DDomp solve against it,
            # so editing the file in place -- changing the diffusion tensor, say
            # -- must invalidate both. Keying on the name alone let a stale copy
            # survive silently, which is issue 3 of ZR3D_GHONIEM_CHANGES all
            # over again: the run does not fail, it answers the wrong question.
            material_sha=self.material_digest,
            temperature_K=self.material.temperature_K,
        ), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()

    @property
    def material_digest(self):
        """Short digest of the material file's contents, or "missing"."""
        try:
            return hashlib.blake2b(self.material.path.read_bytes(),
                                   digest_size=8).hexdigest()
        except OSError:
            return "missing"

    @property
    def case_name(self):
        g = self.geometry
        return (f"{g.type}_{g.size_nm:g}nm_o{self.mesh.element_order}"
                f"_{self.domain_key}")

    @property
    def sim_dir(self):
        """Where the staged MoDELib case lives (see paths.SIM_ROOT)."""
        return paths.SIM_ROOT / self.case_name

    @property
    def tag(self):
        return self.output.tag or self.case_name

    # ── reporting ────────────────────────────────────────────────────────────

    def to_dict(self):
        d = {k: dataclasses.asdict(getattr(self, k))
             for k in ("material", "geometry", "mesh", "boundary", "coupling",
                       "solver", "output")}
        d["derived"] = dict(case_name=self.case_name,
                            domain_key=self.domain_key,
                            sim_dir=str(self.sim_dir),
                            snaps=list(self.coupling.snaps),
                            n_substeps=self.coupling.n_substeps,
                            git_hash=paths.git_hash())
        return d

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                        encoding="utf-8")
        return path

    def summary(self):
        c, g, m = self.coupling, self.geometry, self.mesh
        opt = c.option
        lay = (f"{m.boundary_thickness_nm:g} nm boundary layer, "
               f"{m.boundary_layers_across} across" if m.boundary_layer
               else "uniform")
        bc = ("Dirichlet on every face" if self.boundary.all_dirichlet
              else f"periodic faces {list(self.boundary.periodic_face_ids)}")
        sig = self.boundary.applied_stress_MPa
        load = ("zero applied stress" if not any(sig)
                else "sigma [MPa] " + " ".join(f"{k}={v:g}"
                                               for k, v in zip(VOIGT, sig)
                                               if v))
        return "\n".join([
            f"case        {self.case_name}",
            f"geometry    {g.type}, {g.size_nm:g} nm"
            + (f" x {g.height_nm:g} nm" if g.height_nm else ""),
            f"mesh        order {m.element_order}, "
            f"~{m.target_elements} elements, {lay}",
            f"material    {self.material.file} at "
            f"{self.material.temperature_K:g} K, "
            f"{self.material.dose_rate_dpa_s:g} dpa/s",
            f"boundary    {bc}; {load}",
            f"route       Option {opt.option} — {opt.key}: {opt.integrator}",
            f"dose        seed {c.dose_seed:g} -> "
            f"{', '.join(f'{d:g}' for d in c.doses)} dpa",
            f"march       {c.substeps_per_interval} substeps/interval, "
            f"fast solve every {c.fem_every} "
            f"({c.n_substeps} substeps total)",
            f"sim dir     {self.sim_dir}",
        ])


def sim_for_run(run_dir, verbose=False):
    """The 0-D model chain **as a finished run actually built it**.

    Post-processing that needs the 0-D right-hand side -- `post.volume_average`
    for the conservation channels, `post.boundary_flux` for the surface flux --
    used to call `calibration.build_sim()` with no arguments, which takes the
    applied load from the WORKBOOK. `Material_Environment!sigma_n` stands at
    1.0e8 Pa, so a run staged with `stress_from_boundary` and a zero `BOUNDARY`
    was post-processed at 100 MPa: `f_a` came out 0.4015 in the diagnostic
    against the 1/3 the march itself ran with, and `f_a` sets the aligned /
    non-aligned split whose two families have different capture efficiencies.

    `config.json` records everything needed to reproduce the choice, so the
    diagnostic reconstructs it through the SAME path the driver used rather
    than a parallel one. A run directory without `config.json` -- a standalone
    0-D run, or anything predating the driver -- falls back to the calibrated
    default, which is what those runs were built with.
    """
    from pathlib import Path
    import json as _json

    cj = Path(run_dir) / "config.json"
    if not cj.is_file():
        from dislocluster_code.zerod.calibration import build_sim
        return build_sim(verbose=verbose)

    c = _json.loads(cj.read_text(encoding="utf-8"))
    m, b = c["material"], c.get("boundary", {})
    mat = Material(
        file=m["file"], workbook=m["workbook"],
        temperature_K=float(m["temperature_K"]),
        dose_rate_dpa_s=float(m["dose_rate_dpa_s"]),
        overrides=dict(m.get("overrides") or {}),
        stress_from_boundary=bool(m.get("stress_from_boundary", False)))
    bnd = Boundary(
        periodic_face_ids=tuple(b.get("periodic_face_ids", ())),
        applied_stress_MPa=tuple(b.get("applied_stress_MPa", (0.0,) * 6)),
        applied_stress_rate_MPa_s=tuple(
            b.get("applied_stress_rate_MPa_s", (0.0,) * 6)),
        applied_strain=tuple(b.get("applied_strain", (0.0,) * 6)),
        applied_strain_rate=tuple(b.get("applied_strain_rate", (0.0,) * 6)),
        stiffness_ratio=tuple(b.get("stiffness_ratio", (0.0,) * 6)))
    return mat.build_sim(verbose=verbose, boundary=bnd)
