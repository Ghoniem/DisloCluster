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
           "OUTPUT"]

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
}

SOLVER = {
    "rtol":         1e-6,
    "atol":         1e-20,
    "backend":      "cvode",
    "lmm":          "bdf",
    "linsol":       "dense",
    "analytic_jac": True,
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

    def build_sim(self, verbose=False):
        """The calibrated 0-D chain at this run's T and G.

        Goes through `calibration.build_sim`, never straight to the workbook:
        28 workbook parameters have drifted from the fitted set and 11 are
        absent, which moves N_a by five orders of magnitude.
        """
        from dislocluster_code.zerod.calibration import build_sim
        return build_sim(input_file=self.workbook, T=self.temperature_K,
                         G=self.dose_rate_dpa_s, extra=self.overrides or None,
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

    def validate(self):
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
                              overrides=dict(m["overrides"] or {})),
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
                on_unconverged=c["on_unconverged"]),
            solver=Solver(rtol=float(s["rtol"]), atol=float(s["atol"]),
                          backend=s["backend"], lmm=s["lmm"],
                          linsol=s["linsol"],
                          analytic_jac=bool(s["analytic_jac"])),
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
            temperature_K=self.material.temperature_K,
        ), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()

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
