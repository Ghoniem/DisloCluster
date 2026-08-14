"""generate_mesh.py — parametric MoDELib meshes for DisloCluster.

Builds the simulation domain with Gmsh and writes it in the only format MoDELib
accepts. Two geometries are supported:

    cubic       a rectangular box  (the 1 um cube of the reference run)
    hexagonal   a regular hexagonal prism, the natural cell for hcp Zr, with
                the prism axis along z (the c-axis) and a <a>-type edge along x

What MoDELib requires — verified against its reader, not assumed:

  * `.msh` **version 2.x ASCII**. `GmshReader.cpp` accepts only
    `version>=2.0 && version<3.0`; a v4 file is rejected at load with
    "Unsupported msh version".
  * Only element types **4** (4-node linear tet) and **11** (10-node quadratic
    tet) are consumed — `SimplexReader.h`. Lines and surface triangles in the
    file are read and then ignored, so they are harmless but pointless.
  * The region ID is taken from **`tags[1]`**, the second tag, i.e. Gmsh's
    *elementary entity* tag, not the physical-group tag. The generator therefore
    forces the elementary volume tag to `region_id` so a single-crystal mesh
    lands in region 1.

Coordinate convention. The mesh is written **normalised to the unit bounding
box** `[0,1]^3`, exactly like `Library/Meshes/unitCube_15K.msh`. Physical size is
applied by MoDELib through `F` in `polycrystal.txt`, which maps `x = F*(X-X0)`.
`mesh_deformation_gradient()` returns the matching diagonal `F` in units of the
Burgers vector, so the shape is preserved for the hexagon as well as the cube.

Usage
-----
    from generate_mesh import MeshSpec, generate
    spec = MeshSpec(geometry="cubic", size_nm=1000.0, target_elements=15000)
    result = generate(spec)          # -> MeshResult(path, F, n_nodes, ...)

or from the command line:

    .DisloClusterVenv/Scripts/python.exe ZrClusterDynamics/Gmsh/generate_mesh.py \
        --geometry hexagonal --size-nm 1000 --height-nm 1000 \
        --target-elements 15000 --element-order 2 --boundary-layer
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

GMSH_DIR = Path(__file__).resolve().parent
MESH_DIR = GMSH_DIR / "meshes"

# Burgers vector magnitude [m]. Must match modelib_fields.B_SI and the material
# file; MoDELib works internally in units of b, so F is expressed in b.
B_SI = 3.233e-10


# ── specification ────────────────────────────────────────────────────────────
@dataclass
class MeshSpec:
    """Everything that determines the mesh. Hashed into the output file name."""

    geometry: str = "cubic"          # "cubic" | "hexagonal"

    # Cubic: side length; a single float means a cube, or give (Lx, Ly, Lz).
    # Hexagonal: `size_nm` is the circumradius diameter (corner to corner) and
    # `height_nm` the prism height along z.
    size_nm: float = 1000.0
    height_nm: float | None = None
    aspect: tuple[float, float, float] = (1.0, 1.0, 1.0)   # cubic only

    # Resolution. `target_elements` sets the interior characteristic length by
    # inverting the volume-per-tet estimate; `lc_nm` overrides it outright.
    target_elements: int = 15000
    lc_nm: float | None = None

    # 1 -> 4-node linear tets (msh type 4); 2 -> 10-node quadratic (type 11).
    # The cluster-dynamics trial functions are second order either way; the mesh
    # order controls the geometry, and it is what sets the node count that
    # `evl/cdNodes.txt` must match.
    element_order: int = 1

    # Boundary-layer refinement. The denuded zone lives within ~2 depletion
    # lengths (~150 nm) of the surface and carries all of the spatial structure,
    # so a uniform mesh either wastes elements in the interior plateau or
    # under-resolves the shell.
    #
    # The layer is meshed at a CONSTANT size out to `boundary_thickness_nm` and
    # only then graded up to the interior size over `boundary_transition_nm`.
    # An earlier version ramped the size from the wall outward, which meant the
    # nominal "layer" was spanned by only one or two elements however fine the
    # wall size was — the size had already grown to the interior value a short
    # way in.
    boundary_layer: bool = False
    boundary_thickness_nm: float = 150.0

    # Resolution inside the layer, in order of precedence:
    #
    #   boundary_lc_nm         element size in nm, set outright. Pair it with
    #                          `lc_nm` for the interior to get a plain two-region
    #                          mesh ("20 nm at the wall, 100 nm inside") with no
    #                          ratio arithmetic.
    #   boundary_layers_across element size = thickness / N.
    #   boundary_size_ratio    element size = interior size x ratio.
    boundary_lc_nm: float | None = None
    boundary_layers_across: int | None = 8
    boundary_size_ratio: float = 0.35

    # Growth of the element size beyond the layer, toward the domain centre.
    #
    # `boundary_expansion` is the classic geometric expansion ratio: each
    # successive element layer is this many times thicker than the one before.
    # It maps onto a LINEAR size field, which is what Gmsh's Threshold provides:
    # with h_k = h0*r^k and cumulative depth d_n = h0*(r^n - 1)/(r - 1), it
    # follows that h_n = h0 + (r - 1)*d_n. So a geometric expansion of ratio r
    # is exactly a linear size gradient of (r - 1), and the transition distance
    # needed to climb from the layer size to the interior size is
    # (lc_int - lc_bnd)/(r - 1).
    #
    # When set, it overrides `boundary_transition_nm`.
    boundary_expansion: float | None = None
    boundary_transition_nm: float | None = None   # None -> 2 x layer thickness

    optimize: bool = True
    region_id: int = 1
    verbosity: int = 0

    def resolved_extents_nm(self):
        """Physical bounding-box extents (Lx, Ly, Lz) in nm."""
        if self.geometry == "cubic":
            ax, ay, az = self.aspect
            return (self.size_nm * ax, self.size_nm * ay, self.size_nm * az)
        if self.geometry == "hexagonal":
            R = 0.5 * self.size_nm                    # circumradius
            H = self.height_nm if self.height_nm else self.size_nm
            # Flat-to-flat across y is sqrt(3)*R; corner-to-corner across x is 2R
            return (2.0 * R, math.sqrt(3.0) * R, H)
        raise ValueError(f"unknown geometry {self.geometry!r} "
                         "(expected 'cubic' or 'hexagonal')")

    def volume_nm3(self):
        if self.geometry == "cubic":
            lx, ly, lz = self.resolved_extents_nm()
            return lx * ly * lz
        R = 0.5 * self.size_nm
        H = self.height_nm if self.height_nm else self.size_nm
        return 1.5 * math.sqrt(3.0) * R * R * H       # regular hexagonal prism

    def interior_lc_nm(self):
        """Characteristic element length in the interior [nm]."""
        if self.lc_nm:
            return float(self.lc_nm)
        # A tet of edge h occupies about h^3/(6*sqrt(2)) of volume.
        v_per_tet = self.volume_nm3() / max(self.target_elements, 1)
        return float((6.0 * math.sqrt(2.0) * v_per_tet) ** (1.0 / 3.0))

    def layer_lc_nm(self):
        """Element size inside the boundary layer [nm]."""
        if not self.boundary_layer:
            return self.interior_lc_nm()
        if self.boundary_lc_nm:
            return float(self.boundary_lc_nm)
        if self.boundary_layers_across:
            return float(self.boundary_thickness_nm) / int(self.boundary_layers_across)
        return self.interior_lc_nm() * float(self.boundary_size_ratio)

    def layers_across(self):
        """Elements spanning the layer at the nominal sizes."""
        return self.boundary_thickness_nm / max(self.layer_lc_nm(), 1e-30)

    def transition_nm(self):
        """Distance over which the size climbs from the layer to the interior."""
        if self.boundary_expansion:
            r = float(self.boundary_expansion)
            if r <= 1.0:
                raise ValueError("boundary_expansion must be > 1")
            return max(self.interior_lc_nm() - self.layer_lc_nm(), 0.0) / (r - 1.0)
        if self.boundary_transition_nm:
            return float(self.boundary_transition_nm)
        return 2.0 * self.boundary_thickness_nm

    def estimate_tets(self):
        """Rough element count before meshing — the layer dominates the cost.

        A tet of edge h fills h^3/(6*sqrt(2)). The shell at the wall size plus
        the remaining volume at the interior size is enough to decide whether a
        spec is affordable without waiting for Gmsh.
        """
        def per(h):
            return h ** 3 / (6.0 * math.sqrt(2.0))

        lx, ly, lz = self.resolved_extents_nm()
        V = self.volume_nm3()
        if not self.boundary_layer:
            return V / per(self.interior_lc_nm())

        T = self.boundary_thickness_nm
        if self.geometry == "cubic":
            V_core = max(lx - 2 * T, 0.0) * max(ly - 2 * T, 0.0) * max(lz - 2 * T, 0.0)
        else:
            # Hexagonal prism eroded by T: apothem and height both shrink.
            R = 0.5 * self.size_nm
            H = self.height_nm if self.height_nm else self.size_nm
            a = math.sqrt(3.0) / 2.0 * R - T          # apothem after erosion
            V_core = (2.0 * math.sqrt(3.0) * a * a * max(H - 2 * T, 0.0)
                      if a > 0 else 0.0)
        V_shell = max(V - V_core, 0.0)
        return V_shell / per(self.layer_lc_nm()) + V_core / per(self.interior_lc_nm())

    def slug(self):
        """Stable, readable file stem including a hash of the full spec."""
        h = hashlib.sha1(repr(sorted(asdict(self).items())).encode()).hexdigest()[:8]
        bl = "_bl" if self.boundary_layer else ""
        return (f"{self.geometry}_{self.size_nm:g}nm_"
                f"{self.target_elements}el_o{self.element_order}{bl}_{h}")


@dataclass
class MeshResult:
    path: Path
    spec: MeshSpec
    n_nodes: int
    n_tets: int
    extents_nm: tuple
    F_b: np.ndarray = field(repr=False)      # 3x3 deformation gradient, in b


# ── geometry construction ────────────────────────────────────────────────────
def _build_cubic(gmsh, spec):
    lx, ly, lz = spec.resolved_extents_nm()
    return gmsh.model.occ.addBox(0.0, 0.0, 0.0, lx, ly, lz)


def _build_hexagonal(gmsh, spec):
    """Regular hexagonal prism, axis along z, one vertex on +x.

    Vertex on +x puts a prism facet normal along a <a>-type direction, which is
    the orientation the prismatic <a> loop families are defined against.
    """
    R = 0.5 * spec.size_nm
    H = spec.height_nm if spec.height_nm else spec.size_nm
    pts = [gmsh.model.occ.addPoint(R * math.cos(k * math.pi / 3.0),
                                   R * math.sin(k * math.pi / 3.0), 0.0)
           for k in range(6)]
    lines = [gmsh.model.occ.addLine(pts[k], pts[(k + 1) % 6]) for k in range(6)]
    loop = gmsh.model.occ.addCurveLoop(lines)
    face = gmsh.model.occ.addPlaneSurface([loop])
    out = gmsh.model.occ.extrude([(2, face)], 0.0, 0.0, H)
    vol = [tag for (dim, tag) in out if dim == 3]
    if not vol:
        raise RuntimeError("hexagonal extrusion produced no volume")
    return vol[0]


def _apply_sizing(gmsh, spec, vol_tag):
    """Uniform interior size, optionally refined in a layer at the surface."""
    lc = spec.interior_lc_nm()
    if not spec.boundary_layer:
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        return lc, lc

    lc_bnd = spec.layer_lc_nm()
    trans = spec.transition_nm()
    surfaces = [t for (d, t) in gmsh.model.getBoundary([(3, vol_tag)],
                                                       oriented=False)]
    # Distance from every bounding surface, then a threshold that holds lc_bnd
    # out to the layer thickness (DistMin) and only then grades up to lc.
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", 200)

    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_bnd)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", spec.boundary_thickness_nm)
    gmsh.model.mesh.field.setNumber(thr, "DistMax",
                                    spec.boundary_thickness_nm + trans)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)

    # The background field must win over the point/curve-derived sizes.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_bnd)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    return lc, lc_bnd


# ── writer ───────────────────────────────────────────────────────────────────
_TET_TYPE = {1: 4, 2: 11}          # element order -> msh element type
_TET_NODES = {4: 4, 11: 10}


def _write_msh22(path, coords_unit, tets, tet_type, region_id):
    """Write msh 2.2 ASCII with only tetrahedra, region_id in tags[1].

    Both tags are written as `region_id`: MoDELib reads tags[1], and keeping
    tags[0] the same avoids any ambiguity about which is the physical group.
    """
    n_per = _TET_NODES[tet_type]
    with open(path, "w", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(coords_unit)}\n")
        for i, (x, y, z) in enumerate(coords_unit, start=1):
            f.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(tets)}\n")
        for e, nodes in enumerate(tets, start=1):
            ids = " ".join(str(int(n)) for n in nodes[:n_per])
            f.write(f"{e} {tet_type} 2 {region_id} {region_id} {ids}\n")
        f.write("$EndElements\n")


def generate(spec: MeshSpec, out_dir=None, force=False):
    """Build the mesh and return a MeshResult. Cached on the spec hash."""
    import gmsh

    out_dir = Path(out_dir) if out_dir else MESH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.slug()}.msh"

    if path.is_file() and not force:
        n_nodes, n_tets = _scan(path)
        return MeshResult(path, spec, n_nodes, n_tets,
                          spec.resolved_extents_nm(),
                          mesh_deformation_gradient(spec))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", int(spec.verbosity > 0))
        gmsh.option.setNumber("General.Verbosity", spec.verbosity or 1)
        gmsh.model.add("dislocluster")

        if spec.geometry == "cubic":
            vol = _build_cubic(gmsh, spec)
        elif spec.geometry == "hexagonal":
            vol = _build_hexagonal(gmsh, spec)
        else:
            raise ValueError(f"unknown geometry {spec.geometry!r}")
        gmsh.model.occ.synchronize()

        _apply_sizing(gmsh, spec, vol)
        gmsh.model.mesh.generate(3)
        if spec.optimize:
            gmsh.model.mesh.optimize("Netgen")
        if spec.element_order == 2:
            gmsh.model.mesh.setOrder(2)
        elif spec.element_order != 1:
            raise ValueError("element_order must be 1 (linear) or 2 (quadratic)")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, float).reshape(-1, 3)
        order = np.argsort(node_tags)
        node_tags = np.asarray(node_tags)[order]
        coords = coords[order]
        # Renumber to a dense 1..N so the .msh node IDs are contiguous.
        remap = {int(t): i + 1 for i, t in enumerate(node_tags)}

        want = _TET_TYPE[spec.element_order]
        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        tets = None
        for et, en in zip(etypes, enodes):
            if int(et) == want:
                tets = np.asarray(en, dtype=np.int64).reshape(-1, _TET_NODES[want])
        if tets is None or len(tets) == 0:
            raise RuntimeError(f"no element type {want} produced by Gmsh")
        tets = np.vectorize(remap.__getitem__)(tets)

        lo, hi = coords.min(0), coords.max(0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        coords_unit = (coords - lo) / span          # -> [0,1]^3, F restores size

        _write_msh22(path, coords_unit, tets, want, spec.region_id)
        n_nodes, n_tets = len(coords_unit), len(tets)
    finally:
        gmsh.finalize()

    return MeshResult(path, spec, n_nodes, n_tets, spec.resolved_extents_nm(),
                      mesh_deformation_gradient(spec))


def mesh_deformation_gradient(spec: MeshSpec):
    """Diagonal F in units of b, mapping the unit mesh to physical size.

    Because the mesh is normalised by its own bounding box, using the bounding
    box extents here reproduces the requested shape exactly — including a
    regular hexagon, whose x and y extents differ by sqrt(3)/2.
    """
    lx, ly, lz = spec.resolved_extents_nm()
    return np.diag([lx * 1e-9 / B_SI, ly * 1e-9 / B_SI, lz * 1e-9 / B_SI])


def read_msh(path):
    """Read a msh 2.2 ASCII file written by this module.

    Returns (nodes [N,3] in the file's unit coordinates, tets [M,k]) with node
    indices zero-based. Only tetrahedra (types 4 and 11) are returned, and only
    their first four vertices — the mid-edge nodes of a quadratic tet play no
    part in drawing the element.
    """
    nodes, tets = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("$Nodes"):
                for _ in range(int(next(f).strip())):
                    parts = next(f).split()
                    nodes.append([float(v) for v in parts[1:4]])
            elif line.startswith("$Elements"):
                for _ in range(int(next(f).strip())):
                    parts = next(f).split()
                    etype, ntags = int(parts[1]), int(parts[2])
                    if etype in (4, 11):
                        ids = parts[3 + ntags:3 + ntags + 4]
                        tets.append([int(v) - 1 for v in ids])
    return np.asarray(nodes, float), np.asarray(tets, np.int64)


def _scan(path):
    """(n_nodes, n_tets) of an existing msh 2.2 file, without a full parse."""
    n_nodes = n_elems = 0
    with open(path) as f:
        for line in f:
            if line.startswith("$Nodes"):
                n_nodes = int(next(f).strip())
            elif line.startswith("$Elements"):
                n_elems = int(next(f).strip())
                break
    return n_nodes, n_elems


def polycrystal_F_block(spec: MeshSpec):
    """The `F=...;` block for polycrystal.txt, in MoDELib's exact layout."""
    F = mesh_deformation_gradient(spec)
    rows = ["  ".join(f"{v:.10g}" for v in F[i]) for i in range(3)]
    return "F=" + rows[0] + "\n  " + rows[1] + "\n  " + rows[2] + ";"


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry", choices=("cubic", "hexagonal"), default="cubic")
    p.add_argument("--size-nm", type=float, default=1000.0)
    p.add_argument("--height-nm", type=float, default=None)
    p.add_argument("--target-elements", type=int, default=15000)
    p.add_argument("--lc-nm", type=float, default=None)
    p.add_argument("--element-order", type=int, choices=(1, 2), default=1)
    p.add_argument("--boundary-layer", action="store_true")
    p.add_argument("--boundary-thickness-nm", type=float, default=150.0)
    p.add_argument("--boundary-size-ratio", type=float, default=0.35)
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)
    a = p.parse_args(argv)

    spec = MeshSpec(geometry=a.geometry, size_nm=a.size_nm, height_nm=a.height_nm,
                    target_elements=a.target_elements, lc_nm=a.lc_nm,
                    element_order=a.element_order,
                    boundary_layer=a.boundary_layer,
                    boundary_thickness_nm=a.boundary_thickness_nm,
                    boundary_size_ratio=a.boundary_size_ratio,
                    optimize=not a.no_optimize, verbosity=a.verbose)
    r = generate(spec, force=a.force)
    lx, ly, lz = r.extents_nm
    print(f"mesh      : {r.path}")
    print(f"geometry  : {spec.geometry}")
    print(f"extents   : {lx:.1f} x {ly:.1f} x {lz:.1f} nm")
    print(f"interior lc: {spec.interior_lc_nm():.2f} nm"
          + (f"   boundary lc: {spec.interior_lc_nm()*spec.boundary_size_ratio:.2f} nm"
             f" within {spec.boundary_thickness_nm:g} nm" if spec.boundary_layer else ""))
    print(f"nodes     : {r.n_nodes}")
    print(f"tets      : {r.n_tets}  (msh type {_TET_TYPE[spec.element_order]})")
    print()
    print(polycrystal_F_block(spec))
    return r


if __name__ == "__main__":
    main()
