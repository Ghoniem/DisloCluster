# Gmsh — simulation-domain meshing for DisloCluster

`generate_mesh.py` builds the 3-D domain with [Gmsh](https://gmsh.info) and
writes it in the only format MoDELib accepts. It is driven from the coupling
notebook (Section 6a) or from the command line.

```powershell
.DisloClusterVenv\Scripts\python.exe ZrClusterDynamics\Gmsh\generate_mesh.py `
    --geometry hexagonal --size-nm 1000 --height-nm 1000 `
    --target-elements 15000 --element-order 2 --boundary-layer
```

Gmsh itself comes from the `gmsh` wheel pinned in `requirements.txt`, which
bundles the full SDK — there is nothing to install separately. Generated meshes
are cached in `meshes/` under a hash of the full spec, so an unchanged spec is a
no-op. That directory is gitignored: the meshes are reproducible from the spec.

## Geometries

| `--geometry` | Domain |
|---|---|
| `cubic` | rectangular box; `--size-nm` is the side, scaled per axis by `aspect` |
| `hexagonal` | regular hexagonal prism, axis along **z** (the c-axis), one vertex on **+x** so a prism facet is normal to an ⟨a⟩-type direction. `--size-nm` is the corner-to-corner diameter, `--height-nm` the prism height |

## Mesh controls

| Control | Effect |
|---|---|
| `--target-elements` | sets the interior element size by inverting the volume-per-tet estimate |
| `--lc-nm` | explicit characteristic length; overrides `--target-elements` |
| `--element-order` | `1` → 4-node linear tets (msh type 4); `2` → 10-node quadratic (type 11) |
| `--boundary-layer` | refine a layer at the surface, via a Gmsh Distance + Threshold field |
| `--boundary-thickness-nm` | how deep the refined layer runs (default 150 nm) |
| `--boundary-size-ratio` | element size in the layer ÷ interior size (default 0.35) |
| `--no-optimize` | skip the Netgen optimisation pass |

The boundary layer matters: essentially all of the spatial structure lives within
about two depletion lengths (ℓ ≈ 73 nm) of the surface, so a uniform mesh either
wastes elements on the interior plateau or under-resolves the denuded zone.

## What MoDELib requires

These were read off MoDELib's own reader, not assumed:

- **msh version 2.x ASCII.** `GmshReader.cpp` accepts only `version>=2.0 &&
  version<3.0`; a v4 file is rejected at load with "Unsupported msh version".
- **Only element types 4 and 11** are consumed (`SimplexReader.h`). Lines and
  surface triangles are parsed and then discarded, so the writer omits them.
- **The region ID is `tags[1]`** — Gmsh's *elementary entity* tag, not the
  physical-group tag. The generator forces both tags to `region_id` so a
  single-crystal mesh lands in region 1.

## Coordinate convention

The mesh is written **normalised to the unit bounding box** `[0,1]³`, exactly
like `MoDELib3/Library/Meshes/unitCube_15K.msh`. Physical size is applied by
MoDELib through `F` in `polycrystal.txt`, which maps `x = F·(X − X₀)`.
`polycrystal_F_block(spec)` emits the matching diagonal `F` in units of the
Burgers vector.

Because the normalisation and `F` are exact inverses, the shape is preserved —
including a regular hexagon, whose x and y extents differ by √3/2. For the 1 µm
cube this reproduces the reference run's `F = 3093`.
