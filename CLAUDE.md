# DisloCluster — Project-Level Claude Code Instructions

## What this repository is

**DisloCluster** couples cluster dynamics to dislocation dynamics for irradiated
zirconium. It was assembled from two previously separate repositories and is now
self-contained: nothing outside the repository root is required except the
Python interpreter, SUNDIALS, and (on Windows) WSL.

| Directory | Role | Language |
|---|---|---|
| `ZrClusterDynamics/ZrMicro/` | 0-D reduced cluster dynamics — 19 ODEs (12 physical species, 6 conservation accumulators, ρ_N) | Python + C++/SUNDIALS |
| `ZrClusterDynamics/Gmsh/` | simulation-domain meshing (cubic / hexagonal) | Python + Gmsh SDK |
| `MoDELib3/` | 3-D spatially-resolved cluster dynamics / dislocation dynamics (fork of MoDELib2-NNL) | C++20 |
| `ZrClusterDynamics/Docs/Formulation/` | Formulation LaTeX/PDF, MoDELib WSL build script | — |

Two drivers:

| Driver | What it runs |
|---|---|
| [`ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb`](ZrClusterDynamics/ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb) | the operator-split 0-D ↔ 3-D dose march |
| [`ZrMicro/py_utils/run_zr3d_singlecrystal.py`](ZrClusterDynamics/ZrMicro/py_utils/run_zr3d_singlecrystal.py) | post-processes the standalone 3-D **single cubic crystal** run |

---

## Path resolution — read this before touching any path

**Never hard-code an absolute path.** All locations come from
[`ZrClusterDynamics/ZrMicro/py_utils/paths.py`](ZrClusterDynamics/ZrMicro/py_utils/paths.py),
which walks up to the `.dislocluster_root` marker at the repository root.

```python
from py_utils import paths
paths.REPO_ROOT      paths.ZR_ROOT        paths.ZRMICRO_DIR
paths.INPUT_DIR      paths.OUTPUT_DIR     paths.CPP_UTILS / BUILD_DIR
paths.MODELIB_ROOT   paths.MODELIB_BUILD  paths.MODELIB_MATERIAL
paths.zrmicro_solver_exe()   paths.modelib_ddomp()   paths.venv_python()
paths.git_hash()             paths.windows_to_wsl()  paths.use_wsl()
print(paths.describe())      # resolution report, OK/MISS per location
```

Overrides: `DISLOCLUSTER_ROOT`, `MODELIB_ROOT`, `MODELIB_BUILD`.

If a new module needs a repository location, add it to `paths.py` — do not
re-derive it with `Path(__file__).parent.parent`.

---

## Python environment

- **Venv:** `.DisloClusterVenv/` at the repository root (Python 3.14)
- **Jupyter kernel:** `dislocluster` — "Python 3.14 (DisloCluster)"
- **Dependencies:** `requirements.txt` at the repository root
- **Do not use Anaconda Python** — NumPy 1.x/2.x conflict with SciPy

```powershell
.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute `
    --ExecutePreprocessor.kernel_name=dislocluster <notebook.ipynb>
```

---

## Builds

### 0-D solver (CMake + SUNDIALS 7.1.1)

```powershell
cmake -S ZrClusterDynamics\ZrMicro\cpp_utils -B ZrClusterDynamics\ZrMicro\build -DCMAKE_BUILD_TYPE=Release
cmake --build ZrClusterDynamics\ZrMicro\build --config Release
```

SUNDIALS is found either at `<repo>/Libraries/sundials-7.1.1/` or from the
system install (`C:/Program Files (x86)/SUNDIALS`). Modules used: `cvode`,
`arkode`, `nvecserial`, `sunlinsolband`, `sunlinsolspgmr`. OpenMP is optional
but enables the parallel `--batch_file` mode the coupling march depends on.

### 3-D code (WSL)

```bash
wsl -u root -e bash ZrClusterDynamics/Docs/Formulation/build_modelib_wsl.sh
```

Defaults to `<repo>/MoDELib3`. The script detects and discards a CMake cache
configured under a different absolute path, so a relocated checkout rebuilds
cleanly.

**Moving or renaming the repository invalidates both build trees** — a CMake
cache stores absolute paths, so a reconfigure hard-errors with "the current
CMakeCache.txt directory ... is different". Both sides now detect this and
discard the stale cache themselves: `build_modelib_wsl.sh` for the 3-D tree,
`ensure_zrmicro_solver` in the coupling notebook for the 0-D tree. To do it by
hand, delete `ZrClusterDynamics/ZrMicro/build/` and reconfigure.

---

## The coupling notebook's control blocks

`coupled_0d_3d_ZrMicro.ipynb` is driven entirely from three dicts. Nothing else
in the notebook should need editing for a normal run.

| Cell | Dict | Controls |
|---|---|---|
| 6a | `GEOMETRY`, `MESH` | domain type (`cubic` / `hexagonal`), dimensions, element size and order, boundary-layer refinement |
| 6b | `RUN` | temperature, dose rate, stress, dose schedule, tolerances, substeps, march grid |
| 6c | `PARAMETER_OVERRIDES` | 0-D parameters applied on top of the Excel workbook |

The dose schedule is `n_intervals` equal steps from `dose_seed` to `dose_max`;
the defaults (1 → 26 dpa, 5 intervals) give snapshots at 1, 6, 11, 16, 21, 26 dpa.
`substeps_per_interval` fixes the immobile substep count **per dose interval**
rather than by a wall-clock cap — a 5 dpa step at 1e-7 dpa/s is 5×10⁷ s, which a
fixed 5×10⁴ s cap would have split into a thousand batch subprocesses.

Section 6a writes the mesh through `Gmsh/generate_mesh.py`; the cell after the
run directory materialises a complete MoDELib sim dir (`<run>/sim/`) with that
mesh, an `F` restoring the physical size, and a `DD.txt` dose schedule matching
the snapshots. **The 3-D solve is not launched** — the notebook prints the
command. `EVL_DIR` selects which `evl/` the figures are rendered from; it
defaults to the verification case so the figure cells always have data.

### Figure output

| Directory | Contents |
|---|---|
| `0d/` | the 0-D figure suite, unchanged |
| `3d/` | `fe_mesh.png`, plus Figs 19–23 **one file per quantity per dose**: `Cv_06dpa.png`, `N_c_06dpa.png`, `C_a1_06dpa.png`, `loops_c_06dpa.png`, … |
| `gb/` | Figs 24–27 **one file per quantity**, all doses overlaid (`gb_Cv.png`, `gb_N_a1.png`, `gb_C_c.png`, `gb_d_c.png`), plus `size_dist_<family>_<dose>.png` |

**Loop overlays** (`loops_*`) fill the domain with as many **non-overlapping**
platelets as fit: candidates are visited in random order and accepted only if the
drawn radius clears every platelet already placed. The clearance test uses
bounding spheres, so it is conservative for discs. Radii are exaggerated ×1.5
(⟨c⟩) and ×7 (⟨a⟩) — half the deliverable's factors, which is what makes a dense
non-overlapping fill possible. The two factors differ because ⟨c⟩ loops are ~6×
larger, so sizes are faithful *within* a figure but not *between* the ⟨c⟩ and
⟨a⟩ figures. These are the slow figures: ~40 s each, so ~17 min for six doses.

**`size_dist_*`** is the distribution of the **local mean** loop diameter across
the domain, weighted by local loop density — *not* a per-loop size spectrum. The
model carries one mean size per family per node, so spatial variation is the only
source of spread; the result is strongly bimodal (small numerous loops in the
boundary shell, large sparse ones in the interior). Nodes are weighted as
equal-volume, which over-weights a refined region: `evl/cdNodes.txt` carries
positions only, with no connectivity from which nodal volumes could be formed.

Rendering lives in `py_utils/modelib_report.py`; it reads `evl/cdNodes.txt` +
`evl/evl_<N>.txt` and never runs a solve.

---

## The coupling contract

Operator-split QSSA over each dose step `[γ_n, γ_{n+1}]`:

1. **Fast solve** — steady mobile field `C_M*(x)` with the immobile state frozen
   (`modelib_fem.MoDELibFEMSolver.solve`, or the analytic placeholder).
2. **Slow march** — the immobile ODEs integrated independently at every
   quadrature point with the mobile species frozen
   (`modelib_coupling.run_immobile_step`, one OpenMP batch subprocess).

Splitting error is first order in `Δγ`.

| Direction | Carrier | Module |
|---|---|---|
| 0-D → 3-D | per-family loop density/radius → MoDELib sink field | `modelib_fem.write_sink_field` |
| 3-D → 0-D | `C_M*` at quadrature points | `modelib_fem.solve` |
| 0-D → 3-D | dose-indexed closure table | `modelib_export` |
| both | material constants | `MoDELib3/Library/Materials/Zr3d_ghoniem.txt` (`paths.MODELIB_MATERIAL`) |

State vector (19): `[Cv, Ci, C2i, C3i, CiL, CaiL, CvL, CavL, CiL_i, CaiL_i,
CvL_v, CavL_v, 6 accumulators, rho_N]`.

---

## ODE solver options

Recommended: CVODE BDF + dense.

```python
'solver_method': {'backend': 'cvode', 'lmm': 'bdf', 'linsol': 'dense'}
```

Accuracy benchmark (ZrMicro, `rtol=1e-6`, `atol=1e-20` vs tight reference):

| Config | Backend | LinSol | Max rel. error |
|---|---|---|---|
| A (default) | CVODE BDF | dense | 3.0×10⁻⁴ |
| B | CVODE BDF | band | 3.0×10⁻⁴ |
| C | CVODE BDF | gmres | 1.7×10⁻² |
| G | ARKODE SDIRK_5_3_4 | dense | 3.0×10⁻¹ |
| F, H, I | ARKODE (other tables) | dense | 10⁴–10⁷ (fails) |

ARKODE methods designed as IMEX pairs perform poorly in pure implicit mode here.

---

## The 3-D single-cubic-crystal case

`MoDELib3/tutorials/zrmicro_coupled` is the standalone 3-D verification case: a
1 µm cube (3093 b per side, 24115 FE nodes) with Dirichlet conditions on **all
six faces**, so the whole surface is grain boundary and the `gb/` profiles bin by
the minimum distance to any face. 31 output steps of 1 dpa at G = 1e-7 dpa/s,
T = 573 K, zero stress; nucleation is cascade **plus** homogeneous SIA clustering.

The solve writes `evl/cdNodes.txt` (node coordinates — the CD trial functions
live on second-order elements, so field rows do *not* match `.msh` vertices) and
`evl/evl_<N>.txt` (the CD field block, same node order). Output is written after
`solve()`, so `evl_N` holds the state at N+1 dose steps.

```bash
wsl -e bash MoDELib3/tutorials/zrmicro_coupled/clean_run.sh    # the solve (hours)
```
```powershell
.DisloClusterVenv\Scripts\python.exe `
    ZrClusterDynamics\ZrMicro\py_utils\run_zr3d_singlecrystal.py    # the figures
```

The Python step only reads `evl/`; it never re-runs the solve. It writes a
timestamped `<stamp>_<hash>_zr3d_ghoniem/` with `3d/` (field panels + per-family
platelet overlays), `gb/` (profiles over the first 150 nm) and `provenance.md`
carrying the interior state table.

**Near-boundary loop densities are not quantitative.** Cascade nucleation is
spatially uniform but the only loop-removal channel is coalescence, driven by the
absorbed mobile flux — which vanishes where Dirichlet pins the mobile
concentrations. The boundary is a sink for mobile defects but **not** for loops,
so ⟨a⟩ density climbs steeply toward it and keeps growing linearly in dose. The
interior comparison against the 0-D model is unaffected.

---

## Output reproducibility

Every run creates
`ZrClusterDynamics/ZrMicro/output/<YYYYMMDD_HHMMSS>_<git-hash>[_<tag>]/` with figures and
`provenance.md`. The coupled driver adds `0d/`, `3d/`, `gb/` subdirectories and
`march_state.npz`.

Because the DisloCluster root is not itself a git repository (`ZrClusterDynamics/` and
`MoDELib3/` carry their own `.git`), `paths.git_hash()` falls back root →
`ZrClusterDynamics` → `MoDELib3`.

---

## Known non-correspondences between the 0-D and 3-D models

These are physics, not bugs — do not "fix" them silently:

1. **DAD bias** — 0-D uses phenomenological `δ_i, δ_v` with `Z_a + Z_c = 2`;
   3-D generates the bias from the diffusion tensors via `p_m`. They are
   currently reconciled by a closed-form fit of `(Z0_m, p_m)` in
   `Zr3d_ghoniem.txt`, valid only at the fitted temperature. `Zr4.txt` is the
   standalone 3-D calibration and carries no `dad*` keys — do not check
   against it.
2. **Bi-pyramid vacancy family** — exists only in 3-D.
3. **Size-dependent vacancy-loop thermal emission** — 3-D only; 0-D uses a
   constant surrogate.

---

## See also

- [`README.md`](README.md) — setup and run instructions
- [`ZrClusterDynamics/CLAUDE.md`](ZrClusterDynamics/CLAUDE.md) — 0-D side details
- [`ZrClusterDynamics/ZrMicro/CLAUDE.md`](ZrClusterDynamics/ZrMicro/CLAUDE.md) — model equations, file map
- [`MoDELib3/ZR3D_GHONIEM_CHANGES.md`](MoDELib3/ZR3D_GHONIEM_CHANGES.md) — 3-D side changes
