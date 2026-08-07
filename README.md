# DisloCluster

**Coupled cluster dynamics ↔ dislocation dynamics for irradiated zirconium.**

DisloCluster couples a 0-D reduced cluster-dynamics model of point-defect and
loop evolution to a spatially-resolved 3-D cluster-dynamics / dislocation-dynamics
code, using an operator-split quasi-steady-state scheme in which the fast mobile
species are solved as a steady field and the slow immobile (loop) population is
marched pointwise in dose.

---

## Layout

```
DisloCluster/                        <- repository root (.dislocluster_root marker)
├── .DisloClusterVenv/               Python 3.14 environment for the whole repo
├── requirements.txt                 pinned dependencies
├── ZrClusterDynamics/               0-D cluster dynamics (ZrMicro) + formulation docs
│   ├── Docs/Formulation/            LaTeX/PDF formulation, MoDELib build script
│   └── ZrMicro/
│       ├── code/                    notebooks + solver.cpp
│       │   ├── coupled_0d_3d_ZrMicro.ipynb   ← the coupled 0-D ↔ 3-D driver
│       │   ├── ZrMicro.ipynb                 0-D run + parameter identification
│       │   └── coupling_demo.ipynb           minimal coupling walk-through
│       ├── cpp_utils/               C++ RHS + CMake build (SUNDIALS CVODE)
│       ├── py_utils/                Python model chain
│       │   └── paths.py             ← single source of truth for every path
│       ├── input/                   Excel parameter workbooks
│       ├── build/                   compiled solver.exe
│       └── output/                  timestamped run directories
└── MoDELib3/                        3-D DD / spatially-resolved CD (MoDELib2-NNL fork)
    ├── Library/Materials/            Zr3d_ghoniem.txt is the coupled material
    ├── tutorials/                   simulation-directory templates
    └── build/                       DDomp + pyMoDELib (built inside WSL on Windows)
```

## How the two codes communicate

| Direction | Carrier | Module |
|---|---|---|
| 0-D → 3-D | loop density/radius per family → MoDELib microstructure sink field | `py_utils/modelib_fem.py: write_sink_field` |
| 3-D → 0-D | steady mobile field `C_M*(x)` at quadrature points | `py_utils/modelib_fem.py: solve` |
| 0-D → 3-D | dose-indexed closure table | `py_utils/modelib_export.py` |
| march | pointwise immobile ODE step, mobile frozen | `py_utils/modelib_coupling.py: run_immobile_step` |
| paths | repo root, both codes, binaries, material file | `py_utils/paths.py` |

Nothing in the repository hard-codes an absolute path. `py_utils/paths.py` finds
the root by walking up to the `.dislocluster_root` marker, so the repository can
be cloned or moved anywhere. Two environment variables override the defaults if
MoDELib lives outside the tree:

```
DISLOCLUSTER_ROOT   repository root
MODELIB_ROOT        MoDELib checkout          (default: <root>/MoDELib3)
MODELIB_BUILD       MoDELib build directory   (default: <MODELIB_ROOT>/build)
```

Check the resolution at any time:

```powershell
.DisloClusterVenv\Scripts\python.exe ZrClusterDynamics\ZrMicro\py_utils\paths.py
```

---

## Setup

### 1. Python environment

```powershell
C:\Python314\python.exe -m venv .DisloClusterVenv
.DisloClusterVenv\Scripts\python.exe -m pip install -r requirements.txt
.DisloClusterVenv\Scripts\python.exe -m ipykernel install --user `
    --name dislocluster --display-name "Python 3.14 (DisloCluster)"
```

Do **not** use Anaconda Python — its NumPy 1.x/2.x mix conflicts with SciPy here.

### 2. 0-D solver (C++ / SUNDIALS CVODE 7.1.1)

```powershell
cmake -S ZrClusterDynamics\ZrMicro\cpp_utils -B ZrClusterDynamics\ZrMicro\build -DCMAKE_BUILD_TYPE=Release
cmake --build ZrClusterDynamics\ZrMicro\build --config Release
```

Produces `ZrClusterDynamics\ZrMicro\build\Release\solver.exe`. OpenMP is detected
automatically and enables the parallel batch mode the coupling march relies on.

### 3. 3-D code (MoDELib, built inside WSL on Windows)

```bash
wsl -u root -e bash ZrClusterDynamics/Docs/Formulation/build_modelib_wsl.sh
```

With no argument the script builds `<repo>/MoDELib3`. It produces
`MoDELib3/build/tools/DDomp/DDomp` (a Linux ELF, invoked through `wsl.exe`) and
`MoDELib3/build/tools/pyMoDELib/pyMoDELib*.so`.

---

## Running

Open `ZrClusterDynamics/ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb` with the
**Python 3.14 (DisloCluster)** kernel, or run it headless:

```powershell
.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute `
    --ExecutePreprocessor.kernel_name=dislocluster `
    --output-dir ZrClusterDynamics\ZrMicro\output `
    ZrClusterDynamics\ZrMicro\code\coupled_0d_3d_ZrMicro.ipynb
```

Every run writes `ZrClusterDynamics/ZrMicro/output/<YYYYMMDD_HHMMSS>_<git-hash>_coupled0d3d/`
containing `0d/`, `3d/`, `gb/` figure sets, `march_state.npz`, and
`provenance.md` recording the resolved repository layout, the controls, and the
active fast-solve backend.

### Fast-solve backends

| `FAST_SOLVE_BACKEND` | Behaviour |
|---|---|
| `"auto"` (default) | MoDELib FEM if DDomp is built **and** the sim dir is seeded, else placeholder |
| `"modelib"` | require the real FEM solve; raise if unavailable |
| `"placeholder"` | analytic `C_M*(x) = [1 − exp(−x/ℓ)]·C_M,bulk` — pipeline checks only |

The placeholder is **not** a MoDELib result; its boundary profile is an artifact
(it scales all four mobile species by the same factor) and must not be read
physically.

To activate the real backend, seed a simulation directory once:

```python
from py_utils.modelib_fem import seed_sim_dir_from_tutorial
from py_utils import paths
seed_sim_dir_from_tutorial(paths.MODELIB_ROOT, paths.OUTPUT_DIR / "modelib_sim")
# then run its generateInputFiles.py with MoDELib's MicrostructureGenerator
```

---

## The 3-D single-cubic-crystal run

A standalone 3-D run, independent of the 0-D ↔ 3-D march above: a 1 µm cube
(3093 b per side, 24115 FE nodes), Dirichlet on all six faces, 31 output steps of
1 dpa at G = 1e-7 dpa/s, T = 573 K, zero stress.

```bash
# 1. the solve (hours) — regenerates evl/cdNodes.txt and evl/evl_<N>.txt
wsl -e bash MoDELib3/tutorials/zrmicro_coupled/clean_run.sh
```
```powershell
# 2. the figures (minutes) — reads evl/ only, never re-runs the solve
.DisloClusterVenv\Scripts\python.exe `
    ZrClusterDynamics\ZrMicro\py_utils\run_zr3d_singlecrystal.py `
    --doses 1 5 10 30 --max-nm 150
```

`evl/` is already populated in this checkout, so step 2 runs immediately. Output
goes to `ZrClusterDynamics/ZrMicro/output/<stamp>_<hash>_zr3d_ghoniem/`:

| Path | Contents |
|---|---|
| `3d/mobile_fields.png` | `Cv, Ci, C2i, C3i` on an interior mid-plane cut, one column per dose |
| `3d/immobile_{density,content}_fields.png` | the four loop families, same layout |
| `3d/loops_{c,a1,a2,a3}_overlay.png` | one family per figure: content field + platelets at the true local radius × 3 (⟨c⟩) or × 14 (⟨a⟩) |
| `gb/gb_{density,content,size,mobile}_profiles.png` | radial averages over the first 150 nm from the boundary |
| `provenance.md` | interior state table, size-vs-dose table, exact reproduce command |

Options: `--doses`, `--max-nm`, `--tag`, `--sim-dir`, `--no-overlays` (the
overlays dominate the runtime).

Two things to read carefully:

- Platelet radii are exaggerated and the factor **differs between the ⟨c⟩ and
  ⟨a⟩ figures**, so sizes are faithful *within* a figure but not *between* them.
- **Near-boundary loop density is not quantitative.** The Dirichlet faces are a
  sink for mobile defects but not for loops, so ⟨a⟩ density rises steeply toward
  the boundary and grows linearly in dose there. The interior is unaffected.

---

## Notes inherited from the source repositories

- `ZrClusterDynamics/` and `MoDELib3/` each still carry their own `.git`; the DisloCluster
  root is not itself a git repository. `paths.git_hash()` falls back through
  root → `ZrClusterDynamics` → `MoDELib3` so provenance tags stay meaningful.
- The 0-D regression baseline pinned in the coupling notebook
  (`output/20260622_144021_7959445`) was not copied into this repository, so the
  regression cell reports "skipped" rather than failing. Point `REFERENCE_RUN` at
  any local run directory to re-enable it.
- The 0-D and 3-D DAD (diffusion-anisotropy-difference) parameterisations are
  reconciled by a closed-form fit of `(Z0_m, p_m)` in `Zr3d_ghoniem.txt`
  (`paths.MODELIB_MATERIAL`), tied to the temperature of the fit. Refit if
  `TEMPERATURE_K` changes. `Zr4.txt` is the standalone 3-D calibration and
  carries no `dad*` keys.
