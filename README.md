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
├── requirements.txt                 pinned environment lock
├── pyproject.toml                   the `dislocluster_code` package (pip install -e .)
├── dislocluster_code/                    <- ALL Python lives here
│   ├── paths.py                     single source of truth for every path
│   ├── config.py                    SimulationConfig — the notebook's dicts
│   ├── driver.py                    prepare / march / report
│   ├── build.py                     ensure_zrmicro_solver, ensure_modelib
│   └── zerod/ staging/ coupling/ post/ studies/ fitting/ legacy/
├── Simulations/                     <- where simulations are run
│   ├── run_simulation.ipynb         configure, stage, march, report
│   ├── postprocess.ipynb            re-render an existing run
│   ├── input/                       the Excel workbooks  (INPUT_DIR)
│   └── output/                      run directories      (OUTPUT_DIR)
├── Gmsh/                            simulation-domain meshing
│   ├── generate_mesh.py             cubic / hexagonal, boundary-layer refined
│   └── meshes/                      cache, gitignored (reproducible from the spec)
├── Docs/                            ALL documents live here (see below)
│   ├── DisloCluster Manual/         manuals and development notes
│   ├── Formulation/                 LaTeX/PDF formulation, MoDELib build script
│   ├── Reports/                     deliverables
│   └── Presentations/               slides
├── ZrMicro/                         0-D cluster dynamics
│   ├── code/                        notebooks + solver.cpp
│   │   ├── coupled_0d_3d_ZrMicro.ipynb   superseded by Simulations/
│   │   ├── ZrMicro.ipynb                 0-D run + parameter identification
│   │   └── coupling_demo.ipynb           minimal coupling walk-through
│   ├── cpp_utils/                   C++ RHS + CMake build (SUNDIALS CVODE)
│   ├── py_utils/                    compatibility shim -> dislocluster
│   ├── build/                       compiled solver.exe
│   ├── input/                       the older workbook copies (see below)
│   └── output/                      runs made before Simulations/output/
└── MoDELib3/                        3-D DD / spatially-resolved CD (MoDELib-fullCD fork)
    ├── Library/Materials/            Zr3d_ghoniem.txt is the coupled material
    ├── tutorials/                   simulation-directory templates
    └── build/                       DDomp + pyMoDELib (built inside WSL on Windows)
```

## How the two codes communicate

| Direction | Carrier | Module |
|---|---|---|
| 0-D → 3-D | per-node immobile field → the `evl` CD block | `coupling/field.py: FieldBridge.write_immobile_field` |
| 3-D → 0-D | steady mobile field `C_M*(x)` at the CD nodes | `coupling/qssa.py: MobileQSSASolver.solve` |
| 0-D → 3-D | dose-indexed closure table | `coupling/export.py` |
| march | pointwise immobile ODE step, mobile frozen | `coupling/immobile.py: run_immobile_step` |
| paths | repo root, both codes, binaries, material file | `paths.py` |

The exchange is a genuine per-node field in both directions, through MoDELib's
own CD block in `evl/evl_<N>.txt`. `legacy/modelib_fem.py` is the earlier
scheme, which collapsed the whole immobile state to four scalars per dose step;
it is superseded and kept only so older results stay reproducible.

Nothing in the repository hard-codes an absolute path. `dislocluster_code/paths.py` finds
the root by walking up to the `.dislocluster_root` marker, so the repository can
be cloned or moved anywhere. Two environment variables override the defaults if
MoDELib lives outside the tree:

```
DISLOCLUSTER_ROOT     repository root
DISLOCLUSTER_SIM_ROOT staged simulation cases (default: <MODELIB_ROOT>/tutorials)
MODELIB_ROOT          MoDELib checkout        (default: <root>/MoDELib3)
MODELIB_BUILD         MoDELib build directory (default: <MODELIB_ROOT>/build)
```

Check the resolution at any time:

```powershell
.DisloClusterVenv\Scripts\python.exe -m dislocluster.paths
```

---

## Setup

### 1. Python environment

```powershell
C:\Python314\python.exe -m venv .DisloClusterVenv
.DisloClusterVenv\Scripts\python.exe -m pip install -r requirements.txt
.DisloClusterVenv\Scripts\python.exe -m pip install -e .
.DisloClusterVenv\Scripts\python.exe -m ipykernel install --user `
    --name dislocluster --display-name "Python 3.14 (DisloCluster)"
```

`pip install -e .` puts the `dislocluster_code` package on the path, which is what
lets the notebooks under `Simulations/` import it from anywhere.

Do **not** use Anaconda Python — its NumPy 1.x/2.x mix conflicts with SciPy here.

### 2. 0-D solver (C++ / SUNDIALS CVODE 7.1.1)

```bash
cmake -S ZrMicro/cpp_utils -B ZrMicro/build -DCMAKE_BUILD_TYPE=Release
cmake --build ZrMicro/build --config Release
```

Produces `ZrMicro/build/solver` (`Release\solver.exe` under MSVC). SUNDIALS is
found at `<repo>/Libraries/sundials-7.1.1/`, from a system install, or under the
Homebrew/MacPorts prefix on macOS. OpenMP is detected automatically and enables
the parallel batch mode the coupling march relies on.

### 3. 3-D code (MoDELib)

```bash
bash Docs/Formulation/build_modelib.sh          # Linux, macOS
wsl -u root -e bash Docs/Formulation/build_modelib.sh    # Windows (inside WSL)
```

One script for every platform: it installs the dependencies it needs (apt / dnf
/ pacman on Linux, Homebrew on macOS), discards a build cache configured under a
different absolute path, and builds with Ninja when it is available. With no
argument it builds `<repo>/MoDELib3`, producing
`MoDELib3/build/tools/DDomp/DDomp` and, when pybind11 is installed,
`MoDELib3/build/tools/pyMoDELib/pyMoDELib*.so`.

A C++20 compiler, CMake ≥ 3.16 and Eigen 3 are required. FFTW3, Boost,
SuiteSparse, OpenMP and pybind11 are optional — CMake reports which of them were
found in a configuration summary, and builds without those that are absent. On
Windows the binary is a Linux ELF and everything invokes it through `wsl.exe`;
on Linux and macOS it is executed directly. Nothing above chooses that for you:
`paths.ddomp_cmd()` decides per platform.

`build.preflight()` (the notebook's first cell) builds whatever is missing and
then *runs* both binaries, so a stale or unusable build is reported rather than
trusted.

---

## Running

### The repository is machine-agnostic

Nothing in it is tied to the machine it was last used on. Every location is
resolved by walking up to the `.dislocluster_root` marker, so the checkout can
live anywhere; **no build tree is in git**, so a fresh clone builds its own; and
`build.preflight()` — the notebook's second cell — builds whatever is missing
and then *runs* both binaries to prove they load, because existing on disk is
not the same as being runnable.

```python
from dislocluster_code import build
build.preflight()      # builds if needed, verifies, or raises with the fix
```

It raises with actionable text when it cannot fix things itself: SUNDIALS not
installed, WSL unavailable, no compiler. The MoDELib build runs inside WSL and
takes 10–30 min the first time; it is a no-op afterwards.

Open **[`Simulations/run_simulation.ipynb`](Simulations/run_simulation.ipynb)**
with the **Python 3.14 (DisloCluster)** kernel. Six dicts at the top of §2 set
the material, geometry, mesh, boundary conditions, coupling and output; the rest
of the notebook validates them, stages the MoDELib case and runs the march. See
[`Simulations/README.md`](Simulations/README.md).

Headless:

```powershell
$env:PYTHONIOENCODING="utf-8"
.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute `
    --ExecutePreprocessor.kernel_name=dislocluster `
    --ExecutePreprocessor.timeout=-1 `
    Simulations\run_simulation.ipynb
```

Or from Python, which is all the notebook does:

```python
from dislocluster_code import driver
from dislocluster_code.config import SimulationConfig

cfg    = SimulationConfig.from_dicts(GEOMETRY=dict(size_nm=500.0),
                                     COUPLING=dict(doses=[1, 6, 11, 16, 21]))
run    = driver.prepare(cfg)      # mesh, staged case, CD node set, seed
result = driver.march(run)        # the operator split; resumable
driver.report(run, result)        # figures, movies, report.md
```

Every run writes
`Simulations/output/<YYYYMMDD_HHMMSS>_<git-hash>_<tag>/` containing
`config.json` (the resolved controls), `march_state.npz`, `summary.json`,
`checkpoint/`, `evl_coupled/`, the `3d/` and `gb/` figure sets and `report.md`.

### Where inputs and outputs live

`Simulations/` is the working directory: `input/` beside `output/`, the way
`ZrMicro/` was used before.

| | Path | |
|---|---|---|
| `paths.INPUT_DIR` | `Simulations/input/` | the Excel workbooks |
| `paths.OUTPUT_DIR` | `Simulations/output/` | where a new run is **written** |
| `paths.OUTPUT_DIRS` | both output roots | where a run is **found** |

The 1.4 GB of runs already under `ZrMicro/output/` were left
in place, so `paths.find_runs()` and `paths.latest_run()` search both roots and
`postprocess.ipynb` still reaches them.

The workbooks were **copied**, not moved, so `ZrMicro/input/` still holds its
own set. Only `Simulations/input/` is read. Two copies can drift — and the 0-D
calibration drifted from its workbook once already, by five orders of magnitude
in N_a — so `paths.describe()` hashes them and prints a warning when they stop
agreeing. `paths.workbook_drift()` returns the offending names.

### Interrupting and resuming

The march checkpoints after every substep — about 57 ms against a fast solve of
several minutes — so **an interrupted run is resumed by running the same cell
again**. It continues from the last completed substep.

The checkpoint records the mesh's CD node set, the seed, the material, the
resolved 0-D parameter set and the march settings. If any of those changed,
resuming raises `CheckpointMismatch` instead of silently mixing two
configurations; set `OUTPUT['resume'] = 'never'` to start over deliberately.

`Ctrl-C` now stops the march. It used to corrupt it: the batch solver caught
`KeyboardInterrupt` and returned "every case failed", which the march read as
"keep every node's previous state" and then counted as a completed substep, so
the dose coordinate advanced with no physics applied.

### The fast step

There is no placeholder backend any more. Every substep solves the real steady
mobile field with MoDELib3's `solveMobileClusters`, driven through DDomp with
`useImmobileSolver=0`; `COUPLING['fem_every']` sets how many immobile substeps
run between two of them, and the splitting error is first order in **that**
spacing, not in the snapshot spacing.

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
    -m dislocluster_code.studies.run_zr3d_singlecrystal `
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

- The repository is a single git checkout. The 0-D tree and `MoDELib3/` were
  submodules once but became plain directories in `859af47`.
  `paths.git_hash()` still falls back root → `ZrClusterDynamics` → `MoDELib3`
  so a split checkout would keep working.
- The 0-D regression baseline pinned in the coupling notebook
  (`output/20260622_144021_7959445`) was not copied into this repository, so the
  regression cell reports "skipped" rather than failing. Point `REFERENCE_RUN` at
  any local run directory to re-enable it.
- The 0-D and 3-D DAD (diffusion-anisotropy-difference) parameterisations are
  reconciled by a closed-form fit of `(Z0_m, p_m)` in `Zr3d_ghoniem.txt`
  (`paths.MODELIB_MATERIAL`), tied to the temperature of the fit. Refit if
  `TEMPERATURE_K` changes. `Zr4.txt` is the standalone 3-D calibration and
  carries no `dad*` keys.
