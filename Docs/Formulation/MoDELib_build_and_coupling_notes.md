# Building MoDELib and Coupling it to `ZrMicro`

Companion to `ZrMicro_MoDELib2_two_time_scale_coupling.tex` and
`ZrMicro/coupling_demo.ipynb`. This documents how to build the spatially-resolved
solver and activate the **`"modelib"`** fast-solve backend in the coupling
notebook. The MoDELib checkout lives inside this repository at `DisloCluster/MoDELib3/`
(a fork of `https://github.com/mlm335/MoDELib-fullCD`, which is the
cluster-dynamics-complete upstream; `mlm335/MoDELib2-NNL` has `iSize=0` and no
immobile machinery at all, and is NOT the right baseline for CD comparisons). Its location is resolved
by `dislocluster_code/paths.py` (`paths.MODELIB_ROOT`); never hard-code it.

---

## 1. Why it does not build on the current Windows box

MoDELib is a **C++20** library compiled with **GCC/Clang** flags
(`-Ofast -march=native -fopenmp`, top-level `CMakeLists.txt:18`). This machine
has only **MSVC** (which built `ZrMicro/solver.exe` via the Visual Studio
generator); MSVC rejects those flags. There is **no GCC/Clang/MinGW and no WSL**
installed, and the repo's `CMakeLists.txt` hardcodes macOS paths
(`/opt/local/include/eigen3`, `/Users/matthewmaron/Qt/...`, lines 10–11). So the
final build must happen on **Linux or macOS**, or after installing a GCC/Clang
toolchain here and adapting CMake. A `c:/vcpkg` exists and can supply the
libraries, but a compatible compiler is still required.

Until then, `dislocluster_code/legacy/modelib_fem.py::MoDELibFEMSolver.available()`
returns `False` and the notebook runs the analytic placeholder.

---

## 2. Dependencies

| Dependency | Required? | Notes |
|---|---|---|
| C++20 compiler (GCC ≥ 11 / Clang ≥ 14) | **yes** | the only hard blocker on Windows |
| OpenMP | **yes** | `-fopenmp` |
| Eigen3 | **yes** | header-only; `CMakeLists.txt:10` `EIGEN3_INCLUDE_DIRS` |
| pybind11 + Python | for `pyMoDELib` | in-process Python driver |
| Boost | optional | found via `find_package(Boost)` |
| SuiteSparse (CHOLMOD/UMFPACK) | optional | faster sparse solves; else Eigen `SimplicialLLT`/`SparseLU` |
| FFTW3 | optional | glide-plane noise only; irrelevant to cluster dynamics |

---

## 3. Build recipe (Linux/macOS)

```bash
# 1. install deps  (Ubuntu example)
sudo apt-get install build-essential cmake libeigen3-dev libomp-dev \
                     python3-dev pybind11-dev libsuitesparse-dev libboost-all-dev

# 2. fix the macOS-hardcoded paths in CMakeLists.txt
#    - line 10:  set(EIGEN3_INCLUDE_DIRS /usr/include/eigen3)   # or your path
#    - line 11:  remove/replace the Qt CMAKE_PREFIX_PATH (only DDqt needs Qt)
#    - lines 12-13: drop the libgcc RPATH if not on MacPorts
#    Build only the headless tools (skip DDqt, which needs Qt):
#      in tools/CMakeLists.txt, comment out add_subdirectory(DDqt)

# 3. configure + build
cd <DisloCluster>/MoDELib3       # paths.MODELIB_ROOT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_PYBIND11=ON -DBUILD_TOOLS=ON
cmake --build build -j

# Targets produced:
#   build/tools/DDomp/DDomp                 standalone runner  (file-based)
#   build/tools/pyMoDELib/pyMoDELib*.so     Python module      (in-process)
#   build/tools/MicrostructureGenerator/... initial-microstructure generator
```

### WSL build (chosen path on this machine)

`wsl --install` needs **Administrator elevation + a reboot + interactive Ubuntu
setup**, which cannot be automated. Do these once, in an **elevated PowerShell**:

```powershell
wsl --install            # enables WSL + installs Ubuntu; then REBOOT
# after reboot, Ubuntu opens and prompts for a UNIX username + password
```

Then build (the whole thing is scripted) from a WSL shell:

```bash
bash <DisloCluster>/Docs/Formulation/build_modelib.sh
# with no argument it builds <DisloCluster>/MoDELib3
# installs deps, then builds build/tools/DDomp/DDomp
# (+ build/tools/pyMoDELib/*.so when pybind11 is installed)
```

The same script builds on Linux and macOS — `bash Docs/Formulation/build_modelib.sh`,
no WSL involved. It was `build_modelib_wsl.sh` and used to `sed` the
macOS-specific entries out of `CMakeLists.txt` and comment out the Qt tool on
every run; the CMake files detect all of that themselves now, so the source tree
no longer carries whichever platform was built last. `SKIP_DEPS=1` skips the
package-manager step when the libraries are already in place.

The script reads the repo from the Windows filesystem via `/mnt/<drive>` so the
binaries land at `<DisloCluster>/MoDELib3/build/...`, visible to Windows Python.

**Cross-boundary execution.** A WSL build produces a Linux ELF `DDomp`; Windows
Python cannot exec it directly, and a Linux `pyMoDELib.so` cannot be imported by
Windows Python at all. So use the **DDomp + `wsl_exec=True`** path:

```python
from dislocluster_code import paths
femsolver = MoDELibFEMSolver(sim_dir=...,
                             modelib_root=str(paths.MODELIB_ROOT),
                             modelib_build_dir=str(paths.MODELIB_BUILD),
                             wsl_exec=paths.use_wsl())  # wsl.exe -e <DDomp> <simDir>
```
`modelib_fem.py` translates Windows paths to `/mnt/<drive>/...` automatically
(`windows_to_wsl_path`). Alternatively, run the entire coupling (including the
`ZrMicro` venv) inside WSL to avoid the boundary.

### Native Windows alternative (no WSL)
- Install **LLVM/Clang** or **MinGW-w64 GCC** + Ninja, and Eigen/pybind11 via
  `vcpkg` (`c:/vcpkg`). Configure with `-DCMAKE_CXX_COMPILER=clang++` (not MSVC);
  patch the same macOS-isms. Then `pyMoDELib` is importable by Windows Python.

---

## 4. The coupling interface (already wired in `modelib_fem.py`)

A spatial cluster-dynamics run is a **simulation directory** with an
`inputFiles/` tree, produced by a tutorial's `generateInputFiles.py`
(`tutorials/irradiation_singlecrystal/`). Key pieces:

| File | Role | Set by the coupling |
|---|---|---|
| `DD.txt` | master traits: `useClusterDynamics=1`, `useFEM=1`, `climbSolverType=Galerkin`, `Nsteps`, `timeSteppingMethod`, `dtMax`, `outputQuadraturePoints=1` | once, at setup |
| `Zr4.txt` | material: `doseRate_dpaPerSec`, anisotropic `mobileSpeciesEnergyMigration_eV`, `mobileSpeciesD0_SI`, `elasticBias`, `otherSinks_SI` | match to `ZrMicro` (Section 6) |
| `polycrystal.txt` | mesh + `absoluteTemperature` + box scaling + grain orientation | once |
| `unitCube_15K.msh` | 3-D mesh (`Library/Meshes/`) | once |
| `aLoopsDensity.txt` | `<a>` prismatic loops: `slipSystemIDs`, `targetDensity`, `loopRadiusMean`, `areVacancyLoops=0` | **every dose step** — the `a1,a2,a3` sink field |
| `frankLoopsDensity.txt` | `<c>` Frank/vacancy loops: `targetDensity`, `radiusDistributionMean`, `areVacancyLoops=1` | **every dose step** — the `c` sink field |
| `initialMicrostructure.txt` | lists the microstructure files | once |

**Drive paths** (both in `modelib_fem.py`):
- **`DDomp` executable** (primary, file-based, like `solver.exe`):
  `DDomp <simDir>` reads `inputFiles/`, advances the FEM solve, writes `evl/` and
  `F/` output. Entry point `tools/DDomp/DDomp.cpp`.
- **`pyMoDELib`** (in-process): `DislocationDynamicsBase(simDir)` →
  `DefectiveCrystal(ddBase)` → `runSteps()`. See `python/modelibPy11.py`.

**Output / readback:** with `outputQuadraturePoints=1`, MoDELib writes
per-quadrature-point fields under `F/`. `ClusterDynamics<3>::mobileConcentration(...)`
is the in-code accessor (`include/ClusterDynamics/ClusterDynamics.h`). The exact
column layout for `[x y z Cv Ci C2i C3i]` is self-describing in the output header
and is fixed in `modelib_fem.py::_read_quadrature_mobile_field` on the first real
run.

---

## 5. Activating the coupling in `coupling_demo.ipynb`

1. Build MoDELib (Section 3); set `MODELIB_BUILD` in the notebook's Section 4 to
   `<repo>/build`.
2. Seed a simulation directory and generate its `inputFiles/` once:
   ```python
   from dislocluster_code.legacy.modelib_fem import MoDELibFEMSolver
   from dislocluster_code.legacy.modelib_fem import seed_sim_dir_from_tutorial
   from dislocluster_code import paths
   seed_sim_dir_from_tutorial(paths.MODELIB_ROOT,
                              paths.OUTPUT_DIR / "modelib_sim")
   # then run output/modelib_sim/generateInputFiles.py once (needs the build)
   ```
3. Make `Zr3d_ghoniem.txt` (the coupled material, `paths.MODELIB_MATERIAL`)
   consistent with `ZrMicro` (dose rate `G`, migration energies,
   `D0`, bias) — cross-check against `output/modelib_closure/modelib_cd_meta.json`
   from Section 6.
4. Set `FAST_SOLVE_BACKEND = "modelib"`. `fast_solve_field` then calls
   `femsolver.solve(Q_per_variant)` each dose step. **Geometry note:** the real
   run marches on MoDELib's 3-D mesh quadrature points — replace the notebook's
   illustrative 1-D line (`x`, `f_depl`) with the mesh coordinates returned by
   the solver (`solve(...)["points"]`).

---

## 6. Verification (matches the formulation's staged plan)

1. **Well-mixed regression:** uniform fields + periodic BCs ⇒ MoDELib interior
   must reproduce the `ZrMicro` 0-D `ε_a, ε_c` (the notebook already checks this
   for the placeholder: interior vs 0-D within ~3%).
2. Then activate, one at a time: finite-rate boundary sinks, spatial gradients in
   `T,G,σ`, resolved-stress variant imbalance, local nucleation/variable `N`,
   the bi-pyramid→loop transition — validating each against the previous limit.

---

## 7. File map (cloned repo)

| Component | Path |
|---|---|
| Top CMake | `MoDELib3/CMakeLists.txt` |
| ClusterDynamics | `MoDELib3/include/ClusterDynamics/{ClusterDynamics,ClusterDynamicsFEM,FixedDirichletSolver,SecondOrderReaction}.h` |
| FEM | `MoDELib3/include/FEM/` |
| Runner | `MoDELib3/tools/DDomp/DDomp.cpp` |
| Python module | `MoDELib3/tools/pyMoDELib/pyMoDELib.cxx` |
| Zr materials | `MoDELib3/Library/Materials/Zr{1..4}.txt` |
| Meshes | `MoDELib3/Library/Meshes/unitCube_15K.msh` |
| CD tutorial | `MoDELib3/tutorials/irradiation_singlecrystal/generateInputFiles.py` |
| Python utils | `MoDELib3/python/modlibUtils.py` (`setInputVariable`, `setInputVector`, `PolyCrystalFile`) |
| **Coupling bridge** | `dislocluster_code/legacy/modelib_fem.py` |
