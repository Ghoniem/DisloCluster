# DisloCluster — Project-Level Claude Code Instructions

## What this repository is

**DisloCluster** couples cluster dynamics to dislocation dynamics for irradiated
zirconium. It was assembled from two previously separate repositories and is now
self-contained: nothing outside the repository root is required except the
Python interpreter, SUNDIALS, and (on Windows) WSL.

| Directory | Role | Language |
|---|---|---|
| `dislocluster_code/` | **the Python package** — everything importable | Python |
| `Simulations/` | **where simulations are run** — notebooks, `input/`, `output/` | Jupyter |
| `ZrMicro/` | 0-D reduced cluster dynamics — 19 ODEs (12 physical species, 6 conservation accumulators, ρ_N); C++ solver, workbook, run output | C++/SUNDIALS |
| `Gmsh/` | simulation-domain meshing (cubic / hexagonal) | Python + Gmsh SDK |
| `MoDELib3/` | 3-D spatially-resolved cluster dynamics / dislocation dynamics (fork of MoDELib-fullCD) | C++20 |
| `Docs/` | **All documents** — see the table below | — |

**To run a simulation, open [`Simulations/run_simulation.ipynb`](Simulations/run_simulation.ipynb).**
Six dicts at the top set the material, geometry, mesh, boundary conditions,
coupling and output; the rest of the notebook stages the case and runs the
march. See [`Simulations/README.md`](Simulations/README.md).

| Driver | What it runs |
|---|---|
| [`Simulations/run_simulation.ipynb`](Simulations/run_simulation.ipynb) | **the normal entry point** — configure, stage, march, report |
| [`Simulations/postprocess.ipynb`](Simulations/postprocess.ipynb) | re-render an existing run; never re-solves |
| `python -m dislocluster_code.coupling.seeded_march` | the march as a CLI, on an explicit dose list |
| `python -m dislocluster_code.studies.run_zr3d_singlecrystal` | post-processes the standalone 3-D **single cubic crystal** run |
| [`ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb`](ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb) | the original coupling notebook; superseded by `Simulations/`, kept for reference |

---

## Documents

**Every document lives under the repository-root [`Docs/`](Docs/).** There is no
`Docs/` under any of the sub-codes — earlier revisions of this file claimed one,
which led to documents being written into a stray tree.

| Directory | Contents |
|---|---|
| [`Docs/DisloCluster Manual/`](Docs/DisloCluster%20Manual/) | `DisloCluster_manual`, `MoDELib_manual`, development notes such as `solver_and_coupling_rearchitecture` |
| [`Docs/Formulation/`](Docs/Formulation/) | formulation LaTeX/PDF, `build_modelib.sh`, `MoDELib_build_and_coupling_notes.md` |
| [`Docs/Reports/`](Docs/Reports/) | deliverables |
| [`Docs/Presentations/`](Docs/Presentations/) | slides |

New documents go in the fitting subdirectory above. Write prose and code
comments in **American English** (`-ize`, `-or`, no hyphen after an `-ly`
adverb).

---

## Path resolution — read this before touching any path

**Never hard-code an absolute path.** All locations come from
[`dislocluster_code/paths.py`](dislocluster_code/paths.py), which walks up to the
`.dislocluster_root` marker at the repository root.

```python
from dislocluster_code import paths
paths.REPO_ROOT      paths.ZR_ROOT        paths.ZRMICRO_DIR
paths.INPUT_DIR      paths.OUTPUT_DIR     paths.CPP_UTILS / BUILD_DIR
paths.OUTPUT_DIRS    # every root a run may be FOUND in, newest home first
paths.MODELIB_ROOT   paths.MODELIB_BUILD  paths.MODELIB_MATERIAL
paths.SIM_ROOT       paths.SIMULATIONS_DIR
paths.zrmicro_solver_exe()   paths.modelib_ddomp()   paths.venv_python()
paths.git_hash()             paths.windows_to_wsl()  paths.use_wsl()
paths.run_dir(tag)           # <output>/<stamp>_<hash>_<tag>, created
paths.find_runs()            # every run, across all output roots, oldest first
paths.latest_run()           # the newest one
paths.workbook_drift()       # workbooks that differ between the two input dirs
print(paths.describe())      # resolution report, OK/MISS per location
```

Overrides: `DISLOCLUSTER_ROOT`, `DISLOCLUSTER_SIM_ROOT`, `MODELIB_ROOT`,
`MODELIB_BUILD`.

If a new module needs a repository location, add it to `paths.py` — do not
re-derive it with `Path(__file__).parent.parent`. That idiom broke silently in
four modules when the package moved: `cpp_bridge` resolved its default
`base_dir` that way and stopped finding `solver.exe` at all.

---

## The `dislocluster_code` package

Everything importable lives in `dislocluster_code/` at the repository root, installed
editable into the venv (`pip install -e .`).

| Subpackage | Holds |
|---|---|
| `dislocluster_code.paths` | every repository location (above) |
| `dislocluster_code.config` | `SimulationConfig` — the notebook's six dicts, validated |
| `dislocluster_code.driver` | `prepare` / `march` / `report`, one call per notebook stage |
| `dislocluster_code.build` | `ensure_zrmicro_solver`, `ensure_modelib` |
| `dislocluster_code.zerod` | the 0-D chain: `input_data`, `reaction_rates`, `rate_equations`, `calibration`, `cpp_bridge`, `post_process` |
| `dislocluster_code.staging` | `case` (mesh, stage, bootstrap), `inputs` (DD/polycrystal/ElasticDeformation), `seed` |
| `dislocluster_code.coupling` | `march` (the operator split), `config` (`MarchConfig`), `qssa`, `field`, `immobile`, `checkpoint`, `progress` |
| `dislocluster_code.post` | figures, movies, discrete loops, TEM slices, reports |
| `dislocluster_code.studies`, `.fitting`, `.legacy` | comparison drivers, parameter fits, superseded modules |

**`ZrMicro/py_utils/` still exists as a compatibility shim.**
Each old module aliases `sys.modules[__name__]` to its new home, so
`from py_utils import paths`, `python -m py_utils.X` and the old notebooks all
keep working, and `py_utils.X is dislocluster_code....` is true. New code should
import from `dislocluster_code` directly.

Configuration is a value, not module state. `coupling.march` still carries
`DOSE_SEED`, `SUBSTEPS_PER_INTERVAL` and friends as globals, but only so that
`default_config()` can read them **at call time** for callers that still
monkey-patch. Do not turn those into dataclass field defaults: they would bind
at import and a later `rcs.SUBSTEPS_PER_INTERVAL = 5` would become a silent
no-op.

---

## Python environment

- **Venv:** `.DisloClusterVenv/` at the repository root (Python 3.14)
- **Package:** `pip install -e .` from the repository root (`pyproject.toml`)
- **Jupyter kernel:** `dislocluster` — "Python 3.14 (DisloCluster)"
  (the kernel keeps its original name; only the Python package was renamed)
- **Dependencies:** `requirements.txt` at the repository root
- **Do not use Anaconda Python** — NumPy 1.x/2.x conflict with SciPy

```powershell
.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute `
    --ExecutePreprocessor.kernel_name=dislocluster <notebook.ipynb>
```

---

## Builds

### 0-D solver (CMake + SUNDIALS 7.1.1)

```bash
cmake -S ZrMicro/cpp_utils -B ZrMicro/build -DCMAKE_BUILD_TYPE=Release
cmake --build ZrMicro/build --config Release
```

SUNDIALS is found at `<repo>/Libraries/sundials-7.1.1/`, from the system install
(`C:/Program Files (x86)/SUNDIALS`, `/usr/local`), or — on macOS — under the
Homebrew/MacPorts prefix, which CMake does not search by default and which
differs by architecture (`/opt/homebrew` on Apple silicon, `/usr/local` on
Intel). `brew --prefix` is asked rather than one of them assumed. Modules used:
`cvodes`, `arkode`, `nvecserial`, `sunlinsolband`, `sunlinsolspgmr`. OpenMP is
optional but enables the parallel `--batch_file` mode the coupling march
depends on.

### 3-D code (Linux, macOS, or Windows through WSL)

```bash
bash Docs/Formulation/build_modelib.sh                   # Linux, macOS
wsl -u root -e bash Docs/Formulation/build_modelib.sh    # Windows
```

Defaults to `<repo>/MoDELib3`. The script detects and discards a CMake cache
configured under a different absolute path, so a relocated checkout rebuilds
cleanly. `SKIP_DEPS=1` skips the package-manager step.

**One script, one CMakeLists, every platform.** The script used to be
`build_modelib_wsl.sh` and `sed`-patched `MoDELib3/CMakeLists.txt` on each run —
rewriting the Eigen path, deleting the Qt lines, commenting out `add_subdirectory
(DDqt)`. That made the source tree carry whichever platform's edits had been
applied last, and it was the reason the checked-in CMakeLists hard-coded
Debian's `/usr/include/eigen3`. The CMake files now do the deciding themselves:
every dependency is searched for (`Eigen3Config` then a directory search, with
the Homebrew/MacPorts prefix on `CMAKE_PREFIX_PATH` on macOS), every
optimization flag is probed with `check_cxx_compiler_flag` before use, and
everything optional degrades to a status line. `-DUSE_SUITESPARSE=OFF`,
`-DUSE_FAST_MATH=OFF`, `-DUSE_NATIVE_ARCH=OFF` and `-DUSE_PYBIND11=OFF` turn off
what a comparison across machines might not want.

**MoDELib requires Eigen 3.4.x, and the build now refuses anything newer.**
Homebrew's `eigen` formula is 5.x, and MoDELib built against it *compiles* —
then the BiCGSTAB solve inside the mobile Newton iteration breaks down on the
first step (`Iterative FixedDirichletSolver failed`) on a case that converges
with 3.4.0. That is the worst kind of portability failure: a clean build and a
wrong answer, reported as a solver problem. `build_modelib.sh` fetches 3.4.0
into `Libraries/eigen-3.4.0` (git-ignored, beside the private SUNDIALS) when
the system has only Eigen 5, `CMakeLists.txt` prefers that copy, and the
version check is fatal — `-DEIGEN3_ALLOW_UNTESTED=ON` overrides it for whoever
wants to re-test a newer Eigen. Debian/Fedora/Arch still ship 3.4.x, so Linux
never had to notice. Note the version test reads *both* places Eigen states
its version: 3.4.x puts the macros in `Eigen/src/Core/util/Macros.h`, 5.x in
`Eigen/Version`, and 5.0.1 spells itself `3.5.0` there (`EIGEN_WORLD_VERSION`
stays 3 forever), so a naive `^3\.` test passes it.

The macOS build needed four more things Linux never exposed:

| Symptom | Cause |
|---|---|
| `clang: error: unsupported option '-fopenmp'` | the flag was hard-coded; Apple clang needs `-Xpreprocessor -fopenmp` and Homebrew's libomp, which `find_package(OpenMP)` now supplies |
| `use of undeclared identifier 'assert'`, 103 files | those files never included `<cassert>`; libstdc++ leaks it in, libc++ does not. A `-include cassert` on the whole tree substitutes for editing all 103 |
| `must explicitly initialize the const member 'invTrD'` | Eigen ≥ 3.4.90 declares `Matrix() = default`, so a const member with no initializer is ill-formed. `invTrD` is unused; it is now zero-initialized |
| `no viable conversion from 'RationalLatticeDirection<3>'` | clang resolves the dependent `this->operator+(RationalLatticeDirection(...))` call against the `LatticeVector` overload alone; naming the temporary fixes it |

`DDqt` is built when Qt6 and VTK ≥ 9.4 are both found and skipped otherwise —
it is the only target that needs them, and the coupling never uses it.

**No build tree is in git, and none should be.** `MoDELib3/build_dc/`
(`libMoDELib.so`, `DDomp`, and a `build.ninja` carrying 374 absolute
`/mnt/d/...` paths) used to be committed. A clone on another machine got that
Linux ELF, `paths.modelib_ddomp()` reported MoDELib as built, `ensure_modelib`
skipped the build, and the first DDomp call failed. Both build trees are now
ignored, and `build.preflight()` *runs* each binary rather than trusting its
presence.

**Moving or renaming the repository invalidates both build trees** — a CMake
cache stores absolute paths, so a reconfigure hard-errors with "the current
CMakeCache.txt directory ... is different". Both sides detect this and discard
the stale cache themselves: `build_modelib.sh` for the 3-D tree,
`dislocluster_code.build.ensure_zrmicro_solver` for the 0-D tree.

The failure is quiet, which is what makes it easy to miss: **the already-built
binary keeps working**, so nothing breaks until the next reconfigure, possibly
months later. `build.stale_cache_home()` reports it and `build.describe()`
flags it; `ensure_zrmicro_solver()` discards and rebuilds. That check runs
*before* the "already built" shortcut — putting it after, as an earlier
revision did, made it unreachable in exactly the situation it exists for.

By hand: delete `ZrMicro/build/CMakeCache.txt` and reconfigure.

---

## The control blocks

`Simulations/run_simulation.ipynb` is driven entirely from six dicts in §2, all
defaulted and validated in [`dislocluster_code/config.py`](dislocluster_code/config.py).
Nothing else in the notebook should need editing for a normal run.

| Dict | Controls |
|---|---|
| `MATERIAL` | material file, temperature, dose rate, 0-D parameter overrides |
| `GEOMETRY` | domain type (`cubic` / `hexagonal`), dimensions |
| `MESH` | element size and order, boundary-layer refinement |
| `BOUNDARY` | periodic faces (empty ⇒ Dirichlet everywhere), applied stress and strain |
| `COUPLING` | route, seed dose, snapshot doses, substeps, fast-solve cadence, failure tolerances |
| `SOLVER`, `OUTPUT` | tolerances and backend; tag, figures, movies, `movie_interp`, checkpoint, resume |

`SimulationConfig.from_dicts` rejects an unknown key rather than ignoring it,
and validates before anything expensive runs: a dose grid that does not
increase strictly, a seed past the first snapshot, a boundary layer thicker
than half the domain, an `element_order` outside {1, 2} (MoDELib's
`SimplexReader` consumes only msh element types 4 and 11).

**`BOUNDARY` is a real boundary-condition channel.** `periodic_face_ids` goes
into `polycrystal.txt`; the load goes into `ElasticDeformation.txt`. Stress is
given in **MPa** and converted on staging: MoDELib normalizes stress by
`mu_SI`, so the numbers in that file are multiples of the shear modulus, and
writing MPa straight in would overstate the load by ~33 000× for Zr.

**Staging is keyed on the domain, not the run.** `cfg.domain_key` hashes the
geometry, mesh, boundary conditions, material and temperature — but not the
dose grid — so changing only the doses reuses the staged case, its CD node set
and its bootstrap. `domain.json` in the case directory records what is there.

The older `coupled_0d_3d_ZrMicro.ipynb` is driven from three dicts
(`GEOMETRY`/`MESH`, `RUN`, `PARAMETER_OVERRIDES`) and carries its own inline
copy of the march. It is superseded and kept only for reference.

The dose schedule is `n_intervals` equal steps from `dose_seed` to `dose_max`;
the defaults (1 → 26 dpa, 5 intervals) give snapshots at 1, 6, 11, 16, 21, 26 dpa.
`substeps_per_interval` fixes the immobile substep count **per dose interval**
rather than by a wall-clock cap — a 5 dpa step at 1e-7 dpa/s is 5×10⁷ s, which a
fixed 5×10⁴ s cap would have split into a thousand batch subprocesses.

Section 6a writes the mesh through `Gmsh/generate_mesh.py` (repository root);
the cell after the
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

**Both cut planes go into ONE `Poly3DCollection`.** Every panel draws two
orthogonal mid-cuts, one vertical (normal to y) and one horizontal (normal to
z). Matplotlib depth-sorts 3-D *artists* by a single scalar each, so two
intersecting surfaces drawn as separate `plot_surface` calls cannot interleave
and whichever sorts in front hides the other completely — which is what used to
happen, the vertical cut covering the horizontal one entirely. Sampling both
cuts (and the loop platelets) into one collection sorts them polygon by polygon,
so both stay visible and the platelets read as a cutaway. Do not split them back
into separate artists.

**The panel zoom is fitted, not fixed.** A 3-D axes clips its artists at the
axes rectangle, not at the data cube, so a zoom that projects the domain taller
than the panel silently cuts the ends off the wireframe. `plot_field_panels`
still takes `zoom=1.22`, but `fields._fit_zoom` now reduces it until the
convex hull of the CD nodes fits, measuring through matplotlib's own
`get_proj`. It is the **aspect ratio** that decides this, not the size: a cube
fits 1.22 (0.007..0.960 of the panel) and is left exactly as it was, while a
1 : 0.87 : 1.6 hexagonal prism projects to −0.068..1.054 and loses both end
faces — identically at 200 nm and at 500 nm. Fit to the hull, not to the
bounding box, whose corners for a prism sit out in the six empty wedges. This
applies to `3d/` and to the movie frames, which are the same code;
`discrete_loops.render` uses the default `zoom=1` and was never affected.

**Every panel carries a crystal-orientation triad** — `[10̄10]` (x), `[2̄1̄10]`
(y), `[0001]` (z), drawn by `fields._draw_orientation_triad` from the analytic
projection of matplotlib's own camera, so the arrows always agree with the
viewpoint. On a multi-panel grid only the bottom-left panel gets one. The
discrete-loop renders carry the same triad.

**Nothing in the figure code is specific to one geometry.** Every shape
decision — the outline, the hidden-line test, the volume that turns a density
into a count, the region loops are placed in, the clip — comes from
`fields.domain_faces` / `domain_edges` / `domain_volume`, which are the convex
hull of the CD node cloud. `GEOMETRY['type'] = 'cubic'` gives 6 faces, 12 edges,
3 hidden and `domain_volume == prod(hi - lo)` to machine precision, so the box
case is untouched by the prism corrections; `'hexagonal'` gives 8 faces, 18
edges; a `cubic` case with a non-unit `aspect` follows automatically. Do not
reintroduce a bounding-box assumption or branch on the geometry name. The one
thing that is *not* geometry-derived is the orientation triad, which is a
material property (hcp Zr) and so is the same for every domain shape;
`plot_field_panels(orientation=...)` takes a different crystal frame if one is
ever needed.

**The crystal is drawn as a complete wireframe with hidden lines dotted.**
`fields.draw_domain_wireframe` marks an edge hidden when BOTH faces meeting
there point away from the camera — `domain_edges(return_faces=True)` supplies
the face pair, `camera_direction(view)` the camera. Two things make this
necessary rather than decorative: the axes are built with
`computed_zorder=False` and the wireframe sits above the cut planes, so nothing
in the drawing occludes a back edge any more; and the cuts only read as being
*inside* a body if the body is drawn closed around them. The line where the two
cuts meet is drawn dotted in black over the planes
(`fields._cut_intersections`, clipped to the domain).

**Loop overlays** (`loops_*`) fill the domain with as many **non-overlapping**
platelets as fit: candidates are visited in random order and accepted only if the
drawn radius clears every platelet already placed. The clearance test uses
bounding spheres times a `gap` of 1.4, so neighbours keep visible space between
them; at bare tangency the fill is dense enough to hide the field underneath.
Platelets are **flat discs** with a darker rim, not thick volumes — the rim is
what keeps two overlapping discs from merging into one blob in projection.
Radii are exaggerated ×0.55 (⟨c⟩) and ×2.2 (⟨a⟩), about a third of the
deliverable's factors, which is what a flat disc needs to stay smaller than the
cut plane it sits on. The two factors differ because ⟨c⟩ loops are ~6× larger,
so sizes are faithful *within* a figure but not *between* the ⟨c⟩ and ⟨a⟩
figures. These are the slow figures: ~16 s each.

**`size_dist_*`** is the distribution of the **local mean** loop diameter across
the domain, weighted by local loop density — *not* a per-loop size spectrum. The
model carries one mean size per family per node, so spatial variation is the only
source of spread; the result is strongly bimodal (small numerous loops in the
boundary shell, large sparse ones in the interior). Nodes are weighted as
equal-volume, which over-weights a refined region: `evl/cdNodes.txt` carries
positions only, with no connectivity from which nodal volumes could be formed.

Rendering lives in `dislocluster_code/post/report.py`; it reads `evl/cdNodes.txt` +
`evl/evl_<N>.txt` and never runs a solve.

---

## The coupling contract

Operator-split QSSA, alternating over each substep:

1. **Fast solve** — steady mobile field `C_M*(x)` for the immobile state
   currently held. This is MoDELib3's own `ClusterDynamicsFEM::
   solveMobileClusters`, which has no time derivative at all; the immobile
   population enters only through `ImmobileSinks` evaluated at every quadrature
   point. Driven from Python by `modelib_qssa.MobileQSSASolver.solve`, which
   calls DDomp with `useImmobileSolver=0`.
2. **Slow march** — the immobile ODEs integrated independently at every
   quadrature point with the mobile species frozen
   (`modelib_coupling.run_immobile_step`, one OpenMP batch subprocess).

**The fast solve belongs inside the loop.** If the mobile field is instead
replayed from a previously recorded run, the loop is not an operator split:
`C_M` never responds to the immobile state the march is building, so a seed
away from quasi-steady state can never relax. A 0-D seed is always such a seed —
it is spatially uniform and carries no boundary layer, while the true `C_M*(x)`
is pinned to thermal equilibrium on every Dirichlet face.

Splitting error is first order in the interval between fast solves
(`FEM_EVERY` substeps), not in the dose-snapshot spacing.

| Direction | Carrier | Module |
|---|---|---|
| 0-D → 3-D | per-node immobile field → `evl` CD block | `modelib_field.FieldBridge.write_immobile_field` |
| 3-D → 0-D | `C_M*(x)` at the CD nodes | `modelib_qssa.MobileQSSASolver.solve` |
| 0-D → 3-D | dose-indexed closure table | `modelib_export` |
| both | material constants | `MoDELib3/Library/Materials/Zr3d_ghoniem.txt` (`paths.MODELIB_MATERIAL`) |

`legacy/modelib_fem.py` predates this: it wrote four scalars per dose step
into the microstructure-generator inputs, discarding all spatial structure. It
is superseded by the field channel above and is kept only so older results stay
reproducible.

State vector (19): `[Cv, Ci, C2i, C3i, CiL, CaiL, CvL, CavL, CiL_i, CaiL_i,
CvL_v, CavL_v, 6 accumulators, rho_N]`.

### Why ~1.1 M ODEs per substep are tractable at all

This is the load-bearing design point of the whole scheme, and it is easy to
misread the numbers as claiming something impossible. **A tightly coupled
million-equation stiff system would be hopeless.** This is not one.

**It is 72 494 independent 15-dimensional systems**, not one 1 087 404-dimensional
one (500 nm case: 90 617 nodes, dedup ×1.25). `rate_equations.ode_system(t, y)`
takes a single point's state vector; no neighbor state enters it. Each point is
one case of the OpenMP `--batch_file` mode, dynamically scheduled over 24
threads with a private CVODE workspace each.

That decoupling is legitimate rather than a cheat because **the only thing
coupling neighboring points is diffusion of the mobile species**, and the slow
step freezes those (`freeze_mobile=1`). Freeze the mobile field and the spatial
coupling is gone by construction: a point's immobile ODEs depend only on its
own loop populations and the mobile values pinned into `y[0:4]`.

The cost consequence, for one implicit factorization:

| | flops |
|---|---:|
| split: 72 494 × 15³ | 2.4×10⁸ |
| coupled: (1 359 255)³ | 2.5×10¹⁸ |
| **ratio** | **1.0×10¹⁰** |

Sparsity softens the coupled figure but does not rescue it — a 3-D FE Jacobian
still factorizes at ~O(N²) with fill growing as O(N^4/3).

Stiffness localizes the same way. Each point's system genuinely is stiff (~100
CVODE internal steps per point per substep, which is why BDF), but stiffness
cost scales with the **dimension of the coupled block**, not with how many
independent blocks there are. A 15×15 dense LU is ~1100 flops.

**The coupling did not vanish — it moved to where it is affordable.** The
tightly coupled problem still exists: it is the fast solve, 362 468 unknowns,
**925 s** measured, against 28.3 s for an entire slow sweep over 90 617 points.
What the split buys is *frequency*: `solveMobileClusters` has no time
derivative, so the coupled problem is solved **8 times over the march** instead
of at every timestep. The counterfactual, from the same measurements —
925 s / 5 Newton iterations = 185 s per linear solve:

```
~100 internal steps × 24 substeps ÷ 5 steps per refactorization ≈ 480 factorizations
480 × 185 s ≈ 25 h        ← the 4-species mobile block ALONE, not all 19
measured slow side, all 24 substeps:  11.3 min
```

Three things stacked, then: freezing the mobile field removes the spatial
coupling; the coupled part has no time derivative so it is solved 8 times, not
thousands; and batch OpenMP plus dedup (identical states solved once — ×1.25
here, higher early on when much of the domain shares a state) spreads what is
left over 24 cores.

The bill arrives as **splitting error, first order in the fast-solve spacing**,
which is why `fem_every` is the accuracy dial and why it mattered that a value
not dividing `substeps_per_interval` was skipping whole intervals.

The slow side is linear in node count, as this structure predicts: 9.3 s per
substep at 27 720 nodes and 28.3 s at 90 617 — 3.35×10⁻⁴ vs 3.12×10⁻⁴ s/node.

### The slow step is NOT deterministic

**The same batch of states can fail on one invocation and succeed on the next.**
Measured on the 500 nm anisotropic march: one point of 90 617 failed at the
0.01 → 0.1 dpa substep, and the identical step — same checkpoint, and the
preceding fast solve reproduced to every printed digit
(`Cv 8.46e-15..6.69e-06 Ci 4.11e-27..8.43e-10 …`) — then completed with **zero**
failures when re-run twice.

The mechanism is the per-thread CVODE workspace the batch mode reuses across
cases, which is the optimization that bought 1.67× end to end. Under dynamic
scheduling, which case follows which on a given thread varies between
invocations, so a point sitting near the solver's failure boundary is decided by
scheduling rather than by its own state.

Two consequences:

- `run_immobile_step` takes `retries=2` and re-runs failed points **on their
  own**, which repacks them into a tiny batch and usually clears them. Prefer
  that to raising `max_failed_nodes`, which substitutes an **identity step** —
  it does not recover the point, it freezes it for the substep.
- **Do not assume two runs of the same case agree point for point.** The
  fast-solve optimizations were verified bit-identical and that verification
  stands, because it was the *fast* side; the slow side has never been shown
  reproducible and this measurement says it is not.

### Study drivers

| Module | Role |
|---|---|
| `calibration.py` | **the fitted 0-D parameter set.** `build_sim()` is the only correct way to build the model chain — see below |
| `setup_standalone.py` | stages a calibrated standalone MoDELib3 case at any seed dose, and pins the fast step to the coupled route's solver settings |
| `run_seeded_march.py` | the coupled march from any seed dose on an explicit snapshot-dose list |
| `compare_seeding.py` | coupled vs standalone vs 0-D, interior means, cross-seed ratios |

**The workbook is not the calibrated model** — 28 parameters have drifted and 11
are absent from it entirely. Any driver that builds `InputData` straight from
the workbook runs a different model: `N_a = 2.32e-2` and `c_a = 8.77` (an atom
fraction, so impossible) against the calibrated `8.05e-8` and `1.34e-4`. Always
go through `calibration.build_sim()`.

**Both routes must share the fast step's solver settings.** MoDELib3's default
`mobileSolverClampInLoop=1` makes the mobile solve a projected Newton iteration
that does not converge — it caps out with 53.6% error in `Ci`. `modelib_qssa`
sets `=0` for the coupled route, and `setup_standalone.FAST_STEP_SETTINGS` does
the same for the standalone. Left at the default on one side only, a
route-to-route comparison measures that solver defect rather than the physics.

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
    -m dislocluster_code.studies.run_zr3d_singlecrystal    # the figures
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

**Never report a whole-domain mean of a loop quantity on this geometry.** Of the
24115 CD nodes, **6522 (27%) lie on the six Dirichlet faces** — second-order
elements put that many on the surface — and at 21 dpa they carry **99.8%** of
the total ⟨a⟩ density. Their `c_a/N_a` is 300.8 against `n_iL_nuc = 300`: every
loop there is a fresh nucleus that never absorbed an interstitial, because `C_i`
at those nodes is 4e-27. A domain mean therefore measures the boundary shell,
not the material. Quote the **interior mean** — the innermost quartile by
distance to the nearest face. The difference is not cosmetic: at 21 dpa the
coupled march's ⟨a⟩ density is 474× the 0-D as a domain mean and 0.75× it in the
interior.

The coupled march reproduces this artifact, which is a point *in favor of* the
coupling — the same boundary behavior emerges whether MoDELib3's nodal scheme or
ZrMicro's CVODE march advances the immobile population.

---

## Output reproducibility

Every run creates
`Simulations/output/<YYYYMMDD_HHMMSS>_<git-hash>[_<tag>]/` with figures and
`provenance.md`. The coupled driver adds `0d/`, `3d/`, `gb/` subdirectories and
`march_state.npz`.

`OUTPUT['movies']` adds `movies/` (continuum-field GIFs) and
`OUTPUT['discrete_loops']` adds `discrete_loops/` (per-dose PNG, `loops_*.csv`,
the MoDELib `aLoops_*.txt`, `manifest.json`), `discrete_loops/movies/` and
**`tem_slices/` at the top level of the run**. The TEM micrographs used to be
written to `discrete_loops/tem_slices/`, where — under several hundred loop
PNGs — they were unfindable; `driver.report` now passes `--out
<run>/tem_slices`. Older runs still have them nested, e.g.
`ZrMicro/output/20260809_100409_0dea883_Adaptive_500nm_pristine/discrete_loops/tem_slices/`.
**The march writes ONE snapshot per dose interval**, so `movies/` has as many
frames as `COUPLING['doses']` has entries plus one — an eight-dose run animates
as an eight-frame flipbook however high the fps. `OUTPUT['movie_interp']`
(default 5, `1` disables) subdivides each interval into that many frames,
turning eight doses into 41. **The extra frames are interpolated, not solved**:
`movies.subdivide_doses` + `_blend` blend the two snapshots that bracket each
one, and every synthetic frame is labelled `(interp)` in its title. Two things
about that blend are load-bearing:

- it is **geometric** wherever both endpoints are positive (linear otherwise —
  the pristine seed sits at dose 0). The fields grow by decades through the
  nucleation transient, where a linear interpolant would sit at the upper
  endpoint for the whole interval, and geometric blending commutes with the
  power-law reductions the figures draw, so a size `d ~ (c/N)^(1/3)` formed
  from blended `c` and `N` is the blended `d` to 2e-15;
- it is taken on the **CD block**, not on the 19-column state it is built from.
  `immobile_0d_to_modelib` sums species, and a sum of geometric blends is not
  the geometric blend of the sums, so blending the state first yields frames
  that are *not* bracketed by their neighbours — visible as a population
  overshooting and falling back. Blending what the figures draw keeps every
  interpolant inside its bracket.

`cd_blocks` deliberately keeps `interp=1` as its default: `discrete_loops` and
the `3d/`/`gb/` panels must see solved states only. Only `movies.render` opts
in, through `cd_blocks_interpolated`.

All four steps can be re-run on any finished run without re-solving:

```powershell
.DisloClusterVenv\Scripts\python.exe -m dislocluster_code.post.discrete_loops <run> --doses all
.DisloClusterVenv\Scripts\python.exe -m dislocluster_code.post.loop_movie     <run> --fps 5
.DisloClusterVenv\Scripts\python.exe -m dislocluster_code.post.tem_slices     <run> --out <run>\tem_slices
.DisloClusterVenv\Scripts\python.exe -m dislocluster_code.post.movies         <run> --fps 5 --interp 5
```

`discrete_loops --doses all` takes every snapshot in `march_state.npz`; the
module's own default is four fixed doses, which is not enough frames for
`loop_movie` to produce anything but a flipbook.

**The discrete population lives in the CRYSTAL, not in its bounding box.**
Three things in `discrete_loops` are keyed on the convex body the CD nodes fill
(`fields.domain_faces` / `domain_volume` / `domain_edges`), which for a
`GEOMETRY['type'] = 'hexagonal'` run is the hexagonal prism:

- the volume that turns a density into a count (`domain_volume`, not
  `prod(bounding box)` — for a prism the box is **4/3** the crystal,
  so the loop count used to come out a third too high);
- the nodal volumes, via `voronoi_weights(..., faces=...)`, which rejects the
  Monte-Carlo samples that land in the six empty wedges — every one of those
  used to be charged to whichever boundary node was nearest;
- the region uniform placement draws from, by rejection, so no loop is centred
  outside the crystal.

`render` then outlines that same body and **clips the drawn line at the crystal
surface** (`--no-clip` turns this off). The clip is drawing only —
`loops_*.csv` and `aLoops_*.txt` carry whole loops. It matters at 1–10 dpa on
the 200 nm case, where the ⟨c⟩ radius reaches 47 nm against a 100 nm prism
half-width and a loop centred anywhere but the middle overhangs a wall.

`volume_average.averaged_trajectory` still calls `voronoi_weights` **without**
`faces`, so on a non-box domain its volume weights carry the same bounding-box
error. That is deliberate — fixing it changes already-published figures — but it
is wrong for a hexagonal run and should be revisited.

**Simulations read and write under `Simulations/`.** `input/` holds the Excel
workbooks and `output/` the run directories, the way `ZrMicro/` was used before.
Two consequences worth knowing:

- The 1.4 GB already under `ZrClusterDynamics/ZrMicro/output/` was **not**
  moved, so `OUTPUT_DIR` (writes) and `OUTPUT_DIRS` (reads) differ. Anything
  looking for a run must go through `paths.find_runs()` / `paths.latest_run()`,
  never `OUTPUT_DIR.iterdir()`.
- The workbooks were **copied**, so `ZrMicro/input/` still has its own set and
  only `Simulations/input/` is read. `paths.describe()` hashes both and warns
  when they diverge; `paths.workbook_drift()` names them. Reconcile rather than
  ignore — this is the same failure mode that put the 0-D calibration five
  orders of magnitude off.

The repository is a single git checkout: the 0-D tree and `MoDELib3/` were
submodules once but became plain directories in `859af47`, and neither carries
its own `.git` any more. `paths.git_hash()` still falls back root → `ZR_ROOT` →
`MODELIB_ROOT` so a split checkout would keep working.

A run driven from `Simulations/` also writes `config.json` (the resolved
control dicts) and `checkpoint/`, and reuses the staged case under
`paths.SIM_ROOT` described by that case's `domain.json`.

---

## Which MoDELib upstream, and which is the baseline

**The baseline is `mlm335/MoDELib2-NNL`**, at the point the immobile population
equations were written into it to make the 3-D code consistent with the ZrMicro
0-D model. That is where this work starts, and it started from nothing:
`iSize` was **0**, `solveImmobileClusters()` was an empty function body, there
was no immobile integrator of any kind, and the source carried a literal
`// Missing immobile sinks` placeholder at `ClusterDynamicsFEM.cpp:110`. The
loop populations were **absent, not simplified**.

`mlm335/MoDELib-fullCD` is a **separate cluster-dynamics branch** — `iSize` 8,
with `ImmobileSinkRate.h`, `SpatialODESolver.h`, `FirstOrderReaction.h` and a
continuum→discrete loop conversion. It is a sibling, not the ancestor.

**An earlier revision of this file said the opposite** — that fullCD is the
correct baseline and MoDELib3's immobile solver a *re-discretization* of its
scheme. That was inferred from 31 shared CD parameter keys plus a set of
systematic renames (`loopNucDefects`→`nNuc`, `loopCoalLL/LN`→`cLL/cLN`,
`loopCoalKappa*`→`kappa*`, `loopAnnealTau0_SI`→`tau0_vLoop_SI`,
`loopCoalNetwork_SI`→`rhoNetwork_SI`, `minimumLoopSize`→`r_min`). **The
inference does not follow.** Shared naming across two branches of one code,
developed in one group from one formulation (Po & Ghoniem D1/M1 §2.2), is
evidence of *convergence*, and says nothing about which came first. The
immobile equations here were written to reproduce ZrMicro, not to re-express
fullCD.

fullCD is still worth comparing against, as a sibling:

| | `MoDELib-fullCD` | this branch |
|---|---|---|
| Rate | Galerkin at quadrature points, L2-projected through the consistent mass matrix | nodal collocation, no mass matrix |
| Time update | explicit Euler, then **sequential** implicit loss factors per channel | one **fused** semi-implicit update |
| Sub-cycling | none — one step of `dtMax` | `nSub = 20` per dose step |

The one capability fullCD has and this branch does not is
`clusterDiscretizationTime` / `initializeDiscreteClimbLoops()`, its
continuum→discrete loop transition. That is a **gap, not a regression** —
nothing of the kind existed in MoDELib2-NNL to lose — and it is the piece worth
taking from fullCD. `post/discrete_loops.py` performs the same conversion
offline for export and visualization, which is not the same thing: it does not
feed back into a solve.

Do not cite `Zr4.txt`'s missing immobile keys as evidence about any other
branch: that is a property of one file in this checkout, and fullCD's
`Zr4_Fitted.txt` carries a full set.

Full numbered history: [`Docs/DisloCluster Manual/DisloCluster Code History.tex`](Docs/DisloCluster%20Manual/DisloCluster%20Code%20History.tex).

---

## Conservation is an OPEN-system balance in 3-D

The six accumulators integrate a 0-D atom balance with **no transport term**:

```
d(stored)/dt = production − recombination − sink_absorption
```

On a spatially-resolved march that cannot close, and the residual is not solver
error — it is the atoms that left through the Dirichlet surface. Two things
force it: the domains here are Dirichlet over their whole surface unless faces
are made periodic, and the slow step **freezes** the mobile species while
production keeps accumulating into them (the mobile pool's own balance is
closed by the *fast* solve, which is where the flux to the boundary lives).

The magnitude settles it: the 200 nm march reports a residual of **−71% of
cumulative production**. Nothing in a CVODE run at `rtol=1e-6` is 71% wrong.

So `post_process._calculate_conservation` also publishes `grain_boundary`
(= −residual, positive when atoms have left), plus `mobile_change` and
`immobile_change`. `post.volume_average` presents it as a physical channel
because it knows the surface is absorbing (it reads `periodic_face_ids` from
`config.json`); a standalone 0-D run leaves it as the numerical residual it
genuinely is there, and `plot_conservation` falls back to the old
relative-error figure. **It is closure by difference, not an independent
measurement** — it inherits every other channel's error, and its credibility
rests on those being exact solver integrals rather than reconstructions.

**Figures are in defect COUNTS, not atom fractions.** `n_atoms = V / Ω` with
`V = fields.domain_volume(nodes)` (the crystal, not its bounding box) and
`Ω = cluster_atomic_volume(...)`, both in b³. The 200 nm prism is 6.93×10⁸
atoms, and at 10 dpa:

| channel | interstitials | vacancies |
|---|---:|---:|
| produced | 6.24e9 | 6.24e9 |
| recombined | 1.43e9 (22.9%) | 1.43e9 (22.9%) |
| absorbed at network sinks | 3.51e8 (5.6%) | 3.20e8 (5.1%) |
| **absorbed at grain boundary** | **4.45e9 (71.4%)** | **4.48e9 (71.9%)** |
| stored in microstructure | 4.71e6 (0.1%) | 1.33e6 (0.0%) |

Equal Frenkel production and equal recombination are exact; the storage
asymmetry (4.7e6 vs 1.3e6) is the bias-driven excess that drives growth.

The `*_fractions` figures now include the boundary channel, so their "Sum (=1)"
line is true again — it was reaching ~0.3.

### The boundary channel is measured independently

`post.boundary_flux` gets the same number a second way and never touches the
accumulators. The fast solve returns a **steady** mobile field, so for each
mobile species `0 = ∇·(D∇C) + R`, and by the divergence theorem

```
∫_Ω R dV  =  ∮_∂Ω (−D∇C)·n dA   =  net outward flux
```

`R` is exactly the 0-D right-hand side for the mobile components — that model
has no diffusion, so its mobile equations *are* the reaction terms, and the
grain boundary appears nowhere in it. That is why the volume integral of its
mobile RHS measures the flux to the boundary. Atom weights are (1, 2, 3) for
(Ci, C2i, C3i) and 1 for Cv, matching `I_stored`/`V_stored`.

Measured against closure on the 200 nm march:

| dose | measured | closure | ratio |
|---:|---:|---:|---:|
| 0.01 | 5 666 311 | 5 106 945 | 1.110 |
| 0.1 | 44 999 294 | 44 632 725 | 1.008 |
| 1 | 431 455 063 | 444 694 897 | 0.970 |
| 10 | 4 287 277 443 | 4 450 899 303 | **0.963** |

**Agreement to 3.7% at 10 dpa** — an independent confirmation, not an identity.
The 11% at 0.01 dpa is the trapezoid over a coarse log grid where the rate
falls 36% across the first interval, not a physics discrepancy. Run it alone
with `python -m dislocluster_code.post.boundary_flux <run>`; `march_report`
calls it guarded, so a diagnostic can never cost a report a multi-hour march
earned.

A surface integral inside MoDELib, where the FE gradients already exist, would
be the gold standard. This needs no C++ change and works on runs that already
exist, which is why it came first.

### Ω was corrected — the fit has NOT been redone

Ω used to read **1.2e-29 m³**, which is `V_cell/4`: four atoms in an hcp
primitive cell that holds two. It is now the hcp Zr value, which the input
files already determined twice over, agreeing to **0.14%**:

| route | value |
|---|---:|
| lattice: `(√3/2)a²c / 2`, a = b_a = 3.23 Å, c = b_c = 5.15 Å | **2.326553e-29 m³** |
| mass density: `M_Zr /(ρ·N_A)`, ρ = `rho_SI` = 6520 kg/m³ | 2.3233e-29 m³ |
| former value (`V_cell/4`) | 1.2e-29 m³ |

Changed in **four** places, which must stay equal:

| where | what |
|---|---|
| `Simulations/input/Zr_input_parameters.xlsx` | `Physical_Properties!D5` |
| `ZrMicro/input/Zr_input_parameters.xlsx` | same cell; copied byte-for-byte, so `paths.workbook_drift()` is clean |
| `MoDELib3/Library/Materials/Zr3d_ghoniem.txt` | `atomicVolume_SI` — the 0-D ↔ 3-D bridge converts through this key |
| `zerod/input_data.py` | the fallback default, which read 1.4e-29 — a *third* number for one lattice constant |

The lattice route is the one used, so Ω stays consistent with the same `b_a`
and `b_c` that the loop-radius prefactors `√(Ω/πb)` use.

**THE 28-PARAMETER FIT IS NOW STALE.** Ω is not a reporting factor: every
number density is `C/Ω`, line density is `2πr·C/Ω`, and emission carries
`exp(σΩ/kT)`. Measured on the fitted set at 10 dpa:

```
N_a 0.483x   N_c 0.516x   c_a 0.522x   Cv 1.047x   Ci 1.059x
```

The loop densities **halve**, and those parameters were fitted against
experimental loop densities at the old Ω. The model now has the right lattice
constant and a fit that no longer matches experiment. **Refit before comparing
to data** — a refit is planned separately.

`calibration.LEGACY_OMEGA = 1.2e-29` reproduces a pre-correction run:
`build_sim(extra={"Omega": LEGACY_OMEGA})`, with `atomicVolume_SI` set back
too. `physical_omega=True` is now a no-op, kept so existing callers do not
break. `InputData.check_atomic_volume()` still runs on every `build_sim`; it is
silent now, and is what would catch a regression.

**Runs made before this correction are not comparable to runs made after.**

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
- [`ZrMicro/CLAUDE.md`](ZrMicro/CLAUDE.md) — model equations, state vector, file map
- [`MoDELib3/ZR3D_GHONIEM_CHANGES.md`](MoDELib3/ZR3D_GHONIEM_CHANGES.md) — 3-D side changes
