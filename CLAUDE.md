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
| `dislocluster_code.coupling` | `march` (the operator split), `config` (`MarchConfig`), `qssa`, `field`, `immobile`, `checkpoint`, `progress`, plus the discrete handoff: `transition` (continuum→discrete + the conservation ledger), `neighbors` (screened cutoff, `k²` as `ImmobileSinks` builds it), `ellipse_rom` (the ⟨a⟩ two-parameter loop) |
| `dislocluster_code.post` | figures, movies, discrete loops, TEM slices, reports, `coarsening` (the `d_coarsen` detector) |
| `dislocluster_code.studies`, `.fitting`, `.legacy` | comparison drivers, parameter fits, superseded modules. `dad_sweep` / `dad_window` answer which anisotropy lets ⟨a⟩ and ⟨c⟩ grow together; `compare_anisotropy` diffs two marches; `hardening` turns a march into a DD yield test (below) |

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

- **Venv:** `.DisloClusterVenv/` at the repository root (Python **3.14.3**)
- **Package:** `pip install -e .` from the repository root (`pyproject.toml`)
- **Jupyter kernel:** `dislocluster` — "Python 3.14 (DisloCluster)"
  (the kernel keeps its original name; only the Python package was renamed)
- **Dependencies:** `requirements.txt` at the repository root
- **Do not use Anaconda Python** — NumPy 1.x/2.x conflict with SciPy

**The PATCH version matters, because a notebook records it.** Jupyter and the
VS Code extension write the interpreter into every notebook's metadata —
`kernelspec.display_name` as `.DisloClusterVenv (3.14.3.final.0)` and
`language_info.version` as `3.14.3` — so a venv built on any other patch release
dirties `run_simulation.ipynb` and `hardening_simulation.ipynb` the moment they
are opened, and two machines on different patches churn those lines back and
forth forever. Homebrew is no help here: it carries only the newest 3.14 (3.14.7
at the time of writing) and no way to ask for an older one. Pin the interpreter
instead:

```bash
uv python install 3.14.3          # exact patch, ~17 MB, into ~/.local/share/uv
"$(uv python find --system 3.14.3)" -m venv .DisloClusterVenv   # --system: not the venv
.DisloClusterVenv/bin/python -m pip install -r requirements.txt
.DisloClusterVenv/bin/python -m pip install -e .
.DisloClusterVenv/bin/python -m ipykernel install --user --name dislocluster \
    --display-name "Python 3.14 (DisloCluster)"
```

Every pin in `requirements.txt` installs on 3.14 from a wheel — numpy, scipy,
pandas, matplotlib, pillow and rpds-py ship `cp314` arm64 wheels, and pyzmq,
debugpy, gmsh and fastjsonschema are `abi3` or pure Python — so nothing builds
from source and nothing had to move off its pin.

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
| `COUPLING` | route, seed dose, snapshot doses, substeps, fast-solve cadence, failure tolerances, the coarsening detector (`phi_star`, `frac_star`, `coarsen_hold`) and the discrete handoff (`discrete_transition`, `transition_units`, `climb_cutoff_nL`) |
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

## Anisotropic diffusion, and which anisotropy to use

The diffusional anisotropy difference appears in the material file **twice** — as
`mobileSpeciesEnergyMigration_eV` (the tensor the fast solve diffuses with) and as
`dadAnisotropy` (the `p_m` the closed-form capture efficiencies use). They used to be
independent, so the code simultaneously believed diffusion was isotropic and that the
sinks were biased by its anisotropy. `staging/anisotropy.py` now generates **both from one
`p_m` per species**, split at fixed `D_eff` so a change alters directionality and not
overall mobility:

```
E_m⟨11,22⟩ = E_m_eff + 2 k_BT ln(p_m)      E_m⟨33⟩ = E_m_eff − 4 k_BT ln(p_m)
python -m dislocluster_code.staging.anisotropy --show
python -m dislocluster_code.staging.anisotropy --p-m 1.0 0.91372 0.91372 0.91372 --apply
```

**Simultaneous ⟨a⟩ and ⟨c⟩ growth is possible iff `p_I < p_v`.**

> **WHICH SOLVER THIS CRITERION IS ABOUT.** It is derived from
> `ClusterDynamicsFEM::solveImmobileClusters`, i.e. the 3-D immobile solver
> (`useImmobileSolver=1`), whose capture efficiencies come from the diffusion
> tensor through `loopDADbias`. **The coupled march does not run that solver.**
> Its slow step is ZrMicro's, which carries the *aligned/non-aligned* families
> and *phenomenological* efficiencies `Z_i_a = 1+delta_i`, `Z_v_c = 1+delta_v`
> read from the workbook. In the coupled route the diffusion tensor therefore
> reaches loop growth **only through the mobile concentrations**, never through
> the capture efficiencies. So this criterion predicts the 3-D solver's
> behaviour, and predicts a coupled march only indirectly, through the mobile
> field the fast solve hands over.
>
> **`SOLVER['loop_model'] = 1` removes that gap** — the slow step then carries
> `⟨c⟩ + 3×⟨a⟩` with Woo efficiencies built from the same `p_m`, so the
> criterion applies to the coupled march too. It is **off by default**, because
> the 28-parameter set was fitted against the legacy formulation. See the next
> section.

 Both growth conditions
reduce to bounds on one number — the arrival ratio `A/B = D̄_v c_v / Σ_m D̄_m c_m |m|` — and
the window between them has width `g(p_v)/g(p_I)` with `g(p) = 2/(1+p⁻³)`, strictly
increasing. Derived in full in
[`Docs/DisloCluster Manual/anisotropic_diffusion_and_discrete_coupling_plan.md`](Docs/DisloCluster%20Manual/anisotropic_diffusion_and_discrete_coupling_plan.md) §1.3.1.
Measured co-growth set, 100% of interior nodes at every dose 0.01–10 dpa:

```
p_m = (p_v, p_i, p_2i, p_3i) = (1.000000, 0.913720, 0.913720, 0.913720)
```

i.e. **vacancies isotropic**, interstitials at the already-fitted value; tolerance
`p_v ∈ [0.94, 1.07]`. Li et al.'s structure (DAD on the clusters only) gives no co-growth
**here**, because the clusters carry 3.1% of the interstitial arrival and an isotropic
species is inert in the criterion — in their model the di-interstitial is the *more*
mobile species and carries the flux.

Three limits worth knowing before quoting any of this. The criterion is the 3-D
immobile solver's, as above, and the coupled march runs a different one. It governs the
**absorption flux only**: `ClusterDynamicsFEM.cpp:755` adds cascade nucleation `Gk`, which
no anisotropy affects, so it does not govern net population evolution. And an isotropic
march is **not a zero** of the criterion — it is another parameter point — so a ratio taken
against one cannot test a statement about signs.

## The superposition: why a discrete loop showed no halo

MoDELib solves the mobile species by **superposition**, `c = c_FEM + c_DD`, and `c_DD`
(the analytic Green's-function field of the discrete segments) enters the FEM problem
**only through the Dirichlet values** — `ClusterDynamics.cpp:150`,
`dirichletConditions = bndConcentration − otherConcentration`. The CD block of
`evl_*.txt` is `c_FEM`, the *corrective* part, and it has to be: `initializeConfiguration`
reads it straight back into `mobileClusters`. So **a figure drawn from the CD block is
smooth by construction however many discrete loops exist** — the loop-localized depletion
lives entirely in `c_DD`, which used to be evaluated on demand at DD quadrature points and
discarded.

Three things were needed, and each alone was enough to make the transition inert:

| | |
|---|---|
| `outputSuperposedMobile` | new optional material key, **default 0**. Makes `ClusterDynamics::output` also write `evl/cdTotalMobile_<runID>.txt` (`nNodes × 4`, in `cdNodes.txt` order) holding `c_FEM + c_DD`. `enable_discrete_climb` sets it; the march files one per snapshot into `evl_coupled/`; `movies.cd_blocks` substitutes it for the mobile columns where present. |
| `MobileQSSASolver.adopt_network` | `inject_discrete_loops` *does* merge the generated network into the staged `evl_0` — but `solve` rewrites `evl_0` from its own preserved seed on **every** call, so the next fast solve discarded it. The evl header's first six integers were `0` in that seed, before and after the transfer: **every fast solve ran with no dislocations.** |
| `Nsteps = 2` | `DefectiveCrystal` emplaces `ClusterDynamics` **before** `DislocationNetwork` (`DefectiveCrystal.cpp:38-44`) and `MicrostructureContainer::solve` walks that order, so the CD solve runs first. `c_DD` is **linear in the nodal `climbVelocityScalar`**, which is zero on step 0 — a one-step fast solve sees `c_DD = 0` however many loops are present. |

**This cannot be reimplemented in Python.** The Green's function's amplitude is the nodal
climb velocity, which is the *output* of the climb solve (`clusterConcentration` =
`concentrationMatrices(x) @ [v_source, v_sink]`). There is no offline evaluation.

**Existing figures are correct.** Because `c_DD ≡ 0` in every run made before this,
`c_FEM` *was* the physical concentration. It is once loops genuinely enter a solve that the
CD block becomes the corrective part only — which is why the total is published *alongside*
rather than folded in.

## Two formulations of the slow step — `SOLVER['loop_model']`

The slow step's *integrator* is chosen by route (Option A IMEX / Option B CVODE).
Its **model** is chosen by `SOLVER['loop_model']`, and the two are orthogonal.

| | `0` — legacy (**the default**) | `1` — self-consistent |
|---|---|---|
| families | `iL, aiL, vL, avL` (aligned/non-aligned) | `c, a1, a2, a3` — MoDELib's CD block |
| capture | phenomenological `Z_i_a = 1+delta_i`, one `Z` for all interstitial species | Woo, from the same `p_m` as the tensor, **one `Z` per species** |
| bridge | a lumping + a split, both lossy | an **identity** (only `1/Ω`), round trip exact |
| status | **the fitted model**, bit-identical to the pre-change binary | consistent with the fast solve, **not calibrated** |

Equations:

```
Z_basal(m)     = Z0_m · p_m                      row 0, vacancy-type <c>
Z_prismatic(m) = Z0_m · (p_m + p_m^-2)/2         row 1, interstitial <a>
phi_km         = S_k · Dbar_m · Z(row(k),m) · c_m · |m|
S_k            = (l_k/l) · loopSinkScale_k · sqrt(n_k · c_k)
ydot[4+k]      = nuc_num[k]  - ann_num[k]  - coal_num[k]
ydot[8+k]      = (gain[k] - loss[k]) + nuc_cont[k] - ann_cont[k] - coal_cont[k]
y[4..11]       = [n_c, n_a1, n_a2, n_a3, c_c, c_a1, c_a2, c_a3]
```

identical to `ClusterDynamicsParameters::loopDADbias` and to `ImmobileSinks`'s
`Sk`. `l_k = √(Ω/πb_k)` is the family's **own** radius scale, so ⟨c⟩ is sized
with `b_c` and ⟨a⟩ with `b_a` — the legacy model sizes both with `l_c`.
`Dbar_m = (det D_m)^(1/3)`, which equals the legacy `omega_m` because
`anisotropy.py` splits the migration energies at fixed `D_eff`: **no mobility is
redefined, only its directional weighting.** `p_m`, `Z0_m` and `loopSinkScale`
are read from `Zr3d_ghoniem.txt`, not the workbook — the one file both sides
share, and the file `anisotropy.py` writes.

**The legacy path is bit-identical** (`sha256 8bcc7780…`, 25-point integration),
and that is why the legacy `ydot` assembly in `rate_equations_core.h` is kept
**verbatim** inside `if (P.loop_model == 0)`. Do not tidy it: regrouping
`growth + nuc + G` into `growth + (nuc + G)` — algebraically identical — moves
the 10th significant digit, which was measured (`ff2980c4` vs `8bcc7780`).

### What the formulation change does — measured on the COUPLED MARCH

Two 500 nm hexagonal marches, route A, differing only in `loop_model`, at
10 dpa. **Interior mean** (innermost quartile) — a domain mean is not quotable
on this geometry and would have hidden the effect entirely (`N_a` 1.02× instead
of 3.08×):

| | `N_a` | `c_a` | `N_c` | `c_c` |
|---|---:|---:|---:|---:|
| legacy | 5.830e-08 | 5.156e-05 | 1.777e-08 | 1.648e-03 |
| self-consistent | 1.795e-07 | 5.390e-05 | 1.783e-08 | 1.097e-03 |
| ratio | **3.078×** | 1.045× | 1.003× | **0.665×** |

`N_a` is the **total** in both, summed over the two legacy families or the three
prism variants; the aligned fraction measures 0.404 against `f_a` and each
variant exactly 1/3, and at 1e-4 dpa the two agree to **1.0001**, which is what
proves the totals are comparable.

**LOOP GROWTH IS A NEAR-CANCELLATION, and that is the whole story.** The net
content rate is a small residual of two much larger opposing fluxes:

| ⟨a⟩ prismatic | interstitial gain | vacancy loss | net |
|---|---:|---:|---:|
| legacy | 4.0954e-02 | 3.5677e-02 | 5.277e-03 (**12.9%** of gain) |
| self-consistent | 3.7703e-02 | 3.6854e-02 | 8.492e-04 (**2.25%** of gain) |

so a 5.4% rise in the vacancy efficiency and an 8% fall in the gain cut the net
to **16%**. The loops then never outgrow their nucleation size: `c_a/N_a` is
**300.3** against `n_iL_nuc = 300`, i.e. every ⟨a⟩ loop in the self-consistent
interior is a fresh nucleus, where legacy reaches 884.4.

**`N_a` then follows from two balances, not one.** The number balance pins the
*content*, because 98% of the loop-number loss is the loop-NETWORK channel and
`Σ φ_LN·N_k ∝ Σ r_k²N_k = l_a²·Σ c_k` — proportional to total content and
independent of how it is split (measured loss ratio 1.047 against content ratio
1.045). The growth balance pins the *size*. Density is the quotient:

```
N_a ratio = content ratio / size ratio = 1.045 / (300.3/884.4) = 3.078
measured                                                      = 3.078
```

**Where the efficiency differences come from.** At the FITTED `p_v = 1.178808`
all four efficiencies match the workbook to 6 digits, by construction of the
`dadZ0` fit. Route A sets `p_v = 1`, and because the two Woo forms differ —
basal `Z⁰p` is monotone, prismatic `Z⁰(p+p⁻²)/2` has its minimum at
`p = 2^(1/3) = 1.2599` — that one number pushes the families **opposite ways**:

| | `p_v = 1.1788` | `p_v = 1` |
|---|---:|---:|
| `Z_basal(v)` — ⟨c⟩ gain | 1.107886 (= `Z_v_c`) | 0.939836 → **0.848×** |
| `Z_prismatic(v)` — ⟨a⟩ loss | 0.892114 (= `Z_v_a`) | 0.939836 → **1.053×** |

so it starves ⟨c⟩ growth and accelerates ⟨a⟩ shrinkage at once. The interstitial
efficiencies are untouched (`p_i` unchanged): both ratios exactly 1.000000. In
**legacy** `p_v` cannot reach the efficiencies at all — `Z_v_a`/`Z_v_c` are
workbook constants — so this is the co-growth criterion acquiring teeth in the
coupled march, and it is why `d_coarsen` goes from 0.4 dpa to *never* on this
case.

**Route A therefore does NOT isolate the formulation.** It measures the
formulation *plus* the fact that only mode 1 propagates `p_v` into the capture
efficiencies. At matched `p_v` the residual differences are exactly two:
per-species cluster mobility (legacy gives `C2i`/`C3i` the monomer `ω_i` though
`ω_2i/ω_i = 4.9e-6`, overstating cluster arrival) and three ⟨a⟩ families
instead of two.

### The MATCHED comparison — the formulation isolated

Two further 500 nm hexagonal marches, identical in every respect except
`loop_model`, at **zero applied stress** (so `f_a = 1/3` exactly, the split mode 1
carries — verified `0.333333333333` in both logs) and at the **fitted
`p_v = 1.178808`**, where all four Woo efficiencies reproduce the workbook
constants to 6 digits. Same mesh, same 9-point dose grid, same 90 617 CD nodes,
0.71 h and 0.73 h. Interior mean, `N_a` and `c_a` totalled over families:

| | `N_c` | `N_a` | `c_c` | `c_a` |
|---|---:|---:|---:|---:|
| legacy | 2.5810e-08 | 8.7884e-08 | 1.6061e-03 | 5.1880e-05 |
| self-consistent | 2.5770e-08 | 1.4217e-07 | 1.9143e-03 | 5.4910e-05 |
| ratio | 0.998 | **1.618** | **1.192** | 1.058 |

**Most of what route A showed was `p_v`, not the formulation.** `N_a` falls from
3.078× to 1.618× and `c_c` inverts, 0.665× → 1.192×, once `p_v` is matched — which
is the quantitative form of the claim above, and it confirms that the ⟨c⟩ starvation
was `Z_basal(v)` at `p_v = 1` rather than anything about the families.

The same two-balance closure holds exactly, on both families:

```
<a>   N ratio = content ratio / size ratio = 1.058 / (386.2/590.3) = 1.6177   measured 1.6177
<c>                                       = 1.192 / (74281/62228) = 0.9984   measured 0.9984
```

and the ⟨a⟩ size ratio 0.654 is again the self-consistent loops sitting nearer their
nucleus (386.2 against `n_iL_nuc = 300`, where legacy reaches 590.3).

**`d_coarsen` swaps families outright**, which no ratio would have shown:

| | ⟨c⟩ | ⟨a⟩₁₋₃ |
|---|---|---|
| legacy | **0.4 dpa** | never |
| self-consistent | never | **0.4 dpa** |

so which family the mean-field treatment of coalescence fails for first is a
property of the formulation, not of the material.

> **A CORRECTED CLAIM.** An earlier revision of this section attributed the
> difference to the *variant split* acting through like-loop coalescence,
> `φ_LL = 1−exp(−κ_LL(4/3)πr³n)`, three families of `n/3` overlapping less than
> one of `n`. **That is wrong**: `φ_LL` carries only **2%** of the loop-number
> loss, the loop-network channel carries 98%, and `φ_LN` contains no dependence
> on `n` at all. The same revision quoted 0-D numbers (30.2×, 1.52×) labelled
> "route-A anisotropy" that were in fact taken at the fitted `p_v = 1.178808`,
> because the diagnostics read `dadAnisotropy` from the shared material file
> while concurrent runs were rewriting it. **Pin `p_m` on the command line in
> any diagnostic**; do not inherit it from that file. And note the 0-D is not a
> valid probe of the march's mechanism here — there `Cv`/`Ci` are free and
> re-equilibrate, so 0-D and the march move `c_c` in *opposite* directions.

**The prefactors are not a factor**, and one near-identity is worth recording:
`l_a/l_c = √(b_c/b_a) = 1.2627` and the fitted `loopSinkScale_a = 0.792317` are
reciprocals to 0.05%, so `(l_a/l_c)·s_a = 1.000464` — MoDELib's sink-scale
calibration silently undoes the legacy model's use of `l_c` for ⟨a⟩ sinks. The
⟨c⟩ ratio `s_c/Q = 1.012493`.

**The 28-parameter fit is against mode 0, so it is stale for mode 1** — this
compounds with the Ω correction. Do not compare a mode-1 run with experiment
before the refit.

`COUPLING['discrete_transition'] = True` with `loop_model = 1` **raises at
validation time**: `coupling/transition.py` addresses the immobile state by the
legacy slot names, which name a different family in every slot under the new
layout. The failure would not crash — it would build a well-formed set of
discrete loops from the wrong population. Porting the handoff is the next piece
of work.

`march_state.npz` and `summary.json` record `loop_model`, because **the state
array does not say which layout it is in**. `field.run_loop_model(run)` reads it
and `field.to_legacy_layout` maps mode-1 families onto the legacy slots for the
0-D figure suite and `post.boundary_flux`, which address them by the legacy
names — lossless for every aggregate a figure plots, since the aligned/non-aligned
split it would need does not exist in mode 1.

## The implementation plan's model switches — `SOLVER`, steps 1–5

`loop_model = 1` is the door; these are the rooms behind it. Each defaults to
the value that reproduces the formulation before its step, and each is
**rejected without `loop_model = 1`** — they name families and moments the
legacy slots do not carry, so a run that set one without it would report the new
model and integrate the old one.

| key | 0 / default | 1 (or >4) |
|---|---|---|
| `n_fam` | 4 | 8 adds the three prismatic **vacancy** variants (step 1); 9 adds the stacking-fault pyramid `c_0` (step 4) |
| `chi` | 1.0 | character splitting `X_iI X_vV/(X_iV X_vI)`; the coexistence window has log-width `ln χ` (step 2) |
| `emission_model` | 0 | peripheral emission against each family's own `c^{v,eq}`, replacing the two fitted annealing lifetimes (step 3) |
| `basal_chain` | 0 | `c_0 → c_f → c_p`; needs `n_fam = 9` (step 4) |
| `moments` | 0 | the second content moment `q_k` per family and the log-normal closure on it (step 5) |

`SOLVER['model_params']` carries the numeric inputs those switches need
(`eps_sfp`, `n_sfp_nuc`, `tau_sfp`, `nu_col`, `nu_uf`, `alpha_sfp`, `m_min`,
`m_col`, `m_uf`) straight to the solver command line. It is a dict rather than a
field per parameter because the C++ already defaults every one of them and an
unrecognized key must fail *there* rather than be dropped here. **None of the
basal-chain rates is calibrated** — nothing in `Zr3d_ghoniem.txt` or the
workbook supplies them, and the formulation gives their Arrhenius form without
values for the barriers.

**The state width follows the switches, and the march has to allocate it.**
`immobile.state_width(loop_model, n_fam, moments)` gives 19 / 29 / 38, and the
checkpoint fingerprint records the **actual** width so a resume across a change
of it is refused. The solver defaults every appended `y0` slot to zero, so a
narrower state is always *accepted* — which is exactly why the march allocating
19 while the command line said nine families and three moments would have been
silent.

The demonstration case is
`python -m dislocluster_code.studies.run_200nm_moments` — a 200 nm cube with all
nine families and three moments, writing `uncalibrated_inputs.md` into its own
run directory. `--state <run>` prints the interior per-family table of
`n, c, m̄, Δ` without solving anything.

`Δ_k = q n / c²` on the 0-D side and `q n ω / c²` in `ImmobileSinks` — **one
explicit ω**, because `n` crosses the bridge converted (per atom → per b³) and
`c` does not, which is the same asymmetry that makes MoDELib's mean size
`CI/N/ω`. It reaches the fast solve in exactly one place: `Δ^{-1/8}` on `S_k`.

### Every channel in the `q` equation, and the one that was backwards

`Δ = q n / c² ≥ 1` is Cauchy–Schwarz on a non-negative measure, so `Δ < 1` — and
`q < 0` outright — is not a tight tolerance, it is a **defect**. Three rules keep
the moment set inside it:

| channel | closure | why |
|---|---|---|
| nucleation | at the declared size, `nn·m_nuc²` | it genuinely deposits at `m_nuc`, and `nn(m_nuc − m̄)² ≥ 0` is real broadening |
| the floor current | at `m_min`, `−m_min²·Φ` | a genuine statement about the distribution's lower tail |
| **annealing, coalescence** | **shape-preserving**, `dq = Δ(2m̄·dc − m̄²·dn)` | neither carries a distribution of its own |

**A coalescing loop does not leave the family — it MERGES.** Its defects stay,
only `coal_cont` is lost, and fewer loops holding the same content are *larger*
loops, so coalescence **raises** `q`. Debiting `coal_num·⟨m²⟩` — the
natural-looking reading — inverts the sign on the largest single term in the
equation, because coalescence carries 98% of the loop-number loss. Measured:
`q` crossed zero at 1 dpa on the 200 nm nine-family march and was negative on
**40% of the `c_p` nodes** at 10 dpa.

**The legacy annealing surrogate cannot be made consistent, only avoided.** It
debits content at the fixed size `n_vL_nuc`, so the pair `(ann_n, ann_c)`
declares a removed sub-population of mean `M ≠ m̄`; proportional debiting
over-removes, and debiting at `M` lowers `Δ` by `a(M/m̄ − 1)²` on a narrow
family. No debit is realizable, because the surrogate removes loops of a size
that is not there. `emission_model = 1` deletes the lifetimes outright, so
carrying `moments` against the pre-step-3 annealing is a combination to avoid.

**`plan_step5 verify` goal (vi) is the only test that can see this class of
bug.** The other goals each switch the *other* channels off to isolate one term,
so an inconsistent channel is precisely the one never exercised alongside the
rest. Goal (vi) runs five successive steps with coalescence, nucleation,
emission and the basal chain all live and reports the minimum `Δ` per family.

### After step 3, coalescence is the ONLY sink on vacancy loop number

`emission_model = 1` deletes `tau_avL`/`tau_cvL`, and peripheral emission
replaces them on **content only** — emission moves vacancies out of a loop, it
does not remove the loop. Step 3's own goal (iii) states this as a virtue ("every
density exactly constant"), which it is *as a test*; in a driven run it is a gap.

Measured on one interior node, 0 → 10 dpa, mobile frozen, one switch at a time:

| | `N_a` | `N_c` | ratio |
|---|---:|---:|---:|
| step 2 (`emission_model=0`) | 1.04e-07 | 1.18e-08 | **8.83** |
| step 3 (emission on) | 5.51e-07 | 5.45e-07 | 1.01 |
| steps 4–6 (chain on too) | 2.13e-07 | 1.14e-06 | 0.19 |

**⟨a⟩ should exceed ⟨c⟩ by about an order of magnitude, and step 3 alone
inverts it** — `N_c` rises 46×. The basal chain is a minor contributor and
`eps_sfp` is nearly irrelevant (two decades move `N_c` by 1.5×). If a run shows
⟨c⟩ outnumbering ⟨a⟩, this is why; it is not the placeholders.

**Coalescence is not broken and it is not absent — it is doing the whole job,
and only after step 3.** How much it removes, per family, one node, 0 → 10 dpa
(the factor by which switching it off raises the density):

| | `c_f` | `a1` (i) | `a1v` (v) |
|---|---:|---:|---:|
| `emission_model = 0` | **1.02×** | 806× | 1.01× |
| `emission_model = 1` | **152.9×** | 807× | 91.9× |

**Before step 3 coalescence did essentially nothing to a vacancy loop family** —
not because the channel was missing but because `tau_vL` removed the loops
first, so the population never built up for them to overlap. The interstitial
family, which never had a lifetime, is untouched by the switch (806× either
way). With the basal chain on, `c_p` loses **733×** and `c_f` only **4.6×**.

That is *why* the basal coefficients are small: **the fit never had to make them
large.** `cLL`/`cLN` are **0.162 / 2.96** for basal against **121 / 1131** for
prismatic — ~700× — and the fit that produced them was done where `tau_cvL`
removed the basal loops for them.

The rate is `nu = c_LL · max(drdt, 0) · N^(1/3)` with `drdt ∝ gain/√(nc)` — the
family's own climb speed, and the gain is **net of emission** for a vacancy
family. So a family sitting at its own `c^{v,eq}` has `drdt ≤ 0` and coalesces
not at all. That clamp is a real fragility but it is **not** what is happening
here: `c_p` losing 733× is a family coalescing hard while being fed harder.

**A testable prediction for the refit**, not a free parameter. Scaling the two
basal coefficients alone (`N_a` is insensitive to five digits, so the families
decouple):

| scale | 1 | 10 | 100 | 1000 |
|---|---:|---:|---:|---:|
| `N_a/N_c` | 0.19 | 0.53 | 1.37 | 5.92 |

Recovering the step-2 ratio needs ~10³, landing basal at ~160 / ~2960 — within
30% of the prismatic values. **If the refit lands anywhere else, the missing sink
is physical rather than a coefficient**, and `c_p` below is where to look.

**`c_p` is a terminal population.** Coalescence is its only number loss — it
cannot unfault further, has no dissolution lifetime, and peripheral emission
acts on content and never on number. Under a sustained vacancy supply it is
bounded only by the transfer feeding it. Whether it should have a sink of its
own is an open modelling question, and it must be settled before `nu_uf` is
fitted: the two are not separately identifiable from a steady `c_p` density.

## The continuum → discrete handoff

`post/coarsening.py` decides **when** the mean-field treatment of coalescence stops being
valid (`d_coarsen`, where the Avrami overlap crosses `phi_star`); `coupling/transition.py`
performs the switch, and `coupling/neighbors.py` supplies the cutoff that makes the climb
solve affordable. Armed with `COUPLING['discrete_transition'] = True`, **off by default**.

| invariant across a transfer | status |
|---|---|
| loop number | exact to the integer draw |
| stored defects | **exact to 1e-16** (`rescale_to_defects` removes the rounding) |
| sink strength | **now continuous to ~2%** — was a jump; still reported, never asserted on |

**The Burgers half of the sink jump has been removed rather than accounted for.** It
used to read `√(b_cd/b_dd) = 1.414` — the continuum sized a ⟨c⟩ loop with the full
⟨0001⟩ Burgers vector and DD with the half — times `1/loopSinkScale = 3.43`, product
4.85. Both sides now carry `|b| = c/2`, so a fresh ledger on the 500 nm matched leg
gives `√(b_cd/b_dd) = 1.000000` and measures `S_raw/S_cont` at **0.97–1.04 at every
dose from 1e-4 to 10 dpa**. What remains is `1/loopSinkScale = 3.43`, which is a
convention difference between the two solvers and lives *outside* the S totals —
`continuum_totals` does not apply `loopSinkScale`, which is why it never appeared in
`S_ratio_transfer`. **That one is still a modelling decision to be made before a
transfer feeds a solve.**

Two things the transfer must respect, both measured rather than assumed:

- **`microstructureGenerator` silently refuses loops whose nodes leave the grain.**
  `transition.fits_in_crystal` predicts that decision exactly and keeps the refused share
  in the continuum. The share must be counted in **defects, not loops** — one refusal of
  six was 83% by count and 98% by defects.
- **The domain must be able to hold the loops, and the ⟨c⟩ Burgers correction made
  this WORSE.** `l_c` rose by √2, so ⟨c⟩ radii did too, and a bigger loop is harder to
  fit. Measured on the 500 nm matched leg, ⟨c⟩ only:

  ```
  dpa      1e-4   1e-3   1e-2    0.1      1      2      5     10
  frac_kept 1.000  0.981  0.990  0.757  0.0096 0.0005 0.0144 0.0232
  r_mean nm  3.3    3.5    4.7   19.9    18.2    7.2   42.3   35.8
  ```

  So the ⟨c⟩ handoff is **effectively infeasible on a 500 nm domain above ~0.1 dpa**,
  where the pre-correction figure was 0.55 from 1 dpa on, and it was already exactly 0
  on the 200 nm case. `coalescence_area` tracks `frac_kept` almost exactly (0.00945
  against 0.00964 at 1 dpa), which is the check that the loss is the geometric refusal
  and *not* the coalescence step failing to conserve area; the refused defects stay in
  the continuum, and (I) and (II) still pass. Check `frac_kept` before trusting a
  transfer on any geometry — **including ones where it used to be acceptable.**

**C++ side:** `climbNeighborCutoff_b` (optional material key, 0 or absent = no cutoff)
truncates `GalerkinClimbSolver`'s `O(N_seg²)` pair assembly, which is 100% of the climb
cost. Use `R_c = 4 L_s` — **not 3**: the measured truncation error is 19.3% / 9.6% / 3.9% /
0.25% at `n_L` = 1 / 2 / 3 / 4.

## Irradiation hardening — `studies/hardening.py`

Takes the continuum immobile field a march produced, rebuilds it as discrete
loops in a **periodic cube**, loads that cube and measures the CRSS increment
Δτ(dose). Implements
[`Docs/DisloCluster Manual/architecture/irradiation_hardening_dd_plan.md`](Docs/DisloCluster%20Manual/architecture/irradiation_hardening_dd_plan.md);
what running it corrected and measured is in
[`irradiation_hardening_implementation.md`](Docs/DisloCluster%20Manual/architecture/irradiation_hardening_implementation.md).

**[`Simulations/hardening_simulation.ipynb`](Simulations/hardening_simulation.ipynb)
is the entry point**, six control dicts the way `run_simulation.ipynb` has six,
writing everything into one `Simulations/output/<stamp>_<hash>_<tag>/`: `cases/`,
`figures/`, `snapshots/`, `movies/`, `provenance.{json,md}`.

```bash
python -m dislocluster_code.studies.hardening verify              # the plan's tables, as tests
python -m dislocluster_code.studies.hardening state    [<run>]    # interior N_k, r_k, rho_k
python -m dislocluster_code.studies.hardening routec --alpha-c 0.4 --alpha-a 0.2 --taylor-M 3
python -m dislocluster_code.studies.hardening campaign <root> --doses 0.01 0.1 10 --L 200
```

**Three routes, and only one of them is affordable at every dose.** Route A is a
stress ramp on a frozen obstacle field and gives Δτ directly; route C is the
dispersed-barrier law `Δτ_k = α_k μ b √(N_k d_k)` whose **α is fitted on route A
and then applies at every dose for free**; route D seeds the same cell by
density instead of per-loop export and is the smoke test. Route B (strain rate,
the full σ–ε curve) is days per dose and is deliberately not staged.

**The loops are frozen because they are `SESSILELOOP`s and `climbSolverType=none`**
— `DislocationNode::projectVelocity` zeroes the velocity of every node on one.
That is the largest modelling assumption in the whole measurement: no
absorption, no channelling, so it biases hardening high as strain accumulates.

**Everything the module computes is measured; α and the Taylor factor M are
not.** Neither has a default that would be indistinguishable from a fitted one.

**The interior mean, not the domain mean** — same rule as
`discrete_loops.populate(region="interior")`, for the same reason: the Dirichlet
shell is a sink for mobile defects but not for loops.

Six things that make a case that does not run, or runs and measures nothing.
All were found by running it, and all are in the implementation note:

| | |
|---|---|
| `X0 = 0` | `unitCube24.msh` spans [−0.5, 0.5]³, not [0,1]³ as the plan says |
| `F` to 17 digits | box edges must be lattice vectors; 10 digits leaves `971·c/a` 1.25e-7 short and the run dies with "Input vector is not a lattice vector". `staging/inputs.py:_f_block` had the same limit — no coupled case noticed, none of them is periodic |
| the source's slip system | `periodicDipolesDensity` picks its own, and picked a **basal** one, whose resolved shear under a prismatic load is exactly zero: 3000 steps, γ_p = 8e-22, reading exactly like a pinned source. One `periodicDipoleIndividual` on slip system 6 instead |
| `glideSteps ≠ 0` | the generator inserts the sessile prismatic loop unconditionally and the two **glissile arms** only `if(fabs(glideStep)>FLT_EPSILON)`. At 0 the cell contains nothing that can move |
| sign of `n` | resolving on (s, +y) when the system is (s, −y) leaves τ positive and γ_p negative, so the offset criterion never fires |
| the ramp rate | the plan's "1 MPa per 1e4 b/cs" and its "~7e11 Pa/s" differ by 1e3; the DD-unit form is the one implemented |

**A few realizations per thousand are refused by MoDELib** — a network node
lands a few b outside the primary cell and `DislocationNode` rejects it at
startup. It is a property of the draw (500 nm at 1e-2 dpa: seeds 0 and 1 fail,
2 and 3 run), so `run_case` raises `PlacementRejected` and `campaign` redraws
with the next seed and records which one it used.

**A zero-byte `F/F_0.txt` makes the NEXT run segfault**, so any first failure
turns later attempts into an unrelated-looking crash;
`staging.inputs.clear_empty_F` removes the stubs and both `case.bootstrap` and
`hardening.run_case` call it. `microstructureGenerator` is the mirror image — it
segfaults if `evl/evl_0.txt` already exists — so `run_case` clears `evl/` and
`F/` first.

**The offset criterion is a choice and it matters.** An irradiated cell creeps
below its breakaway stress, so on the 200 nm cell at 10 dpa γ_p crosses 1e-4 at
~3 MPa and the knee is at 25–30 MPa. Every offset in `OFFSETS` is reported;
`1e-3` is the default and the criterion is quoted with the number.

**`post/dd_frames.py` draws what the SOLVER has**, where `discrete_loops.py`
draws what a continuum field implies: the `evl_<N>` configurations, two panels
each (the cell, and the projection along the glide-plane normal where bowing and
pinning are visible), plus one movie per dose from the solved frames. Four
things in the evl format decide whether it draws dislocation or nonsense —
`loopType` is column **11** (column 10 is the grain, and reading it marks the
gliding dislocation sessile), `hasNetworkLink` separates real segments from the
links that merely close a periodic loop, wrapped polylines must be cut or
they draw straight lines across the cell, and **a network link's dislocation is
the SUM of the loop links on it, which can be zero**.

`hasNetworkLink` is not the whole of that last test. A loop boundary that runs
out and back along the same node pair — the loop pinched to a sliver of zero
area — writes both records with `hasNetworkLink = 1` and opposite sense, so the
link between those two nodes carries `b = 0` and holds no dislocation.
`periodicDipoleIndividual` makes exactly that: each glissile arm closes back
onto the anchor node it shares with the sessile prismatic loop, and those
anchors are pinned for the whole run (`V = 0` in every frame — they sit on a
SESSILE loop, so `projectVelocity` zeroes them, the same mechanism that freezes
the irradiation loops). Drawn, it is a straight red segment from the cell centre
out to wherever the line has bowed to; and because the two arms' anchors differ
only in `y`, the axis the glide panel projects along, the two segments
superimpose at the origin and read as **one line joining the two
dislocations** — which is what a reader reports as "why are these two
dislocations connected".

`_null_pairs` sums the loop Burgers vectors per network-node pair and
`drop_null` (default **on**) cuts the polyline wherever the sum is zero. On the
200 nm case at 0.1 dpa, step 1600, that is **8 pairs of 1576**: the dipole's
`x = 0` lines, its tie bar along `y`, the two tie-backs, and one contact where
the gliding line has zipped onto an ⟨a⟩ loop and locally annihilated it — 121 nm
of the 457 nm the glissile loops appeared to carry. The whole sessile prismatic
source loop is null and now draws nothing, so the panel's `obstacles` count is
the irradiation loops alone and agrees with `n_obstacles`; it read one too many
before. **The control frames never showed the tie-back**, because there the line
has traversed the cell and the tail runs out to the `x` faces in jumps of
309.5 b against `_split_wrapped`'s 309.2 b threshold — so which frames showed it
was the wrap heuristic, not the physics.

**The line walk used to stop one node short, which broke each dislocation into
disjoint stubs.** `loop_polylines` follows the real-link chain with
`while node in nxt and node not in visited` and appended only the nodes it
*entered*, never the sink it exited on, so every open run was one segment short
at its far end. That alone would only clip a tip — what makes it a gap in the
MIDDLE of a line is `visited`: a seed landing mid-chain leaves the stretch
behind it to be walked separately, and that walk then terminates on the
already-visited node without drawing the segment into it. On the 200 nm case at
0.1 dpa, step 7950, the gliding dislocation came out as **four disjoint stubs
with three gaps** — every second segment of it missing. The terminal node is now
appended, and the check is that each dipole arm spans the cell: the two arms
draw **exactly 400.0 nm** at step 0 (two straight lines through a 200 nm cell)
against 366.7 nm before, and 403–448 nm once bowed against 230–290 nm — as
little as **54%** of the line was being drawn. The piece count rises and the
drawn length lands on the geometric minimum, which is what says the recovered
segments are line and not new spurious chords.

**Δτ is NOT monotone in dose on the 200 nm cell, and that is a cell-size
artifact.** The first notebook run measured Δτ = −0.33 / 8.79 / 8.45 MPa at
0.01 / 0.1 / 10 dpa (offset 1e-3, control τ = 4.612 MPa). The fall at 10 dpa is
not saturation: the ⟨c⟩ loops there come out at **r = 55.6 nm, 111 nm across in
a 200 nm cell**, so most are geometrically refused and the obstacle field
collapses from 99 loops to 39 while the stored content rises. It is the same
refusal `transition.fits_in_crystal` reports for the continuum→discrete handoff,
reaching hardening through `cube_population`. **A ⟨c⟩ hardening number at 10 dpa
needs the 500 nm cell.**

**`n_loops` counts the source's three loops too.** `evl_0` holds every loop the
generator wrote, and the periodic dipole adds one sessile prismatic loop plus
two glissile arms, so quoting it against a dose overstates the obstacle field by
exactly three. `run_case` records `n_obstacles`/`n_source_loops`, and
`obstacle_count(row)` recovers the count for older campaigns from the case's
`hardening.json`. Nothing measured moves: `fit_alpha` takes `N_k` from the
continuum state.

**Measured, where the plan estimated:** `useSubCycling=1` is **5.4×** on the
200 nm cell (0.026 vs 0.138 s/step), not the ~100× estimated — the saving is the
ratio of total to active segments, and this cell has ~600 segments against the
~25 000 of the 500 nm case at 0.1 dpa, so re-measure before sizing on it. The
⟨a⟩ loops **do** survive the remesh (46 loops at step 0 and at step 190, through
20 passes), as §6 predicts from `isGeometricallyRemovable`. Cost on 8 cores:
0.026 s/step at 200 nm, **2.5–3.3 s/step at 500 nm** — the 500 nm campaign wants
a 24-core machine.

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

**It grows with the plan's switches, and every index above keeps its meaning.**
Steps 1 and 4 append five families at 19..28 (numbers 19..23, contents 24..28)
rather than interleaving them, and step 5 appends nine second moments at 29..37:

| width | carries |
|---|---|
| 19 | the four legacy slots — `loop_model = 0`, or `loop_model = 1` at `n_fam = 4` |
| 29 | `n_fam` 8 or 9 |
| 38 | plus `moments = 1` |

The CD block is ordered the other way — **moment-major**, all nine numbers, then
all nine contents, then all nine `q` — so the bridge is a scatter and not a
copy. `coupling/field.immobile_into_state` is the one place that knows the
mapping; doing it at a call site is what put an 18-wide block into an 8-wide
slice.

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

### A point can fail the slow step, and where

**Measured on the 500 nm anisotropic march**: node 41804 fails the immobile
integration at the 0.01 → 0.1 dpa substep, and it fails **deterministically** —
in the full batch, again when re-run on its own, twice, across two separate
march invocations.

It sits at `z = 2423.5 b` in a `2474.5 b` domain: **51 b from the basal-pole
face**, inside the boundary layer, with entirely unremarkable concentrations
(30th–50th percentile in every mobile species). It is not an extreme-value node.
That location is the tell — anisotropic interstitial diffusion narrows the
boundary layer along ⟨c⟩ and steepens its gradient, which is what Li et al.
report and what puts this point beyond CVODE's reach at `rtol=1e-6`.

`run_immobile_step` takes `retries=2` and re-runs failed points on their own.
That is worth keeping — it costs a handful of cases against tens of thousands
and it distinguishes a transient from a real failure — but it does **not** help
here, and the retry log says so explicitly (`1 case(s) re-run alone, 0
recovered`). A point that fails alone needs a solver or a model answer, not
another attempt.

Note what `max_failed_nodes` does before reaching for it: it substitutes an
**identity step**, freezing the point for that substep. It does not recover the
point; it declares the march successful without it.

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

**`tem_slices/` renders four conditions**, each as per-dose panels plus a montage. A
circular loop of habit normal `n` viewed along `B` projects to an ellipse of axis ratio
`|n·B|`, and that one number is the whole story:

| view | ⟨c⟩ | ⟨a⟩₁ | ⟨a⟩₂ | ⟨a⟩₃ |
|---|---:|---:|---:|---:|
| `B_0001` — down the c-axis | 1.000 face-on | 0 | 0 | 0 |
| `B_0110` — prism zone | 0 | 0 | 0.5 | 0.5 |
| `B_1120` — the ⟨a⟩ counterpart of `B_0001` | 0 | 0.5 | **1.000 face-on** | 0.5 |
| `B_0001_t45` — tilted 45° about [2̄1̄10] | **0.707** | 0 | 0.612 | 0.612 |

`B_0001` and `B_1120` between them measure both populations at true size. The tilted view
exists because neither zone axis shows a basal loop as an ellipse, and the ellipse is what
identifies one; `tilted_view(deg)` builds any other angle. Note that tilting opens ⟨a⟩₂/₃
while foreshortening ⟨c⟩, so at 45° the two are only 0.095 apart in axis ratio and are
told apart by ellipse *orientation* and the Burgers markers rather than by shape.
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

**BUT THAT AGREEMENT IS NOT GENERAL — it has only ever been demonstrated on the
200 nm march.** Run on the 500 nm hexagonal matched leg (9 doses, 90 617 nodes),
the same diagnostic gives:

| dose | measured | closure | ratio |
|---:|---:|---:|---:|
| 0.1 | 25 059 785 | 163 678 581 | 0.153 |
| 1 | 292 106 555 | 1 687 484 085 | 0.173 |
| 10 | 5 000 524 226 | 17 910 221 185 | **0.279** |

a factor of 3.6, not 3.7%. Three things are established about it and one is not:

- **it is not a regression.** Re-run today on the 200 nm march the ratios are
  1.116 / 0.922 / 0.901 / **0.900** against the 1.110 / 1.008 / 0.970 / 0.963
  recorded above. Both columns fell by the same ~1.93× — the Ω correction, 0.516×
  — so the *ratio* survived and the diagnostic still works.
- **it is not the trapezoid.** The 500 nm grid is the finer of the two (9 points
  against 4), so grid error would go the other way.
- **it is not `sim_for_run`.** With the node states fixed, the workbook and
  run-specific models give *bit-identical* mobile rows; see that fix's commit.
- **what it is, is open.** The leading candidate is the fast-solve cadence: this
  march does 1 FEM solve per dose interval and freezes the mobile field across 3
  substeps, so the snapshot field the measured side integrates is not the field
  the accumulators saw. That predicts the right sign but has not been shown to
  give a factor of 3.6.

So quote the 200 nm number as what it is — one case — and **re-measure before
relying on the boundary channel on any new geometry**. The closure value itself is
unaffected either way: it is the accumulators' own integral.

A surface integral inside MoDELib, where the FE gradients already exist, would
be the gold standard, and this disagreement is the argument for building it. The
Python route needs no C++ change and works on runs that already exist, which is
why it came first.

### The loop Burgers vectors — one value per family, four places

α-Zr's two loop families, and the values every part of the code now uses:

| family | Burgers vector | \|b\| |
|---|---|---:|
| prismatic ⟨a⟩, interstitial | `⅓⟨11̄20⟩` | `a` = **3.23 Å** = 0.323 nm |
| basal ⟨c⟩, vacancy | `½[0001]` | `c/2` = **2.575 Å** = 0.2575 nm |

with `a = 3.23 Å`, `c = 5.15 Å`, hence `c/a = 1.5944272` — **the physical ratio, not
the ideal √(8/3) = 1.6329932**. Ω is `(√3/4)a²c = 2.326553e-29 m³` from the same two
constants (see below); the mass-density route agrees to 0.139%.

**⟨a⟩ was already right everywhere** — `|lat × (1,0,0)| = 1` exactly for all three
variants, `l_a = √(Ω/πb_a)`, and `b_cd = b_dd = 1` in `discrete_loops.FAMILIES`. Which
is why the transfer ledger only ever reported a Burgers discontinuity for ⟨c⟩.

**⟨c⟩ was wrong in three different ways at once**, and the numbers are worth keeping
because they show how far apart the three sides had drifted:

| side | was | using |
|---|---:|---|
| DD (discrete) | 2.6397 Å | ½ × **ideal** c |
| 3-D CD (continuum) | 5.2795 Å | **full** ideal c |
| 0-D | 5.1500 Å | **full** physical c |
| all now | **2.5750 Å** | ½ × physical c |

So the dominant error was the **factor of 2** — only the DD side treated a ⟨c⟩ loop as a
½[0001] loop at all — and the residual 2.4% was ideal-versus-physical `c`.

Four places, which must stay equal:

| where | what |
|---|---|
| `MoDELib3/.../HEXlattice.cpp` | `getLatticeBasis(material)` takes `c/a` from the **optional** material key `c_SI`; absent ⇒ ideal √(8/3), so every other HEX material is unchanged bit-for-bit |
| `Zr3d_ghoniem.txt` | `c_SI = 5.15e-10`, `b_SI = 0.323e-9`, and `immobileSpeciesBurgers` ⟨c⟩ column z = **0.5** (was 1.0 — the full [0001]) |
| `post/discrete_loops.py` | `FAMILIES["c"]`: `b_cd = b_dd = 0.7972136`, `d_plane = 1.5944272` |
| `zerod/input_data.py` | `b_cL = b_c/2` feeds `l_c`. **`b_c` is deliberately NOT reused** — it is also the lattice constant `c` behind Ω, and halving it would halve Ω, the exact error corrected below |

**The ⟨c⟩ Burgers vector derives from the lattice basis and from nowhere else.**
`aLoopGenerator` builds a discrete basal loop's `b` from it, and
`immobileSpeciesBurgers` is multiplied by it to give `immobileSpeciesBurgersMagnitude`,
which `rloop()` sizes every continuum ⟨c⟩ loop with. That is why no input-file edit
could fix the DD side — `write_microstructure` emits plane IDs, radii, side counts and
centres, never a Burgers vector.

`b_SI` moved 3.233 → 3.23 Å so the two codes share one `a`. It is MoDELib's length
unit, so **every staged case re-stages** — automatic, because `domain_key` hashes the
material content. It also removes a 0.28% error in Ω\_b³: the 3-D was converting an Ω
built from `a = 3.23` by `b_SI³` with `a = 3.233`.

**THE ⟨c⟩ PART OF THE FIT IS NOW STALE.** The continuum ⟨c⟩ radius rises ×1.4312 and
the 0-D `l_c` ×1.4142, and sink strength rises with `r`. This is the **third**
stale-fit warning, alongside Ω and `loop_model`, and they compound. What it buys is
`f_burgers = √(b_cd/b_dd) = 1.000000` — no Burgers discontinuity at the
continuum → discrete transfer, where it was 1.414214. Verified on all four families;
the residual ⟨c⟩ sink jump is `1/loopSinkScale = 3.43`, down from 4.85, and that
factor is a separate modelling decision rather than a crystallography error.

**A hard-coded ideal `c/a` in `aLoopGenerator` made the whole change a silent
no-op, and this is the failure mode to remember.** `generateSingle` chose its
branch by comparing the plane spacing against constants — `√3/2` for prismatic,
**`√(8/3)` for basal** — and the basal spacing *is* `c/a`. With `c_SI` present the
spacing became 1.5944272, matched neither branch, and fell off the end of the
`if`/`else if` chain: **every requested basal loop vanished with no message, no
exit code and no empty-output warning**, and `evl_0.txt` was written with zero
loops for the solve to run on. Measured A/B on one case, the two inputs differing
only by the `c_SI` line: 5 basal loops with the key absent, **0** with it present.

Two things were changed, and the second matters more than the first:

- `cOverA` now comes from `grain.singleCrystal->latticeBasis.col(2).norm()`. That
  is `Q·A`'s third column, so it is `c/a` in any crystal orientation, and an
  ideal-ratio material still takes exactly the path it took before.
- the fall-through now **throws**, naming the spacing and both accepted values. A
  requested loop that cannot be built is a failure, not a no-op.

The 0-D consequence of `|b_⟨c⟩| = c/2` is **not** confined to ⟨c⟩, and the reason is
a pre-existing conflation worth knowing about: `l_c` plays two roles in
`loop_model = 0`. It forms the genuine ⟨c⟩ radius (`r_vL = l_c√(CvL_v/CvL)`, and
`r_iL` correctly uses `l_a`), *and* it is the model's single sink prefactor `l_c/l`,
applied to ⟨a⟩ as well through `pref_iL`, `pref_aiL` and `v_abs_a`. Full
self-consistency was chosen deliberately over pinning that prefactor, so at 10 dpa:

```
N_a 0.716x   c_a 0.714x   N_c 0.993x   c_c 0.499x
```

The ⟨a⟩ column moves because of the shared prefactor, not because of anything about
the ⟨c⟩ Burgers vector. `loop_model = 1` has no such conflation — it uses each
family's own `l_k`. No C++ or solver rebuild was needed for this: `l_c` reaches the
C++ as a CLI parameter.

**One hard-coded ideal `c/a` is deliberately left**:
`DislocationMobilityHEXprismatic.cpp:37,62` sets `h = √(8/3)/2`, the kink-pair
height for prismatic *glide*. It is not a Burgers vector and it was fitted at the
ideal ratio; the climb loops here are `SESSILELOOP` and never use it. Revisit it
with the glide kinetics, not with this.

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
