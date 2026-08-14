# Simulations

Notebooks that set up and run DisloCluster simulations.

| Notebook | What it does |
|---|---|
| [`run_simulation.ipynb`](run_simulation.ipynb) | Set the material, geometry, mesh, boundary conditions and coupling parameters, then run the operator-split march to completion. |
| [`postprocess.ipynb`](postprocess.ipynb) | Re-render figures, movies, discrete loops and TEM slices from a run that already exists. Never re-solves. |

Open either with the **Python 3.14 (DisloCluster)** kernel.

§1 resolves every path from the repository marker and §2 calls
`build.preflight()`, which builds either C++ code if its binary is absent,
discards a CMake cache left pointing at a previous location, and runs both
binaries to confirm they load. Nothing depends on the machine the repository
was last used on, and no build tree is committed — a fresh clone builds its
own.

---

## The short version

Everything you edit is in §2 of `run_simulation.ipynb`, as six dicts:

| Dict | Controls |
|---|---|
| `MATERIAL` | material file, temperature, dose rate, 0-D parameter overrides |
| `GEOMETRY` | cubic or hexagonal, size |
| `MESH` | element size and order, boundary-layer refinement |
| `BOUNDARY` | periodic faces (else Dirichlet), applied stress and strain |
| `COUPLING` | route, seed dose, snapshot doses, substeps, fast-solve cadence |
| `SOLVER`, `OUTPUT` | tolerances; what to render, checkpoint and resume |

Then run the cells in order. The configuration is validated in §3 — before
anything expensive — so a dose grid that does not increase, a boundary layer
thicker than the domain or a mistyped key fails immediately rather than three
hours in.

---

## Resuming

§5 is the long cell. It checkpoints after every substep (~57 ms against a fast
solve of a few minutes), so **if the kernel dies or you interrupt it, just run
the cell again** — it continues from the last completed substep.

The checkpoint records what it was written for: the mesh's CD node set, the
seed, the material, the resolved 0-D parameter set, and the march settings. If
any of those changed, resuming raises `CheckpointMismatch` rather than silently
mixing two configurations. To start over deliberately, set
`OUTPUT['resume'] = 'never'`.

A resumed march re-enters at exactly the state it left. It is not guaranteed
*bit*-identical to an uninterrupted run afterwards, because DDomp and the CVODE
batch are both OpenMP-parallel; pin `OMP_NUM_THREADS` if you need that.

---

## Where things go

| Path | Contents |
|---|---|
| `Simulations/input/` | the Excel workbooks (`paths.INPUT_DIR`) |
| `Simulations/output/<stamp>_<hash>_<tag>/` | one run: `config.json`, `march_state.npz`, `summary.json`, `checkpoint/`, `evl_coupled/`, and the figure sets |
| `MoDELib3/tutorials/<case>/` | the staged MoDELib case: mesh, input files, CD node set, seed. Reused across runs that share a domain — see `domain.json`. Relocate with `DISLOCLUSTER_SIM_ROOT`. |
| `ZrMicro/output/` | runs made before `Simulations/output/` existed. Still searched — see below. |

Run directories are gitignored; only this README and `output/README.md` are
tracked. Everything a run needs to be reproduced is in its `config.json`.

**Finding runs.** New runs are written to `Simulations/output/`, but the older
ones were left under `ZrMicro/output/`, so use

```python
paths.find_runs()     # every run, both roots, oldest first
paths.latest_run()    # the newest
```

rather than listing `paths.OUTPUT_DIR` — that would miss them.
`postprocess.ipynb` already does this.

**The workbooks were copied, not moved.** `ZrMicro/input/` still holds its own
set and only `Simulations/input/` is read. `paths.describe()` hashes both and
warns when they stop agreeing, because a workbook silently diverging from the
model that reads it is exactly how the 0-D calibration ended up five orders of
magnitude off in N_a.

Changing only the dose grid does **not** re-mesh: the staged case is keyed on
the geometry, mesh, boundary conditions, material and temperature, so a new
dose schedule reuses the existing domain and its bootstrap.

---

## Two things to read carefully in the output

- **Near-boundary loop densities are not quantitative.** Cascade nucleation is
  spatially uniform, but the only loop-removal channel is coalescence, driven by
  the absorbed mobile flux — which vanishes where Dirichlet pins the mobile
  concentrations. The boundary is a sink for mobile defects but *not* for loops,
  so ⟨a⟩ density climbs steeply toward it and keeps growing in dose.
- **Never quote a whole-domain mean of a loop quantity.** On a Dirichlet cube
  the surface nodes carry almost all of the total ⟨a⟩ density, so a domain mean
  measures the boundary shell rather than the material. Quote the interior mean;
  `report.md` does.

---

## Running headless

```powershell
.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute `
    --ExecutePreprocessor.kernel_name=dislocluster `
    --ExecutePreprocessor.timeout=-1 `
    Simulations\run_simulation.ipynb
```

Set `PYTHONIOENCODING=utf-8` first on Windows: several modules print non-ASCII
status characters, and the default console code page cannot encode them.
