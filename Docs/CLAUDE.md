# DisloCluster Docs — Claude Code Instructions

This file used to describe a two-codebase repository (`ZrMicro` and `Creep`)
with a 12-equation model. Both statements are now wrong: `Creep/` was stripped
when DisloCluster was assembled and does not exist in this checkout, and the
0-D state vector is 19 long (12 physical species, 6 conservation accumulators,
`rho_N`). Rather than keep a third divergent copy of the architecture, the
duplicated material has been removed and this file now holds only what is not
documented better elsewhere.

**Authoritative sources**

| For | Read |
|---|---|
| repository layout, the package, the coupling contract | [`../CLAUDE.md`](../CLAUDE.md) |
| the 0-D model equations, state vector, file map | [`../ZrMicro/CLAUDE.md`](../ZrMicro/CLAUDE.md) |
| running a simulation | [`../Simulations/README.md`](../Simulations/README.md) |
| setup and build | [`../README.md`](../README.md) |
| 3-D side changes vs upstream | [`../MoDELib3/ZR3D_GHONIEM_CHANGES.md`](../MoDELib3/ZR3D_GHONIEM_CHANGES.md) |

Documents themselves live under this `Docs/` tree: `DisloCluster Manual/`
(manual + development notes), `Formulation/` (LaTeX/PDF formulation and
`build_modelib.sh`, the cross-platform MoDELib build script), `Reports/`,
`Presentations/`.

---

## Python environment

- **Venv:** `.DisloClusterVenv/` at the repository root (Python 3.14)
- **Package:** `dislocluster_code`, installed editable with `pip install -e .`
- **Jupyter kernel:** registered as `dislocluster` ("Python 3.14 (DisloCluster)")
  — the kernel keeps its original name; only the Python package was renamed
- **Do not use Anaconda Python** — NumPy 1.x/2.x conflict with SciPy
- **Dependencies:** `requirements.txt` at the repository root

---

## ZrMicro C++ Solver — Integration Method Options

The C++ solver (`ZrMicro/code/solver.cpp`) supports two backends and multiple linear solvers, selectable from the notebook via the optional `solver_method` sub-dict inside `SIMULATION_CONFIG`.

### Usage example

```python
SIMULATION_CONFIG = {
    't_begin': 1e-1,
    't_end':   1e9,
    'n_points': 1000,
    'rtol':    1e-6,
    'atol':    1e-20,
    'log_time': True,

    # Optional — omit entirely to use defaults (CVODE BDF + dense)
    'solver_method': {
        'backend':   'cvode',               # 'cvode' (default) | 'arkode'
        'lmm':       'bdf',                 # CVODE only: 'bdf' (default) | 'adams'
        'linsol':    'dense',               # 'dense' (default) | 'band' | 'gmres'
        'mu':        11,                    # band solver upper bandwidth (default N-1)
        'ml':        11,                    # band solver lower bandwidth (default N-1)
        'max_order': 0,                     # 0 = solver default; BDF→5, Adams→12
        'ark_table': 'ARK548L2SA_DIRK_8_4_5',  # ARKODE only — DIRK table name
    }
}
```

### Backend options

| `backend` | Description |
|---|---|
| `'cvode'` | CVODE linear multistep — BDF or Adams (default) |
| `'arkode'` | ARKODE ARKStep — implicit Runge-Kutta DIRK methods |

### CVODE `lmm` options

| `lmm` | Method | Order | Use for |
|---|---|---|---|
| `'bdf'` | Backward Differentiation Formula | 1–5 | **stiff problems (default, recommended)** |
| `'adams'` | Adams-Moulton | 1–12 | non-stiff problems |

### Linear solver options (both backends)

| `linsol` | Solver | Notes |
|---|---|---|
| `'dense'` | Direct LU factorisation | **default; optimal for N=19** |
| `'band'`  | Banded LU factorisation | set `mu`, `ml`; equivalent to dense when `mu=ml=N-1` |
| `'gmres'` | SPGMR iterative Krylov | matrix-free; less accurate than dense at equal tolerances |

### ARKODE DIRK table options (`ark_table`)

Only used when `backend='arkode'`. All tables are L-stable or A-stable DIRK methods for stiff systems.

| `ark_table` string | Order | Notes |
|---|---|---|
| `'SDIRK_2_1_2'` | 2 | lightweight 2-stage SDIRK |
| `'SDIRK_5_3_4'` | 4 | L-stable 5-stage SDIRK — **best-performing ARKODE option on this problem** |
| `'KVAERNO_7_4_5'` | 5 | 7-stage A-stable SDIRK |
| `'ARK548L2SA_DIRK_8_4_5'` | 5 | 8-stage L-stable (default) |
| `'ESDIRK547L2SA_7_4_5'` | 5 | 7-stage stiffly-accurate ESDIRK |

Integer IDs (ARKODE_DIRKTableID, range 100–126) are also accepted directly as `ark_table`.

### Accuracy benchmark (ZrMicro radiation-damage ODE, rtol=1e-6, atol=1e-20)

Max relative error vs. tight reference (CVODE BDF dense, rtol=1e-10, atol=1e-30):

| Config | Backend | Method/Table | LinSol | Max order | Max rel. error |
|---|---|---|---|---|---|
| A | CVODE | BDF | dense | default | 3.0×10⁻⁴ |
| B | CVODE | BDF | band (full) | default | 3.0×10⁻⁴ |
| C | CVODE | BDF | gmres | default | 1.7×10⁻² |
| D | CVODE | BDF | dense | 3 | 2.2×10⁻⁴ |
| E | CVODE | Adams | dense | default | 3.8×10⁻⁵ |
| F | ARKODE | ARK548L2SA_DIRK_8_4_5 | dense | default | 1.7×10⁴ |
| G | ARKODE | SDIRK_5_3_4 | dense | default | 3.0×10⁻¹ |
| H | ARKODE | KVAERNO_7_4_5 | dense | default | 3.6×10⁷ |
| I | ARKODE | ESDIRK547L2SA_7_4_5 | dense | default | 2.2×10³ |
| J | ARKODE | SDIRK_2_1_2 | dense | default | ❌ failed |

**Recommendation:** Use CVODE BDF + dense (configs A/B) for production runs. ARKODE SDIRK_5_3_4 (config G) is the only ARKODE table that completes with borderline accuracy at these tolerances; ARKODE methods designed as IMEX pairs (ARK, ESDIRK) perform poorly in pure implicit mode on this problem.

### Rebuild after changes

```powershell
cmake -S ZrMicro\cpp_utils -B ZrMicro\build -DCMAKE_BUILD_TYPE=Release
cmake --build ZrMicro\build --config Release
```

Use **Release**: the parameter-identification utilities call the solver
thousands of times, and the OpenMP `--batch_file` mode the coupled march
depends on is only worth having optimized. `dislocluster_code.build.
ensure_zrmicro_solver()` does this for you, and discards a CMake cache left
pointing at a previous location.
