# ZrClusterDynamics Project — Claude Code Instructions

## Project Objectives

This repository contains two coupled physics simulation codebases for modeling irradiated zirconium alloy behavior.

### ZrMicro
Develops a microstructure model for neutron-irradiated zirconium. A system of coupled rate equations evolves the concentrations of point defects and defect clusters (vacancies, interstitials, dislocation loops, and their complexes) under irradiation conditions. The evolved microstructure is then used to derive macroscopic irradiated properties:
- **Irradiation growth** — dimensional change under irradiation at zero stress
- **Irradiation creep** — stress-induced dimensional change under irradiation
- **Void swelling** — volume change from vacancy cluster accumulation
- **Radiation hardening** — increase in yield stress from dislocation loop obstacles

### Creep
Models the thermal (non-irradiation) creep rate of zirconium alloys at arbitrary temperature and stress. Conservation equations for four dislocation density populations are solved — mobile, immobile (static), boundary, and subgrain boundary — along with subgrain radius evolution. From these, the macroscopic creep strain rate as a function of temperature and applied stress is determined.

---

## Codebase 1: ZrMicro

### Location
`ZrMicro/`

### Solution Method
A 12-equation ODE system is integrated forward in time. Two solver backends are available:

- **C++ backend (primary):** `code/solver.cpp` uses **CVODE (SUNDIALS 7.1.1)** with BDF and a dense linear solver. The Python notebook invokes it via `py_utils/cpp_bridge.py`, which passes all pre-computed parameters as CLI `--key=value` arguments and parses the stdout results. This mirrors the Creep codebase architecture exactly.
- **Python backend (fallback):** `py_utils/simulation.py` uses **SciPy's `solve_ivp`** (LSODA). Still functional but retained mainly as a reference/fallback.

The 12-element state vector contains:
- `Cv`, `Ci` — free vacancy and interstitial concentrations
- `C2i`, `C3i` — di- and tri-interstitial clusters
- `CiL`, `CaiL` — interstitials in interstitial loops and alloyed interstitial loops
- `CvL`, `CavL` — vacancies in vacancy loops and alloyed vacancy loops
- Four corresponding defect-in-loop variables (`CiL_i`, `CaiL_i`, `CvL_v`, `CavL_v`)

Rate equations include: production from cascade damage, recombination, annihilation at sinks (grain boundaries, dislocations, loops), loop nucleation and growth, and thermal annealing. All sink concentrations and fluxes are cached per ODE call via `reaction_rates.update_state()`. Post-processing is fully vectorized.

### Module Architecture

| File | Language | Role |
|---|---|---|
| `code/solver.cpp` | C++ | CVODE ODE system; RHS function; CLI arg parsing; stdout output |
| `cpp_utils/parameters.h` | C++ | `Parameters` struct |
| `cpp_utils/rate_equations.cpp/.h` | C++ | C++ RHS implementation |
| `py_utils/cpp_bridge.py` | Python | Collects parameters, launches `solver.exe`, parses output |
| `py_utils/simulation.py` | Python | `ZrMicroSimulation` — init; Python `solve_ivp` fallback path |
| `py_utils/rate_equations.py` | Python | `RateEquations` — Python ODE RHS |
| `py_utils/reaction_rates.py` | Python | `ReactionRates` — all reaction rate methods; per-call caching |
| `py_utils/input_data.py` | Python | `InputData` — reads Excel, computes derived parameters |
| `py_utils/pre_process.py` | Python | Find input file, validate setup |
| `py_utils/post_process.py` | Python | `calculate_derived_quantities` — vectorized post-processing |
| `py_utils/visualization.py` | Python | `ZrMicroVisualizer` — generates all plots |

### Build System
- **CMake:** `ZrMicro/cpp_utils/CMakeLists.txt` — C++17, finds SUNDIALS in `../../Libraries/sundials-7.1.1/`, links `cvode`, `nvecserial`; outputs `build/Debug/solver.exe`
- Build commands:
  ```
  cd ZrMicro/cpp_utils
  cmake -S . -B ../build -DCMAKE_BUILD_TYPE=Debug
  cmake --build ../build --config Debug
  ```

### Input
- **File:** `ZrMicro/input/input_parameters.xlsx`
- **Sheets:**
  - `Material_Environment` — irradiation dose rate, temperature, neutron flux, alloy composition
  - `Physical_Properties` — Burgers vector, atomic volume, diffusion coefficients, recombination factor, etc.
  - `Model_Parameters` — rate equation model constants

### Output
Each run creates a unique directory: `ZrMicro/output/<YYYYMMDD_HHMMSS>_<git-hash>/`

| File | Contents |
|---|---|
| `provenance.md` | All input parameters + software environment + solver stats (markdown tables) |
| `point_defects.png` | Free vacancy and interstitial concentrations [cm⁻³] vs. dose [dpa] |
| `loop_density.png` | Dislocation loop number density [m⁻³] vs. dose |
| `loop_sizes.png` | Mean loop radii vs. dose |
| `irradiation_growth.png` | Growth strain vs. dose |
| `irradiation_creep.png` | Creep strain rate vs. dose |
| `radiation_hardening.png` | Yield stress increment vs. dose |
| `flux_evolution.png` | Defect flux to sinks vs. dose |
| `net_flux_balance.png` | Net vacancy/interstitial flux balance vs. dose |

### Concentration Units
- Internal: dimensionless atomic fraction
- `point_defects.png`: divided by `Omega * 1e6` → cm⁻³
- `loop_density.png`: divided by `Omega` → m⁻³

### Known Physics Notes
- `CvL_v` and `CavL_v` can go slightly negative numerically (clamped in ODE wrapper; harmless)
- `phi_v_emission_avL` uses `e_sigma_v_avL` (not `e_sigma_v_vL`)
- Annealing is applied correctly in both `dCvL_dt` and `dCavL_dt`
- `recom` parameter must be present in `Physical_Properties` sheet; default fallback = 1.0

---

## Codebase 2: Creep

### Location
`Creep/`

### Solution Method
A 5-equation ODE system is solved using **CVODE (SUNDIALS 7.1.1)** via a compiled C++ executable. CVODE uses BDF (Backward Differentiation Formula) for stiff systems with a dense linear solver. The Python notebook launches the C++ solver as a subprocess, passing all material parameters as command-line arguments, and parses the stdout results as a numpy array.

The state vector is:
1. `rho_m` — mobile dislocation density [m⁻²]
2. `rho_s` — static (immobile) dislocation density [m⁻²]
3. `rho_b` — boundary dislocation density [m⁻²]
4. `R_sb` — subgrain radius [m]
5. `strain` — cumulative creep strain [-]

Creep strain rate is computed as `d(strain)/dt = rho_m * b * v_g`, where `v_g` is the dislocation glide velocity (exponential function of effective stress and activation energy). Climb velocities couple temperature-dependent diffusion with jog-pair formation energy.

**Solver settings:** rel. tol = 1e-3, abs. tol = 1e-6, max steps = 100,000.

### Module Architecture

| File | Language | Role |
|---|---|---|
| `code/creep.ipynb` | Python | Main orchestration: read input, launch C++, parse output, plot |
| `code/ode_solver.cpp` | C++ | CVODE ODE system; RHS function; CLI arg parsing; stdout output |
| `cpp_utils/parameters.h` | C++ | `Parameters` struct (21 material params + derived quantities) |
| `py_utils/read_data.py` | Python | Read Excel, parse C++ output, create run directory, save provenance |
| `py_utils/creep_plots.py` | Python | `FigureGenerator` class; produces 5 output PNGs |

### Input
- **File:** `Creep/input/material_data.xlsx` (sheet: `steel`)
- **21 material parameters:** Burgers vector `b`, Young's modulus `E`, Poisson's ratio `nu`, octahedral stress `tau_oct`, diffusion pre-factor `D_o`, activation energies (`E_core`, `E_s`, `E_m`), Boltzmann constant `k`, precipitate density `N_p` and radius `r_p`, atomic volume `Omega`, stacking fault energy `SFE`, and model coefficients (`delta`, `sigma_o`, `W_g`, `a1`, `c_jog`, `K_c`, `Beta`, `xi`, `zeta`)
- **Simulation parameters (set in notebook):** temperature T [K], time domain [t0, tf], number of time evaluation points

**Initial conditions (hardcoded in C++):**
- `rho_m(0)` = 1×10¹⁴ m⁻²
- `rho_s(0)` = `rho_b(0)` = 1×10¹¹ m⁻²
- `R_sb(0)` = 10 μm
- `strain(0)` = 0

### Output
Each run creates a unique directory: `Creep/output/<YYYYMMDD_HHMMSS>_<git-hash>/`

| File | Contents |
|---|---|
| `provenance.md` | Material parameters + simulation config + software environment (markdown tables) |
| `mobile_dislocation_density.png` | `rho_m` [m⁻²] vs. time |
| `static_dislocation_density.png` | `rho_s` [m⁻²] vs. time |
| `boundary_dislocation_density.png` | `rho_b` [m⁻²] vs. time |
| `subgrain_radius.png` | `R_sb` [m] vs. time |
| `strain.png` | Creep strain [%] vs. time [hours] |

### Build System
- **CMake:** `Creep/cpp_utils/CMakeLists.txt` — C++17, finds SUNDIALS in `../../Libraries/sundials-7.1.1/`, links `cvode`, `nvecserial`; outputs `build/Debug/ode_solver.exe`
- **VSCode task:** `Creep/.vscode/tasks.json` — alternative g++ build with manual include/lib paths

---

## Python Environment

- **Venv:** `DisloCluster/.DisloClusterVenv/` (Python 3.14, shared by both codebases)
- **Jupyter kernel:** registered as `dislocluster` ("Python 3.14 (DisloCluster)")
- **Run notebooks:** `.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=dislocluster <notebook.ipynb>`
- **Do not use Anaconda Python** — NumPy 1.x/2.x conflict with SciPy
- **Dependencies:** `requirements.txt` at repo root

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
| `'dense'` | Direct LU factorisation | **default; optimal for N=12** |
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

### Accuracy benchmark (ZrMicro radiation damage ODE, rtol=1e-6, atol=1e-20)

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

```
cd ZrMicro/cpp_utils
cmake -S . -B ../build -DCMAKE_BUILD_TYPE=Debug
cmake --build ../build --config Debug
```
