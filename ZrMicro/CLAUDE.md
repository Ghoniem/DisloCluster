# ZrMicro — Claude Code Instructions

## Objectives

Model the **microstructure evolution of neutron-irradiated zirconium alloys** by solving a coupled system of rate equations for point defects and defect clusters. The evolved microstructure is used to predict macroscopic irradiated properties:

- **Irradiation growth** — dimensional change under irradiation at zero applied stress
- **Irradiation creep** — stress-induced dimensional change under irradiation
- **Void swelling** — volume change from vacancy cluster accumulation
- **Radiation hardening** — yield stress increment from dislocation loop obstacles

The model accounts for both pure Zr loops and alloyed (solute-decorated) loop variants, and includes stress-dependent vacancy emission from dislocation loops.

---

## Physical Model & Equations

### State Vector (12 equations)

```
y[12] = [Cv, Ci, C2i, C3i, CiL, CaiL, CvL, CavL, CiL_i, CaiL_i, CvL_v, CavL_v]
```

| Index | Variable | Description |
|---|---|---|
| 0 | Cv | Free vacancy concentration (atom fraction) |
| 1 | Ci | Free interstitial concentration |
| 2 | C2i | Di-interstitial cluster |
| 3 | C3i | Tri-interstitial cluster |
| 4 | CiL | Interstitial loop number density |
| 5 | CaiL | Alloyed interstitial loop number density |
| 6 | CvL | Vacancy loop number density |
| 7 | CavL | Alloyed vacancy loop number density |
| 8 | CiL_i | Interstitials in interstitial loops |
| 9 | CaiL_i | Interstitials in alloyed interstitial loops |
| 10 | CvL_v | Vacancies in vacancy loops |
| 11 | CavL_v | Vacancies in alloyed vacancy loops |

### Indices 4–11 depend on `loop_model`

`SOLVER['loop_model']` (CLI `--loop_model`, default 0) selects which immobile
formulation the solver integrates. **Only slots 4–11 change; the mobile species,
the six accumulators and `rho_N` are identical in both.**

| slot | `loop_model = 0` (default) | `loop_model = 1` |
|---|---|---|
| 4–7 | `CiL, CaiL, CvL, CavL` | `n_c, n_a1, n_a2, n_a3` |
| 8–11 | `CiL_i, CaiL_i, CvL_v, CavL_v` | `c_c, c_a1, c_a2, c_a3` |

Mode 1 is MoDELib's own CD immobile block, in MoDELib's order — one basal ⟨c⟩
and three prismatic ⟨a⟩ variants, no aligned/non-aligned split — with capture
efficiencies built from the same `p_m` that sets the diffusion tensor:

```
Z_basal(m)     = Z0_m * p_m                            row 0, <c>
Z_prismatic(m) = Z0_m * (p_m + p_m^-2)/2               row 1, <a>
phi_km         = S_k * Dbar_m * Z(row(k),m) * c_m * |m|
S_k            = (l_k/l) * loop_sink_scale_k * sqrt(n_k * c_k)
ydot[4+k]      = nuc_num[k] - ann_num[k] - coal_num[k]
ydot[8+k]      = (gain[k] - loss[k]) + nuc_cont[k] - ann_cont[k] - coal_cont[k]
```

The decisive difference is **resolution, not formula**: every mobile species
carries its own efficiency and its own `Dbar_m`, where the legacy model lumps
`Ci, C2i, C3i` into `omega_i*(Ci + 2*C2i + 3*C3i)` under a single `Z_i_a` — so
it gives the clusters the *monomer* mobility and cannot represent an anisotropy
given to the clusters alone. Parameters arrive as `--dad_p_<m>`, `--dad_Z0_<m>`,
`--loop_sink_scale_<k>`, `--variant_frac_<k>`, emitted by
`zerod/cpp_bridge.py` from the MoDELib material file.

**`loop_model = 0` was bit-identical to the pre-change solver** (`sha256
8bcc7780…` on a 25-point integration) until the two homogeneous-nucleation
channels were corrected — `R_2i_3i` now debits `ydot[2]`, carries content weight
5.0 and feeds the loop-number current, and `2i+2i` feeds it at `0.5*R_2i_2i`
(one loop per event, not per 2i lost). Embryo sizes are now 4i, 4i and 5i;
the objective moves 3.2e-7 relative, and the root
[`CLAUDE.md`](../CLAUDE.md#two-formulations-of-the-slow-step--solverloop_model)
carries the numbers and the `2i+2i` counterpart that was deliberately left
alone. Its `ydot` assembly and paired
accumulator sums in `rate_equations_core.h` are kept **verbatim** inside
`if (P.loop_model == 0)` for that reason: regrouping `growth + nuc + G` into
`growth + (nuc + G)` is algebraically identical and moves the 10th significant
digit, which was measured. The fit was made against the original association —
do not tidy that branch. Full discussion in the root
[`CLAUDE.md`](../CLAUDE.md#two-formulations-of-the-slow-step--solverloop_model).

### Rate Equations (key physics terms)

**Free vacancy:**
```
dCv/dt = G_v - α·Cv·Ci - ω_v·[Z_N·ρ_N·Cv + Z_iL·CiL·Cv + Z_vL·CvL·Cv + ...]
         + emission_from_loops - capture_by_loops
```

**Free interstitial:**
```
dCi/dt = G_i - α·Cv·Ci - ω_i·[Z_N·ρ_N·Ci + Z_iL·CiL·Ci + ...]
         - K_2i·Ci² (nucleation into di-clusters)
```

**Loop number densities (CiL, CaiL, CvL, CavL):**
```
dCiL/dt = K_nuc_i (loop nucleation from C3i) - annealing · CiL
dCvL/dt = G_vL (cascade-produced loops) - annealing · CvL - emission_terms
```

**Defects in loops (CiL_i, etc.):**
```
dCiL_i/dt = ω_i·Z_iL·Ci·CiL - ω_v·Z_iL·Cv·CiL - annealing·CiL_i + emission_iL
```

### Stress-Dependent Vacancy Emission

Applied stress σ modifies vacancy emission from loops via activation volume ΔV:

```
Γ_v(σ, loop) = Γ_v(0, loop) · exp(±ΔV · σ / kT)
```

Sign depends on loop orientation (a-loops, c-loops) and stress direction.

### Derived Macroscopic Properties

**Irradiation growth strain:**
```
ε_growth = (1/3) · [Σ(f_a · CiL_i) - Σ(f_a · CvL_v)] · b_a / a + c-component
```

**Irradiation creep rate:**
```
dε_creep/dt = Φ · σ · B_0  (creep compliance × dose rate × stress)
```

**Radiation hardening (dispersed barrier model):**
```
Δτ = M · α_obs · μ · b · √(N_loop · d_loop)
```

### Material Parameters

Loaded from `input/Zr_input_parameters.xlsx`, three sheets:

**The workbook alone is NOT the calibrated model** — 28 of its parameters have
drifted from the fitted set and 11 are absent entirely, which moves N_a by five
orders of magnitude. Always build the chain through
`dislocluster_code.zerod.calibration.build_sim()`, which applies `REFERENCE_OVERRIDES`
on top of it.

| Sheet | Contents |
|---|---|
| `Material_Environment` | Dose rate G [dpa/s], temperature T [K], neutron flux |
| `Physical_Properties` | Burgers vectors b_a/b_c, atomic volume Ω, Z_N/Z_iL/Z_vL bias factors, recombination factor, stacking fault energy |
| `Model_Parameters` | ODE tolerances rtol/atol, loop fraction f_a |

---

## Solution Algorithm

1. **Input:** `InputData` reads the 3-sheet Excel file, computes derived parameters:
   - Jump frequencies: `ω_v = ν_v · exp(-E_m_v / kT)`, `ω_i = ν_i · exp(-E_m_i / kT)`
   - Equilibrium concentrations: `Cv_eq = exp(-E_f_v / kT)`
   - Length scales: `l`, `l_a`, `l_c` from dislocation geometry
   - Stress-dependent emission factors

2. **Rate constants:** `ReactionRates.update_state(concentrations, time)` caches sink concentrations `(C_v_s, C_i_s)` at each ODE call; no expensive recomputation inside the RHS

3. **ODE integration:**
   - **C++ backend (primary):** `cpp_bridge.py` collects all parameters → `solver.exe` → CVODE BDF + dense linear solver → parse stdout
   - **Python fallback:** `simulation.py` uses `scipy.integrate.solve_ivp(method='LSODA')`

4. **Post-processing:** `post_process.calculate_derived_quantities()` performs vectorized NumPy transformations:
   - Converts concentrations (atom fraction) to physical units (m⁻³, cm⁻³)
   - Computes loop radii, loop densities, growth/creep strains, hardening

5. **Output:** `visualization.py` writes 8 PNG figures + `provenance.md` to a timestamped run directory

### Solver Configuration Example

```python
SIMULATION_CONFIG = {
    't_begin':  1e-1,        # s
    't_end':    1e9,         # s
    'n_points': 1000,
    'rtol':     1e-6,
    'atol':     1e-20,
    'log_time': True,
    # Optional (defaults to CVODE BDF + dense)
    'solver_method': {
        'backend': 'cvode',
        'lmm':     'bdf',
        'linsol':  'dense',
    }
}
```

### Solver Backend Options

| `backend` | `lmm` | `linsol` | Notes |
|---|---|---|---|
| `'cvode'` | `'bdf'` | `'dense'` | **Default, recommended** |
| `'cvode'` | `'bdf'` | `'band'` | Set `mu`, `ml` bandwidths |
| `'cvode'` | `'bdf'` | `'gmres'` | Matrix-free; less accurate |
| `'arkode'` | — | `'dense'` | Use `ark_table='SDIRK_5_3_4'` only |

---

## File Map

```
ZrMicro/
├── code/
│   ├── ZrMicro.ipynb             Main orchestration notebook: init, run, post-process, plot
│   └── solver.cpp                C++ entry point: integrate_one() + single-case CLI
│                                  and OpenMP --batch_file mode; CVODE/ARKODE setup, stdout
│
├── cpp_utils/
│   ├── CMakeLists.txt            CMake build: C++17, SUNDIALS 7.1.1, links cvode/nvecserial/band/gmres
│   ├── parameters.h              Parameters struct: ω_i/ω_v/e_v/e_2i/e_3i, bias factors,
│   │                             stress-dependent emissions, generation rates, y0[12], solver settings
│   ├── rate_equations.h          C++ RHS declarations
│   └── rate_equations.cpp        C++ 12-equation ODE RHS (mirrors Python _rhs_full)
│
├── py_utils/                 COMPATIBILITY SHIM ONLY. Every former module is
│                             an alias for its new home in the repository-root
│                             `dislocluster_code/` package:
│                               input_data, reaction_rates, rate_equations,
│                               calibration, cpp_bridge, pre_process,
│                               post_process, simulation  -> dislocluster_code.zerod
│                               setup_domain, setup_standalone      -> .staging
│                               run_coupled_vs_standalone   -> .coupling.march
│                               modelib_qssa/_field/_coupling      -> .coupling
│                               modelib_report/_fields/_gb, movies,
│                               discrete_loops, tem_slices,
│                               visualization                          -> .post
│                             See ../../CLAUDE.md for the full table.
│
├── input/
│   ├── Zr_input_parameters.xlsx  THE workbook actually read: 3 sheets
│   │                             (Material_Environment, Physical_Properties,
│   │                             Model_Parameters). `input_parameters.xlsx` is
│   │                             tried first, does not exist, and
│   │                             calibration.default_input_file() therefore
│   │                             always falls back to this one.
│   └── Zr_input_parameters_{New,old}.xlsx   older revisions, unused
│
├── build/                        CMake artifacts
└── output/                       Timestamped run directories (<YYYYMMDD_HHMMSS_git-hash>/)
```

### File Relationships

```
ZrMicro.ipynb
  ├── InputData (Zr_input_parameters.xlsx, + calibration.REFERENCE_OVERRIDES)
  │     └── derives: ω_i, ω_v, Cv_eq, l, l_a, l_c, stress emissions
  ├── ReactionRates (input_data)
  │     └── update_state() called on each ODE step
  ├── RateEquations (input_data, reaction_rates)
  │     └── ode_system(t, y) → _rhs_full(y)
  ├── ZrMicroSimulation (input_data, reaction_rates, rate_equations)
  │     ├── run_with_cpp_solver()
  │     │     └── cpp_bridge.py → solver.exe → parse stdout
  │     └── run_with_python_solver()
  │           └── solve_ivp(LSODA)
  └── post_process → ZrMicroVisualizer → output/<run_dir>/
```

### Output Files (per run)

| File | Contents |
|---|---|
| `provenance.md` | All input parameters + software environment + solver stats |
| `point_defects.png` | Cv, Ci [cm⁻³] vs dose [dpa] |
| `loop_density.png` | Loop number density [m⁻³] vs dose |
| `loop_sizes.png` | Mean loop radii vs dose |
| `irradiation_growth.png` | Growth strain vs dose |
| `irradiation_creep.png` | Creep strain rate vs dose |
| `radiation_hardening.png` | Yield stress increment vs dose |
| `flux_evolution.png` | Defect flux to sinks vs dose |
| `net_flux_balance.png` | Net vacancy/interstitial flux balance vs dose |

### Concentration Unit Conversions

- Internal ODE units: dimensionless atom fraction
- `point_defects.png`: divided by `Omega * 1e6` → cm⁻³
- `loop_density.png`: divided by `Omega` → m⁻³

---

## Build

```bash
cd ZrMicro/cpp_utils
cmake -S . -B ../build -DCMAKE_BUILD_TYPE=Release
cmake --build ../build --config Release
```

Use **Release** for the parameter fit (the optimiser calls the solver thousands
of times). CMake auto-detects OpenMP; if found, the parallel batch mode below is
enabled (the build still works serially without it).

### Parallel batch mode (`--batch_file`)

`solver.cpp` has two invocation modes, both calling the shared `integrate_one()`:

- **Single case** (default): one integration, parameters as `--key=value` CLI
  args. Unchanged interface; used by the main run cell and the optimised re-run.
- **Batch** (`solver.exe --batch_file=<path>`): the file holds one case per line
  (same `key=value` tokens). All cases are integrated **concurrently with
  OpenMP** — each on its own `SUNContext` / vectors / integrator memory (SUNDIALS
  objects are never shared across threads) — and written back in input order,
  each block prefixed by `=== CASE <i> status=<s> ===`.

The fit objective uses batch mode to solve **all ~16 temperatures of one
parameter set in a single subprocess** instead of one subprocess per
temperature. This amortises the (Windows-dominant) process-spawn + DLL-load cost
and parallelises the integrations — roughly a 6× speedup per objective
evaluation. Bridge: `cpp_bridge.run_cpp_solver_batch()`; notebook driver:
`model_history_batch()`. Results are bit-identical to the per-case path.

---

## Known Physics Notes

- `CvL_v` and `CavL_v` can go slightly negative numerically — clamped in ODE wrapper; harmless
- `phi_v_emission_avL` uses `e_sigma_v_avL` (not `e_sigma_v_vL`)
- Annealing applied correctly in both `dCvL_dt` and `dCavL_dt`
- `recom` parameter must be present in `Physical_Properties` sheet; default fallback = 1.0
