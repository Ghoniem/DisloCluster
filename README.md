# DisloCluster

**Coupled cluster dynamics and dislocation dynamics for irradiated zirconium.**

DisloCluster is a physics-based simulation suite for the spatially-resolved evolution of
radiation-induced defect populations — free vacancies and self-interstitial atoms (SIAs),
small SIA clusters, and the two dislocation-loop families of the hcp lattice: prismatic
⟨a⟩ loops on $\{10\bar{1}0\}$ and basal ⟨c⟩ loops on $(0001)$ — in irradiated
α-zirconium, together with the mechanical consequences of that microstructure. The
reference host material is **α-Zr** at reactor-relevant temperature and dose rate, the
cladding material of water-cooled fission reactors, whose irradiation growth is
controlled by the co-evolution of those two loop families.

The distinguishing feature of the code is that the microstructure is not evolved by one
solver at one scale. A **0-D reduced cluster-dynamics model** and a **3-D
spatially-resolved cluster-dynamics / dislocation-dynamics code** are coupled by an
operator-split quasi-steady-state (QSSA) scheme: the fast mobile species are solved as a
*steady* finite-element field over the crystal, and the slow immobile loop population is
marched in dose independently at every quadrature point. The two exchange full per-node
fields in both directions, not scalars. Because the fast problem carries no time
derivative and the slow problem carries no spatial coupling, ~10⁶ stiff ODEs per substep
become tractable — see [§2.3](#23-why-a-million-stiff-odes-are-tractable), the
load-bearing design point of the scheme.

A second structural feature: **there is only one immobile model.** The 0-D right-hand
side and the march's slow step are the same templated C++ function with the same
switches, so a parameter set fitted in 0-D is one the 3-D code runs term for term — not a
reimplementation that must be kept in agreement.

- **Languages** — Python (orchestration, staging, coupling, post-processing), C++17
  (0-D and immobile ODE solver, SUNDIALS CVODES), C++20 (MoDELib3 — finite-element
  cluster dynamics and discrete dislocation dynamics).
- **License** — MIT for DisloCluster's own code; the vendored `MoDELib3/` is third-party,
  see [§13](#13-license-and-contact).
- **Primary reference** — N. M. Ghoniem, G. Po, M. Maron, R. Escobar,
  B. Ramirez Flores, K. Baker and T. Black, *A Self-Consistent Spatially-Resolved
  Cluster Dynamics Model for Irradiated Zirconium*, Journal of Nuclear Materials (in
  preparation);
  [`Docs/Formulation/self-consistent/SC_manuscript/`](Docs/Formulation/self-consistent/SC_manuscript/).
- **Working reference** — [`CLAUDE.md`](CLAUDE.md) at the repository root: every measured
  number, corrected claim and known limitation, with the case it was measured on. This
  README summarizes it.

---

## Contents

1. [Capabilities](#1-capabilities)
2. [Methodology](#2-methodology)
3. [Repository structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [Inputs](#6-inputs)
7. [Outputs and provenance](#7-outputs-and-provenance)
8. [Examples, tests and verification](#8-examples-tests-and-verification)
9. [Computational aspects](#9-computational-aspects)
10. [Documentation](#10-documentation)
11. [Publications and how to cite](#11-publications-and-how-to-cite)
12. [Repository conventions](#12-repository-conventions)
13. [License and contact](#13-license-and-contact)

---

## 1. Capabilities

| Capability | Where formulated | Notes |
|---|---|---|
| Operator-split QSSA coupling | §2.2; `coupling/march.py` | steady mobile FE field alternating with a pointwise immobile dose march; splitting error first order in the fast-solve spacing |
| Per-node field exchange, both directions | §2.4 | the immobile state enters the FE problem at every quadrature point; the steady mobile field returns at every CD node |
| 0-D reduced cluster dynamics | `ZrMicro/`, `zerod/` | 19–38 ODEs: 4 mobile species, 4–9 loop families with number and content, 6 conservation accumulators, network density $\rho_N$ |
| 3-D spatially-resolved cluster dynamics | `MoDELib3/` | second-order FE mobile solve by superposition, $c = c_{\rm FEM} + c_{\rm DD}$ |
| Discrete dislocation dynamics | `MoDELib3/` | sessile prismatic and basal loops, glissile sources, periodic cells |
| Two immobile formulations | §2.5 | *legacy* aligned/non-aligned families with phenomenological capture, or *self-consistent* ⟨c⟩ + 3×⟨a⟩ with Woo efficiencies from the diffusion tensor |
| Anisotropic diffusion (DAD) | §2.6; `staging/anisotropy.py` | one $p_m$ per species generates **both** the migration-energy tensor and the capture efficiencies, split at fixed $D_{\rm eff}$ |
| ⟨a⟩ / ⟨c⟩ co-growth criterion | §2.6 | simultaneous growth possible iff $p_I < p_v$; reduces to bounds on a single arrival ratio |
| Five model extensions behind switches | §2.5 | vacancy ⟨a⟩ variants, character splitting $\chi$, peripheral emission, the basal stacking-fault-pyramid chain, second content moments with a log-normal closure |
| Continuum → discrete loop handoff | §2.7; `coupling/transition.py` | dose-triggered conversion of the mean-field population to discrete loops, with a conservation ledger |
| Coarsening detector | `post/coarsening.py` | $d_{\rm coarsen}$ — the dose at which the mean-field treatment of coalescence stops being valid |
| Irradiation hardening | §2.8; `studies/hardening.py` | rebuilds a march's loops in a periodic cube and measures $\Delta\tau$(dose) by stress ramp, or by a dispersed-barrier law whose $\alpha$ is fitted on it |
| Cubic and hexagonal domains | `Gmsh/generate_mesh.py` | boundary-layer-refined meshes; every figure and volume keyed on the convex hull of the node cloud, not a bounding box |
| Real boundary conditions | §6.2 | periodic face sets and applied stress/strain, converted to MoDELib's normalization on staging |
| Conservation as an open-system balance | §2.9 | six accumulators plus an independently measured grain-boundary channel |
| Parameter identification | `fitting/` | 28-parameter 0-D fit against 78 experimental points over 13 temperatures; one objective evaluation is 0.2–0.5 s |
| TEM-comparable micrographs | `post/tem_slices.py` | four diffraction conditions; a loop of habit normal $n$ viewed along $B$ projects to an ellipse of axis ratio $\lvert n\cdot B\rvert$ |
| Provenance-stamped output | §7 | timestamped directory with git SHA, resolved controls, state array, figures, report |
| Checkpoint and resume | §5.1 | after every substep (~57 ms against a fast solve of minutes); a fingerprint mismatch refuses rather than mixes configurations |

**Known-stale calibration.** Three corrections since the 28-parameter fit was made
compound and are **not** yet refitted: the atomic volume Ω (was $V_{\rm cell}/4$, a
factor 1.94), the ⟨c⟩ Burgers vector ($\lvert b\rvert = c/2$, not $c$), and the
self-consistent loop model now being the default. Loop densities move by ~2× under the
first alone. **Do not compare against experiment before the refit** —
[§8.4](#84-what-is-not-calibrated).

---

## 2. Methodology

### 2.1 The two models

**0-D (reduced).** Spatially uniform rate equations. In the legacy layout the state is

$$y = [\,C_v,\ C_i,\ C_{2i},\ C_{3i},\ \underbrace{C_{iL},\ C_{aiL},\ C_{vL},\ C_{avL}}_{\text{loop number}},\ \underbrace{C_{iL}^{i},\ C_{aiL}^{i},\ C_{vL}^{v},\ C_{avL}^{v}}_{\text{loop content}},\ \underbrace{\cdots}_{\text{6 accumulators}},\ \rho_N\,]$$

19 components. Concentrations are atomic fractions, loop content is defects per atom,
energies in eV, temperature in K. Mean loop size is content/number and radius is
$r_k = l_k\sqrt{c_k/n_k}$ with each family's own $l_k = \sqrt{\Omega/\pi b_k}$.

**3-D (spatially resolved).** The same species as fields. The mobile species obey a
steady reaction–diffusion problem solved by second-order finite elements, with
anisotropic transport through a tensor $D_m$ whose basal and axial components differ; the
immobile species obey the 0-D equations *at every point*, with the local mobile values as
coefficients.

### 2.2 The operator split

One substep is

1. **Fast solve** — the steady mobile field $C_M^\*(x)$ for the immobile state currently
   held. This is MoDELib3's `solveMobileClusters`, which has **no time derivative at
   all**; the immobile population enters only through `ImmobileSinks`, evaluated at every
   quadrature point.
2. **Slow march** — the immobile ODEs integrated independently at every point with the
   mobile species *frozen* at $C_M^\*$.

Splitting error is first order in the interval between fast solves (`fem_every`
substeps), **not** in the dose-snapshot spacing, so `fem_every` is the accuracy dial.

**The fast solve belongs inside the loop.** Replay the mobile field from a recorded run
and the scheme is not an operator split: $C_M$ never responds to the immobile state the
march is building, so a seed away from quasi-steady state can never relax. A 0-D seed is
always such a seed — spatially uniform, with no boundary layer, where the true
$C_M^\*(x)$ is pinned to thermal equilibrium on every Dirichlet face.

### 2.3 Why a million stiff ODEs are tractable

A tightly coupled million-equation stiff system *would* be hopeless. This is not one.

On the 500 nm case — 90 617 CD nodes, 1 087 404 immobile ODEs per substep — the slow step
is **72 494 independent 15-dimensional systems** (after deduplicating identical states,
×1.25 here), not one 1 087 404-dimensional system. No neighbor state enters a point's
right-hand side. That decoupling is legitimate rather than a cheat because **the only
thing coupling neighboring points is diffusion of the mobile species**, which the slow
step freezes. For one implicit factorization:

| | flops |
|---|---:|
| split: $72\,494 \times 15^3$ | $2.4\times10^{8}$ |
| coupled: $(1\,359\,255)^3$ | $2.5\times10^{18}$ |
| **ratio** | $\mathbf{1.0\times10^{10}}$ |

Sparsity softens the coupled figure without rescuing it. Stiffness localizes the same
way: each point's system genuinely is stiff (~100 CVODE internal steps per point per
substep, which is why BDF), but stiffness cost scales with the dimension of the **coupled
block**, not with how many independent blocks there are.

**The coupling did not vanish — it moved to where it is affordable.** The tightly coupled
problem is the fast solve: 362 468 unknowns, **925 s** measured, against 28.3 s for an
entire slow sweep over 90 617 points. What the split buys is *frequency* — because
`solveMobileClusters` has no time derivative, the coupled problem is solved **8 times over
the march** instead of at every timestep. Integrating the mobile block in time instead
would need ~480 factorizations at 185 s each, **~25 h** for the 4 mobile species alone,
against a measured 11.3 min for the whole slow side over all 24 substeps.

### 2.4 The exchange

| Direction | Carrier | Module |
|---|---|---|
| 0-D → 3-D | per-node immobile field → the `evl` CD block | `coupling/field.py` |
| 3-D → 0-D | steady mobile field $C_M^\*(x)$ at the CD nodes | `coupling/qssa.py` |
| 0-D → 3-D | dose-indexed closure table | `coupling/export.py` |
| both | material constants | `MoDELib3/Library/Materials/Zr3d_ghoniem.txt` |

The state array and the CD block are ordered differently — the state appends families,
the CD block is moment-major (all numbers, then all contents, then all second moments) —
so the bridge is a scatter, not a copy. `coupling/field.immobile_into_state` is the one
place that knows the mapping.

**The mobile field MoDELib writes is the corrective part, not the total.** MoDELib solves
by superposition, $c = c_{\rm FEM} + c_{\rm DD}$, where $c_{\rm DD}$ is the analytic
Green's-function field of the discrete segments and enters only through the Dirichlet
values. The CD block is $c_{\rm FEM}$, so a figure drawn from it is smooth by
construction however many discrete loops exist; the optional key
`outputSuperposedMobile` makes MoDELib also write the total.

### 2.5 Two formulations of the slow step, and five extensions

The slow step's *integrator* is chosen by route; its **model** by `SOLVER['loop_model']`,
and the two are orthogonal.

| | `0` — legacy | `1` — self-consistent (**default**) |
|---|---|---|
| families | $iL, aiL, vL, avL$ (aligned / non-aligned) | $c, a_1, a_2, a_3$ — MoDELib's CD block |
| capture | phenomenological $Z_i^a = 1+\delta_i$, one $Z$ for all interstitial species | Woo, from the same $p_m$ as the tensor, **one $Z$ per species** |
| bridge to 3-D | a lumping plus a split, both lossy | an **identity** (only $1/\Omega$); round trip exact |
| status | the **fitted** model | consistent with the fast solve, **not calibrated** |

In mode 1,

$$Z_{\rm basal}(m) = Z^0_m\,p_m, \qquad
  Z_{\rm prismatic}(m) = Z^0_m\,\frac{p_m + p_m^{-2}}{2},$$

$$\phi_{km} = S_k\,\bar{D}_m\,Z(\text{row}(k),m)\,c_m\,\lvert m\rvert, \qquad
  S_k = \frac{l_k}{l}\,s_k\sqrt{n_k c_k},$$

identical to MoDELib's own `loopDADbias` and `ImmobileSinks`, with $l_k$ each family's
**own** radius scale — ⟨c⟩ sized with $b_c$ and ⟨a⟩ with $b_a$, where the legacy model
sizes both with $l_c$.

Five further switches each default to the value that reproduces the formulation before
its step, and each is **rejected without** `loop_model = 1`:

| key | 0 / default | 1 (or more) |
|---|---|---|
| `n_fam` | 4 | 8 adds the three prismatic **vacancy** variants; 9 adds the stacking-fault pyramid $c_0$ |
| `chi` | 1.0 | character splitting $X_{iI}X_{vV}/(X_{iV}X_{vI})$; the coexistence window has log-width $\ln\chi$ |
| `emission_model` | 0 | peripheral emission against each family's own $c^{v,\rm eq}$, replacing two fitted annealing lifetimes |
| `basal_chain` | 0 | $c_0 \to c_f \to c_p$; needs `n_fam = 9` |
| `moments` | 0 | the second content moment $q_k$ per family and a log-normal closure on it |

State width follows the switches — 19 / 29 / 38 — and the checkpoint fingerprint records
the **actual** width, because the solver defaults every appended slot to zero and a
too-narrow state would otherwise be accepted silently.

With moments on, $\Delta_k = q_k n_k / c_k^2 \ge 1$ by Cauchy–Schwarz, so $\Delta < 1$ is
a defect, not a loose tolerance. Nucleation deposits at its declared size, the floor
current at $m_{\rm min}$, and **annealing and coalescence are shape-preserving** — a
coalescing loop merges rather than leaving the family, so coalescence *raises* $q$.
Debiting it the natural-looking way inverts the sign on the largest term in the equation.

### 2.6 Anisotropic diffusion, and which anisotropy to use

The anisotropy difference appears twice — as the migration-energy tensor the fast solve
diffuses with, and as the $p_m$ the closed-form capture efficiencies use. These were once
independent, so the code simultaneously believed diffusion was isotropic and that the
sinks were biased by its anisotropy. `staging/anisotropy.py` now generates **both from one
$p_m$ per species**, split at fixed $D_{\rm eff}$ so a change alters directionality and
not overall mobility:

$$E_m^{\langle 11,22\rangle} = E_m^{\rm eff} + 2k_BT\ln p_m, \qquad
  E_m^{\langle 33\rangle} = E_m^{\rm eff} - 4k_BT\ln p_m.$$

**Simultaneous ⟨a⟩ and ⟨c⟩ growth is possible iff $p_I < p_v$.** Both growth conditions
reduce to bounds on one number — the arrival ratio
$A/B = \bar{D}_v c_v / \sum_m \bar{D}_m c_m \lvert m\rvert$ — and the window between them
has width $g(p_v)/g(p_I)$, $g(p) = 2/(1+p^{-3})$, strictly increasing. The measured
co-growth set, 100% of interior nodes at every dose 0.01–10 dpa, is

```
p_m = (p_v, p_i, p_2i, p_3i) = (1.000000, 0.913720, 0.913720, 0.913720)
```

**vacancies isotropic**, interstitials at the already-fitted value, tolerance
$p_v \in [0.94, 1.07]$.

Three limits before quoting this. The criterion is derived from the 3-D *immobile* solver,
which **the coupled march does not run** — there the diffusion tensor reaches loop growth
only through the mobile concentrations, unless `loop_model = 1` closes the gap. It governs
the **absorption flux only**, and cascade nucleation is added separately. And an isotropic
march is **not a zero** of the criterion but another parameter point, so a ratio taken
against one cannot test a statement about signs.

### 2.7 The continuum → discrete handoff

`post/coarsening.py` decides **when** the mean-field treatment of coalescence stops being
valid ($d_{\rm coarsen}$, where the Avrami overlap crosses `phi_star`);
`coupling/transition.py` performs the switch; `coupling/neighbors.py` supplies the
screened cutoff that makes the resulting climb solve affordable. Off by default.

| invariant across a transfer | status |
|---|---|
| loop number | exact to the integer draw |
| stored defects | exact to $10^{-16}$ |
| sink strength | continuous to ~2%; reported, never asserted on |

Two things the transfer must respect, both measured. MoDELib's `microstructureGenerator`
**silently refuses loops whose nodes leave the grain**, so `transition.fits_in_crystal`
predicts that decision exactly and keeps the refused share in the continuum — counted in
**defects, not loops**, since one refusal of six was 83% by count and 98% by defects. And
the domain must be able to hold the loops: on a 500 nm domain the ⟨c⟩ kept fraction falls
to 0.01–0.02 above 1 dpa, making the ⟨c⟩ handoff effectively infeasible there. Check it
before trusting a transfer on any geometry.

### 2.8 Irradiation hardening

`studies/hardening.py` takes the continuum immobile field a march produced, rebuilds it as
discrete loops in a **periodic cube**, loads that cube and measures the CRSS increment
$\Delta\tau$(dose).

| Route | What it does | Cost |
|---|---|---|
| **A** | stress ramp on a frozen obstacle field; gives $\Delta\tau$ directly | affordable at a few doses |
| **C** | dispersed-barrier law $\Delta\tau_k = \alpha_k\mu b\sqrt{N_k d_k}$, $\alpha$ **fitted on route A** | free at every dose |
| **D** | seeds the cell by density instead of per-loop export | smoke test |
| B | strain rate, the full $\sigma$–$\varepsilon$ curve | days per dose; deliberately not staged |

**The loops are frozen because they are sessile and climb is off**, so there is no
absorption and no channeling. That is the largest modelling assumption in the measurement
and it biases hardening high as strain accumulates. $\alpha$ and the Taylor factor $M$ are
the two quantities the module does **not** compute.

### 2.9 Conservation is an open-system balance

The six accumulators integrate a 0-D atom balance with **no transport term**,

$$\frac{d(\text{stored})}{dt} = \text{production} - \text{recombination} - \text{sink absorption},$$

which on a spatially-resolved march cannot close. The residual is not solver error — it is
the atoms that left through the Dirichlet surface. The magnitude settles it: the 200 nm
march reports **−71% of cumulative production**, and nothing in a CVODE run at
$r_{\rm tol}=10^{-6}$ is 71% wrong. Post-processing therefore publishes `grain_boundary`
as a physical channel wherever it knows the surface is absorbing.

`post/boundary_flux.py` obtains the same number a second way, never touching the
accumulators: the fast solve returns a **steady** field, so $0 = \nabla\cdot(D\nabla C)+R$
and by the divergence theorem $\int_\Omega R\,dV = \oint_{\partial\Omega}(-D\nabla C)\cdot
n\,dA$. Here $R$ is exactly the 0-D right-hand side for the mobile components — that model
has no diffusion, so its mobile equations *are* the reaction terms. On the 200 nm march
the two routes agree to **3.7%** at 10 dpa: an independent confirmation, not an identity.

**That agreement is not general.** On the 500 nm hexagonal case the same diagnostic gives
a factor of 3.6. It is not a regression and not the trapezoid rule; the leading candidate
is the fast-solve cadence, unproven. **Re-measure before relying on the boundary channel
on any new geometry.** The closure value itself is unaffected — it is the accumulators'
own integral.

---

## 3. Repository structure

```
DisloCluster/                        <- repository root (.dislocluster_root marker)
├── CLAUDE.md                    physics and solver reference — measured numbers, limits
├── pyproject.toml               the `dislocluster_code` package (pip install -e .)
├── requirements.txt             pinned lock for .DisloClusterVenv
├── .DisloClusterVenv/           Python 3.14.3 environment (gitignored)
│
├── dislocluster_code/           <- ALL importable Python lives here
│   ├── paths.py                     single source of truth for every location
│   ├── config.py                    SimulationConfig — the six control dicts, validated
│   ├── driver.py                    prepare / march / report, one call per notebook stage
│   ├── build.py                     ensure_zrmicro_solver, ensure_modelib, preflight
│   ├── zerod/                       the 0-D chain
│   │   ├── input_data.py, reaction_rates.py, rate_equations.py
│   │   ├── calibration.py               the fitted parameter set — build_sim()
│   │   ├── cpp_bridge.py               subprocess bridge to the C++ solver
│   │   └── post_process.py, simulation.py, pre_process.py
│   ├── staging/                     case construction
│   │   ├── case.py                     mesh, stage, bootstrap
│   │   ├── inputs.py                   DD / polycrystal / ElasticDeformation files
│   │   ├── anisotropy.py               one p_m -> tensor + capture efficiencies
│   │   └── seed.py, domain.py, stress.py, standalone.py
│   ├── coupling/                    the operator split
│   │   ├── march.py                    the split itself
│   │   ├── qssa.py                     MobileQSSASolver — the fast solve
│   │   ├── immobile.py                 run_immobile_step — the slow sweep, batched
│   │   ├── field.py                    FieldBridge — per-node exchange both ways
│   │   ├── transition.py               continuum -> discrete, with the ledger
│   │   ├── neighbors.py                screened cutoff; k^2 as ImmobileSinks builds it
│   │   └── checkpoint.py, progress.py, export.py, ellipse_rom.py, seeded_march.py
│   ├── post/                        figures, movies, reports, diagnostics
│   │   ├── fields.py                   domain hull, panels, orientation triad
│   │   ├── discrete_loops.py            continuum field -> discrete loops (offline)
│   │   ├── dd_frames.py                 what the SOLVER has, from evl_<N>
│   │   ├── tem_slices.py                four diffraction conditions
│   │   ├── coarsening.py                the d_coarsen detector
│   │   ├── boundary_flux.py             the independent grain-boundary measurement
│   │   └── movies.py, report.py, march_report.py, volume_average.py, gb.py, …
│   ├── studies/                     comparison and campaign drivers
│   │   ├── hardening.py                 march -> DD yield test
│   │   ├── dad_sweep.py, dad_window.py  which anisotropy allows co-growth
│   │   ├── plan_step1..11.py            the model-extension verification suite
│   │   └── run_200nm_moments.py, run_zr3d_singlecrystal.py, compare_*.py
│   ├── fitting/                     parameter identification
│   └── legacy/                      superseded, kept for reproducibility
│
├── Simulations/                 <- where simulations are RUN
│   ├── run_simulation.ipynb         the normal entry point: configure, stage, march, report
│   ├── hardening_simulation.ipynb   the hardening campaign, six dicts of its own
│   ├── postprocess.ipynb            re-render an existing run; never re-solves
│   ├── input/                       the Excel workbooks            (paths.INPUT_DIR)
│   └── output/                      run directories                (paths.OUTPUT_DIR)
│
├── ZrMicro/                     0-D reduced cluster dynamics
│   ├── code/solver.cpp              the CVODES driver (batch OpenMP mode)
│   ├── cpp_utils/                   rate_equations_core.h — ONE templated RHS
│   │   ├── dual.h                       forward-AD for the exact Jacobian
│   │   └── jac_check.cpp                AD vs central difference, same residual
│   ├── py_utils/                    compatibility shim -> dislocluster_code
│   ├── input/, output/, build/       older workbook copies, earlier runs, binary
│   └── CLAUDE.md                    model equations, state vector, file map
│
├── MoDELib3/                    3-D spatially-resolved CD / DD (C++20)
│   ├── Library/Materials/Zr3d_ghoniem.txt   the shared material file
│   ├── tutorials/zrmicro_coupled/           the standalone verification case
│   ├── ZR3D_GHONIEM_CHANGES.md              what this fork changed
│   └── build*/                              gitignored
│
├── Gmsh/                        simulation-domain meshing
│   ├── generate_mesh.py             cubic / hexagonal, boundary-layer refined
│   └── meshes/                      cache, gitignored (reproducible from the spec)
│
└── Docs/                        <- ALL documents live here
    ├── DisloCluster Manual/         manuals, architecture notes, numbered history
    ├── Formulation/                 the manuscript, derivations, build scripts
    ├── Reports/                     deliverables
    └── Presentations/               slides
```

### Module status

| Module | Status | Description |
|---|---|---|
| `dislocluster_code/` | **Active** | The Python package: everything importable, installed editable into the venv. |
| `Simulations/` | **Active** | The working directory — notebooks, inputs, run output. Recommended entry point. |
| `ZrMicro/` | **Active** | The 0-D model and the C++ solver that is *also* the march's slow step. |
| `MoDELib3/` | **Active** | The 3-D FE cluster-dynamics and DD code, forked from `mlm335/MoDELib-fullCD`. |
| `Gmsh/` | Active | Mesh generation for both domain types. |
| `ZrMicro/py_utils/` | Compatibility shim | Each old module aliases `sys.modules` to its new home, so `from py_utils import paths` and the old notebooks keep working. New code imports `dislocluster_code` directly. |
| `dislocluster_code/legacy/` | Superseded | `modelib_fem.py` wrote four scalars per dose step, discarding all spatial structure. Kept so older results stay reproducible. |
| `ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb` | Superseded | The original coupling notebook, with its own inline copy of the march. Kept for reference. |

### Branches

| Branch | Kind | Contents |
|---|---|---|
| `main` | production | The suite. A fast-forward target. |
| `simulations-notebook` | working | Where development lands. |

The repository is a **single git checkout**. The 0-D tree and `MoDELib3/` were submodules
once and became plain directories in `859af47`; neither carries its own `.git` any more.
`paths.git_hash()` still falls back root → `ZR_ROOT` → `MODELIB_ROOT`, so a split
checkout would keep working.

**No build tree is in git, and none should be.** `MoDELib3/build_dc/` used to be
committed, carrying a Linux ELF and a `build.ninja` with 374 absolute paths. On another
machine `paths.modelib_ddomp()` then reported MoDELib as built, the build was skipped, and
the first call failed. Both build trees are now ignored, and `build.preflight()` *runs*
each binary rather than trusting its presence.

---

## 4. Installation

Nothing outside the repository root is required except the Python interpreter, SUNDIALS,
and — on Windows — WSL.

### 4.1 Python environment

**The patch version matters, because a notebook records it.** Jupyter writes the
interpreter into every notebook's metadata, so a venv on a different patch release
dirties `run_simulation.ipynb` the moment it is opened, and two machines on different
patches churn those lines forever. Pin it:

```bash
uv python install 3.14.3                                          # exact patch, ~17 MB
"$(uv python find --system 3.14.3)" -m venv .DisloClusterVenv     # --system: not the venv
.DisloClusterVenv/bin/python -m pip install -r requirements.txt
.DisloClusterVenv/bin/python -m pip install -e .
.DisloClusterVenv/bin/python -m ipykernel install --user --name dislocluster \
    --display-name "Python 3.14 (DisloCluster)"
```

On Windows the same `pip` steps run through `.DisloClusterVenv\Scripts\python.exe`.

Requirements: Python ≥ 3.12 with `numpy`, `scipy`, `pandas`, `matplotlib`, `openpyxl` and
`pillow`; `gmsh` (mesh generation) and `jupyterlab`/`ipykernel` (notebooks) are extras.
Every pin in `requirements.txt` installs on 3.14 from a wheel, so nothing builds from
source. **Do not use Anaconda Python** — its NumPy 1.x/2.x mix conflicts with SciPy here.

`pip install -e .` is what lets the notebooks import the package from anywhere. Check
path resolution at any time with `python -m dislocluster_code.paths` (or
`dislocluster-paths`), which prints an OK/MISS report per location.

### 4.2 The 0-D solver (C++17, SUNDIALS CVODES)

```bash
cmake -S ZrMicro/cpp_utils -B ZrMicro/build -DCMAKE_BUILD_TYPE=Release
cmake --build ZrMicro/build --config Release
```

**SUNDIALS 7.1.1** is found at `<repo>/Libraries/sundials-7.1.1/`, from a system install,
or — on macOS — under the Homebrew/MacPorts prefix, which CMake does not search by default
and which differs by architecture; `brew --prefix` is asked rather than one of them
assumed. Modules used: `cvodes`, `arkode`, `nvecserial`, `sunlinsolband`,
`sunlinsolspgmr`. OpenMP is optional but enables the parallel `--batch_file` mode the
march depends on.

### 4.3 The 3-D code (Linux, macOS, or Windows through WSL)

```bash
bash Docs/Formulation/build_modelib.sh                   # Linux, macOS
wsl -u root -e bash Docs/Formulation/build_modelib.sh    # Windows
```

**One script, one CMakeLists, every platform.** The script used to `sed`-patch
`CMakeLists.txt` on each run, so the source tree carried whichever platform's edits had
been applied last. The CMake files now decide for themselves: every dependency is searched
for, every optimization flag probed with `check_cxx_compiler_flag` before use, and
everything optional degrades to a status line. `SKIP_DEPS=1` skips the package-manager
step. Required: a C++20 compiler, CMake ≥ 3.16, and **Eigen 3.4.x**.

> **Eigen 5 is rejected, and the reason is worth reading.** Built against Homebrew's
> `eigen` (5.x) MoDELib *compiles* — then the BiCGSTAB solve inside the mobile Newton
> iteration breaks down on the first step of a case that converges with 3.4.0: a clean
> build and a wrong answer, reported as a solver problem. The script fetches 3.4.0 into
> `Libraries/` when the system has only Eigen 5, and the check is fatal
> (`-DEIGEN3_ALLOW_UNTESTED=ON` overrides it). It reads *both* places Eigen states its
> version, because 5.0.1 spells itself `3.5.0` in one of them.

FFTW3, Boost, SuiteSparse, OpenMP and pybind11 are optional and auto-detected;
`-DUSE_FAST_MATH=OFF` and `-DUSE_NATIVE_ARCH=OFF` turn off what a cross-machine comparison
might not want. On Windows the binary is a Linux ELF invoked through `wsl.exe`;
`paths.ddomp_cmd()` decides per platform.

### 4.4 Verify the installation

```python
from dislocluster_code import build
build.preflight()      # builds what is missing, then RUNS both binaries
```

It raises with actionable text when it cannot fix things itself — SUNDIALS not installed,
WSL unavailable, no compiler. The MoDELib build takes 10–30 min the first time and is a
no-op afterwards.

**Moving or renaming the repository invalidates both build trees**, because a CMake cache
stores absolute paths. Both sides detect this and discard the stale cache. The failure is
quiet — the already-built binary keeps working — which is why the check runs *before* the
"already built" shortcut.

---

## 5. Quick start

### 5.1 Notebook (recommended)

Open [`Simulations/run_simulation.ipynb`](Simulations/run_simulation.ipynb) with the
**Python 3.14 (DisloCluster)** kernel. Six dicts at the top of §2 set the material,
geometry, mesh, boundary conditions, coupling and output; the rest of the notebook
validates them, stages the MoDELib case, runs the march and writes the report. Nothing
else should need editing for a normal run. See
[`Simulations/README.md`](Simulations/README.md).

**An interrupted run is resumed by running the same cell again.** The march checkpoints
after every substep — ~57 ms against a fast solve of several minutes — and continues from
the last completed one. The checkpoint records the CD node set, the seed, the material,
the resolved 0-D parameter set, the march settings and the state width; if any changed,
resuming raises `CheckpointMismatch` rather than silently mixing two configurations. Set
`OUTPUT['resume'] = 'never'` to start over deliberately.

Headless:

```powershell
$env:PYTHONIOENCODING="utf-8"
.DisloClusterVenv\Scripts\python.exe -m nbconvert --to notebook --execute `
    --ExecutePreprocessor.kernel_name=dislocluster `
    --ExecutePreprocessor.timeout=-1 Simulations\run_simulation.ipynb
```

### 5.2 Python API

The notebook is a thin wrapper over three calls:

```python
from dislocluster_code import driver
from dislocluster_code.config import SimulationConfig

cfg = SimulationConfig.from_dicts(
    MATERIAL = dict(temperature_K=573.0, dose_rate_dpa_s=1e-7),
    GEOMETRY = dict(type='hexagonal', size_nm=500.0),
    MESH     = dict(element_size_nm=25.0, element_order=2),
    BOUNDARY = dict(periodic_face_ids=[], applied_stress_MPa=0.0),
    COUPLING = dict(dose_seed=1.0, dose_max=26.0, n_intervals=5,
                    substeps_per_interval=3, fem_every=3),
    SOLVER   = dict(loop_model=1, rtol=1e-6, atol=1e-20),
    OUTPUT   = dict(tag='500nmHex', figures=True, movies=True),
)

run    = driver.prepare(cfg)      # mesh, staged case, CD node set, seed
result = driver.march(run)        # the operator split; resumable
driver.report(run, result)        # figures, movies, report.md
```

`from_dicts` **rejects an unknown key** rather than ignoring it, and validates before
anything expensive runs: a dose grid that does not increase strictly, a seed past the
first snapshot, a boundary layer thicker than half the domain, an `element_order` outside
$\{1,2\}$ (MoDELib's `SimplexReader` consumes only msh element types 4 and 11).

**Staging is keyed on the domain, not the run.** `cfg.domain_key` hashes the geometry,
mesh, boundary conditions, material and temperature — but *not* the dose grid — so
changing only the doses reuses the staged case, its CD node set and its bootstrap.

### 5.3 Command line

```bash
python -m dislocluster_code.coupling.seeded_march             # the march on an explicit dose list
python -m dislocluster_code.studies.run_zr3d_singlecrystal    # the standalone 3-D figures
python -m dislocluster_code.studies.run_200nm_moments         # 9 families + 3 moments
python -m dislocluster_code.studies.hardening verify          # the hardening plan, as tests
python -m dislocluster_code.staging.anisotropy --show         # the current p_m and tensor
```

Post-processing re-runs on any finished run without re-solving:

```bash
python -m dislocluster_code.post.discrete_loops <run> --doses all
python -m dislocluster_code.post.movies         <run> --fps 5 --interp 5
python -m dislocluster_code.post.tem_slices     <run> --out <run>/tem_slices
python -m dislocluster_code.post.boundary_flux  <run>
```

---

## 6. Inputs

Inputs arrive on three channels, and which one a quantity belongs to is not arbitrary.

### 6.1 The workbook — 0-D physics parameters

`Simulations/input/Zr_input_parameters.xlsx`, read by `zerod/input_data.py`: lattice and
elastic constants, atomic volume, formation and migration energies, loop Burgers vectors,
capture-efficiency biases, nucleation and coalescence coefficients, sink densities, dose
rate and temperature.

> **The workbook is not the calibrated model.** 28 parameters have drifted from it and 11
> are absent from it entirely. Any driver that builds `InputData` straight from the
> workbook runs a *different* model — $N_a = 2.3\times10^{-2}$ and $c_a = 8.77$ (an atom
> fraction, so impossible) against the calibrated $8.05\times10^{-8}$ and
> $1.34\times10^{-4}$. **Always go through `calibration.build_sim()`.**

The workbooks under `ZrMicro/input/` are copies; only `Simulations/input/` is read. Two
copies can drift, and this one drifted by five orders of magnitude once already, so
`paths.describe()` hashes both and `paths.workbook_drift()` names any that disagree.

### 6.2 The six control dicts — per-run choices

| Dict | Controls |
|---|---|
| `MATERIAL` | material file, temperature, dose rate, 0-D parameter overrides |
| `GEOMETRY` | domain type (`cubic` / `hexagonal`), dimensions |
| `MESH` | element size and order, boundary-layer refinement |
| `BOUNDARY` | periodic faces (empty ⇒ Dirichlet everywhere), applied stress and strain |
| `COUPLING` | route, seed dose, snapshot doses, substeps, fast-solve cadence, failure tolerances, the coarsening detector, the discrete handoff |
| `SOLVER`, `OUTPUT` | model switches, tolerances, backend; tag, figures, movies, checkpoint, resume |

`BOUNDARY` is a real boundary-condition channel: `periodic_face_ids` goes into
`polycrystal.txt` and the load into `ElasticDeformation.txt`. **Stress is given in MPa and
converted on staging** — MoDELib normalizes stress by $\mu_{\rm SI}$, so writing MPa
straight in would overstate the load by ~33 000× for Zr.

`substeps_per_interval` fixes the immobile substep count **per dose interval** rather than
by a wall-clock cap — a 5 dpa step at $10^{-7}$ dpa/s is $5\times10^7$ s, which a fixed
$5\times10^4$ s cap would have split into a thousand batch subprocesses.

### 6.3 The material file — what both codes share

`MoDELib3/Library/Materials/Zr3d_ghoniem.txt` (`paths.MODELIB_MATERIAL`) is the one file
both sides read: lattice constants, the mobile diffusion tensors, the DAD parameters $p_m$
and $Z^0_m$, loop sink scales, the Burgers vectors. `staging/anisotropy.py` writes it.

Four quantities must stay equal across the workbook, the material file, the Python defaults
and the C++ — Ω, $a$, $c$, and the loop Burgers vectors — and each has been wrong in at
least one of those places. `InputData.check_atomic_volume()` runs on every `build_sim` and
would catch a regression.

```python
PARAM_OVERRIDES = {'T': 573.0, 'Omega': 2.326553e-29}   # per-run, recorded in provenance
```

Overrides apply to that run only and are written into its provenance. Permanent changes
belong in the workbook or the material file.

---

## 7. Outputs and provenance

Each run writes a self-describing directory:

```
Simulations/output/YYYYMMDD_HHMMSS_<git-hash>_<tag>/
├── config.json           the resolved control dicts, after validation
├── provenance.md         inputs with overrides applied, machine, runtime, interior state
├── march_state.npz       the per-node immobile state at every snapshot dose
├── summary.json          end-of-run summary, including `loop_model`
├── checkpoint/           per-substep resume state
├── evl_coupled/          the CD blocks handed to and from MoDELib
├── 0d/ 3d/ gb/           the figure sets — 0-D suite, field panels, boundary profiles
├── movies/               continuum-field GIFs               (OUTPUT['movies'])
├── discrete_loops/       per-dose PNG, loops_*.csv, aLoops_*.txt, manifest.json
├── tem_slices/           four diffraction conditions, per-dose panels plus montages
└── report.md
```

`march_state.npz` and `summary.json` record `loop_model`, **because the state array does
not say which layout it is in.** `field.run_loop_model(run)` reads it and
`field.to_legacy_layout` maps mode-1 families onto the legacy slots for the figure suite,
which addresses them by the legacy names.

Because the git SHA and the complete resolved parameter set are written with every run, a
figure can always be traced back to the code and inputs that produced it. Runs never
overwrite one another.

**Where runs are found is not where they are written.** 1.4 GB of earlier runs were left
under `ZrMicro/output/`, so `paths.OUTPUT_DIR` (writes) and `paths.OUTPUT_DIRS` (reads)
differ. Anything looking for a run must go through `paths.find_runs()` or
`paths.latest_run()` — never `OUTPUT_DIR.iterdir()`.

### Reading the figures

Three rules, each of which has produced a wrong number:

- **Quote the interior mean, not the domain mean.** On the single-crystal case 27% of CD
  nodes lie on Dirichlet faces and at 21 dpa carry **99.8%** of the total ⟨a⟩ density. The
  coupled march's ⟨a⟩ density is 474× the 0-D as a domain mean and 0.75× it in the interior.
- **Near-boundary loop density is not quantitative.** Cascade nucleation is uniform but the
  only loop-removal channel is coalescence, driven by a mobile flux that vanishes where
  Dirichlet pins the concentrations — the boundary is a sink for mobile defects but **not**
  for loops.
- **Loop overlay radii are exaggerated, and the factor differs between the ⟨c⟩ and ⟨a⟩
  figures**, so sizes are faithful *within* a figure and not *between* them.

Movie frames between snapshots are **interpolated, not solved**, and labelled `(interp)`.
The blend is geometric where both endpoints are positive, and is taken on the CD block
rather than the state behind it — a sum of geometric blends is not the geometric blend of
the sums, so blending the state first produces frames not bracketed by their neighbors.

---

## 8. Examples, tests and verification

**Driver notebooks** — [`run_simulation.ipynb`](Simulations/run_simulation.ipynb) (the
full workflow), [`hardening_simulation.ipynb`](Simulations/hardening_simulation.ipynb)
(the hardening campaign), [`postprocess.ipynb`](Simulations/postprocess.ipynb) (re-render
without re-solving), and `ZrMicro/code/ZrMicro.ipynb` (0-D runs and parameter
identification).

**The standalone 3-D verification case** — `MoDELib3/tutorials/zrmicro_coupled` is a 1 µm
cube (3093 b per side, 24 115 FE nodes), Dirichlet on all six faces, 31 output steps of
1 dpa at $G = 10^{-7}$ dpa/s and $T = 573$ K.

```bash
wsl -e bash MoDELib3/tutorials/zrmicro_coupled/clean_run.sh      # the solve (hours)
python -m dislocluster_code.studies.run_zr3d_singlecrystal       # the figures (minutes)
```

The coupled march **reproduces this case's boundary artifact**, which is a point in favor
of the coupling: the same behavior emerges whether MoDELib3's nodal scheme or ZrMicro's
CVODE march advances the immobile population.

**Physics checks** — `studies/plan_step1.py` … `plan_step11.py` verify the five model
extensions of §2.5. One is worth singling out: **`plan_step5 verify` goal (vi) is the only
test that can see an inconsistent moment channel**, because the other goals each switch
the other channels off to isolate one term, so a bad channel is precisely the one never
exercised alongside the rest. Goal (vi) runs five successive steps with coalescence,
nucleation, emission and the basal chain all live and reports the minimum $\Delta$ per
family.

`ZrMicro/cpp_utils/jac_check.cpp` differences the forward-AD Jacobian against a central
difference of the **same** residual, with a relative step per column because the state
spans ~34 decades. Components on the concentration floor are reported but not counted as
failures: the floor makes the residual non-differentiable there, so AD returns the
one-sided derivative while a central difference straddles the kink.

### 8.4 What is *not* calibrated

Read this before comparing anything to experiment.

| | Status |
|---|---|
| the 28-parameter 0-D set | fitted — but against **mode 0**, at the **old Ω**, with the **old ⟨c⟩ Burgers vector**. All three corrections compound. |
| `loop_model = 1` | consistent with the fast solve, **not calibrated** |
| the basal-chain rates | **none is calibrated** — neither the material file nor the workbook supplies them, and the formulation gives their Arrhenius form without barrier values |
| $m_{\rm vanish}$, $\nu_{\rm vanish}$ | where a sweep put them, not where a measurement did |
| $\alpha$ and the Taylor factor $M$ in hardening | not computed; neither has a default that would be indistinguishable from a fitted one |

A refit is tractable — one objective evaluation is 0.2–0.5 s over 13 temperatures and 78
experimental points — and `fitting/fit_cloops.py` drives it:

```python
from dislocluster_code.fitting import fit_cloops as F
F.set_model(loop_model=1, n_fam=9, moments=1, emission_model=1)
J, breakdown = F.objective_full(F.OPT, want_breakdown=True)
```

At the *unrefitted* reference parameters the second moment **improves** the fit with
nothing refitted ($J$ 0.5375 → 0.4735), while peripheral emission blows the ⟨c⟩ term up by
12× — the same finding as the march's inverted $N_a/N_c$, arriving independently against
experiment rather than against a reference run.

What a refit has already established: **the missing channel is physical.** The data want
vacancy loops that are *fewer and larger*, and coalescence can only buy fewer by also
buying smaller — $d_c$ stays at 0.07–0.11× of experiment in every fit. Something that
removes vacancy loop **number** without shortening the survivors' lives is absent, which is
what the deleted annealing lifetimes used to do.

---

## 9. Computational aspects

**Solvers.** The slow step uses SUNDIALS CVODES with variable-order BDF and an exact
forward-AD Jacobian. Measured accuracy at $r_{\rm tol}=10^{-6}$ against a tight
reference:

| Config | Backend | Linear solver | Max relative error |
|---|---|---|---|
| **A (default)** | **CVODE BDF** | **dense** | $3.0\times10^{-4}$ |
| B | CVODE BDF | band | $3.0\times10^{-4}$ |
| C | CVODE BDF | GMRES | $1.7\times10^{-2}$ |
| G | ARKODE SDIRK_5_3_4 | dense | $3.0\times10^{-1}$ |
| F, H, I | ARKODE (other tables) | dense | $10^4$–$10^7$ (fails) |

ARKODE methods designed as IMEX pairs perform poorly in pure implicit mode here.

**Parallelism.** The slow sweep is one OpenMP `--batch_file` job: each point an
independent case, dynamically scheduled with a private CVODE workspace per thread, and
identical states deduplicated (×1.25 on the 500 nm case, higher early on when much of the
domain shares a state). The fast solve is MoDELib's own parallel FE assembly.

**Cost, measured.** 500 nm hexagonal case, 90 617 CD nodes, 24 cores: 925 s per fast
solve, 28.3 s per slow sweep, 11.3 min for all 24 substeps. Hardening: 0.026 s/step on a
200 nm cell on 8 cores, 2.5–3.3 s/step at 500 nm — the 500 nm campaign wants a 24-core
machine. `useSubCycling=1` is **5.4×** on the 200 nm cell, not the ~100× estimated,
because the saving is the ratio of total to active segments; re-measure before sizing on
it.

**A point can fail the slow step.** On the 500 nm anisotropic march one node of 90 617
fails deterministically at the 0.01 → 0.1 dpa substep, sitting 51 b from a basal-pole face
inside the boundary layer with unremarkable concentrations — anisotropic interstitial
diffusion narrows that layer along ⟨c⟩ and steepens its gradient. `run_immobile_step`
re-runs failures alone (the batch step is *not* deterministic, so a point near the failure
boundary is decided by thread scheduling) and then with tightened tolerances;
$r_{\rm tol}=10^{-8}$ *with* $a_{\rm tol}=10^{-30}$ integrates this one in 0.083 s.
Tighter, never looser — loosening also "works", by accepting a worse answer at the one
point already known to be difficult. And note what `max_failed_nodes` does before reaching
for it: it substitutes an **identity step**, freezing the point, and declares the march
successful without it. Diagnosing this class of failure:
[`cvodes_corrector_convergence_failure.md`](Docs/DisloCluster%20Manual/architecture/cvodes_corrector_convergence_failure.md).

**Reproducibility** is part of the solver contract: provenance stamping, timestamped
directories, per-substep checkpoints with a fingerprint that refuses a mismatched resume,
and no build tree in git.

---

## 10. Documentation

| Location | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | the working reference — every measured number, corrected claim and known limitation, with the case it was measured on |
| [`Docs/Formulation/self-consistent/`](Docs/Formulation/self-consistent/) | the manuscript, its sections, figures and bibliography |
| [`Docs/Formulation/Bias Factors/`](Docs/Formulation/Bias%20Factors/) | finite-loop transport and capture efficiencies in α-Zr |
| [`Docs/Formulation/annealing/`](Docs/Formulation/annealing/) | the loop-annealing literature survey behind the emission model |
| [`Docs/Formulation/architecture/`](Docs/Formulation/architecture/) | the 3-D code architecture |
| [`Docs/Formulation/data_fitting_operator_split/`](Docs/Formulation/data_fitting_operator_split/) | parameter identification; the two-time-scale coupling derivation |
| [`Docs/Formulation/key_references/`](Docs/Formulation/key_references/) | the reference library underlying the parameter set |
| [`Docs/DisloCluster Manual/manuals/`](Docs/DisloCluster%20Manual/manuals/) | the DisloCluster and MoDELib manuals |
| [`Docs/DisloCluster Manual/architecture/`](Docs/DisloCluster%20Manual/architecture/) | development notes: the hardening plan and what running it corrected, the anisotropy and discrete-coupling plan, solver notes |
| [`Docs/DisloCluster Manual/history/`](Docs/DisloCluster%20Manual/history/) | the full numbered code history |
| [`ZrMicro/CLAUDE.md`](ZrMicro/CLAUDE.md) | the 0-D model: equations, state vector, file map |
| [`MoDELib3/ZR3D_GHONIEM_CHANGES.md`](MoDELib3/ZR3D_GHONIEM_CHANGES.md) | what this MoDELib fork changed |

**All documents live under the repository-root [`Docs/`](Docs/)** — there is no `Docs/`
under any sub-code. LaTeX sources sit next to their compiled PDFs and `.bib` files, and
each document compiles from inside its own folder.

### Which MoDELib upstream, and which is the baseline

**The baseline is `mlm335/MoDELib2-NNL`**, at the point the immobile population equations
were written into it — and it started from nothing: `iSize` was 0,
`solveImmobileClusters()` was an empty function body, and the source carried a literal
`// Missing immobile sinks` placeholder. The loop populations were **absent, not
simplified**.

`mlm335/MoDELib-fullCD` is a **separate cluster-dynamics branch** — a sibling, not the
ancestor. Shared parameter names across two branches of one code, developed in one group
from one formulation, are evidence of convergence and say nothing about precedence. It is
still worth comparing against:

| | `MoDELib-fullCD` | this branch |
|---|---|---|
| Rate | Galerkin at quadrature points, L2-projected through the consistent mass matrix | nodal collocation, no mass matrix |
| Time update | explicit Euler, then sequential implicit loss factors per channel | one fused semi-implicit update |
| Sub-cycling | none — one step of `dtMax` | `nSub = 20` per dose step |

The one capability fullCD has and this branch does not is its in-solver continuum→discrete
loop transition — a **gap, not a regression**, since nothing of the kind existed in
MoDELib2-NNL to lose, and the piece worth taking from it. `post/discrete_loops.py` performs
the same conversion offline, which is not the same thing: it does not feed back into a solve.

### Known non-correspondences between the 0-D and 3-D models

These are physics, not bugs — do not "fix" them silently.

1. **DAD bias** — the 0-D uses phenomenological $\delta_i, \delta_v$ with
   $Z_a + Z_c = 2$; the 3-D generates the bias from the diffusion tensors via $p_m$. They
   are reconciled by a closed-form fit of $(Z^0_m, p_m)$, **valid only at the fitted
   temperature**. `loop_model = 1` removes this one.
2. **Bi-pyramid vacancy family** — exists only in 3-D.
3. **Size-dependent vacancy-loop thermal emission** — 3-D only; the 0-D uses a constant
   surrogate unless `emission_model = 1`.

---

## 11. Publications and how to cite

If you use DisloCluster, please cite the framework manuscript:

> N. M. Ghoniem, G. Po, M. Maron, R. Escobar, B. Ramirez Flores, K. Baker and T. Black,
> *A Self-Consistent Spatially-Resolved Cluster Dynamics Model for Irradiated Zirconium*,
> Journal of Nuclear Materials (in preparation).

```bibtex
@article{Ghoniem2026DisloCluster,
  author  = {Ghoniem, N. M. and Po, G. and Maron, M. and Escobar, R. and
             Ramirez Flores, B. and Baker, K. and Black, T.},
  title   = {A Self-Consistent Spatially-Resolved Cluster Dynamics Model for
             Irradiated Zirconium},
  journal = {Journal of Nuclear Materials},
  year    = {2026},
  note    = {In preparation}
}
```

The formulation continues a long line of work on defect cluster kinetics:

- N. M. Ghoniem and D. D. Cho, *The simultaneous clustering of point defects during
  irradiation*, Phys. Status Solidi A **54** (1979) 171–178.
  [doi:10.1002/pssa.2210540122](https://doi.org/10.1002/pssa.2210540122)
- N. M. Ghoniem and S. Sharafat, *A numerical solution to the Fokker–Planck equation
  describing the evolution of the interstitial loop microstructure during irradiation*,
  J. Nucl. Mater. **92** (1980) 121–135.
  [doi:10.1016/0022-3115(80)90148-8](https://doi.org/10.1016/0022-3115(80)90148-8)
- N. M. Ghoniem, *Stochastic theory of diffusional planar-atomic clustering and its
  application to dislocation loops*, Phys. Rev. B **39** (1989) 11810–11819.
  [doi:10.1103/PhysRevB.39.11810](https://doi.org/10.1103/PhysRevB.39.11810)
- C. H. Woo, *The sink strength and bias of dislocation loops* (1981) and *Intrinsic bias
  differential between vacancy and interstitial loops* (1982) — the capture efficiencies
  `loop_model = 1` uses.

The key reference set for the parameter values, including the anisotropic-diffusion and
coupled CD–DD literature this work builds on, is collected under
[`Docs/Formulation/key_references/`](Docs/Formulation/key_references/); full reference
lists are in the `.bib` files under [`Docs/Formulation/`](Docs/Formulation/).

---

## 12. Repository conventions

- **Never hard-code an absolute path.** Every location comes from
  [`dislocluster_code/paths.py`](dislocluster_code/paths.py), which walks up to the
  `.dislocluster_root` marker. If a new module needs a repository location, **add it to
  `paths.py`** — do not re-derive it with `Path(__file__).parent.parent`. That idiom broke
  silently in four modules when the package moved; `cpp_bridge` resolved its default
  `base_dir` that way and stopped finding the solver at all. Overrides:
  `DISLOCLUSTER_ROOT`, `DISLOCLUSTER_SIM_ROOT`, `MODELIB_ROOT`, `MODELIB_BUILD`.
- **Configuration is a value, not module state.** Legacy globals are read at call time so
  monkey-patching callers keep working; making them dataclass field defaults would bind
  them at import and turn a later assignment into a silent no-op.
- **Outputs are immutable and stamped.** Runs never overwrite one another; `output/` and
  every build tree are gitignored.
- **The 0-D *is* the march's slow step.** One templated C++ function, instantiated for the
  residual and for the exact Jacobian, so new physics is added once and both scales get
  it. The only difference is `freeze_mobile`.
- **Do not tidy the legacy `ydot` assembly.** It is kept verbatim inside
  `if (loop_model == 0)` because regrouping algebraically identical terms moves the 10th
  significant digit, which was measured.
- **Nothing in the figure code branches on the geometry name.** Every shape decision — the
  outline, the hidden-line test, the volume that turns a density into a count, the region
  loops are placed in — comes from the convex hull of the CD node cloud. Do not
  reintroduce a bounding-box assumption: for a hexagonal prism the box is **4/3** of the
  crystal, and the loop count came out a third too high.
- **Write prose and code comments in American English** (`-ize`, `-or`, no hyphen after an
  `-ly` adverb). New documents go under the repository-root [`Docs/`](Docs/), in the
  fitting subdirectory.

Issues and pull requests are welcome. Changes that touch reaction rates, stoichiometry or
the shared constants (Ω, $a$, $c$, the Burgers vectors) should state which of the four
places they were changed in, and come with the conservation or Jacobian check that covers
them.

---

## 13. License and contact

Released under the [MIT License](LICENSE), © 2026 Nasr Ghoniem.

That covers DisloCluster's own contributions. **`MoDELib3/` is a fork of third-party code**
— derived from `mlm335/MoDELib2-NNL` (see [§10](#which-modelib-upstream-and-which-is-the-baseline))
— and carries no license file of its own in this tree, so redistributing that subtree is
governed by its upstream terms rather than by the license above. Settle those terms with
the MoDELib authors before releasing the repository as a whole.

**Nasr M. Ghoniem** — Mechanical and Aerospace Engineering Department,
University of California, Los Angeles · ghoniem@ucla.edu
