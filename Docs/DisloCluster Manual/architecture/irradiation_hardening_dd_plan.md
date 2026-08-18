# Irradiation hardening from a coupled march — a DD plan

**Source run:** `Simulations/output/20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent`
(500 nm hexagonal prism, route Adaptive, `loop_model = 1`, T = 573 K, G = 1e-7 dpa/s,
zero applied stress, 9 snapshots 0 → 10 dpa, 90 617 CD nodes)

**Goal:** at every output dose, apply a remote load and run discrete dislocation
dynamics to measure the increase in flow stress the irradiation microstructure causes,
Δτ(dose) → Δσ_y(dose).

Everything below was read out of `MoDELib3/` (tutorials, `Library/`, the sources named
inline, and `manual_beta/tex/MoDELib.tex` §2.2) and out of the run itself. Numbers
marked *measured* were computed from `march_state.npz`; numbers marked *estimated* are
arithmetic on those, and need one timing calibration run before they are trusted.

---

## 1. What the run gives, and what DD needs

The march produces a **continuum** immobile field on an unstructured mesh: per CD node,
four loop number densities `n_k` and four stored-defect contents `c_k`
(`c, a1, a2, a3`). DD needs **discrete loops** — centre, habit plane, Burgers vector,
radius, polygon — inside a domain a DD cell can be.

Two mismatches have to be resolved, and they are independent.

### 1.1 Geometry: the marched domain is the wrong domain for a yield test

The march ran on a hexagonal prism with **Dirichlet conditions on the whole surface**
(`periodic_face_ids = []`). Two consequences:

- **The domain mean is not the material.** CLAUDE.md already records this for the cubic
  case and it holds here: the boundary shell is a sink for mobile defects but not for
  loops, so ⟨a⟩ density climbs steeply toward the wall. The RVE must be built from the
  **interior mean** — the innermost quartile by distance to the nearest face, which is
  7 340 of the 90 617 nodes (*measured*).
- **A free-surface prism measures a nano-pillar, not bulk yield.** Image forces, surface
  sources and the absence of a periodic image sum all change the answer. For a bulk
  yield strength the DD cell must be **fully periodic**.

So: extract interior-mean statistics, and rebuild them in a **periodic cube**.

### 1.2 Population: the continuum → discrete conversion already exists

`dislocluster_code/post/discrete_loops.py` does exactly this conversion and already
writes `aLoops_<dose>.txt` in MoDELib's `microstructureGenerator` input format. The run
has four of them (`discrete_loops/aLoops_{0p0001,0p01,1,10}dpa.txt`).

`populate(..., region="interior")` even selects `positions="uniform"`, and its docstring
says why: *"This is the right mode for a bulk DD cell, where the point is to carry the
interior state without the boundary layer attached to it."* The one thing it does not do
is place them in a **cube of our choosing** — it fills the marched crystal's own convex
hull. That is the single new piece of Python needed (§7).

**Two things must be regenerated, not reused.**

1. The `loops_*.csv` in this run carry `b_mag_b = 0.816497` for ⟨c⟩ — that is
   `½·√(8/3)`, the **ideal** c/a. The current `FAMILIES["c"]` carries `0.7972136`
   (= c/2a physical). Those files predate commit `48f5ae4`. Re-run
   `python -m dislocluster_code.post.discrete_loops <run> --doses all` before using
   anything from `discrete_loops/`.
2. There are only four doses on disk. The march has **nine**. `--doses all` fixes that.

---

## 2. The interior state, dose by dose

*Measured* from `march_state.npz`, interior quartile, ⟨a⟩ summed over the three prism
variants, radii from `r = √(mΩ/πb)` with `m = c/(nΩ)`:

| dpa | N_c [m⁻³] | r_c [nm] | N_a [m⁻³] | r_a [nm] | ρ_c [m⁻²] | ρ_a [m⁻²] |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 5.18e19 | 3.41 | 4.76e19 | 2.10 | 1.11e12 | 6.28e11 |
| 1e-3 | 3.86e20 | 3.53 | 4.68e20 | 2.14 | 8.55e12 | 6.29e12 |
| 1e-2 | 7.67e20 | 5.41 | 3.77e21 | 2.51 | 2.61e13 | 5.94e13 |
| 0.1 | 7.66e20 | 25.37 | 1.18e22 | 2.74 | 1.22e14 | 2.03e14 |
| 1 | 7.60e20 | 56.78 | 6.68e21 | 3.32 | 2.71e14 | 1.39e14 |
| 2 | 7.61e20 | 55.56 | 4.18e21 | 3.76 | 2.66e14 | 9.89e13 |
| 5 | 7.61e20 | 55.52 | 4.19e21 | 3.77 | 2.65e14 | 9.92e13 |
| 10 | 7.61e20 | 55.55 | 4.20e21 | 3.77 | 2.66e14 | 9.93e13 |

`ρ = 2πrN` is the loop line density. Total line density saturates near **3.6e14 m⁻²**,
which is a strong obstacle field, and the ⟨c⟩ half of it dominates from 0.1 dpa on.

**Read the ⟨c⟩ column above 0.1 dpa with suspicion.** `N_c^(-1/3)` = 110 nm against a
loop **diameter** of 111 nm: the loops are geometrically touching, `2r/d = 1.005`, which
the manifest flags as `saturated: true`. Separately, `summary.json` records
`d_coarsen = 0.4 dpa` for all three ⟨a⟩ variants — the dose above which the Avrami
mean-field treatment of coalescence is no longer valid on its own terms. **Both families
are outside their validated range over most of the dose list**, and that is a statement
about the source data, not about the DD.

---

## 3. Route A — CRSS by stress ramp on a frozen obstacle field *(recommended)*

This is the standard dispersed-barrier DD measurement, and it is the only one of the
three that is affordable at every dose.

### The physics

Hold the irradiation loops fixed, drive a single gliding dislocation (or a small source
population) through them under a **pure resolved shear stress ramped in time**, and
record the stress at which the line achieves steady propagation across the cell. That
stress is the CRSS increment Δτ contributed by the loop field.

MoDELib does the freezing for us, and it is worth knowing that it is not optional:
`DislocationNode::projectVelocity` (`src/DislocationDynamics/DislocationNode.cpp:141`)
sets `velocity.setZero()` for **any node belonging to a `SESSILELOOP`** on a non-climbing
step, and `aLoopGenerator` inserts every ⟨a⟩ and ⟨c⟩ loop as `SESSILELOOP`
(`aLoopGenerator.cpp:125,235,274`). So with `climbSolverType=none` the loops are rigid
obstacles that contribute stress and can be cut into junctions, but never move.

### Why a stress ramp and not a strain rate

A strain-rate test has to accumulate macroscopic strain; this one has to move one
dislocation across one cell. The strains differ by three orders of magnitude:

```
one dislocation sweeping a 500 nm cell:  Δε_p = b/L = 0.323/500 = 6.5e-4
0.2% offset yield:                       Δε_p = 2.0e-3, reached only after
                                         many sweeps of many dislocations
```

and the step count follows the same ratio (§6).

### Loading

Apply the load through the **uniform load controller**, not the FEM. In a fully periodic
domain that happens automatically: `DislocationDynamicsBase.cpp:107` builds the FE space
only when `!isPeriodicDomain && useFEM`, and `ElasticDeformation.cpp:67` falls back to
`getUniformEDcontroller` when `fe` is null. The controller is documented in
`manual_beta/tex/MoDELib.tex` §2.2:

```
sigma = C (C+D)^-1 [ sigma_0 + D (eps_0 - eps_p) ],   D_i = alpha_i C_ii
alpha_i = 0        -> component i is under PURE STRESS control, sigma_i = sigma_0i
alpha_i -> infinity -> component i is under PURE STRAIN control, eps_i = eps_0i
```

For a clean CRSS there is no reason to involve a Schmid factor. Put the shear directly
on the target slip system with `stiffnessRatio = 0 0 0 0 0 0` (pure stress control
throughout) and

```
ExternalStress0     = 0 0 0 0 0 0
ExternalStressRate  = tau_dot * (Voigt components of s (x) n + n (x) s)
```

Voigt order is `11 22 33 12 23 13` (`Library/ElasticDeformation/ElasticDeformation.txt`,
confirmed by `DislocationDynamicsBase.cpp:78`'s `voigtTraits`). Stress is normalized by
`mu_SI`, so the file carries `tau/mu`, not MPa — the same trap `staging/inputs.py`
already handles for the coupled route (33 GPa, so a MPa value written raw is out by
~33 000×).

Ramp rate: slow enough to be quasi-static. `τ̇ = 1 MPa` per `1e4 b/cs` is a reasonable
first choice (`b/cs = 1.436e-13 s`, so that is ~7e11 Pa/s — fast in SI, quasi-static in
DD terms because the line relaxes in ~1e2 b/cs).

### Output and reduction

`ElasticDeformation::output` (`src/ElasticDeformation/ElasticDeformation.cpp:264`)
appends `epsil` then `sigma`, six each, into `F/F_0.txt` with labels `e_11 … s_13` in
`F/F_labels.txt` — but **only** on the uniform-controller branch, which is exactly the
branch a periodic run takes. `DislocationNetwork` writes `betaP` (the plastic
distortion) and, when `outputLoopLength`/densities are on, `glissile density`,
`sessile density`. So the reduction is:

```
tau(t)      = s . sigma(t) . n          resolved shear on the target system
gamma_p(t)  = s . betaP(t) . n          resolved plastic shear
Delta_tau   = tau at which dgamma_p/dt becomes and stays non-zero
              (or the 0.01% offset on the tau-gamma_p curve)
```

Do this at each dose **and once with an empty cell** (loops removed, same sources, same
seed). The difference is the irradiation increment; the absolute value carries the
Peierls law and is not what is being measured.

### Converting to yield strength

Δσ_y = M · Δτ. The cell is a single crystal, so M is a modelling choice, not an output.
For prismatic ⟨a⟩ slip in textured Zr cladding M ≈ 2.2–3.0; quote the Δτ and the M
separately rather than folding them together.

---

## 4. Route B — full stress–strain curve *(the physical answer, ~10³× the cost)*

Same cell, same loops, but `ExternalStrainRate` non-zero and `stiffnessRatio` large on
the loading component — the `periodic_density` tutorial is the working template
(`tutorials/periodic_density/generateInputFiles.py`), where
`ExternalStrainRate = [0,0,0,0,0,7.185e-11]` is 500 s⁻¹ in SI (divide by
`b/cs = 1.436e-13 s`).

This gives the whole curve: elastic slope, yield, hardening rate, and — if the loops are
allowed to be absorbed rather than frozen — defect-free channel formation. It is the
right experiment and the wrong budget. See §6 for why.

**Route B is the one to run at two or three doses once Route A has produced the trend**,
as a check that the CRSS increment and the 0.2% offset yield increment agree.

---

## 5. Route C — analytical dispersed-barrier model *(free, immediate, all nine doses)*

```
Delta_tau_k = alpha_k * mu * b * sqrt(N_k d_k)          per family
Delta_tau   = sqrt( sum_k Delta_tau_k^2 )               root-sum-square superposition
```

With μ = 33 GPa, b = 0.323 nm, α_c = 0.4 (sessile basal loops, strong), α_a = 0.2
(prismatic loops sharing the glide Burgers vector, weaker), *estimated*:

| dpa | Δτ_c [MPa] | Δτ_a [MPa] | Δτ [MPa] | Δσ_y at M = 3 [MPa] |
|---:|---:|---:|---:|---:|
| 1e-4 | 2.5 | 1.0 | 2.7 | 8 |
| 1e-3 | 7.0 | 3.0 | 7.7 | 23 |
| 1e-2 | 12.3 | 9.3 | 15.4 | 46 |
| 0.1 | 26.6 | 17.2 | 31.6 | 95 |
| 1 | 39.6 | 14.2 | 42.1 | 126 |
| 2–10 | 39.2 | 12.0 | 41.0 | 123 |

**This is not a substitute for the DD — it is the thing the DD calibrates.** The α's are
the unknowns, and DD at two or three doses fixes them. Once fixed, the law gives every
dose for free, including doses that were never marched. That is the practical way to
deliver "hardening at every output dose" without eight DD campaigns.

The magnitude is credible: ~120 MPa of shear-stress increment saturating by ~1 dpa is
the right order for α-Zr at 573 K, though the absolute densities behind it are not
experiment-calibrated (§8).

---

## 6. Computational requirements

### Cell size and loop count

The cell must hold several ⟨c⟩ loops of radius 55 nm and several spacings. `L = 500 nm`
is the smallest defensible choice at high dose; at low dose it can shrink.

The box edges must be **lattice vectors of the real crystal**, or the periodic shifts do
not map glide planes onto themselves. For hcp with `a1 = (1,0,0)`, `a2 = (½,√3/2,0)`,
`a3 = (0,0,c/a)` in units of b, an orthogonal commensurate cell is

```
e1 = n * a1            = (n, 0, 0)
e2 = m * (2 a2 - a1)   = (0, m*sqrt(3), 0)
e3 = p * a3            = (0, 0, p*c/a)
```

and with **`c/a = 1.5944272`** (the physical ratio this material carries through `c_SI`,
not the ideal 1.6329932):

| target | n | m | p | `F = diag(...)` [b] | actual [nm] |
|---:|---:|---:|---:|---|---|
| 200 nm | 619 | 357 | 388 | 619.0, 618.3, 618.6 | 199.94, 199.72, 199.82 |
| 300 nm | 929 | 536 | 583 | 929.0, 928.4, 929.6 | 300.07, 299.87, 300.24 |
| 500 nm | 1548 | 894 | 971 | 1548.0, 1548.5, 1548.2 | 500.00, 500.15, 500.06 |
| 800 nm | 2477 | 1430 | 1553 | 2477.0, 2476.8, 2476.1 | 800.07, 800.02, 799.79 |

> **Do not build this cell with `modlibUtils.PolyCrystalFile`.** Its HEX lattice matrix
> is hard-coded `A = [[1,½,0],[0,√3/2,0],[0,0,√(8/3)]]` (`python/modlibUtils.py:50`) —
> the **ideal** c/a — while `HEXlattice::getLatticeBasis` takes c/a from `c_SI`. Its
> `compute()` would pick `p = 948` for a 500 nm target, which against the real basis is
> a 488 nm edge: a 2.4% error, and a shift vector that is not a lattice vector.
> Use `dislocluster_code.staging.inputs.write_polycrystal`, which patches `F` directly
> and is what the coupled route already uses.

Loop counts at `L = 500 nm` (*estimated* from §2):

| dpa | ⟨c⟩ loops | ⟨a⟩ loops | ⟨c⟩ segments | ⟨a⟩ segments | total segments |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | 6 | 6 | 78 | 95 | ~170 |
| 1e-3 | 48 | 58 | 579 | 936 | ~1 500 |
| 1e-2 | 96 | 471 | 1 151 | 7 532 | ~8 700 |
| 0.1 | 96 | 1 478 | 1 149 | 23 640 | ~24 800 |
| 1 | 95 | 835 | 1 140 | 13 366 | ~14 500 |
| 2–10 | 95 | 523 | 1 141 | 8 380 | ~9 500 |

(⟨c⟩ as 12-gons refined to `Lmax = 100 b`, ⟨a⟩ as the 16-gons `discrete_loops` writes.)

### The remesh trap, and why it does not fire

An ⟨a⟩ loop at 10 dpa has `r = 3.77 nm = 11.7 b`; a 16-gon of that circumradius has a
chord of **4.6 b**, far below a sensible `Lmin = 25`. The natural fear is that the first
remesh collapses every ⟨a⟩ loop. It does not, and the reason is worth writing down:

`DislocationLoopNode::isGeometricallyRemovable`
(`src/DislocationDynamics/DislocationLoopNode.cpp:414`) removes a node by
`deltaArea/loopArea < relAreaTh` only when `loopArea > Lmin²`. The ⟨a⟩ loop area is
430 b² against `Lmin² = 625`, so it takes the *other* branch — which requires
`loopAreaRate < 0`. A frozen sessile loop has zero velocity, hence zero area rate, hence
returns `false`. **The loops survive precisely because they are frozen.** Verify this on
the first run rather than trusting the reading; if it does fire, drop `Lmin` to 4 and
accept the ⟨c⟩ refinement cost.

### Per-step cost, and the one setting that decides it

`DislocationNetwork.cpp:544-590` creates and updates quadrature points on **every**
network link, frozen or not, and each quadrature point sums stress from every segment in
every periodic image. With `periodicImageSize = 1 1 1` that is 27 images.

```
naive:  N_qp (~3e4) x N_seg (1e4) x 27 images = 8e9 evaluations/step
                                             ~ 30 s/step at 24 threads
```

which at 1e4–1e5 steps is 3–35 days per dose. **`useSubCycling = 1` removes essentially
all of it.** `DislocationSegment::velocityGroup`
(`src/DislocationDynamics/DislocationSegment.cpp:496`) returns
`*subcyclingBins.rbegin()` — the largest bin, 100 with the default
`subcyclingBins = 1 2 5 10 50 100` — for any segment whose average nodal velocity is
below `FLT_EPSILON`. Every frozen loop segment is exactly that. So the loops are
re-evaluated once per 100 steps while the moving line is evaluated every step:

```
subcycled: N_qp_active (~150) x N_seg (1e4) x 27 = 4e7 evaluations/step
                                                 ~ 0.2 s/step
```

a ~100× saving, *estimated*. **This must be measured on a short run before any campaign
is sized.** The loops still act as stress sources every step — only their own quadrature
update is deferred.

### Step counts

`b/cs = 1.436e-13 s` with `cs = √(μ/ρ) = 2250 m/s`.

| | strain to reach | steps *(estimated)* | wall at 0.2 s/step |
|---|---|---:|---|
| Route A, one sweep of a 500 nm cell | 6.5e-4 | ~310 at `dxMax = 5` | ~1 min |
| Route A, full ramp with pinning waits | — | 1e4 – 1e5 | 0.5 – 6 h **per dose** |
| Route B, 0.5% at 1e3 s⁻¹ | 5e-3 | ~2e6 | ~4.6 days per dose |
| Route B, 0.2% at 1e5 s⁻¹ | 2e-3 | ~1e4 | ~35 min per dose |

Route B at 1e5 s⁻¹ is affordable but the rate is 10⁸× a mechanical test; the yield it
reports is rate-dominated. Route B at a defensible rate is not affordable on a
workstation. **That asymmetry is the whole argument for Route A.**

### Software and hardware

- `DDomp` and `microstructureGenerator`, both already built under
  `MoDELib3/build_dc/tools/` (Linux ELF, run through WSL — `paths.use_wsl()`,
  `paths.modelib_ddomp()`, `paths.modelib_generator()`).
- OpenMP across all available cores; the march already uses 24.
- Memory is modest (~GB); the cost is CPU.
- Storage: set `outputFrequency` to 50–200. Per-step `evl` output on a 25 000-segment
  network is the fastest way to fill a disk.

---

## 7. Implementation plan

Nothing here needs a C++ change. The pieces already exist; what is missing is one
module that assembles them.

### Step 1 — regenerate the discrete populations at all nine doses

```powershell
.DisloClusterVenv\Scripts\python.exe -m dislocluster_code.post.discrete_loops `
    "Simulations\output\20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent" --doses all
```

This picks up the corrected `b_⟨c⟩ = 0.7972136` and gives nine `aLoops_*.txt` instead of
four. Check `manifest.json`'s `saturated` and `ratio_after` fields against §2 before
going on.

### Step 2 — new module `dislocluster_code/studies/hardening.py`

Roughly 300 lines, reusing what exists:

| function | does | reuses |
|---|---|---|
| `interior_state(run, dose)` | interior-quartile `N_k`, `r_k` per family | `movies.cd_blocks`, `discrete_loops.gb_distance` |
| `commensurate_box(L_nm, material)` | `(n, m, p)` and `F` from the real `c/a` | §6 table |
| `populate_cube(state, F, seed)` | uniform non-overlapping loop centres in the cube | `discrete_loops.LoopPopulation`, `coalesce` |
| `stage(run, dose, cfg)` | writes the whole sim dir | `discrete_loops.write_microstructure`, `staging.inputs.write_polycrystal` / `write_dd` / `write_elastic_deformation` |
| `run(sim_dir)` | generator then `DDomp` | `paths.generator_cmd`, `transition.inject_discrete_loops` pattern |
| `reduce(sim_dir)` | `F/F_0.txt` + labels → τ–γ_p curve, Δτ | new |

`populate_cube` is the only genuinely new physics-facing code, and it is a variant of
`discrete_loops.sample_family(positions="uniform")` with `box` set to the cube instead
of the marched crystal's hull.

**The `fits_in_crystal` refusal does not apply in a periodic cell** — and this is a real
simplification. `MicrostructureGenerator::insertJunctionLoop`
(`src/DislocationMicrostructure/MicrostructureGenerator.cpp:349`) reads
`ddBase.isPeriodicDomain ? true : allPointsInGrain(...)`, so in a periodic domain every
loop is accepted and wraps through the faces. The `frac_kept = 0.0096` catastrophe
CLAUDE.md records for ⟨c⟩ at 1 dpa on the 500 nm prism **is a free-surface artefact and
vanishes here**. Keep the ledger anyway; it costs nothing and it is the check that the
cube carries the same stored defects the interior does.

### Step 3 — the input files

`polycrystal.txt` — via `write_polycrystal`:

```
meshFile        = unitCube24.msh          # Library/Meshes; FEM is off, so 24 elements suffice
F               = diag(1548.0, 1548.5, 1548.2)
X0              = 0.5 0.5 0.5             # unitCube24 spans [0,1]^3; centre it
periodicFaceIDs = 0 1 2 3 4 5             # all six -> isPeriodicDomain = true
```

`DD.txt` — the deltas from `Library/DislocationDynamics/DD.txt`:

```
useFEM=0;  useElasticDeformation=1;  useElasticDeformationFEM=0;
useDislocations=1;  useClusterDynamics=0;  useInclusions=0;
glideSolverType=Galerkin;   climbSolverType=none;
timeSteppingMethod=adaptive;  dxMax=5;  dtMax=1e25;
periodicImageSize=1 1 1;  EwaldLengthFactor=1;
useSubCycling=1;  subcyclingBins=1 2 5 10 50 100;      # <- the 100x, see 6
quadPerLength=0.1;  alphaLineTension=0.1;
remeshFrequency=10;  Lmin=25;  Lmax=100;
maxJunctionIterations=1;  crossSlipModel=1;
outputFrequency=100;  outputQuadraturePoints=0;
Nsteps=100000;
```

`ElasticDeformation.txt` — §3, remembering the `mu_SI` normalization.

`initialMicrostructure.txt` — two entries: the loop file from step 1, and a mobile
source population. `Library/Microstructures/periodicDipolesDensity.txt` is the natural
choice for a periodic cell; seed ρ_m ≈ 1e12–1e13 m⁻², which in 1.25e-19 m³ is only
0.1–1.3 µm of line, i.e. a handful of sources. Use the **same seed and the same source
configuration at every dose**, so the only thing that changes is the loop field.

### Step 4 — the control run

Identical in every respect except `initialMicrostructure.txt` lists only the sources.
Δτ is a difference; without this run there is no measurement.

### Step 5 — calibrate α, then apply Route C everywhere

Run Route A at 1e-2, 0.1 and 10 dpa. Fit α_c and α_a in §5. Report Δτ at all nine doses
from the fitted law, with the three DD points marked. This is the deliverable, and it
costs three DD campaigns rather than nine.

---

## 8. Modelling requirements and limitations

State these with the result; several of them move the answer by more than the DD
statistics do.

**Frozen loops.** No absorption, no dragging, no defect-free channel formation. Correct
for *initial* yield, increasingly wrong as strain accumulates — it biases hardening high
and suppresses the strain softening that channelling produces. This is the single
largest modelling assumption in Route A.

**⟨a⟩ loops are frozen more wrongly than ⟨c⟩ loops.** A prismatic ⟨a⟩ loop carries
`b = ⅓⟨11̄20⟩` — the *same* Burgers vector as the gliding dislocations — so in reality it
both reacts strongly (junctions, absorption) and is itself glissile along its prism axis.
MoDELib marks it `SESSILELOOP` because `flow · n ≠ 0` (`DislocationLoop.cpp:28`), which
correctly forbids glide *on the habit plane* but also forbids the prism-axis glide that
is physical. ⟨c⟩ loops, `½[0001]` on basal planes, are genuinely sessile and are modelled
correctly.

**Only basal and prismatic slip exist.** `Zr3d_ghoniem.txt:28` sets
`enabledSlipSystems = fullBasal fullPrismatic`, and pyramidal drag is
`B0e_SI_py = 50e20 Pa·s` — immobile by construction. ⟨c+a⟩ slip is absent, so any load
with a large c-axis component has no accommodating slip system and the answer is
meaningless. Keep the resolved shear on prismatic ⟨a⟩.

**A hard-coded ideal c/a in the glide mobility.**
`DislocationMobilityHEXprismatic.cpp:37,62` sets the kink-pair height `h = √(8/3)/2`
= 0.8165 b, where the physical value is c/2a = 0.7972 b — 2.4% high. CLAUDE.md flags this
and says to revisit it *with the glide kinetics*. **This plan is the glide kinetics.** It
largely cancels in the increment Δτ, but it does not cancel in the absolute CRSS, so fix
it before quoting an absolute flow stress.

**The 28-parameter fit is stale, three times over.** CLAUDE.md carries three separate
warnings that compound: the Ω correction (loop densities halved), `|b_⟨c⟩| = c/2`
(⟨c⟩ radii ×1.4142), and `loop_model = 1` (fitted against mode 0). Loop densities and
sizes are internally self-consistent but **not calibrated against experiment**. The
hardening inherits that in full. Report Δτ as a *model* prediction; a comparison with
measured Δσ_y needs the refit first.

**Both loop families are outside their validated range over most of the dose list.**
`d_coarsen = 0.4 dpa` for ⟨a⟩; the ⟨c⟩ population is geometrically saturated
(`2r/d = 1.005`) from 0.1 dpa on. Doses at and below 1e-2 dpa are the ones the CD state
supports cleanly — and they are also the ones where Δτ is small. There is no way around
this that does not involve the continuum → discrete handoff running *inside* the march
(`COUPLING['discrete_transition']`, which currently raises under `loop_model = 1`).

**Single crystal → Taylor factor.** Δσ_y = M·Δτ, and M is assumed, not computed.

**One realization per dose.** Loop placement is a random draw. Δτ has a spread across
seeds of typically 10–20% in cells this size. Run three seeds per dose or quote the
single-realization caveat.

**Absent physics:** solute and oxygen hardening, SFTs, grain boundaries, hydrides,
second phases, and the free surfaces of a real cladding.

---

## 9. Alternatives, ranked

| # | Approach | Cost | What it buys | What it costs you |
|---|---|---|---|---|
| **A** | Stress-ramp CRSS on frozen loops, periodic cube | 0.5–6 h/dose | Δτ(dose) directly, at every dose | no channelling, no hardening rate |
| **C** | Dispersed-barrier law calibrated on 3 A-points | minutes | all nine doses, extrapolates beyond them | α is fitted, not predicted |
| **B** | Strain-rate periodic cell, full σ–ε | days/dose | the actual yield and hardening rate | rate 10⁵–10⁸× experiment |
| **D** | Density-seeded loops via `aLoopsDensity.txt` instead of per-loop export | as A | no export pipeline at all — two numbers per family per dose | loses the spatial correlation of the CD field |
| **E** | Free-surface 500 nm prism, the marched geometry itself, `useFEM=1` | as B | comparable to a micro-pillar test; uses the real spatial field | not a bulk yield strength; surface sources dominate |
| **F** | Allow loop absorption (`climbSolverType=Galerkin`, or unfreeze) | ≫ B | channel formation, strain softening | far beyond a workstation; needs the runtime handoff first |

**D deserves a second look.** `tutorials/irradiation_singlecrystal/generateInputFiles.py`
already seeds both families by density — `aLoopsDensity.txt` with `slipSystemIDs`,
`targetDensity`, `loopRadiusMean` per variant, and `frankLoopsDensity.txt` for the basal
family. Feeding it the four `(N_k, r_k)` pairs from §2 reproduces the RVE with **no new
Python at all**, and is the fastest possible route to a first number. It discards the
spatial correlation the CD field carries — but the interior quartile is nearly
homogeneous anyway, so for Route A the loss is small. **Do D first as a smoke test, then
build the export in §7 for the production runs.**

---

## 10. What to do first

1. `discrete_loops --doses all` on the run, and check `manifest.json` against §2.
2. **Route D smoke test at 10 dpa**: 200 nm periodic cube, `aLoopsDensity` +
   `frankLoopsDensity` at the §2 numbers, one periodic dipole, stress ramp, 2 000 steps.
   The point is not the answer — it is to measure the per-step cost with and without
   `useSubCycling`, and to confirm the ⟨a⟩ loops survive the first remesh (§6).
3. Size the campaign from that measurement, then build `studies/hardening.py` (§7).
4. Route A at 1e-2, 0.1, 10 dpa plus the loop-free control; fit α; publish all nine
   doses through Route C with the three DD anchors marked.
