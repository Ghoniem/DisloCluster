# Zr3d_ghoniem — differences from upstream MoDELib

**Branch:** `zr3d_ghoniem`  ·  **Status:** in progress, see §6

This tree entered DisloCluster as a git submodule pointing at
`https://github.com/Ghoniem/MoDELib2-NNL.git`, branch `zr3d_ghoniem`. Commit
`859af47` converted it to a plain directory, so that `.gitmodules` is gone and
this line is the only remaining record of where the checkout came from.
**MoDELib2-NNL is both the fork's name and the baseline this file is written
against** — see below, and note that an earlier revision claimed otherwise.

Zr3d_ghoniem is a variant of MoDELib whose cluster-dynamics equations and parameters
are taken from the **ZrMicro 0-D code** (`ZrClusterDynamics/ZrMicro`, relative to
the DisloCluster repository root), so that the 3-D spatially-resolved solve
reduces to the 0-D result in the well-mixed limit.

The governing equations are those of Po & Ghoniem, Deliverable D1/M1
(`Docs/Formulation/GW_Phase4_D1M1_NG.pdf`), §2.2 in particular.

## Which upstream — read this before comparing anything

**The baseline for this file is [`mlm335/MoDELib2-NNL`](https://github.com/mlm335/MoDELib2-NNL)**,
at the point the immobile population equations were written into it. At that point it solved
the mobile steady-state cluster-dynamics PDE and nothing else: `iSize` was **0**,
`solveImmobileClusters()` was an empty function body, there was no `ImmobileSinkRate.h` and no
immobile integrator of any kind, and the source carried a literal `// Missing immobile sinks`
placeholder at `ClusterDynamicsFEM.cpp:110`. The loop populations were **absent, not present in
a simplified form**.

[`mlm335/MoDELib-fullCD`](https://github.com/mlm335/MoDELib-fullCD) is a **separate**
cluster-dynamics branch — `iSize` 8, with `ImmobileSinkRate.h`, `SpatialODESolver.h`,
`FirstOrderReaction.h` and a continuum→discrete loop conversion. It is a **sibling, not the
ancestor**.

### A correction to earlier revisions of this file

Earlier revisions declared fullCD the correct baseline and described the immobile solver below
as a **re-discretization** of fullCD's scheme. That was inferred from 31 CD parameter keys
shared verbatim plus a set of systematic renames (`loopNucDefects`→`nNuc`,
`loopCoalLL/LN`→`cLL/cLN`, `loopCoalKappa*`→`kappa*`, `loopAnnealTau0_SI`→`tau0_vLoop_SI`,
`loopCoalNetwork_SI`→`rhoNetwork_SI`, `minimumLoopSize`→`r_min`).

**That inference does not follow.** Shared naming across two branches of one code, developed in
one group from one formulation (Po & Ghoniem D1/M1 §2.2), is evidence of *convergence*, and says
nothing about which came first. The immobile equations here were written to reproduce ZrMicro,
not to re-express fullCD.

### Comparing against the sibling

fullCD remains worth comparing against wherever the comparison is informative. The two immobile
schemes differ at each of three decision points:

| | MoDELib-fullCD (sibling) | this branch |
|---|---|---|
| Rate | Galerkin: assembled at quadrature points, then L2-projected through the consistent mass matrix (CG to 1e-4, `SpatialODESolver`) | nodal collocation; no mass matrix |
| Time update | explicit Euler `dof += rate*dt`, then **sequential** implicit loss factors `(1+dt·λ_k)^-1` per channel | one **fused** semi-implicit update `(n + dt·nucRate)/(1 + dt·lossN)` |
| Sub-cycling | none — one step of `dtMax` | `nSub = 20` per dose step |

New here relative to the baseline, beyond the immobile equations themselves:
`dadAnisotropy`/`dadZ0` (DAD reconciliation with the 0-D), `atomicVolume_SI`,
`concentrationFloor`, `loopSinkScale`, size-dependent vacancy-loop emission, the bi-pyramid
family.

The one capability fullCD has and this branch does not is `clusterDiscretizationTime` /
`initializeDiscreteClimbLoops()`, which converts the continuum loop field into discrete climb
loops once the dose passes a threshold. That is a **gap, not a regression** — nothing of the
kind existed in MoDELib2-NNL to lose — and it is the piece most worth taking from fullCD.

Full numbered history: `Docs/DisloCluster Manual/DisloCluster Code History.tex`.
Plan for closing the gap: `Docs/DisloCluster Manual/anisotropic_diffusion_and_discrete_coupling_plan.md`.

`Library/Materials/Zr4.txt` is **untouched**. The Zr3d_ghoniem material definition is a
new file, `Library/Materials/Zr3d_ghoniem.txt`. The C++ changes are global: `iSize`
goes from 0 to 8 relative to MoDELib2-NNL, so `Zr4.txt` cannot be run from this branch.
Note that `Zr4.txt` here carries no immobile keys, but that is a property of this one file
and **not** evidence about upstream: fullCD's `Zr4_Fitted.txt` and `Zr4_BMD19_nuc.txt` each
carry the full sixteen-key immobile set.

---

## 1. Build and environment

Not physics — required to build on WSL/Ubuntu rather than macOS.

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` | `EIGEN3_INCLUDE_DIRS` `/opt/local/include/eigen3` → `/usr/include/eigen3`; dropped the hard-coded Qt prefix path and macOS `RPATH` | Paths were absolute to a macOS MacPorts install |
| `tools/CMakeLists.txt` | `DDqt` subdirectory commented out | Needs Qt; not required for CD runs |

Two build traps worth recording (handled in `Docs/Formulation/build_modelib_wsl.sh`):

- The repo declares `cmake_minimum_required(VERSION 3.1.0)`. CMake ≥ 4.0 **rejects**
  anything below 3.5 outright, so `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is mandatory,
  not defensive — Ubuntu 26.04 ships CMake 4.2.3.
- `libfftw3-dev` is **not optional**, despite `CMakeLists.txt:65` warning that the noise
  generator is merely "disabled" when it is missing: line 117 still links
  `${FFTW3_LIBRARIES}` unconditionally, so the generate step fails without it.

---

## 2. C++ changes

### 2.1 `include/DislocationDynamicsBase/ClusterDynamicsParameters.h`

- **`iSize` 0 → 8.** Four immobile families × (number density, content):
  `[N_c, N_a1, N_a2, N_a3 | c_c, c_a1, c_a2, c_a3]`. This is what turns on the
  immobile machinery throughout; it is the reason `Zr4.txt` is incompatible.
- **New parameters** for the §2.2 kinetics: `loopCascadeFractions` (ε_k, Eq. 51),
  `nNuc` (Eq. 36), `cLL`/`cLN` (Eq. 99), `kappaLL`/`kappaLN` (Eq. 98),
  `tauVac` (Eq. 95), `rhoNetwork`, `dadAnisotropy` (p_m), `dadZ0` (Z⁰_m).
- **New method `loopDADbias()`** — the Eq. (15) capture efficiencies,
  `Z_c = Z⁰·p`, `Z_a = Z⁰·(p + p⁻²)/2`, row 0 = ⟨c⟩ (vacancy) loops, row 1 = ⟨a⟩.
- **New static `getClusterAtomicVolume()`** — see issue 6 below.
- **New member `loopNucChannels` and method `getLoopNucChannels()`** — the subset
  of `reactionMap` whose product size is carried by no mobile species, i.e. the
  homogeneous-clustering reactions that nucleate loops. See issue 17 below.

### 2.2 `src/DislocationDynamicsBase/ClusterDynamicsParameters.cpp`

- Initializers for all of the above, each guarded by `iSize>0` and converted to
  MoDELib units on read (`tauVac` seconds → b/cs, `rhoNetwork` m⁻² → b⁻²).
- `omega(ddBase.poly.Omega)` → `omega(getClusterAtomicVolume(ddBase))`.
- `r_min` now divided by `b_SI` on read (issue 5).
- `getLoopNucChannels()` implemented, and the selected channels printed at
  start-up with their coefficient in 1/s so they can be checked against the 0-D
  jump frequencies by eye.

### 2.2b `mBWF` / `dmBWF` scale with `cdp.omega`, not `ddBase.poly.Omega`

`FluxMatrix` divides `D` by `cdp.omega`, and the weak form multiplies the flux back by
an atomic volume, so the two must be the *same* quantity for the flux to come out as
`−D·∇c`. They were identical until the CD atomic volume became settable independently
of the lattice cell volume (issue 6); leaving `poly.Omega` here would have scaled the
entire diffusion operator by `poly.Omega/cdp.omega` = 3.98 relative to every reaction
term. See issue 13.

### 2.3 `include/ClusterDynamics/ImmobileSinks.h` — **new file**

An `EvalFunction` mirroring the existing `SecondOrderReaction`, supplying the loop
absorption sink that was previously missing (the source carried a
`// Missing immobile sinks` comment). For each mobile species *m* it assembles

    k²_m = Σ_k Z(type(k), m) · S_k ,        S_k from clusterDensity() = 2πr_k N_k

and returns `−k²_m · D̄_m` on the diagonal, matching the sign convention of `getR1()`.

Per your instruction, the sink set is the 0-D one **except the grain boundary**,
which is imposed spatially by the Dirichlet BC on the mobile species and must not be
double-counted here. The network dislocation sink is not in this file either — it
reuses the existing `otherSinks_SI` plumbing (§3, `otherSinks_SI`), which required no
new code.

### 2.4 `src/ClusterDynamics/ClusterDynamicsFEM.cpp`

- **`ImmobileSinks` wired into the mobile Newton system** — `bWF_RI` into the
  Jacobian and `lWF_RI` into the residual, alongside the existing `R1`/`R2` terms.
- **`solveImmobileClusters()` implemented** (was an empty stub). Node-wise
  sub-stepped integration of Eqs. (34)–(56) and (97)–(102) over one dose step:
  cascade nucleation, absorption growth with the Eq. (91) shrink gate, Avrami
  coalescence in two channels (like-loop conserves content, loop-network removes
  both), vacancy-loop dissolution, and thermal emission. DOF layout is
  `index = node*iSize + component`, `[0,nF)` densities and `[nF,2nF)` contents.
- **Homogeneous-clustering (SIA) nucleation added** to both the density and the
  content equations, evaluated per node from `cdp.loopNucChannels` and the local
  mobile field, and split across the families of matching polarity in proportion
  to `loopCascadeFractions`. See issue 17 below.

### The mobile fixed-point loop: an inherited non-termination bug

`solveMobileClusters` iterates `while(cError>cTol)` with `cTol = 1e-5`, **no iteration
cap and no stagnation test**. That code is inherited verbatim — MoDELib2-NNL and
MoDELib-fullCD both have the identical loop — and it is safe only while the iteration
always converges. It does not always converge.

Solving the mobile field against an immobile state supplied by an external march (the
ZrMicro coupled route), `cError` falls from 5.8e5 to 1.3e-4 in nine iterations and then
**stagnates**, oscillating between 4.7e-5 and 1.2e-4 in near-identical successive pairs.
With no cap, DDomp never returns. Diagnostics show the worst species is always `Ci` and
that 7090 of 96460 mobile dofs sit on the positivity floor, a count that does not change
from iteration to iteration.

The one structural difference from upstream is the cause. fullCD has **no mobile clamp at
all**; this branch inserted `clampMobileClusters()` between the Newton update and the error
test, which turns the iteration into a *projected* Newton method.

The precise failure is an inconsistency between the linear system and the constraint, and
it is worth stating exactly, because the obvious explanation is wrong. The active set does
**not** oscillate: an instrumented run shows it freeze at 7090 dofs by iteration 3 and never
change again (`set +0/-0` on every subsequent iteration). What fails is that the Newton
matrix `dmBWF+bWF_R1+bWF_R2+bWF_RI` is the **unconstrained** Jacobian — it does not know any
dof is pinned. Each iteration it computes an increment for all dofs, including moves for the
pinned ones; the projection then discards exactly those moves, but the increment for the
**free** dofs was computed assuming the pinned ones would move. The free dofs therefore
respond every iteration to a coupling that is immediately cancelled, and that inconsistency
does not decay.

Solve-then-project is simply not a consistent method for a constrained system. The
consistent form is an active-set method: eliminate the pinned rows and solve the reduced
system. Removing the projection avoids the issue entirely (convergence becomes quadratic);
under-relaxation only makes the inconsistent map a contraction, which is why it converges
linearly and needs four times as many iterations.

Two further observations follow from the same mechanism and were both confirmed:

- Changing the **error metric** cannot help. A floored dof has `cOld = cNew = floor` and so
  contributes exactly zero to the step norm already; excluding floored and Dirichlet dofs
  (`mobileSolverErrorMode=1`) reproduces the baseline iteration *bit for bit*, all 16 digits,
  iteration for iteration. The free dofs are genuinely oscillating.
- Relaxing `cTol` to 1e-4 "converges" at iteration 19 — but only because the oscillation
  happens to dip below the loosened bar, and the field it then accepts is **49.9% wrong**
  in `Ci` at the worst node. It is not a fix; it declares victory on the bad field.

The comment on `clampMobileClusters()` states the intent exactly, and shows why the
placement is wrong: it mirrors ZrMicro, "which applies `y = max(y, C_floor)` **before every
rate evaluation**". In ZrMicro the floor is part of the *rate evaluation*. Here it mutates
the *state* inside the iteration, which is a different operation, and only the second one
breaks convergence.

**Measured, all from the identical saved immobile state (24 115 CD nodes, 96 460 mobile dofs):**

| configuration | iters | converged | final `cError` | wall |
|---|---:|---|---:|---:|
| clamp in loop, undamped (as shipped) | 30 (capped) | no | 6.4e-5, cycling | 392 s |
| **clamp deferred out of the loop** (fullCD) | **5** | **yes** | **6.83e-7** | **112 s** |
| clamp in loop, damped `w=0.5` | 30 (capped) | not yet | 9.4e-5, still halving | 400 s |
| clamp in loop, damped `w=0.7` | 20 | yes | 8.2e-6 | 283 s |
| clamp in loop, `errorMode=1` | 40 (capped) | no | 5.1e-5 (= baseline exactly) | 492 s |
| clamp in loop, `cTol=1e-4` | 19 | "yes" | 9.2e-5 | 258 s |

Accuracy of each resulting field, as the maximum relative difference in `Ci` against the
converged solution, over the 22 344 nodes above the floor. Only the two genuinely converged
configurations recover it:

| configuration | max rel. difference in `Ci` |
|---|---:|
| clamp deferred (converged, 6.8e-7) | — (reference) |
| damped `w=0.7` (converged, 8.2e-6) | **0.19%** |
| `errorMode=1` (capped) | 21.1% |
| `cTol=1e-4` ("converged") | 49.9% |
| baseline (capped) | 53.6% |

Deferring the clamp restores **quadratic** convergence — 2.95e4 → 1.23e3 → 6.77e1 → 5.73e-2
→ 6.83e-7 — i.e. a genuine Newton method, and it is faster than the solve has ever been.
Under-relaxation also converges, but only linearly (a factor ≈2 per iteration in the tail),
so it needs ~40+ iterations to reach the same tolerance.

Two checks on the deferred-clamp result, both passed:

- **Physicality.** The resulting field has zero negative entries, exactly 7090 dofs on the
  floor, and the same min/max as the clamped runs. Deferring the floor does not leak
  negative concentrations into the answer.
- **Same fixed point.** The damped clamp-in-loop iterate agrees with it to within 4% on
  every species while still 1e-4 from convergence, and their means agree to 4–5 figures.

The stalled iterate is not merely unrecognized as converged: against the converged field its
means agree to four figures, but **2 nodes exceed 1% and one reaches 53.6% on `Ci`**
(3.11e-16 converged against 4.77e-16 stalled). A handful of nodes hovering at the clamp
boundary drives the cycle, and because the error test is a max-norm over nodes, those few
alone prevent it from ever settling.

Five optional DD.txt scalars now control the loop. All default to the historical behavior
except the iteration cap, which converts a hang into a bounded, reproducible result:

| key | default | meaning |
|---|---|---|
| `mobileSolverTolerance` | `1e-5` | `cTol` |
| `mobileSolverMaxIterations` | `200` | cap; `0` = unlimited (the old behavior). On hitting the cap the **best** iterate seen is restored, not the last |
| `mobileSolverRelaxation` | `1.0` | `w` in `c += w·increment`; `w<1` damps the cycle |
| `mobileSolverErrorMode` | `0` | `0` all dofs (historical), `1` exclude floored and Dirichlet dofs, `2` max nodal `\|dc\|/(\|c\|+floor)` |
| `mobileSolverClampInLoop` | `1` | `1` clamp every iteration (this branch), `0` clamp once after the loop (fullCD) |

Each iteration now prints the iteration number, which species attains the maximum, and how
many dofs the floor moved.

### `useImmobileSolver` — running the fast step alone

`ClusterDynamicsFEM::solve()` already *is* the two-time-scale split:

```cpp
solveMobileClusters();        // FAST — steady C_M*(x). No time derivative
                              // anywhere; the immobile population enters only
                              // through ImmobileSinks, evaluated at every
                              // quadrature point from the LOCAL state.
solveImmobileClusters();      // SLOW — dt update of the nodal n_I, c_I
```

A new DD.txt scalar `useImmobileSolver` (default 1 when the key is absent, so
every pre-existing case is unaffected) makes the second call conditional. With
`useImmobileSolver=0` a DDomp run performs the fast step and passes the immobile
field through untouched, which lets an external driver own the slow step. That
is how the ZrMicro coupled march works: it writes its per-node immobile state
into the `evl` CD block, calls DDomp for one step to obtain C_M*(x) for *that*
state, freezes it, and integrates the immobile ODEs at every node with CVODE.

Because `runSingleStep` writes output *before* incrementing `runID`, a run with
`startAtTimeStep=0` and `Nsteps=1` reads `evl_0.txt` and overwrites the same
file — an in-place round trip. `py_utils/modelib_qssa.py` drives it and asserts
on the `immobile solver SKIPPED` line, so a binary predating this flag fails
loudly rather than advancing the immobile field twice.

Files: `include/ClusterDynamics/ClusterDynamicsFEM.h`,
`src/ClusterDynamics/ClusterDynamicsFEM.cpp`.

---

## 3. `Library/Materials/Zr3d_ghoniem.txt` vs `Zr4.txt`

Everything outside the `# ClusterDynamics` section (elasticity, mobility laws,
gamma-surface) is inherited unchanged. Original values are preserved in comments
in the file itself.

### 3.1 Values changed to match ZrMicro

| Key | Zr4 (Po) | Zr3d_ghoniem (ZrMicro) | Basis |
|---|---|---|---|
| `atomicVolume_SI` | *(absent)* | `1.2e-29` | ZrMicro `physical_props['Omega']` |
| `mobileSpeciesSurvivingEfficiency` | `0.01` | `1.0` | 0-D loses defects only to the in-cascade cluster fractions |
| `mobileSpeciesCascadeFractions` | `1.0 0.6 0.3212 0.0788` | `0.995 0.97788714 0.005 0.00333333` | `G_v/G = 1−ε_vL`, `G_i/G = 1−ε_2i−ε_3i−ε_iL`, `G_2i/G = ε_2i/2`, `G_3i/G = ε_3i/3` |
| `mobileSpeciesEnergyFormation_eV` | `1.5 …` | `1.6 …` | 0-D `E_F_v`, gives `C_v_eq = 8.45e-15` at 573 K |
| `mobileSpeciesEnergyMigration_eV` | `0.9 / 0.09 / 0.041–0.187` | `1.2 / 0.759101 / 0.759101 / 0.759101`, isotropic | 0-D `E_m_v`, `E_m_i`; clusters transport at the monomer value (issue 11) |
| `mobileSpeciesD0_SI` | `4.9639e-10`, `8.1882e-10` | `5.21645e-7` for all, isotropic | `D = (a²/z_c)·ω = a²ν e^(−E_m/kT)` ⇒ `D0 = a²ν`, a = 3.23e-10 m, ν = 5e12 s⁻¹ |
| `reactionPrefactorMap` | `2.856`, `1.0`, `0.0` | all ten recomputed | §3.3 |
| `Eb_eV` | `5.0 5.0 5.0 5.0` | `5.0 5.0 0.957033 2.0` | index 0 → loop emission (0-D uses e⁻⁵ᵉⱽ/ᵏᵀ); 2, 3 → 0-D `E_b_2i`, `E_b_3i` |
| `otherSinks_SI` | `0 0 0 0` | `1.88e14  1.8988e14 ×3` | network dislocation sink, `k²_v = ρ_N`, `k²_i = Z_N ρ_N` |

The migration-energy row order matters: components are `[11 12 13 22 23 33]`, and the
off-diagonal `D0` entries are zero, so the tensors are diagonal.

### 3.2 New §2.2 block

`immobileSpeciesVector`, `immobileSpeciesRelRelaxVol`, `immobileSpeciesBurgers`,
`alpha_bp`, `delVPyramid`, `w0`, `n_s`, `evc`, `Nvmax`, `nmin`, `nmax`, `r_min`,
`distanceFactor`, `loopCascadeFractions`, `nNuc`, `cLL`, `cLN`, `kappaLL`, `kappaLN`,
`tau0_vLoop_SI`, `Ea_vLoop_eV`, `rhoNetwork_SI`, `dadAnisotropy`, `dadZ0`.

The coalescence coefficients, nucleation sizes and dissolution constants are the
fitted 0-D values from run `20260622_144021_7959445`.

### 3.3 How the reaction prefactors were derived

MoDELib forms `K_ab = p_ab·4π(r_a+r_b)(D_a+D_b)/Ω` in `getR2()`, with
`r = (3Ω/4π)^(1/3)` for `|m| = 1` and `r = √(|m|Ω/(bπ))` otherwise.
ZrMicro forms `K_ab = f_ab(ω_a+ω_b) = f_ab(z_c/a²)(D_a+D_b)`, with `f = 0.5` for v+i
(`physical_props['recom']`) and 1 otherwise, the like-species channels carrying the
extra factors of `R_i_i = 2ω_i C_i²` and `R_2i_2i = 4ω_2i C_2i²`. Equating,

    p_ab = f_ab · (z_c/a²) · Ω / (4π(r_a+r_b)) · (D⁰ᴰ_a+D⁰ᴰ_b)/(D³ᴰ_a+D³ᴰ_b)

The last ratio is unity except for pairs involving 2i/3i, where it restores the
ω_2i-based rate now that those species transport at the monomer diffusivity
(ω_2i/ω_i = 4.886e-6).

**This derivation was confirmed empirically.** MoDELib prints the assembled
second-order interaction matrices in 1/s at start-up:

| entry | MoDELib | ZrMicro | agreement |
|---|---|---|---|
| (v,i) | 2.526876e7 | 0.5(ω_i+ω_v) = 2.52560e7 | 0.05% |
| (i,i) | −2.021161e8 | −4ω_i = −2.020216e8 | 0.05% |
| (v,2i) | 6.940722e3 | ω_2i+ω_v = 6.93473e3 | 0.09% |

This also settles a factor-of-2 ambiguity: `SecondOrderReaction.h` documents the rate
as `ċ = ½R₂cc` while `getR2()` fills both `(a,b)` and `(b,a)`, and it was not clear
from inspection whether the ½ survives into the weak form. The match above shows it does.

### 3.4 `loopSinkScale` — reproducing ZrMicro's loop sink convention exactly

ZrMicro's loop absorption is *not* the purely geometric `S_k = 2πr_k N_k`. It departs
from it in two ways, and `loopSinkScale` reproduces both exactly:

**(a) One radius prefactor for all four families.** `input_data.py` derives both
`l_a = √(Ω/πb_a)` and `l_c = √(Ω/πb_c)`, but `reaction_rates.loop_absorption()` uses
`lc_l = l_c/l` for *every* family — `l_a` is computed and never used anywhere in the
codebase. The 3-D builds `r_k` from each family's own `|b_k|`, so each needs
`√(|b_k|_MoDELib / b_c_ZrMicro)`, with b_c = 5.15e-10 m against MoDELib's
1.632993·b_SI = 5.279467e-10 m (⟨c⟩) and 1.0·b_SI = 3.233e-10 m (⟨a⟩):

| family | factor |
|---|---|
| ⟨c⟩ | √(5.279467/5.15) = 1.012495 |
| ⟨a⟩ | √(3.233/5.15) = 0.792317 |

**(b) The fitted `Q` on the ⟨c⟩ channel only:** ⟨c⟩ → 1.012495 × 0.287931 = **0.291528**.

    loopSinkScale = 0.291528  0.792317  0.792317  0.792317

Implemented as an optional material key (default `1 1 1 1` = purely geometric),
applied identically in `ImmobileSinks.h` (the sink seen by the mobile species) and in
`solveImmobileClusters()` (the growth flux it causes) — mirroring the 0-D, where
`loop_absorption()` and the `loop_growth_rate_*` methods share one prefactor. Applying
it in only one place would break the mass-conserving coupling between them.

---

## 4. Simulation case `tutorials/zrmicro_coupled/` — new

CD-only verification case: `useFEM=1`, `useDislocations=0`, `useClusterDynamics=1`,
`dtMax = 6.95868964e19` (= 1 dpa at G = 1e-7), `outputFrequency=1`,
`startAtTimeStep=0`, `absoluteTemperature=573`, non-periodic, zero target loop
densities. `clean_run.sh` refreshes the material file and regenerates `evl_0` on every
invocation — see issues 1 and 2.

---

## 5. Issues encountered, and how each was solved

**A recurring hazard, stated once:** almost every failure in this stack **exits 0** —
the build script, `DDomp` with a missing `evl_0`, a missing material key, a CRLF mesh,
and the solver failures below. Verification has been by artifacts on disk and log
tails, never by exit status.

### 1. `microstructureGenerator` never terminates
**Symptom:** hangs indefinitely at start-up.
**Cause:** it is the *discrete-dislocation* path. It inserts loops one at a time; the
log showed it had reached 2.34e17 m⁻³ against a ZrMicro target of 1.48e21 — 0.016% of
the way, needing ~10¹¹ discrete loops. CD loop densities are not representable as
discrete dislocations, which is precisely why they belong in the continuum fields.
**Fix:** `useDislocations=0` and `targetDensity=0` in `frankLoopsDensity.txt` /
`aLoopsDensity.txt`, giving an empty 20-byte `evl_0.txt` in about a second. `DDomp`
still requires that file to exist even with dislocations off.

### 2. Restart silently double-counted dose
**Symptom:** ⟨a⟩ loop content came out at exactly 3× the expected value.
**Cause:** `startAtTimeStep=-1` means "last available step", and `evl_0.txt` serves
double duty as both the initial configuration *and* the runID=0 output. A bare re-run
therefore resumed from the previous run's accumulated dose.
**Diagnosis:** `c_a` max = 2.112e-3 = exactly 3 × (G₀·ε_a·1 dpa).
**Fix:** pinned `startAtTimeStep=0`, and `clean_run.sh` now regenerates `evl_0` every
time. Also note output is written **after** `solve()`, so `evl_N` holds the state at
*(N+1)* dose steps.

### 3. Stale material file in the simulation directory
**Symptom:** run died on `does not cointain line with format immobileSpeciesVector=…`
**Cause:** `inputFiles/` holds a *copy* of the material file, made before the §2.2
block existed.
**Fix:** `clean_run.sh` re-copies it from `Library/` on every run.

### 4. CRLF line endings silently produced a zero-node mesh
**Symptom:** downstream periodicity failure with no parse error.
**Cause:** `unitCube_15K.msh` checked out with CRLF; the `.msh` parser returned zero
nodes without complaint.
**Fix:** `sed -i "s/\r$//"` on the simulation-directory copies only, leaving the
`Library/` originals untouched.

### 5. `r_min` never converted from metres
**Cause:** read raw from the material file in metres, but compared against radii from
`rloop()`/`rpyr()`, which are in units of b. The Eq. (91) shrinking gate was therefore
permanently wide open.
**Fix:** divide by `b_SI` on read. It is used only by the new code, so the conversion
is safe.

### 6. Atomic volume was the lattice *cell* volume
**Cause:** `Polycrystal::Omega` is `det(latticeBasis)` = 1.41421 b³ for HEX — the
volume of a cell containing **two** atoms, not the atomic volume the CD equations
assume. Every content ↔ radius ↔ density conversion inherited the error, and it
differs from ZrMicro's 1.2e-29 m³ by roughly a factor of 4.
**Fix:** new **optional** material key `atomicVolume_SI`, read by
`getClusterAtomicVolume()`; material files that omit it keep the previous behaviour,
so `Polycrystal` — which the elastic/DD side also uses — is left alone.
**Noted for the record:** ZrMicro's 1.2e-29 m³ implies ~12.6 g/cm³ for Zr, about half
the true atomic volume. It is baked into the fitted parameter set, so Zr3d_ghoniem matches it
rather than "correcting" it.

### 7. ⟨c⟩ loops collapsed onto the positivity floor
**Symptom:** `n_vL` pinned at 1e-30 while ⟨a⟩ loops nucleated and grew normally.
**Cause:** τ_vL ≈ 9.95e16 MoDELib time units against a sub-step of 3.48e18 — the decay
time is **35× shorter than the step**, so explicit Euler gave `n(1−35) < 0`, clamped
every iteration.
**Confirmation it was integration and not physics:** the analytic balance
`n* = ṅ_nuc·τ_vL = 1.79e-8` matches the 0-D `CvL+CavL = 1.78e-8`.
**Fix:** integrate the linear losses implicitly — unconditionally stable, and returns
`n* = ṅ_nuc·τ_vL` exactly as `dt/τ → ∞`. Verified: `n_vL = 1.79161e-08`.

### 8. Thermal emission written as a rate, not a coefficient
**Cause:** the Eq. (55)–(56) emission term was computed as an absolute rate rather
than proportional to the stored content, so it could not enter the implicit update.
**Fix:** per loop `α_v = 2πr·D_v/Ω·e^(−E_b/kT)` (the convention of `getR1()`); the
content lost per unit volume is `α_v·N·Ω = S_k·D_v·e^(−E_b/kT)`, the Ω's cancelling.
Since that does not scale with *c*, it is divided by *c* to form an equivalent
first-order coefficient.

### 9. Loop densities stored per atom instead of per b³
**Cause:** `clusterRadius()`, `clusterDensity()` and `sigmoid()` all form
`n = c/(N·Ω)`, i.e. they expect **N in loops per b³**. The new integration was
accumulating loops *per atom*.
**Consequence:** the helpers read the mean loop size as 2.82× too large — enough to
push the ⟨c⟩ family across the bi-pyramid→loop sigmoid onto the wrong branch.
**Fix:** store per b³ (`nucRate` divided by Ω), `Nvol = n(k)` directly, and
`dLL = (1/N)^(1/3)`. Content needs no conversion: a volume fraction is numerically
equal to the 0-D per-atom content.

### 10. ⟨a⟩ Burgers vectors were not equivalent
**Symptom:** `n_a1, n_a2, n_a3` = 2.84e-7, 2.32e-7, 3.16e-7 at **zero applied stress**,
where the three prismatic families must be identical.
**Cause:** `immobileSpeciesBurgers` columns are multiplied by `latticeBasis`, i.e.
they are **lattice** components; Cartesian ones had been supplied. The resulting
magnitudes were |b| = 1.0, 0.753, 1.197 instead of all 1.0.
**Fix:** `(1,0,0), (0,1,0), (−1,1,0)` for ⟨a⟩ and `(0,0,1)` for ⟨c⟩. Verified:
`immobileSpeciesBurgersMagnitude: 1.633, 1.0, 1.0, 1.0`.
**Standing test:** at zero stress the three ⟨a⟩ family densities must agree.

### 11. Cluster diffusivity made the Newton matrix singular
**Symptom:** `FixedDirichletSolver failed.` — thrown from `compute()`, i.e. the
factorization, not the solve.
**Cause:** setting `E_m_2i = 1.36292 eV` drops D_2i to ~5e-6 of D_i. The mobile solve
is a **pure steady-state (QSSA) system** — `dmBWF` is diffusion only, with no mass
term — so the 2i/3i block had nothing on its diagonal but a vanishing Laplacian.
**Fix, and why it is *more* faithful to ZrMicro:** the 0-D is inconsistent about which
frequency it applies where, but unambiguous — `R_2i_s`/`R_3i_s` (network sinks) and
`flux_i` (loop absorption) all use **ω_i**, and ω_2i appears *only* in the
mobile–mobile cluster reactions. So 2i/3i transport at the monomer diffusivity, and
the ω_2i/ω_i factor is carried in the reaction prefactors instead (§3.3), where it
belongs. The network-sink and loop-absorption terms now reproduce the 0-D exactly,
which they did not before.

### 12. Mobile Newton iteration does not converge on the first step — **OPEN**
**Symptom:** `convergenceError = 2.9e5` after the first Newton iteration, then the
iterative (BiCGSTAB) solve fails on the second.
**Cause:** cold start. `mobileClusters` is initialized to the *thermal equilibrium*
concentration (~1e-15) and must reach the irradiation steady state (~1e-6) — nine
orders — in one Newton solve of a quadratically nonlinear system. It converged before
only because production was 100× smaller.
**Attempted and rejected:** switching `rSolver` to the direct SparseLU branch. That
fails *earlier*, in `compute()` itself, on the ~96k-dof 3-D matrix, while the LLT
factorization of the pure diffusion operator in `mSolver` succeeds — so the direct
path is not viable here at this size. Reverted, with a comment at the call site.
**Note:** `dt` does not enter the mobile equation at all (it is a steady-state solve),
so ramping the dose step cannot help. The remedy has to be a better initial guess —
seeding the mobile field from the 0-D steady state, which is exactly what the
two-time-scale coupling provides — or damping/continuation in G.

### 13. Diffusion operator scaled 3.98× relative to the reactions
**Cause:** introduced by the fix to issue 6. `FluxMatrix` returns `−D/cdp.omega`, and
`mBWF`/`dmBWF` multiplied it back by `ddBase.poly.Omega`. Those two were the same
number until the CD atomic volume became independently settable; afterwards the
diffusion term carried a spurious factor of `poly.Omega/cdp.omega` = 1.41421/0.355105
= 3.98 against every reaction term.
**Fix:** use `cdp.omega` in both weak forms, restoring the exact cancellation so the
flux is `−D·∇c`. Found by reading the constructor while chasing issue 12 — a factor
of four on the diffusion operator is a plausible contributor to that non-convergence,
so issue 12 should be re-tested now that this is corrected.

### 14. Coalescence was integrated explicitly and overshot
**Symptom:** ⟨a⟩ mean loop size m_a = 5042 at 1 dpa, then exactly 300.0 (= n_nuc) at
2 dpa — content collapsing 11× while density kept climbing.
**Cause:** the same stiffness failure as issue 7, in the channel left explicit.
`nu_LN*phi_LN*dt_sub` reaches ~3.6, so each sub-step overshoots and clamps to the floor.
**Fix:** Eq. (97) removes loops in proportion to n and content in proportion to c, so
both are first-order coefficients; moved into the implicit update.
**Effect:** ⟨a⟩ loop density went from 3–4× the 0-D to within ~30%.

### 15. Vacancy-loop annealing released the wrong content
**Symptom:** ⟨c⟩ loop content 350× below the 0-D; m_c = 487 against n_nuc = 400,
i.e. pinned at the nucleation size, versus 1.7e5 in the 0-D.
**Cause:** dissolution was implemented as removing content in proportion to the
content, `c/tau_vL`. ZrMicro's `annealing_content_vL()` returns
`n_vL_nuc * CvL / tau_vL` — each dissolving loop releases its **birth** content, not
its mean, because tau_vL describes dissolution of still-small EMBRYO loops while
loops that survive and grow are stable. Removing the mean content instead caps the
mean size at ~n_nuc.
**Fix:** content sink proportional to the DENSITY, `n_nuc*n*Omega/tau_vL`.
**Self-check:** at number saturation `n* = Ndot_nuc*tau_vL`, so the release equals
`n_nuc*Ndot_nuc*Omega = Gk` and exactly cancels the cascade content seed, leaving net
content growth flux-driven — precisely what the 0-D docstring describes.
**Effect:** ⟨c⟩ content from 350× low to within 20–26%.

### 16. `Eb` read before initialization in `getR1()` — **pre-existing upstream bug**
**Symptom:** C2i and C3i ~1e-17 against the 0-D's ~2.5e-11, six orders low.
**Cause:** `getR1()` reads `Eb(k)` to build the cluster dissociation rates, but `Eb`
was declared *after* `R1` in the class. C++ initializes members in DECLARATION order,
not in the order of the constructor's initializer list, so this was an uninitialized
read. It evaluated to `exp(0) = 1` instead of `exp(-Eb/kT)`, inflating the 2i/3i
dissociation rate to the bare attempt frequency.
**Evidence:**
- `R1(2,2)` = −5.0535e7 1/s where the network sink alone gives −20.86.
- `R1(3,3)/R1(2,2)` = 0.4802 = `p(1,2)/p(1,1)` exactly, so both dissociation rates
  had lost their Boltzmann factor.
- Predicted C2i under the bug, `G_2i/alpha` = 9.89e-18, against the observed 1.06e-17.
**Fix:** move the `Eb` declaration (and its initializer) ahead of `R1`, with the
reasoning recorded at the declaration site.
**Effect:** `R1(2,2)` → −21.05 1/s; C2i +1.2% and C3i −2.3% against the 0-D.
**NOTE:** this is a bug in the ORIGINAL code, not in the Zr3d_ghoniem changes. It was
invisible in Po's calibration because `Eb_eV = 5.0` for every species makes
`exp(-5/kT)` ~ 1e-44, negligible whether or not the factor is applied. Giving 2i its
physical 0.957 eV is what exposed it. **Worth reporting upstream**: it affects any
MoDELib run with a non-negligible cluster binding energy.

### 17. ⟨a⟩ loop nucleation was cascade-only
**Symptom:** ⟨a⟩ loop density 0.53× the 0-D at 30 dpa, flat in dose. ⟨c⟩ density
matched to 0.1% throughout, so this was specific to the interstitial families.
**Cause:** the 0-D nucleates ⟨a⟩ loops from *two* sources — cascades
(`G·ε_iL/n_iL_nuc`, `reaction_rates.nucleation_rate_iL/aiL`) and homogeneous SIA
clustering (`R_i,3i + R_2i,2i`, with the matching content deposit
`4R_i,3i + 2R_2i,2i + 3R_2i,3i` from `nucleation_content_i`). Only the first was
wired. At the 0-D saturated state the budget is

| source | loops/atom/s | share |
|---|---|---|
| cascade `G·ε_iL/n_iL_nuc` | 7.0429e−13 | 61.6% |
| clustering `R_i,3i + R_2i,2i` | 4.3969e−13 | 38.4% |

so cascade-only nucleation predicts a density ratio of 0.62 against the 0.53
measured — the right size and sign for the whole discrepancy. The interstitials
those reactions consume were simply disappearing: with no 4i species to receive
the product, `getR2()` debits the reactants and credits nothing. That is the same
truncation that raises `Warning: Sum of R2 is not zero` at start-up, previously
recorded as benign.

**Fix:** take the nucleation flux from the reaction network the mobile solve
already uses, so the loops gain exactly what the mobile field loses.
`getLoopNucChannels()` selects the pairs `(a,b)` that

1. have reactants of the same polarity (a mixed pair is recombination), and
2. have a product size `m_a+m_b` carried by **no** mobile species, so the product
   is off the end of the ladder and must become a loop embryo.

For `{v,i,2i,3i}` that selects `i+3i`, `2i+2i` and `2i+3i`, and rejects `i+i`
(→2i), `i+2i` (→3i) and every v-bearing pair. The 0-D counts the first two in the
number source and all three in the content source; the third is ~2e−7 of the total
here, so including it in both is immaterial and keeps the split mass-consistent.

**Factor-of-two trap:** `getR2()` writes `−K_ab` into *both* `R2[a](a,b)` and
`R2[a](b,a)`, so `cᵀR2[a]c = −2K_ab c_a c_b` — but the residual is assembled as
`R2*(0.5*mobileClusters)` (`lWF_R2`), the quadratic-form factor that makes
`bWF_R2 = R2*c` its exact Jacobian. The per-species loss rate is therefore
`K_ab c_a c_b`, not `2K_ab c_a c_b`.

**Verification** — the channel coefficients printed at start-up against the 0-D
jump frequencies (`ω_m = z_c ν exp(−E_m/kT)`, T = 573 K):

| channel | Zr3d_ghoniem K [1/s] | ZrMicro | error |
|---|---|---|---|
| i+3i → 4i | 5.0535398e7 | `ω_i + ω_3i` = 5.0535398e7 | 4e−6 |
| 2i+2i → 4i | 9.8765170e2 | `4ω_2i` = 9.8812394e2 | 0.05% |
| 2i+3i → 5i | 4.9382578e2 | `ω_2i + ω_3i` = 4.9406197e2 | 0.05% |

**Content vs number:** cascade-borne loops are born at `nNuc` defects, clustering-
borne embryos carry only the `|m_a|+|m_b|` atoms of the event that made them, so
the two sources are *not* divided by the same number — exactly as in the 0-D,
where the cascade term is `G_iL/n_iL_nuc` while the clustering term is the bare
reaction rate. This dilutes the mean loop size, and it should: `m_a` moved from
1.164× the 0-D to 1.114×.

**Family split:** in proportion to `loopCascadeFractions` within each polarity.
The 0-D splits the clustering flux with `f_na`/`f_a`, the very fractions that split
`G_iL` into `G_iL`/`G_aiL`, so this reproduces the 0-D split exactly and
generalizes to any number of families.

**Effect (30 dpa, median interior):**

| quantity | before | after | 0-D |
|---|---|---|---|
| ⟨a⟩ loop density | 0.533× | **0.704×** | 1.0 |
| ⟨a⟩ loop content | 0.620× | **0.784×** | 1.0 |
| ⟨a⟩ defects/loop | 1.164× | **1.114×** | 1.0 |
| ⟨c⟩ loop density | 0.999× | 0.999× | 1.0 |

### 18. C2i/C3i depleted ~2.5× once the loops have grown — **OPEN**
**Symptom:** at 1 dpa C2i and C3i agree with the 0-D to 1–2%; by 5 dpa they have
fallen to 0.41–0.45× and they stay there. The 0-D values are flat in dose.
**Cause:** ZrMicro's loops absorb 2i and 3i for GROWTH —
`flux_i = ω_i(C_i + 2C_2i + 3C_3i)`, `reaction_rates.flux_i` — but `dC2i_dt` and
`dC3i_dt` never debit those pools. Their only sinks there are the network
(`R_2i_s`, `R_3i_s`), recombination with Cv, and the cluster reactions.
Zr3d_ghoniem's `ImmobileSinks` *does* debit them, so the 3-D carries a 2i/3i loop
sink the 0-D lacks. The onset coincides with loop growth, which is the tell.
**Magnitude:** C2i is linear in its own loss coefficient, so

    C2i_3D/C2i_0D = Λ_0D / (Λ_0D + D_i k²_loops)

with, at 30 dpa, `Λ_0D` = 20.92 1/s (of which `R_2i_s` = 20.86) and
`D_i k²_loops` = 17.86 1/s, giving 0.54 against the 0.41 measured. Same
mechanism, right order; the residual is the sigmoid sink law of `clusterDensity()`
and the 2i↔3i coupling, neither of which the one-line estimate carries.
**Why it matters here:** the clustering nucleation flux of issue 17 is
`K·C_i·C_3i`, so a depleted C3i feeds it directly. The 3-D's own nucleation rate
is 0.782× the 0-D's for this reason, and `0.704 / 0.782 = 0.90` — i.e. with the
cluster concentrations matched, the ⟨a⟩ density would land at ~0.90×.
**Not fixed, because it is a physics decision, not a bug in either code as such.**
Zr3d_ghoniem is the mass-conserving one: a loop that absorbs a di-interstitial
must remove it from the mobile pool. The reconciling change belongs in ZrMicro —
add the loop sink to `dC2i_dt`/`dC3i_dt` — but that shifts the 0-D fit that the
experiments are calibrated against, so it is your call.

### 19. The tree only built on the platform it was last built on

**Symptom:** on macOS the build stopped at the first translation unit with
`clang: error: unsupported option '-fopenmp'`, and behind that another three
errors, none of which Linux ever shows.

**Cause:** the build encoded one machine's answers instead of asking. Four of
them, in order of appearance:

1. `CMAKE_CXX_FLAGS` hard-coded `-march=native -fopenmp -Ofast`. Apple clang
   rejects `-fopenmp` outright (it wants `-Xpreprocessor -fopenmp` plus
   Homebrew's libomp), and `-march=native` is rejected on arm64 by older Apple
   clang. Each flag is now probed with `check_cxx_compiler_flag` before it is
   used, and OpenMP comes from `find_package(OpenMP)` — with a libomp fallback
   on macOS — rather than from a raw flag.
2. `set(EIGEN3_INCLUDE_DIRS /usr/include/eigen3)` — Debian's location, nowhere
   else's, and it overrode the caller's `-D`. Now `Eigen3Config.cmake` first,
   then a directory search that includes the Homebrew/MacPorts prefix.
3. 103 files under `include/` and `src/` call `assert()` without including
   `<cassert>`. libstdc++ pulls the header in transitively; libc++ does not.
   A tree-wide `-include cassert` stands in for editing all 103 and keeps the
   fork mergeable with upstream.
4. `const Eigen::Matrix<double,mSize,mSize> invTrD;` in `ClusterDynamicsFEM.h`
   had no initializer. Eigen ≥ 3.4.90 declares `Matrix() = default` rather than
   user-provided, which makes an uninitialized const member ill-formed; the
   member is unused and is now zero-initialized. This is an Eigen-version
   dependency, not a platform one — a new enough Eigen breaks it on Linux too.

**And one that was not a build failure at all.** With those four fixed the tree
built clean against Homebrew's Eigen — which is 5.x — and then the mobile
Newton iteration failed at its first step with `Iterative FixedDirichletSolver
failed`, the BiCGSTAB solve breaking down on a 500 nm case that had staged
successfully on Linux/Eigen 3.4.0 two months earlier. Rebuilding the same
source against Eigen 3.4.0 and re-running the same case bootstraps in 138 s.
The build now refuses Eigen newer than 3.4.x rather than producing a binary
that compiles and does not work; `build_modelib.sh` fetches 3.4.0 into
`<repo>/Libraries/` where the system has only Eigen 5. Watch the version test
itself: Eigen 5.0.1 keeps `EIGEN_WORLD_VERSION` at 3 and reports `3.5.0`, in
`Eigen/Version` rather than the `Macros.h` that 3.4.x uses, so the obvious
check passes it on both counts.

A fifth compile error was a genuine overload-resolution difference:
`RationalLatticeDirection::operator+(const LatticeVector&)` called
`this->operator+(RationalLatticeDirection<dim>(...))`, and clang resolves that
dependent call against the `LatticeVector` overload alone — the one being
defined — then rejects the argument. Naming the temporary resolves it, with no
change in meaning.

`cmake_minimum_required` also went from 3.1.0 to 3.16: CMake ≥ 4.0 refuses to
run a project asking for less than 3.5, so the old floor was a time bomb on any
current toolchain.

`tools/CMakeLists.txt` now decides about `DDqt` from `find_package(Qt6)` +
`find_package(VTK 9.4)` instead of having the line commented out by a `sed` in
the build script, and `pyMoDELib` is added only when pybind11 was actually
found — `find_package(pybind11 CONFIG REQUIRED)` used to turn a missing
optional dependency into a hard configure failure that took DDomp down with it.

**Effect:** one `Docs/Formulation/build_modelib.sh` builds on Linux, WSL and
macOS; `DDomp`, `microstructureGenerator` and `libMoDELib.a` all build clean on
Apple silicon, and the CD Newton iterates reproduce bit-identically between two
differently configured builds of the same tree.

### 20. The Galerkin climb assembly had no distance cutoff

**Symptom:** none, until a discrete population is present — and then the climb solve
dominates everything. `GalerkinClimbSolver::computeClimbScalarVelocitiesBulk` runs
`clusterStiffnessMatrix(fieldSegment, sourceSegment)` over **every ordered pair** of
segments, each costing an `mSize`-wide `concentrationMatrices` evaluation summed over
`periodicShifts`.

**Why the linear solve is not the cost.** The sparse path in that file is commented out
and would throw; only the *lumped* path is live, and the "solve" is one scalar division
per node, `nodeV[n](kc) = Fc[kc](n)/KKc[kc](n)`. **100% of the cost is the pairwise
assembly**, so reducing the number of unknowns buys nothing and reducing the number of
*pairs* is the only lever.

**Why truncating is legitimate.** A bare `1/r` kernel could not be truncated — a growing
loop is a net sink, its monopole does not vanish, and a shell at `r` contributes `~r`, so
the sum grows with the cutoff. The physical kernel is screened by the sink field as
`exp(−kr)/r`, with exactly the `k²` `ImmobileSinks.h` already assembles. Independently:
the continuum field `cCD` already carries the mean-field response of the whole
population, so the discrete sum must supply **only** the near-field correction the mean
field misses — extending it further would double count.

**Change:** new optional material key `climbNeighborCutoff_b`, read through the same
`try`/`catch` idiom as `atomicVolume_SI` and `concentrationFloor`. **Zero or absent means
no cutoff**, so every material file written before this key existed keeps all-pairs
behaviour exactly. The test is deliberately conservative — midpoint distance against
`R_c` plus *both* half-chords, so a pair is dropped only when no point of one segment can
lie within `R_c` of any point of the other — and takes the minimum over `periodicShifts`,
because a pair far in the primary cell may be near in an image. The **self term is never
truncated**: its midpoint distance is zero and it passes any `R_c ≥ 0`.

**Verification**, on a real discrete case (200 nm, 0.1 dpa, 6 ⟨c⟩ loops handed over by
`coupling/transition.py`, `useDislocations=1`, `climbSolverType=Galerkin`):

| run | cutoff | result |
|---|---|---|
| a | key absent | the new branch is never taken |
| b | `1e12` | branch taken, excludes nothing |
| c | `3 L_s` | branch taken, excludes pairs |

**a and b agree bit-for-bit** in `evl` and in every `F` field, which is the test of the
distance logic itself; c differs, so the cutoff bites.

**Convergence**, against the uncut assembly on `dotBetaP_33` — the basal
plastic-distortion rate, i.e. exactly what ⟨c⟩ vacancy loops produce:

| `n_L` = `R_c/L_s` | 1 | 2 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|
| error | 19.3% | 9.6% | **3.9%** | 0.25% | 0.000% |

**Use `n_L = 4`, not 3.** The zeros at `n_L ≥ 6` are a finite-size artifact — the cutoff
there exceeds the extent of the loop cloud, so nothing is excluded. With six loops in a
small domain this sweep is per-case, not once-and-for-all.

### 21. The superposed mobile field was never published, so `c_DD` was invisible

**Symptom:** on a march that had transferred 53 ⟨c⟩ loops to a discrete population at
1 dpa, `3d/Cv_10dpa.png` is as smooth as a continuum-only run — no depletion around any
loop. The expectation was the opposite: a discrete loop is a sink, and its field should
show a halo.

**Not a bug in the physics — a missing output, plus two upstream blockers.**

MoDELib solves the mobile species by **superposition**. The physical concentration is

```
c(x) = c_FEM(x) + c_DD(x)
```

with `c_DD` the analytic (Green's function) field of the discrete segments. `c_DD` enters
the FEM problem **only through the Dirichlet values** — `ClusterDynamics.cpp:150`:

```cpp
otherConcentration += microstructure->mobileConcentration(node->P0,node,nullptr,nullptr);
...
mobileClusters.dirichletConditions().at(mSize*node->gID+k)
        = bndConcentration(k) - otherConcentration(k);
```

so the FEM carries the **corrective** part and the sum meets the true boundary condition.
The CD block of `evl_*.txt` is therefore `c_FEM`, and it must be: `initializeConfiguration`
reads it straight back into `mobileClusters`, so writing the total there would corrupt the
restart. `c_DD` was evaluated on demand at DD quadrature points
(`DislocationQuadraturePoint.cpp:267`) and never stored anywhere. **A figure drawn from the
CD block is smooth by construction, whatever the dislocation state.**

**Fix:** optional material key `outputSuperposedMobile` (**default 0**, so no existing case
writes the file or pays for it). `ClusterDynamics::output` then also writes
`evl/cdTotalMobile_<runID>.txt` — `nNodes × mSize`, in `evl/cdNodes.txt` order — holding
`c_FEM + c_DD`. Cost is `O(nNodes × nSegments)`, OpenMP-parallel over nodes, and the FE
node pointer is passed to `mobileConcentration` so `pointGrains` resolves the grain from
the node's own elements instead of doing ~90 000 mesh searches.

**With no discrete dislocations every `mobileConcentration()` returns zero and this file
reproduces the CD block's mobile columns exactly** — worth keeping as a self-check.

**Two blockers on the Python side, both of which alone made the transition inert:**

| blocker | why |
|---|---|
| the fast solve ran with an **empty** network, always | `inject_discrete_loops` merges the generated network into the staged `evl_0`, but `MobileQSSASolver.solve` rewrites `evl_0` from its own preserved seed on every call — which is why that seed exists. The evl header's first six integers were `0` in the seed, before and after the transfer. `adopt_network` re-seeds from the merged evl. |
| one step is not enough | `DefectiveCrystal` emplaces `ClusterDynamics` **before** `DislocationNetwork` (`DefectiveCrystal.cpp:38-44`) and `MicrostructureContainer::solve` walks that order, so the CD solve precedes the climb-velocity computation. `c_DD` is **linear in the nodal `climbVelocityScalar`** (`DislocationSegment::clusterConcentration` = `concentrationMatrices(x) @ [v_source, v_sink]`), which is zero on step 0. `adopt_network` raises `Nsteps` to 2. |

**Why this cannot be done in Python.** The Green's function's *amplitude* is the nodal
climb velocity, and that is the **output** of the climb solve. There is no way to evaluate
`c_DD` offline without first reproducing the solve that sets it.

**Read existing figures correctly.** Because `c_DD ≡ 0` in every run made before this,
`c_FEM` *was* the physical concentration and every published field figure is right as it
stands. It is once loops genuinely enter a solve that the CD block becomes the corrective
part only — and the same figure code would then be wrong rather than merely featureless.
That asymmetry is why the total is published **alongside** the CD block, not folded into it.

### 22. The ⟨c⟩ loop Burgers vector was the FULL [0001], in an IDEAL lattice

**Symptom:** the continuum→discrete transfer ledger reported a sink-strength jump with a
`f_burgers = √(b_cd/b_dd) = 1.414214` factor in it, for ⟨c⟩ only. ⟨a⟩ was always clean.

**Cause, in two layers.**

`HEXlattice<3>::getLatticeBasis()` hard-coded the **ideal** hcp ratio:

```cpp
temp << 1.0, 0.5,           0.0,
        0.0, 0.5*sqrt(3.0), 0.0,
        0.0, 0.0,           sqrt(8.0/3.0);   // 1.6329932; alpha-Zr is 1.5944272
```

and `immobileSpeciesBurgers`' ⟨c⟩ column was `(0,0,1)` — the **full** [0001]. So the
three sides of the model sized the same vacancy loop three different ways:

| side | \|b_⟨c⟩\| | using |
|---|---:|---|
| DD (discrete) | 2.6397 Å | ½ × ideal c |
| 3-D CD (continuum) | 5.2795 Å | full ideal c |
| 0-D | 5.1500 Å | full physical c |
| **all now** | **2.5750 Å** | ½ × physical c, `b = ½[0001]`, `c = 5.15 Å` |

The dominant error is the **factor of 2** — only DD treated a ⟨c⟩ loop as ½[0001] at all
— with a residual 2.4% from ideal-versus-physical `c`.

**Why no input file could fix the discrete side.** The ⟨c⟩ Burgers vector derives from
the lattice basis and from nothing else: `aLoopGenerator::generateSingle` builds a basal
loop's **b** from the lattice, and `discrete_loops.write_microstructure` emits only plane
IDs, radii, side counts, vacancy flags and centres. There is no Burgers vector in the
microstructure file to override.

**Fix.** `getLatticeBasis` now takes the material and reads an **optional** key:

```cpp
double cOverA(sqrt(8.0/3.0));
try { cOverA = TextFileParser(material.materialFile).readScalar<double>("c_SI",true)/material.b_SI; }
catch(const std::runtime_error&) {}     // absent -> ideal, bit-for-bit unchanged
```

`Zr3d_ghoniem.txt` gains `c_SI = 5.15e-10`, sets `immobileSpeciesBurgers`' ⟨c⟩ column z
to **0.5**, and moves `b_SI` from 0.3233e-9 to **0.323e-9** so the 3-D and the workbook
share one `a = 3.23 Å`. Only the `c` axis moves: the ⟨a⟩ directions lie in the basal
plane, so `|b_⟨a⟩| = a = 3.23 Å` in either convention, exactly as `⅓⟨11̄20⟩` requires.

**Consequences.** `f_burgers` becomes **1.000000**. The continuum ⟨c⟩ radius rises
×1.4312 and the 0-D `l_c` ×1.4142, so **the ⟨c⟩ part of the 28-parameter fit is stale** —
the third such warning after Ω and `loop_model`. `b_SI` is MoDELib's length unit, so
every staged case re-stages; that is automatic because `domain_key` hashes the material
content, and it also removes a 0.28% error in Ω\_b³ (the 3-D had been converting an Ω
built from `a = 3.23` using `a = 3.233`).

**`HEXlattice_OLD.cpp` carries its own `getLatticeBasis()` and was NOT touched** — it is
absent from `src/PolycrystallineMaterials/CMakeLists.txt` and is not compiled. BCC and
FCC keep their no-argument signatures.

---

## 5b. Verification against the 0-D

Interior nodes, selected by the top decile of Cv (the grain boundary is a Dirichlet
sink, so Cv is depressed near it and saturates in the interior). This selection must
be made ONCE from the mobile field and applied to all quantities: loop density
anti-correlates with Cv (corr = −0.98), because near the boundary the mobile species
are depleted so loops neither grow nor coalesce and simply accumulate at their
nucleation value. A per-quantity percentile picks the boundary for the immobile
fields and the interior for the mobile ones.

Ratios 3-D / 0-D, from the 31-step run to 30 dpa. "before" is cascade-only
nucleation, "after" is with the homogeneous SIA clustering flux of issue 17.

| quantity | 1 dpa | 5 dpa | 10 dpa | 30 dpa |
|---|---|---|---|---|
| Cv | 1.073 | 0.982 | 0.994 | 1.004 |
| Ci | 1.145 | 1.112 | 1.133 | 1.148 |
| C2i | 1.012 | 0.412 | 0.394 | 0.388 |
| C3i | 0.977 | 0.400 | 0.383 | 0.377 |
| ⟨c⟩ loop density | 0.999 | 0.999 | 0.999 | 0.999 |
| ⟨a⟩ loop density — before | 0.426 | 0.547 | 0.540 | 0.533 |
| ⟨a⟩ loop density — **after** | **0.662** | **0.739** | **0.718** | **0.704** |
| ⟨a⟩ loop content — before | 0.486 | 0.634 | 0.627 | 0.620 |
| ⟨a⟩ loop content — **after** | **0.639** | **0.811** | **0.795** | **0.784** |
| ⟨a⟩ defects/loop — before | 1.141 | 1.159 | 1.159 | 1.164 |
| ⟨a⟩ defects/loop — **after** | **0.965** | **1.098** | **1.106** | **1.114** |

Cv agrees to within 2% beyond 5 dpa and the ⟨c⟩ loop density to 0.1% at every dose.
Ci runs 11–15% high throughout, at every dose and in both runs.

The two remaining discrepancies are one story, not two. The ⟨a⟩ loop density is
0.704× and the 3-D's own nucleation rate — evaluated from its own, depleted
cluster concentrations — is 0.782× the 0-D's, so `0.704 / 0.782 = 0.90`: at
matched C2i/C3i the density would land within 10%. C2i/C3i are depleted because
of the 2i/3i loop sink of issue 18, which the 0-D does not carry. Closing issue 18
should therefore close most of what is left on the ⟨a⟩ families as well.

---

## 6. Status and open items

**Working:** MoDELib builds and runs the §2.2 physics. The mobile Newton iteration
converges cleanly (7.5e6 → 6.2e-8 in ten iterations, quadratic tail). ⟨a⟩ families are
symmetric at zero stress. The reaction network matches ZrMicro to ~0.1%, including the
three loop-nucleation channels (§5, issue 17). Cv and the ⟨c⟩ loop density agree with
the 0-D to 2% and 0.1%; the ⟨a⟩ loop density is 0.70× and its content 0.78× (§5b). No
negative concentrations anywhere, at any dose.

**Two further outputs added for post-processing:**
- `evl/cdNodes.txt` — finite-element node coordinates, written once per run. Without
  it the fields cannot be placed in space: the CD trial functions live on second-order
  elements (24115 nodes for a 3392-node mesh), so the field rows do not correspond to
  the `.msh` vertices and the ordering is not reproducible externally.
- `ZrClusterDynamics/ZrMicro/py_utils/modelib_fields.py` — Fig. 4-style panel plots, one row
  per species and one column per dose, with loops overlaid as discrete platelets at
  sampled sites (visualization only).

**Known modelling gaps, all documented in the coupling `.tex`:**

1. ~~**Loop sink strengths differ by construction.**~~ **Resolved** — see §3.4.
2. ~~**Nucleation is cascade-only.**~~ **Resolved for the clustering flux**
   `Φ_clus = R_i,3i + R_2i,2i` — see §5, issue 17. The SIPN stress weights `w_k(σ)`
   are still not wired; exact at zero stress, which is the configured condition.
3. **DAD is a fitting form, not a generated one.** `dadAnisotropy`/`dadZ0` are fitted
   to reproduce the 0-D symmetric bias forms rather than derived from `D_c/D_a`, and
   the diffusion tensors are isotropic to match ZrMicro. These runs therefore exercise
   none of the tensorial-DAD mechanism of the deliverable, and the fit is tied to 573 K.
4. **`Warning: Sum of R2 is not zero`** — expected, and now *meaningful* rather than
   merely benign. The reaction map contains `i+3i` but there is no 4i species to
   receive the product, so that channel does not conserve interstitials within the
   mobile ladder. Those interstitials are no longer unaccounted: they are exactly the
   loop embryos of issue 17, and the immobile solve now credits them. ZrMicro
   truncates the cluster ladder at the same place, for the same reason.
5. **2i/3i are absorbed at the loops here but not in the 0-D** — issue 18. This is
   the largest remaining source of disagreement and it is a physics decision, not a
   bug: Zr3d_ghoniem is the mass-conserving one, but the reconciling change belongs
   in ZrMicro and would move a fit the experiments are calibrated against.

**Performance note:** `useElasticDeformation=1` costs ~42 s/step on a 72,345-dof solve
for a case with no dislocations and no applied stress — roughly 20 minutes of a 30-step
run. Worth disabling for production runs.
