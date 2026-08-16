# Plan — anisotropic defect diffusion and discrete↔continuum loop growth

**Status:** 16 August 2026. **Phases 1 and 2 and sub-step 4a are implemented**
(see the per-phase markers below); Phase 3 and sub-steps 4b--4f are not.
Phase 5's refit is deferred pending experimental data.
**Source paper:** Y. Li, M. Maron, K. Baker, B. Ramirez Flores, T. Black, J. Hollenbeck,
I. Lalani, N. Ghoniem, G. Po, *Coupled cluster and dislocation dynamics modeling of
microstructure evolution in irradiated materials*, J. Mech. Phys. Solids **206** (2026)
106366 — [`Docs/Formulation/Li-CoupledCD-DD_1-s2.0-S0022509625003400-main.pdf`](../Formulation/Li-CoupledCD-DD_1-s2.0-S0022509625003400-main.pdf).

---

## The headline: most of this is already in the tree, and it is switched off

The single most important finding of this audit is that **the machinery the paper
describes is already present in `MoDELib3/`** — inherited from the same MoDELib2-NNL
line the paper was implemented in — and DisloCluster does not use any of it. This is
not a port. It is a wiring, calibration and cost-control problem.

| Paper feature | Where it already lives | State in DisloCluster |
|---|---|---|
| Full anisotropic diffusion tensor per species | `ClusterDynamicsParameters::getDlocal/getD`, rotated per grain as `C2G·D·C2Gᵀ` | present, **fed isotropic numbers** |
| Anisotropic flux in the FEM fast solve | `FluxMatrix::operator()` — uses the whole 3×3 block | present, **already anisotropic-capable** |
| Green's function `c∞` for a segment, Eq. (48)/(A.1)–(A.3) | `DislocationSegment::concentrationMatrices` | present, **never called** |
| Dislocation bias `Z⋆` | `discreteDislocationBias`, 2×mSize | **already in `Zr3d_ghoniem.txt`** |
| Superposition climb solve, Eq. (50) | `GalerkinClimbSolver::computeClimbScalarVelocitiesBulk` | present, **never constructed** |
| Discrete↔continuum concentration handoff | `DislocationQuadraturePoint::cCD` / `cDD` | present, **never populated** |
| A working reference configuration | `MoDELib3/greatWhitePlots/sims/simA`, `simB` | **the paper's own input files, vendored** |

`greatWhitePlots/sims/simA/inputFiles/Zr3.txt` carries the paper's `p_m = 0.7`
di-interstitial row verbatim, and its `DD.txt` has `useDislocations=1` and
`climbSolverType=Galerkin`. Every DisloCluster case stages `useDislocations=0`, so the
climb solver is never built, `concentrationMatrices` is never evaluated, and `cCD`/`cDD`
stay at zero.

---

## Part 1 — What the paper does

### 1.1 Anisotropic diffusion in the fast solve

The mobile problem is the steady-state, spatially resolved reaction–diffusion system
(Eq. 36 with `ċ = 0`), decomposed by superposition into an infinite-medium part carried
by the discrete dislocations and a finite-medium correction solved by Galerkin FEM:

```
c⋆ = c∞⋆ + c̃⋆                                                          (44)
J⋆ ≈ −(D⋆/Ω₀) Z⋆ ∇c∞⋆   −  (D⋆/Ω₀) ∇c̃⋆                                 (46)
                ⌞ Green's function ⌟      ⌞ FEM ⌟
```

Two points matter for us:

- **The anisotropy is carried explicitly by the tensor `D⋆`, not by a bias factor.**
  The paper is emphatic that `Z⋆` is the *dislocation* (elastic) bias, absorbing the
  drift term and the strain dependence of the saddle point, "while the anisotropy of
  the diffusion tensor is considered explicitly."
- **`D⋆ = D₀⋆ exp(−E_m⋆/k_BT)` component by component**, with `D₀⋆` a positive-definite
  pre-exponential tensor. Anisotropy is introduced through the *migration energies*,
  keeping `D₀` common — see 1.2.

The finite-medium correction `c̃⋆` is the piece DisloCluster's fast solve already
computes. Making it anisotropic requires no new equation: it is the same weak form with
a non-scalar `D`.

### 1.2 Migration energies for the diffusion tensor — the exact recipe

The anisotropy factor is defined (Woo 1988) as

```
p_m = ( D⟨c⟩ / D⟨a⟩ )^(1/6)
```

with `D⟨c⟩` along the basal pole and `D⟨a⟩` in the basal plane. The paper varies `p_m`
**at fixed effective diffusivity** `D_eff = (D⟨a⟩² · D⟨c⟩)^(1/3)`, which fixes both
components uniquely. Writing `E_m^eff` for the isotropic migration energy that
reproduces `D_eff`:

```
E_m⟨11,22⟩ = E_m^eff + 2 k_B T ln(p_m)
E_m⟨33⟩    = E_m^eff − 4 k_B T ln(p_m)
```

**Verified against the paper's Table 2** (`E_m^eff = 0.09` eV, T = 553 K) — reproduced
to the last quoted digit:

| `p_m` | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|
| `E_m⟨33⟩` derived | 0.0900 | 0.1101 | 0.1325 | 0.1580 | 0.1874 | 0.2221 |
| `E_m⟨33⟩` Table 2 | 0.090 | 0.110 | 0.132 | **0.158** | 0.187 | 0.222 |
| `E_m⟨11,22⟩` derived | 0.0900 | 0.0800 | 0.0687 | 0.0560 | 0.0413 | 0.0239 |
| `E_m⟨11,22⟩` Table 2 | 0.090 | 0.080 | 0.069 | **0.056** | 0.041 | 0.024 |

The `p_m = 0.7` column is exactly the third row of `simA`'s
`mobileSpeciesEnergyMigration_eV`, which independently confirms both the recipe and the
component order `[11 12 13 22 23 33]`.

**Other values from the paper**, for reference (Tables 1 and 3):

| quantity | value |
|---|---|
| `E_m` vacancy | 0.9 eV, isotropic |
| `E_m` interstitial | 0.09 eV, isotropic |
| `E_m` di-interstitial | anisotropic, Table 2; **only 2i carries the DAD** |
| `D₀` vacancy | 4.9639×10⁻¹⁰ m²/s |
| `D₀` interstitial, di-interstitial | 8.1882×10⁻¹⁰ m²/s |
| `E_f` v / i / 2i | 1.5 / 3.0 / 6.0 eV |
| `Z_v⟨c,a⟩` | 1.0 |
| `Z_{i,2i}⟨c⟩` | 1.05 |
| `Z_{i,2i}⟨a⟩` | varied, 1.0–1.5 |
| `f_2i` | 0.4 |
| T, G₀ | 553 K, 10⁻⁷ dpa/s |

**Physics conclusion to carry into the calibration:** the paper finds DAD dominates over
both dislocation bias and production bias, and that **`p_m < 0.8` is required** to produce
simultaneous ⟨a⟩-axis expansion and ⟨c⟩-axis contraction — with lower `p_m` more
appropriate at TEM-realistic loop densities.

**A consequence worth stating up front:** because the anisotropy is stored as *energies*
and not as a ratio, `p_m` becomes temperature-dependent, `p_m(T) = exp(−ΔE_m/6k_BT)`.
A pair fixed to give `p_m = 0.7` at 573 K gives:

| T (K) | 473 | 523 | **573** | 623 | 673 |
|---|---|---|---|---|---|
| `p_m` | 0.649 | 0.677 | **0.700** | 0.720 | 0.738 |

This is physically correct — anisotropy weakens as temperature rises — and it is
precisely the defect in the present fitted DAD, which is **frozen at one temperature**
(known non-correspondence #1 in `CLAUDE.md`). Moving the anisotropy into the tensor
fixes it for free.

### 1.3 How DAD enters the growth equations

In the paper, growth of a *discrete* loop is not a bias-weighted sink term at all. The
climb velocity comes from the superposition condition evaluated on the dislocation line,
Eq. (50):

```
∫_L  [ w'⋆ · (b' × dℓ') ] / [ 4π √det(Z⋆D⋆) √( (x−x')·(Z⋆D⋆)⁻¹(x−x') ) ]  =  c_eq⋆(x) − c̃⋆(x)
```

solved for the per-species scalar velocities `w⋆`, then summed with sign, Eq. (51):
`w_c = Σ_v w_nv − Σ_i w_ni`. The DAD enters through `D⋆` inside the kernel — through
both `det(D⋆)` and `D⋆⁻¹` — so **different segments of the same loop climb at different
rates**, which is what produces the elliptical ⟨a⟩ loops the paper reports (conclusion 2).
A scalar capture efficiency cannot produce that; it is a shape degree of freedom the
continuum model does not have.

The equilibrium line concentration is set by the Peach–Koehler force, Eq. (49):

```
c_eq_nv = c₀_nv exp[ +nΩ f_PK·(b×ξ) / (k_BT ‖b×ξ‖²) ]
c_eq_ni = c₀_ni exp[ −nΩ f_PK·(b×ξ) / (k_BT ‖b×ξ‖²) ]
```

**For the continuum immobile families this is not how DisloCluster works, and does not
need to change.** DisloCluster uses Woo's closed-form capture efficiencies,
`Z_c = Z⁰p`, `Z_a = Z⁰(p+p⁻²)/2` (`loopDADbias()`), which are the analytic *surrogate*
for the same anisotropic absorption problem, valid for a randomly oriented population.
The critical requirement is **consistency**: see §3.2.

### 1.3.1 The DAD window — a closed-form criterion for simultaneous ⟨a⟩ and ⟨c⟩ growth

Irradiated Zr holds prismatic interstitial ⟨a⟩ loops and basal vacancy ⟨c⟩ loops at the
same time, and reproducing that is the reason the DAD mechanism exists at all. Because
DisloCluster's continuum families absorb through Woo's two closed forms rather than
through a line integral, the condition for co-growth can be **derived** rather than
searched for numerically. It is implemented and checked in
[`dislocluster_code/studies/dad_window.py`](../../dislocluster_code/studies/dad_window.py).

#### Step 1 — the growth flux, and what cancels out of it

`ClusterDynamicsFEM.cpp:617–645` forms, for each immobile family `k`:

```
gdot_k = φ_like,k − gate_k · φ_opp,k
φ_·,k  = Σ_m D̄_m · Z(row(k), m) · S_k · c_m · |m_m|
```

with `row(k) = 0` for a vacancy family and `1` for an interstitial family, `D̄_m =
(det D_m)^(1/3)`, and `S_k = clusterDensity·loopSinkScale`. Two structural facts do the
work:

- **`S_k` multiplies both channels identically**, so it cancels from the *sign* of
  `gdot_k`. Whether a family grows is therefore independent of its own density and
  size — the microstructure sets the *rate*, never the *direction*. This is what makes
  the criterion below universal rather than state-dependent.
- **`gate_k ∈ [0,1]`** is a smoothstep in `(r_k/r_min − 1)/0.3` applied to the
  *shrinking* channel only. Setting `gate_k = 1` is therefore the conservative choice
  for a growth criterion: a family this analysis calls shrinking would shrink at least
  as fast with the gate active. §Step 6 returns to what the gate changes.

Set `gate_k = 1` and drop the common `S_k > 0`.

#### Step 2 — collapse the mobile species into two numbers

Define the vacancy and interstitial **arrival rates**

```
A = D̄_v c_v                                  (the single vacancy species)
B = Σ_{m interstitial} D̄_m c_m |m_m|         (i, 2i, 3i, weighted by cluster size)
```

A ⟨c⟩ loop is vacancy-type: vacancies are its `like` channel, interstitials its `opp`.
A ⟨a⟩ loop is interstitial-type and the roles swap. Writing the Eq. (15) efficiencies as

```
Z_c(m) = Z⁰_m · p_m                          row 0 — basal, vacancy-type
Z_a(m) = Z⁰_m · f(p_m),   f(p) ≡ (p + p⁻²)/2 row 1 — prismatic, interstitial-type
```

The whole criterion rests on these two responding *differently* to `p`. `Z_c` is linear
and monotone. `f` is not: it is a rising plus a falling term with

```
f′(p) = (1 − 2p⁻³)/2  = 0   at   p = 2^(1/3) = 1.259921,  where f = 0.944941
f(p)  = 1                    at   p = 1  and  p = (1+√5)/2 = 1.618034
```

so `f` is strictly decreasing on `(0, 2^(1/3))` — **still falling at `p = 1`**, where
`f′(1) = −1/2` — and dips below `Z⁰` on `1 < p < 1.618`. It is that asymmetry, not the
anisotropy as such, that lets one family gain while the other does not.

Taking for the moment **one anisotropy `p_I` shared by the interstitial species**
(legitimate here because `Zr3d_ghoniem.txt` already gives `i`, `2i` and `3i` a common
`Z⁰_I = 1.015504`, against `Z⁰_v = 0.939836`):

```
gdot_c > 0   ⟺   Z⁰_v p_v A       >  Z⁰_I p_I B
gdot_a > 0   ⟺   Z⁰_I f(p_I) B    >  Z⁰_v f(p_v) A
```

#### Step 3 — both conditions are bounds on the same single number

Divide each by `B` and by the leading constants. Both become bounds on the **arrival
ratio** `A/B`:

```
⟨c⟩ grows   ⟺   A/B  >  (Z⁰_I/Z⁰_v) · p_I / p_v          ← lower bound
⟨a⟩ grows   ⟺   A/B  <  (Z⁰_I/Z⁰_v) · f(p_I) / f(p_v)    ← upper bound
```

Co-growth requires `A/B` to sit in the window between them. That the two conditions
reduce to bounds on one scalar — rather than to two independent constraints — is the
reason the problem has a clean answer.

#### Step 4 — when is the window non-empty?

The window has width

```
W  =  upper / lower  =  [f(p_I)/f(p_v)] · [p_v/p_I]  =  g(p_v) / g(p_I)
```

where

```
g(p)  ≡  p / f(p)  =  2p / (p + p⁻²)  =  2p³/(p³ + 1)  =  2 / (1 + p⁻³)
```

The last form makes the monotonicity immediate: `p⁻³` is strictly decreasing in `p` for
`p > 0`, so `1 + p⁻³` is strictly decreasing and `g` is **strictly increasing**, with
`g(0⁺) = 0`, `g(1) = 1`, `g(∞) = 2`. Hence `W > 1 ⟺ g(p_v) > g(p_I) ⟺ p_v > p_I`:

> **Simultaneous growth of ⟨a⟩ and ⟨c⟩ loops is possible if and only if `p_I < p_v`** —
> the interstitial species must be biased into the basal plane *relative to* the
> vacancies.

This is the whole DAD argument in one inequality. Note what it does **not** contain: no
dose, no loop density, no loop size, no temperature except through `p(T)`, and no
property of the mobile field. It is a statement about the capture efficiencies alone.
What the dose and the field decide is only whether the realized `A/B` lands *inside* the
window.

Two corollaries worth keeping in view:

- **Only the contrast matters, never the absolute anisotropy.** Setting `p_v = p_I` gives
  `W = 1` — a degenerate window no `A/B` satisfies — however extreme the common value.
  An "anisotropic" model in which every species shares one tensor cannot produce
  co-growth at all.
- **`A/B` is not a fixed target.** The anisotropy reshapes the mobile field as well as
  re-weighting the absorption, so moving `p` moves the ratio and the bounds together.
  This is exactly the coupling that cannot be represented in 0-D, where the mobile
  concentrations come from a well-mixed balance with no direction in it — and it is why
  each point of the sweep in `dad_sweep.py` costs a fast solve.

#### Step 5 — the general case: interstitial species with different `p`

Li et al.'s own structure gives the DAD to the di- and tri-interstitial and leaves the
single interstitial isotropic, so the shared-`p_I` reduction of Step 2 does not apply.
Carrying the species sum through instead, and writing the **arrival shares**
`w_m = D̄_m c_m |m_m| / B` (so `Σ w_m = 1` over the interstitial species), the same two
conditions become

```
⟨c⟩ grows   ⟺   A/B  >  (Z⁰_I/Z⁰_v) · p̄ / p_v ,        p̄ = Σ_m w_m p_m
⟨a⟩ grows   ⟺   A/B  <  (Z⁰_I/Z⁰_v) · f̄ / f(p_v) ,     f̄ = Σ_m w_m f(p_m)
```

— identical in form, with `p_I → p̄` and `f(p_I) → f̄`. The window is non-empty iff
`p_v f̄ > f(p_v) p̄`, which reduces to Step 4 when all `p_m` are equal. The new content is
in the difference

```
f(p) − p  =  (p + p⁻² )/2 − p  =  (p⁻² − p)/2  =  (1 − p³) / (2p²)
```

which is **positive for `p < 1`, zero at `p = 1`, negative for `p > 1`**. So at `p_v = 1`
the window opens by `f̄ − p̄ = Σ_m w_m (1 − p_m³)/(2p_m²)`, and:

> **An isotropic species is inert.** A species left at `p_m = 1` enters `p̄` and `f̄` with
> the same value and contributes nothing to opening the window.

The anisotropic species therefore help only in proportion to the share of the
interstitial arrival they actually deliver. For `p = (p_v, p_i, p_2i, p_3i) =
(1, 1, 0.7, 0.7)`:

| cluster share `w_2i + w_3i` | `p̄` | `f̄` | `W` |
|---:|---:|---:|---:|
| 0.05 | 0.9850 | 1.0185 | 1.034 |
| 0.20 | 0.9400 | 1.0741 | 1.143 |
| 0.60 | 0.8200 | 1.2222 | 1.491 |
| 1.00 | 0.7000 | 1.3704 | 1.958 |

A window always exists (any `p_m < 1` opens one), but a cluster-only DAD is nearly inert
if the single interstitial carries the flux — no value of `p_2i`, however extreme, can
widen a window whose weight `w_2i` is small. **Whether Li et al.'s structure can work in
this model is therefore an empirical question about the arrival shares**, which
`dad_window.run` measures.

#### Step 6 — assumptions, and what relaxing them does

The derivation is exact for the flux as coded, under three stated simplifications:

| assumption | effect of relaxing it |
|---|---|
| `gate_k = 1` (Step 1) | The gate multiplies the shrinking channel by `≤ 1`, so it scales the lower bound down and the upper bound up: `W_gated = W / (gate_c · gate_a) ≥ W`. **The gate can only widen the window**, so `p_I < p_v` is *necessary and sufficient* for the ungated criterion but only *sufficient* once the gate is active. Empirically the distinction has not mattered: every grid point with `p_I > p_v` gave co-growth on exactly 0.000 of the interior nodes. |
| thermal emission omitted | Vacancy-loop emission is a loss for ⟨c⟩ that does not scale with `A`, so it raises the lower bound by an `A/B`-independent amount and **narrows** the window from below. The criterion is therefore optimistic for ⟨c⟩ by that margin. |
| `Z⁰` common to the interstitial species | True as `Zr3d_ghoniem.txt` stands (`1.015504` for all three). If a refit breaks it, Step 5's weighted form still holds with `Z⁰_I → Σ w_m Z⁰_m` in each mean. |

**The largest limitation is not in that table: `gdot` is only one term of the content
equation.** `ClusterDynamicsFEM.cpp:755` updates content as

```cpp
cNew = (c + dt*(gdot + Gk + clusShare*clusC - contentSink)) / (1 + dt*lossC)
```

where `Gk` is cascade nucleation — an independent positive source that carries no
dependence on `A/B` whatever. So:

> The window governs whether existing loops **grow or dissolve by absorption**. It does
> **not** govern the net population evolution, which nucleation, coalescence and thermal
> emission also drive.

Run (b) shows the gap concretely: the sweep predicts `ġ_c < 0` at `p = (1,1,0.7,0.7)`, yet
that march's interior ⟨c⟩ content rises strongly through 1 dpa because nucleation supplies
it. Both statements are true and they are about different quantities. Absorption is still
the only channel the anisotropy enters, so the criterion remains the right instrument for
choosing the DAD — but a claim about an observed loop *population* must account for `Gk`
as well.

#### Step 7 — the inversion, which is the useful direction

In practice `A/B` is measured and the question is what anisotropy it demands. Inverting
the ⟨a⟩ condition for the largest admissible `p_I`, at a given `p_v`:

```
f(p_I) = (A/B) · (Z⁰_v/Z⁰_I) · f(p_v)        solve for p_I ∈ (0, 1]
```

`f` is strictly decreasing on `(0,1]`, so the root is unique and a bracketed solve is
robust. For `A/B ≫ 1` the `p⁻²/2` term dominates `f`, giving the asymptote

```
p_I  ≲  1 / √( 2 · (A/B) · Z⁰_v/Z⁰_I · f(p_v) )
```

so the required anisotropy tightens only as the **square root** of the vacancy excess —
doubling the vacancy surplus costs a factor 1.41 in `p_I`, not 2. Checked against the
exact root:

| `A/B` | exact `p_I` | asymptote |
|---:|---:|---:|
| 1.5 | 0.6928 | 0.6001 |
| 3 | 0.4423 | 0.4244 |
| 5 | 0.3348 | 0.3287 |
| 10 | 0.2339 | 0.2324 |

#### Step 8 — from `p` back to what the material file actually holds

`p_m` is a derived quantity, not an input. The file carries
`mobileSpeciesEnergyMigration_eV`, and `staging/anisotropy.py` splits it at fixed
`D_eff = (D⟨a⟩² D⟨c⟩)^(1/3)` — so that a change in `p` changes the *directionality* of
diffusion and not the overall mobility the existing calibration was fitted at:

```
E_m⟨11,22⟩ = E_m,eff + 2 k_BT ln(p_m)
E_m⟨33⟩    = E_m,eff − 4 k_BT ln(p_m)
```

Because the anisotropy is stored as energies, `p_m(T) = exp(−ΔE_m/6k_BT)` is
temperature-dependent, and a window that is open at one temperature can close at
another. Any co-growth statement is therefore a statement at a stated `T`.

### 1.3.2 Vacancy ⟨a⟩ loops — what the window argument forbids, and what it allows

Experiments in irradiated Zr report ⟨a⟩ loops of **both** vacancy and interstitial
character. That looks at first like a contradiction of §1.3.1, and resolving it sharpens
what the criterion does and does not claim.

#### The complementarity is exact — no DAD parameters can produce both

`Z_h(m)` depends on the loop's **habit plane** `h` and on the diffusing species `m`. It
does **not** depend on whether the loop is vacancy- or interstitial-type: a basal loop
presents its face to the c-axis flux and a prismatic loop to the basal flux, and that is
geometry. So the two ⟨a⟩ types share one `Z_prism(m) = Z⁰_m f(p_m)`, and

```
interstitial ⟨a⟩ grows  ⟺  Z_prism(I)·B > Z_prism(v)·A  ⟺  A/B < U
vacancy      ⟨a⟩ grows  ⟺  Z_prism(v)·A > Z_prism(I)·B  ⟺  A/B > U
```

with the *same* `U = (Z⁰_I/Z⁰_v) f(p_I)/f(p_v)` from §1.3.1 Step 3. The `Z⁰` and `f(p)`
factors cancel identically because they multiply both sides. Applying the same argument
to the basal habit at threshold `L` partitions the `A/B` axis into three regimes:

| regime | growing populations |
|---|---|
| `A/B < L` | interstitial ⟨a⟩ + interstitial ⟨c⟩ |
| `L < A/B < U` | **vacancy ⟨c⟩ + interstitial ⟨a⟩** — the classic Zr microstructure |
| `A/B > U` | vacancy ⟨c⟩ + **vacancy ⟨a⟩** |

> **A vacancy ⟨a⟩ loop grows exactly where an interstitial ⟨a⟩ loop dissolves**, for any
> `p_v`, `p_I` and `Z⁰`. The two populations cannot both grow at one place and time.

So the observation cannot be reached by tuning the DAD, and it is not evidence that the
criterion is wrong — it is a statement about a different regime of the same axis.

#### The inhomogeneity required is small, which inverts into a constraint on `p_I`

The gap between regimes 2 and 3 is the window width `W = g(p_v)/g(p_I)`. At the
co-growth point measured in `dad_sweep` (`p_v = 1`, `p_I = 0.9137`), `L = 0.9873` and
`U = 1.1407` — a **16% band in `A/B`**. A local or transient excursion of that size moves
the material from regime 2 into regime 3 and makes vacancy ⟨a⟩ loops.

This inverts into a useful inference:

> The coexistence of both ⟨a⟩ types is **evidence for a narrow window**, i.e. for `p_I`
> only slightly below `p_v`.

At the paper's `p_I = 0.70` against `p_v = 1` the window is 96% wide and a factor-2
excursion in `A/B` would be needed. The fitted `p_I = 0.9137` against `p_v = 1` sits in
exactly the narrow-window regime that makes both populations easy to obtain — which is an
argument in favour of the fitted anisotropy over the paper's, on evidence independent of
the parameter fit itself.

#### Candidate mechanisms, in order of how much new physics each needs

1. **Temporal.** `A/B` is vacancy-rich in the early transient before the sink density
   builds, then falls. Loops nucleated in regime 3 are vacancy-type; when `A/B` drops
   into regime 2 they shrink — but slowly, because the rate carries the common `S_k` and
   the minimum-size gate suppresses the shrinking channel near `r_min`. Predicts vacancy
   ⟨a⟩ as a **relic** population, declining at high dose. Needs no new physics.
2. **Cascade-scale.** A vacancy-rich cascade core collapses directly to a vacancy loop in
   picoseconds at nanometre scale — a **nucleation** channel that bypasses `A/B`
   altogether, and neither spatial nor temporal at the continuum scale. The channel
   already exists in form (`loopCascadeFractions`); what is missing is a family for it to
   feed.
3. **Continuum-scale spatial.** `A/B` varies across the specimen; the Dirichlet boundary
   layer is an extreme case. Sufficient in principle, but strained if the two ⟨a⟩ types
   are observed intermixed within one grain.
4. **Type-dependent capture.** The complementarity is exact only because `Z` factorizes
   into `Z⁰_m` times a habit shape. Any mechanism making capture depend on the loop's own
   type — its own strain field, faulting, solute decoration — splits the thresholds:
   writing `β_v`, `β_i` for type-specific like-defect enhancements, coexistence needs
   `β_v β_i > 1`. **Note the sign problem**: the standard dislocation bias (SIAs have the
   larger relaxation volume, so all loops prefer them) gives `β_i > 1` but `β_v < 1`, and
   it is not obvious the product exceeds 1. This route therefore needs a specific
   mechanism, not merely "more physics".

One piece of extra physics already in the 3-D model has the **wrong sign**: vacancy-loop
thermal emission raises the threshold for vacancy loops of both habits, opening a dead
band in which *neither* ⟨a⟩ type grows rather than a coexistence band.

#### The code cannot currently represent a vacancy ⟨a⟩ loop

`ClusterDynamicsFEM.cpp:484` builds `Z` with rows labelled by **habit**
(`// row0 <c>, row1 <a>`), but line 621 selects the row by **polarity**:

```cpp
const int row(isVacancyFamily(k)?0:1);
```

With `immobileSpeciesVector = -1 1 1 1` the two labellings coincide one-to-one, so this
is latent rather than a live bug. But adding a vacancy ⟨a⟩ family would silently give it
the **basal** capture row — the wrong geometry, and precisely the quantity the whole
argument turns on. Testing hypothesis 1 or 2 therefore requires carrying a habit index
per family alongside the polarity index. It is a small change, but it is a C++ change and
a rebuild, and it must land before any vacancy-⟨a⟩ family is added.

### 1.4 The superposition principle

Applied twice, identically in structure:

| | mechanics, Eq. (42)–(43) | diffusion, Eq. (44)–(47) |
|---|---|---|
| decomposition | `u = u∞ + ũ + ζ` | `c⋆ = c∞⋆ + c̃⋆` |
| infinite-medium part | dislocation fields, Green's functions | `c∞⋆ = −G ∗ Ω₀p⋆/Z⋆`, Eq. (48) |
| finite-medium correction | Galerkin FEM, tractions/displacements corrected on ∂Ω | Galerkin FEM, `c̃⋆ = c_bc − c∞⋆` on ∂Ω |

The infinite-medium concentration solves the *anisotropic Poisson equation*
`(D⋆∇c∞⋆)·∇ + Ω₀p⋆/Z⋆ = 0`, whose Green's function follows from the isotropic one by a
change of variables — which is exactly why `√det(D)` and `D⁻¹` appear in the kernel and
why `invD`/`detD` are precomputed and cached per grain in the code.

Discretized on straight segments with linear shape functions (Appendix A), the
per-segment contribution reduces to two closed-form integrals `I₀`, `I₁` of
`1/√(au²+bu+c)`, regularized by `ε²`. **This is implemented verbatim** in
`DislocationSegment::concentrationMatrices` — including the `ε² → a2/det(D)^(1/3)`
regularization and a sum over `periodicShifts`.

---

## Part 2 — Audit of the existing implementation

### 2.1 Anisotropic diffusion — present and unused

`ClusterDynamicsParameters.cpp`:

```cpp
Dlocal <<  msD0(k,0)*exp(-msEm(k,0)/kB/T), msD0(k,1)*exp(-msEm(k,1)/kB/T), ...
Dglobal = grainPair.second.singleCrystal->C2G * Dlocal * C2G.transpose();   // per grain
invD  = D.inverse()  (per species, per grain)
detD  = D.determinant()
```

`FluxMatrix::operator()` inserts `−D.at(grainID)[k]/cdp.omega` as a full `dim×dim`
block. **The fast solve is already anisotropic; only the input is isotropic.**
`Zr3d_ghoniem.txt` currently has, for every species,

```
mobileSpeciesEnergyMigration_eV = 1.2 0.0 0.0 1.2 0.0 1.2      (and 0.759101 ×3 for i, 2i, 3i)
mobileSpeciesD0_SI             = 5.21645e-7 0.0 0.0 5.21645e-7 0.0 5.21645e-7
```

### 2.2 DAD — a fitted surrogate, disconnected from the tensor

```
dadAnisotropy = 1.178808  0.913720  0.913720  0.913720     # p_m per mobile species
dadZ0         = 0.939836  1.015504  1.015504  1.015504
```

These were fitted to reproduce the 0-D symmetric bias forms (`Z_a + Z_c = 2`) at 573 K.
They are **free parameters with no link to `msEm`**, so the code currently believes the
diffusion is isotropic and the sinks are biased — two statements that cannot both be
true. `ImmobileSinks.h` and `solveImmobileClusters()` already use
`Dbar(m) = detD(m)^(1/3)`, which is *exactly* the paper's invariant `D_eff`; so the
continuum side is structurally ready for an anisotropic tensor and simply needs `p_m`
derived rather than fitted.

### 2.3 Discrete↔continuum — complete, dormant

- `GalerkinClimbSolver::clusterForceKernel` builds the RHS from
  `quadraturePoint.cDD − quadraturePoint.cCD` — i.e. `c_eq − c̃` of Eq. (50).
- `DislocationQuadraturePoint.cpp:267` fills `cCD` by summing
  `microstructure->mobileConcentration(r, …, includingSimplex)` over CD microstructures;
  line 270 fills `cDD` from `cdp.dislocationMobileConcentration(burgers, rl, pkForce, stress)`.
- `clusterStiffnessKernel` calls `sourceSegment.concentrationMatrices(...)` for every
  (field segment, source segment) pair — the dense `O(N_seg²)` interaction.
- `DislocationNetwork.cpp:638` calls `computeClimbScalarVelocities()`, and the solver is
  constructed only if `climbSolverType` is read as `Galerkin`.
- **The system is solved lumped, not sparse.** The `lhsT` triplet path is commented out
  throughout and its `else` branch would `throw`; only `KKc`/`Fc` — plain vectors of
  length `N_nodes` — are accumulated, and the solve is one division per node,
  `nodeV[n](kc) = Fc[kc](n)/KKc[kc](n)`. This is the single most consequential audit
  finding for cost: **the linear algebra is free, and every optimization has to attack
  the pairwise assembly instead.** It also means reducing the number of shape unknowns —
  an ellipse in place of a polygon, say — buys nothing by itself; see §4.3.

**Missing entirely:** the continuum→discrete transition. fullCD's
`clusterDiscretizationTime` / `initializeDiscreteClimbLoops()` have no counterpart here.
`dislocluster_code/post/discrete_loops.py` performs the same conversion **offline, for
export and visualization only** — it never feeds back into a solve.

---

## Part 3 — The plan

Five phases, ordered so that each is independently verifiable and each leaves the code in
a runnable state. Phases 1–2 are cheap and high value; phase 4 is where the cost is.

### Phase 1 — Turn on anisotropic diffusion in the fast solve

> ✅ **Implemented** as `dislocluster_code/staging/anisotropy.py`. It generates the
> six migration-energy components per species from one `p_m`, and generates
> `dadAnisotropy` from the *same* `p_m` in the same pass — which is Phase 2, and
> the two are done together because doing either alone leaves the tensor and the
> capture efficiencies describing different physics. Verified: `p_m = 1`
> reproduces the isotropic tensor exactly, the recipe reproduces the paper's
> Table 2, and the anisotropic fast solve converges (48 s, 0 unconverged, 200 nm).
>
> The anisotropy shipped is `p_m = (1.178808, 0.913720, 0.913720, 0.913720)` —
> the values `dadAnisotropy` already carried. Adopting them as the *tensor's*
> anisotropy makes the two agree for the first time **without moving the
> calibration**. It is deliberately *not* the paper's choice (DAD on 2i only,
> `p_m < 0.8`), which needs the refit.

**Change:** replace the isotropic `mobileSpeciesEnergyMigration_eV` rows in
`Zr3d_ghoniem.txt` with `p_m`-derived pairs using the §1.2 recipe at T = 573 K,
`E_m^eff = 0.759101` eV for the interstitial family:

| `p_m` | `E_m⟨11,22⟩` | `E_m⟨33⟩` |
|---|---|---|
| 0.9 | 0.748696 | 0.779911 |
| 0.8 | 0.737065 | 0.803174 |
| **0.7** | **0.723878** | **0.829548** |
| 0.6 | 0.708655 | 0.859994 |
| 0.5 | 0.690650 | 0.896004 |

Following the paper, apply DAD to the **cluster species (2i, 3i) only** at first, leaving
`v` and `i` isotropic; the paper's justification is that single defects diffuse
isotropically in Zr while clusters do not.

**Code work:** none in C++. Add a generator in
`dislocluster_code/zerod/calibration.py` (or a new `materials.py`) that writes the six
components from `(E_m^eff, p_m, T)` so the two can never drift apart, and expose `p_m`
per species in the notebook's `MATERIAL` dict.

**Efficiency:** the FEM assembly, sparsity and node count are **unchanged** — a full 3×3
block replaces a scalar in the same slot. The cost is entirely in *conditioning*:

| `p_m` | `D⟨c⟩/D⟨a⟩` | anisotropy |
|---|---|---|
| 0.9 | 0.531 | 1.9 : 1 |
| 0.8 | 0.262 | 3.8 : 1 |
| 0.7 | 0.118 | 8.5 : 1 |
| 0.6 | 0.047 | 21 : 1 |
| 0.5 | 0.016 | **64 : 1** |

A 64:1 anisotropy raises the condition number of the diffusion operator by roughly the
same factor, and the fast solve is already the bottleneck and already fragile: the mobile
Newton iteration has an open non-convergence issue from a cold start, and BiCGSTAB has
been observed to break down outright against Eigen 5. **Budget for the diagonal
preconditioner becoming inadequate above `p_m ≈ 0.7`.** Mitigation, in order of
preference: (i) keep the incumbent preconditioner and measure; (ii) switch the
Dirichlet-reduced solve to an incomplete-Cholesky/ILU preconditioner, which for a
diffusion operator is a small change to `FixedDirichletSolver`; (iii) rescale coordinates
per species by `D^(-1/2)` to restore isotropy in the solve — mathematically exact but
species-dependent, so it costs one operator per species.

**Verification:** at `p_m = 1` the answer must be **bit-identical** to today's. That is a
hard check and it is available for free, because setting the three components equal
reproduces the current scalar exactly.

### Phase 2 — Make DAD derived, not fitted

> ✅ **Implemented**, together with Phase 1 and in Python rather than in C++.
> `staging/anisotropy.py` writes `mobileSpeciesEnergyMigration_eV` and
> `dadAnisotropy` from one `p_m` in one pass, which makes them consistent by
> construction with no MoDELib rebuild. `--show` reports both and whether they
> agree; before this change it reported `agree: NO` for every species.
>
> A second defect surfaced while wiring it and is fixed: `cfg.domain_key` hashed
> the material file's *name*, so editing the tensor in place did **not**
> invalidate the staged case — which holds a *copy* of the material file and a
> bootstrap solved against it. The key now includes the file's content digest.
> That is issue 3 of `ZR3D_GHONIEM_CHANGES.md` in a new place: the run does not
> fail, it answers the wrong question.

**Change:** compute `dadAnisotropy` from the tensor rather than reading it,
`p_m = (D₃₃/D₁₁)^(1/6)` in `getDlocal()`, with the material key retained only as an
override. `loopDADbias()` is then a *function of the diffusivities*, and `Dbar =
detD^(1/3)` is automatically `D_eff` — so the continuum immobile families see one
consistent anisotropy instead of two independent ones.

**Why this is the correct fix and not a double count:** Woo's `Z_c = Z⁰p`,
`Z_a = Z⁰(p+p⁻²)/2` are derived *assuming* an anisotropic `D` and are applied to a flux
built from `D_eff`. Provided `p` is the tensor's own anisotropy, the product
`Z·D_eff` reproduces the directional absorption once. Leaving `dadAnisotropy` free while
`D` is anisotropic is what would double count.

**Cost:** zero. **Consequence:** this changes the answer, and it invalidates the fitted
0-D correspondence — which is already stale after the Ω correction. Fold it into the
planned refit rather than refitting twice.

### Phase 3 — Reproduce the paper's own case as a regression test

`greatWhitePlots/sims/simA` and `simB` are the paper's configurations and they are
already in the tree. Stage and run them against the current build before touching the
coupling. This is the cheapest possible validation of phases 1–2 and of the dormant
climb path, and it is the only place where a published number is available to check
against.

**Note the model differences** before comparing: `simA` has `mSize = 3` (v, i, 2i)
against DisloCluster's 4, `mobileSpeciesSurvivingEfficiency = 0.01` against our 1.0, and
`otherSinks_SI = 0` against our network sink. These are the paper's choices, not errors.

### Phase 4 — Discrete loops that climb, coupled to the continuum

This is the substantial phase. Four design decisions dominate it and are set out before
the sub-steps, because they determine the cost, the correctness and the amount of new
code: **when to switch** (§4.1), **nearest-neighbor truncation** (§4.2), the **split
representation by loop family** (§4.3), and **the transfer ledger that prevents double
counting** (§4.5).

#### 4.1 When to switch: the transition dose `d_coarsen`

The run stays purely continuum up to `d_coarsen` and switches to the coupled mode there.
The question is not *when coarsening starts* — the continuum already models it, in both
channels — but **when the continuum's mean-field treatment of coarsening stops being
trustworthy.**

**The criterion is the Avrami argument itself, and it is already computed.**
`solveImmobileClusters()` forms

```
phi_LL = 1 − exp(−kappaLL · Λ_LL),   Λ_LL = (4/3)π r³ N   = (π/6)(2r/d)³
phi_LN = 1 − exp(−kappaLN · Λ_LN),   Λ_LN = π r² ρ_N
```

with `kappaLL = 2.0`, `kappaLN = 0.4`, `ρ_N = 1.88×10¹⁴ m⁻²` (network spacing 72.9 nm).
The identity `Λ_LL = (π/6)(2r/d)³` is worth noting: **the Avrami overlap argument and the
geometric crowding ratio are the same criterion**, so this is not a new or arbitrary
measure — it is the model's own estimate of the effect in question.

`1 − exp(−κΛ)` is a Poisson estimate that assumes *independent random placement*. Its
leading error is the two-body correlation, entering relatively at `O(Λ)` — and coarsening
is precisely the process that correlates positions. So the Avrami form is asymptotically
exact as `Λ → 0` and progressively self-invalidating as `Λ → 1`.

**Evaluated on the measured trajectory** by
`dislocluster_code/post/coarsening.py` (interior nodes, 500 nm OPT run, `phi_gate`
= the 90th percentile of `phi` over interior nodes):

| dose | family | `2r/d` | `Λ_LL` | `φ_LL` | `Λ_LN` | `φ_LN` | **`φ_gate`** |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10⁻⁴ | ⟨c⟩ | 0.018 | 0.0000 | 0.0000 | 0.0033 | 0.0013 | 0.0013 |
| 10⁻⁴ | ⟨a⟩ | 0.011 | 0.0000 | 0.0000 | 0.0027 | 0.0011 | 0.0011 |
| 0.01 | ⟨c⟩ | 0.055 | 0.0001 | 0.0002 | 0.0053 | 0.0021 | 0.0021 |
| 0.01 | ⟨a⟩ | 0.078 | 0.0002 | 0.0005 | 0.0089 | 0.0035 | 0.0036 |
| 0.1 | ⟨c⟩ | 0.190 | 0.0036 | 0.0071 | 0.0633 | 0.0250 | 0.0256 |
| 0.1 | ⟨a⟩ | 0.091 | 0.0004 | 0.0008 | 0.0132 | 0.0053 | 0.0053 |
| 1 | ⟨c⟩ | 0.628 | 0.1295 | 0.2282 | 0.6956 | 0.2429 | **0.2443** |
| 1 | ⟨a⟩ | 0.091 | 0.0004 | 0.0008 | 0.0134 | 0.0054 | 0.0054 |
| 10 | ⟨c⟩ | 0.611 | 0.1197 | 0.2129 | 0.6596 | 0.2319 | **0.2320** |
| 10 | ⟨a⟩ | 0.091 | 0.0004 | 0.0008 | 0.0142 | 0.0056 | 0.0057 |

**The radius must be the one the continuum model itself uses** — the CD-internal
`b_cd`, not the `b_dd` that `discrete_loops` reports. The cluster-dynamics material
file gives ⟨c⟩ a full ⟨0001⟩ Burgers vector (1.632993 b) while MoDELib3's aLoop
generator builds a half-⟨0001⟩ loop (0.8165 b); defect count is what is conserved, not
radius, so the same loop has `r_dd/r_cd = √2`. Using the discrete radius inflates `Λ_LL`
by 2.83 and `Λ_LN` by 2 and fires the detector roughly a decade of dose too early. An
earlier draft of this table did exactly that.

Three things fall out, and they settle the design:

1. **⟨a⟩ never leaves the mean-field regime** — `φ_gate ≤ 0.006` at every dose, and it
   plateaus there. On coarsening grounds ⟨a⟩ never needs discretizing. It needs the
   ellipse ROM for its *shape* response to DAD, which is a separate argument (§4.3).
2. **⟨c⟩ crosses between 0.1 and 1 dpa**, both channels together, with the loop-network
   channel leading slightly.
3. **The existing dose grid cannot resolve it.** A decade of dose separates the last
   sub-threshold snapshot from the first over-threshold one. **The detector has to run
   at immobile-substep resolution, not at snapshot resolution.**

The criterion also **independently selects the same family the crowding analysis
assigned to nodal treatment** — ⟨c⟩ to discrete, ⟨a⟩ to the cheap ROM. Two unrelated
arguments agreeing is the best evidence available that the split is real and not an
artifact of either.

**The detector.** At each immobile substep, on **interior nodes only** (the boundary
shell carries 99.8% of the ⟨a⟩ density and would trigger immediately on an artifact —
the same reason whole-domain loop means are never reported):

```
phi_max(x, k) = max( 1−exp(−kappaLL·Λ_LL), 1−exp(−kappaLN·Λ_LN) )
trigger family k  when  frac{ x ∈ interior : phi_max(x,k) ≥ phi* }  ≥  f*
```

Defaults `phi* = 0.2`, `f* = 0.10`, sustained for **two consecutive substeps**. `phi* =
0.2` corresponds to a ~20% error in the coalescence rate from the neglected pair
correlation, and lands at `2r/d = 0.597` in the like-loop channel and `r = 30.7 nm` in
the loop-network channel — both comfortably inside the ⟨c⟩ transition. Requiring a
*fraction* of nodes rather than any single node prevents one outlier from triggering the
switch; the two-substep hold prevents chatter.

Cost: `Λ_LL` and `Λ_LN` are already formed inside the substep loop, so the detector is a
comparison and a reduction — immeasurable.

**Threshold the smooth quantity, not the step.** "A fraction `f*` of nodes exceed `phi*`"
and "`phi_gate` ≥ `phi*`", with `phi_gate` the `1−f*` quantile of `phi`, are the same
statement — but the second is smooth in dose while the first is a step. On the 500 nm run
the fraction goes 0.000 to 1.000 between two snapshots, because the interior field is
smooth enough that every interior node crosses at once. Interpolating the fraction gives a
meaningless crossing dose (0.13 dpa); interpolating `phi_gate` gives 0.63 dpa. Trigger on
the fraction, interpolate on `phi_gate`.

**Measured `d_coarsen`**, from `post/coarsening.py` at `phi* = 0.2`, `f* = 0.10`:

| run | ⟨c⟩ | ⟨a⟩ | ⟨c⟩ plateau `phi_gate` |
|---|---:|---:|---:|
| 500 nm hex, `..._OPT` | **0.63 dpa** | never | 0.244 |
| 200 nm hex | **0.24 dpa** | never | 0.483 |

**And a caveat the sweep exposes.** `phi` *saturates* — growth and coarsening come into
balance — so a threshold set near the plateau is either never reached or reached at a dose
that moves a lot for a small change in `phi*`:

| `phi*` | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---:|---:|---:|---:|---:|
| 500 nm ⟨c⟩ | 0.22 | 0.37 | **0.63** | — | — |
| 200 nm ⟨c⟩ | 0.14 | 0.19 | **0.24** | 0.31 | 0.41 |

The 200 nm case is robust across the whole range. **The 500 nm case is not**: its plateau
is 0.244, so `phi* = 0.25` never fires and `d_coarsen` nearly triples between 0.10 and
0.20. Do not treat 0.63 dpa as a converged number — treat it as bracketing the switch to
within a factor of ~3, and prefer the substep-resolution detector before committing to it.

(The 200 nm run predates the atomic-volume correction of §Phase 5, so `Ω` in its stored
fields differs from the one this module reads. Its φ values are indicative, not
comparable with the 500 nm run's.)

**A corroborating detector, reported but not used to trigger.** The ratio of the
coalescence loss to the nucleation source in the number balance,
`coalN·n / nucRate`, states literally when the population stops being
nucleation-controlled and becomes coarsening-controlled; it reaches 1 at saturation. It
answers a slightly different question from the geometric detector — *whether coarsening
matters* rather than *whether the mean-field estimate of it is valid* — and the second is
what governs the switch. Print both: if they disagree by much, the physics has changed
in a way worth understanding before trusting either.

**Switching mechanics.** The trigger is evaluated per substep but acted on at the next
fast-solve boundary, where a DDomp call happens anyway and the state is already
synchronized. `d_coarsen` is recorded in `summary.json` with the `φ` trajectory that
produced it, so the choice is auditable after the fact rather than a hidden branch.

#### 4.2 The interaction is screened — nearest neighbors are enough

The Galerkin climb solve is `O(N_seg²)`: `clusterStiffnessMatrix(fieldSegment,
sourceSegment)` runs over every ordered pair, each costing an `mSize`-wide
`concentrationMatrices` evaluation summed over `periodicShifts`. It is OpenMP-parallel
over field segments.

**The linear solve is not the cost, and this is worth stating because it is
counter-intuitive.** The sparse path in `GalerkinClimbSolver` is commented out and would
throw; only the *lumped* path is live, and the "solve" is one scalar division per node,
`nodeV[n](kc) = Fc[kc](n)/KKc[kc](n)`. **100% of the cost is the pairwise assembly.**
Reducing the number of unknowns therefore buys nothing; reducing the number of *pairs*
is the only lever that matters.

A bare `1/r` kernel cannot simply be truncated. A growing loop is a net sink, so its
monopole does *not* vanish — `∮ w·(b×dℓ) = w|b|·perimeter > 0` — and a shell at radius
`r` contributes `~r²·(1/r) = r`, so the sum grows with the cutoff instead of converging.

**But the physical kernel is not bare: it is screened by the sink field, `e^{−kr}/r`,
where `k²` is the total sink strength — exactly the `k²` that `ImmobileSinks.h` already
assembles.** Evaluating `L_s = 1/k` from the measured population of the 500 nm OPT run
plus `otherSinks_SI = 1.88×10¹⁴ m⁻²`:

| dose | `S_⟨c⟩` (m⁻²) | `S_⟨a⟩` ×3 (m⁻²) | `k²` (m⁻²) | **`L_s`** | spacing `d` | `L_s/d` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 1.14×10¹² | 6.33×10¹¹ | 1.90×10¹⁴ | 72.6 nm | 265 nm | 0.27 |
| 0.01 | 2.05×10¹³ | 7.49×10¹³ | 2.83×10¹⁴ | 59.4 nm | 109 nm | 0.54 |
| 1 | 2.31×10¹⁴ | 7.73×10¹³ | 4.96×10¹⁴ | 44.9 nm | 110 nm | 0.41 |
| 10 | 2.24×10¹⁴ | 7.44×10¹³ | 4.87×10¹⁴ | 45.3 nm | 110 nm | 0.41 |

**`L_s/d < 1` at every dose** — the interaction is screened before it reaches the nearest
neighbor. At 10 dpa the first shell is down to `e^{−2.4} ≈ 9%` and the second to
`e^{−4.9} ≈ 0.7%`.

**Truncation here is correct rather than merely tolerable, and the reason matters.** The
continuum field `c̃` already carries the mean-field response of the entire loop
population, through `ImmobileSinks`. The discrete Green's-function sum must therefore
supply *only the near-field correction the mean field misses*; extending it further
would double count — the same failure mode as §4.5. And this is why `L_s` is the natural
matching radius rather than a tuned parameter: the `k²` of the screened kernel **is** the
`ImmobileSinks` sink strength, so the mean field and the discrete correction meet at
`L_s` by construction.

**Implementation.** Set `R_c = 3/k`, which is adaptive for free since `k²` is recomputed
each step. Cell list on loop centers, cell size `R_c`; reject whole loop pairs on center
distance minus radii before descending to segments. Three things to get right:

- **Always keep the self-term** — a segment against the other segments of its own loop.
  That is the dominant contribution and it is what makes a loop's own shape respond to
  the DAD.
- Apply the cutoff **per periodic image**, not to the base position.
- Because the kernel stays bare inside `R_c`, residual error appears as **sensitivity to
  `R_c`**, which is measurable: sweep `R_c/L_s ∈ [2,4]` and require the answer to be
  flat. If it is not, the fix is the screened kernel itself — at the price of losing the
  closed-form `I₀`/`I₁` of Appendix A, which have no screened counterpart.

#### 4.3 Split the implementation by loop family

The two families are in completely different geometric regimes, and the split follows
from measured numbers rather than preference. From the 10 dpa interior population:

| family | habit | `r` | spacing `d` | `2r/d` | `r/d` | point-sink error `~(r/d)²` |
|---|---|---:|---:|---:|---:|---:|
| ⟨c⟩ | basal, vacancy | 47.3 nm | 110 nm | 0.86 | 0.43 | **19%** |
| ⟨a⟩₁₋₃ | prismatic, interstitial | 4.9 nm | 107 nm | 0.091 | 0.046 | **0.2%** |

**There is also a symmetry argument, and it points the same way.** A basal ⟨c⟩ loop's
line lies entirely in the basal plane, where a transversely isotropic `D` (`D₁₁ = D₂₂ ≠
D₃₃`) is isotropic — so every segment sees the same diffusivity and **⟨c⟩ loops stay
circular under DAD**. A prismatic ⟨a⟩ loop's line samples both the basal plane and the
`c` axis, so its segments see different diffusivities and **⟨a⟩ loops must go
elliptical**. This is exactly what the paper reports (conclusion 2), and it reports it
for ⟨a⟩ only.

So:

**⟨a⟩ — a two-parameter ellipse ROM, in the slow step.** Carry semi-axes
`(a_x, a_z)` per loop, giving **2 ODEs per loop per species** — embarrassingly
parallel, no topology, no remesh, no junctions, and structurally identical to the
per-node CVODE systems the slow step already solves. This captures the dominant shape
degree of freedom the paper says the circular approximation misses, while remaining a
strict subset of its full nodal freedom (no non-planar climb, no faceting, no
coalescence by contact). The formulation is set out in §4.3.1.

Because `r/d = 0.046`, a neighboring ⟨a⟩ loop can be treated as a **point sink** — its
`c∞` is a monopole, `O(1)` per pair, accurate to 0.2%. The neighbor term is lagged
(explicit), so there is no coupled solve.

##### 4.3.1 The ⟨a⟩ ellipse ROM, in equations

This is a reduced-order model proposed here; it is **not** in Li et al., who carry full
nodal freedom. What follows separates what the geometry fixes from the one place a
modeling choice enters.

**Kinematics — fixed by the crystallography, no freedom.** An ⟨a⟩ loop is prismatic and
pure climb, so its Burgers vector is normal to its own habit plane, and that plane
therefore *contains* the c-axis. With in-plane axes `ê_z ∥ [0001]` and
`ê_x = b̂ × ĉ` (an in-plane ⟨10̄10⟩-type direction):

```
r(θ)  = a_x cosθ ê_x + a_z sinθ ê_z
ξ(θ)  ∝ (−a_x sinθ,  a_z cosθ)                      tangent
n̂(θ) = (a_z cosθ,  a_x sinθ) / sqrt(a_z²cos²θ + a_x²sin²θ)      climb direction
```

`n̂` is the in-plane outward normal because `b × ξ` lies in the plane and is
perpendicular to `ξ`. Defect count is

```
N_defects = π a_x a_z |b| / Ω
```

which is **the same invariant `discrete_loops` conserves**, so the ROM and the discrete
export agree by construction rather than by calibration.

**The two ODEs.** Evaluate the normal velocity at the two axis endpoints:

```
ȧ_x = (Ω/|b|) Σ_m  (s_m Z_m D_m^(x) / λ_x) [ c_m − c_m⁰ exp(−s_m Ω σ_n^(x) / k_B T) ]
ȧ_z = (Ω/|b|) Σ_m  (s_m Z_m D_m^(z) / λ_z) [ c_m − c_m⁰ exp(−s_m Ω σ_n^(z) / k_B T) ]
```

with `s_m = ±1` by species polarity, `Z_m` the dislocation bias
(`discreteDislocationBias`), and `λ` a capture length. The bracket is Eq. (49) of the
paper written as a driving force.

**Where the DAD enters — and the one modeling choice.** The capture diffusivity is set
by the plane the flux arrives through, which is the plane perpendicular to the local
tangent:

| endpoint | tangent `ξ` | capture plane | `D` |
|---|---|---|---|
| θ = 0, end of `a_x` | ∥ `[0001]` | basal | `D_a` |
| θ = π/2, end of `a_z` | in basal plane | spans `ĉ` and `b̂` | `√(D_a D_c)` or `D_c` |

The second row is the choice. A geometric-mean closure (the natural one for a line sink
in a transversely isotropic medium) and a normal-projection closure give different
steady aspect ratios:

| closure | `a_x/a_z` | at `p_m = 0.7` |
|---|---|---:|
| geometric mean, `D^(z) = √(D_a D_c)` | `√(D_a/D_c) = p_m⁻³` | 2.9 |
| normal projection, `D^(z) = D_c` | `D_a/D_c = p_m⁻⁶` | 8.5 |

Both put the **major axis in the basal plane** for `p_m < 1`, which is the paper's
conclusion 2. **Which closure is correct should be settled by measurement, not by
argument** — that is exactly validation item 6, the ROM against a full nodal solve on
the same case. Until it is, the ROM has a factor-of-3 ambiguity in the ellipticity it
predicts, and that ambiguity should be quoted with any result it produces.

**What limits the ellipticity.** Without a restoring term the ratios above are
attractors and the loop elongates indefinitely. The restoring term is the self-stress
entering `c^eq`, through the curvature of the ellipse:

```
σ_n^self(θ) = −[ μ|b| / (4π(1−ν)) ] κ(θ) ln( 1 / (κ(θ) r_c) )
κ(0) = a_z / a_x²          κ(π/2) = a_x / a_z²
```

As the loop elongates, curvature rises at the major-axis ends, the back-stress there
grows, and growth throttles. This is the mechanism the paper refers to when it notes
that "the change of the loop shape will re-distribute the Peach–Koehler force,
influencing the shrinking/growing rate of the loop". **A ROM without this term is not
merely less accurate — it has no steady shape at all.**

**Driving concentration.** `c_m = c̃_m + Σ_neighbors c∞_m`, with `c̃` from the FEM at
the loop's location and the neighbor sum a lagged monopole over loops inside `R_c`
(§4.2). Explicit in the neighbor term, so no coupled solve.

**What the ROM cannot represent**, and should assert against at runtime rather than
assume: coalescence by contact, junction formation, non-planar climb, and any departure
from a centered ellipse. `2r/d = 0.091` says none of these are active for ⟨a⟩ at the
doses run so far — but that is a property of this dose and dose rate, not of the model.

**⟨c⟩ — full nodal Galerkin, `DD_SIDES = 12`.** Crowding (`2r/d = 0.86`) breaks
axisymmetry through the *neighbor* field even though DAD does not, and `r/d = 0.43`
makes a point-sink source 19% wrong. These need resolved segments both as field and as
source. There are only ~99 of them, so this is affordable.

**A common interface is required.** The two families are neighbors of each other, so both
representations must be able to answer "evaluate `c∞` at `x`" — the ellipse by monopole,
the nodal loop by segment sum. Design that interface first; it is the one piece of
structure the split imposes.

**Do not hand the 64-sided rendering polygons to the climb solver.**
`discrete_loops.CIRCLE_SIDES = 64` was chosen for figures, where it costs nothing.
`DD_SIDES = 12` is a separate constant; the two need not agree, because the exported
`loopRadii_SI` is area-matched for any `n`. Confusing them costs a factor of 28.

#### 4.4 Cost of the split

Interior population at 10 dpa (99 ⟨c⟩ + 315 ⟨a⟩), `k = 33` neighbors at `R_c = 3L_s`:

| scheme | work |
|---|---:|
| **⟨c⟩ nodal, 12 sides, NN** | 470 448 segment-pairs |
| **⟨a⟩ ellipse, point-sink neighbors** | 10 395 `O(1)` evaluations |
| **split total** | **≈ 4.8×10⁵** |
| monolithic nodal, 12 sides, NN | 1.96×10⁶ |
| monolithic nodal, 12 sides, all pairs | 2.46×10⁷ |
| monolithic nodal, 64 sides, all pairs | 6.99×10⁸ |

**The split plus nearest-neighbor truncation is ~1450× cheaper than the naive
64-sided all-pairs assembly**, and ~4× cheaper than a monolithic nodal treatment with
the same cutoff. Note that ⟨c⟩ alone would be affordable even without the cutoff
(1.4×10⁶ pairs); the cutoff is what makes the ⟨a⟩ population — three times as numerous —
free.

Two further cost controls, both cheap:

1. **Reuse the operator-split cadence.** The climb solve belongs on the `fem_every`
   cadence beside the fast solve, not on every immobile substep — it is the part that
   carries the spatial coupling, for exactly the same reason.
2. **Transfer only the interior population.** The boundary shell carries 99.8% of the
   ⟨a⟩ density and is a known artifact (loops accumulate at nucleation size because there
   is no removal channel where the mobile field is pinned). `region='interior'` is
   already the default in `discrete_loops`.

**Anticipated envelope.** With the split, `DD_SIDES = 12`, `R_c = 3/k` and an
interior-only transfer, the climb solve should sit well below one fast solve (~3–4 min on
the 500 nm case) rather than dominating it. **These are preconditions, not later
optimizations**: without them the assembly is 10²–10³× larger and the coupled march
stops being runnable.

#### 4.5 No double counting: the transfer ledger

Once a loop is discrete it is a sink through the Green's function. If it is *also* still
in the continuum immobile field it is a sink through `ImmobileSinks` as well, and it
absorbs twice. Everything in this subsection exists to make that impossible rather than
unlikely.

**The three invariants.** For each family `k`, across a transfer event, define totals
over the whole crystal regardless of which carrier holds them:

```
(I)   loop number      N_tot = ∫ n_k dV            +  (count of discrete loops)
(II)  stored defects   C_tot = ∫ c_k/Ω dV          +  Σ_i π r_i² |b_k| / Ω
(III) sink strength    S_tot = ∫ Z·2π r̄_k n_k dV   +  Σ_i Z_i · 2π r_i
```

(I) and (II) are **exact** and must not move at all — assert, hard-fail. (III) is a
modeling statement and is the one that will actually catch mistakes; see below.

**Transfer at the mean size, or transfer everything.** The continuum carries *one* mean
size per family per node, so a size-selective transfer is not representable — there is no
size spectrum to split. Removing `Δn` loops must remove content in the same proportion,
`Δc/Δn = c/n`, which leaves `r̄` unchanged and makes (III) exact by construction. The
simplest policy that satisfies this trivially is to **transfer the whole family and zero
its continuum field**; `ImmobileSinks` then stops counting it with no partial bookkeeping
and no drift.

**Nucleation keeps refilling the continuum, and that is the right architecture.** Cascade
and clustering nucleation continue after `d_coarsen`, so the continuum ⟨c⟩ population
regrows from zero. Re-evaluate the §4.1 detector each fast-solve interval and transfer
again whenever the *continuum* population re-crosses `phi*`. The division of labor that
results is the physically natural one: **the continuum owns nucleation and early growth,
where it is valid and cheap; the discrete population owns coarsening, where it is
needed.** Loops that shrink below `r_min` or coalesce away are removed on the discrete
side.

**The subtle double count, stated explicitly.** `DislocationQuadraturePoint::cCD` samples
the continuum mobile field *at the loop line*. If the continuum still carried that loop's
sink, the loop would see its own depletion twice — once in `c̃` through `ImmobileSinks`,
and once again in its own `c∞`. Zeroing the transferred family fixes both this and the
absorption double count in one step, and they are the same bug.

**A concrete discontinuity that must be resolved before the first transfer.** The two
sides scale loop absorption by different, unrelated factors:

| side | parameter | ⟨c⟩ value |
|---|---|---:|
| continuum | `loopSinkScale` (per immobile family) | **0.291528** |
| discrete | `discreteDislocationBias` row 0 (vacancy loop) | **1.0** |

`loopSinkScale` exists to reproduce ZrMicro's sink convention exactly — one radius
prefactor for all families, plus the fitted `Q` on the ⟨c⟩ channel — while
`discreteDislocationBias` is the elastic bias `Z⋆` of the paper. **As things stand, a
⟨c⟩ loop's absorption would jump by 1/0.2915 = 3.4× at the moment it becomes discrete.**
Invariant (III) is exactly the check that catches this. Three ways out, and it must be a
deliberate choice:

1. Carry `loopSinkScale` into the discrete representation, setting the transferred
   loop's bias to `Z_DAD(k,m) · loopSinkScale_k`. Preserves (III) exactly; keeps a 0-D
   fitting convention alive in a model that no longer needs it.
2. Drop `loopSinkScale` at transfer and accept the jump, absorbing it in the refit. Most
   defensible physically — `loopSinkScale` is a convention artifact, and the discrete
   treatment computes the absorption from first principles — but it makes runs across
   `d_coarsen` non-comparable until the refit lands.
3. Retire `loopSinkScale` on both sides as part of the refit, so continuum and discrete
   agree by construction. Cleanest, largest scope.

Note that the *DAD* part is not a discontinuity: it leaves `loopDADbias()` on the
continuum side and re-enters through the anisotropic `D` in the Green's function on the
discrete side. That is the paper's own division and it is consistent — the two
representations carry the same physics by different routes, which is precisely why
invariant (III) has to be measured rather than assumed.

**What must *not* be removed at transfer.** `otherSinks_SI` is the *network* dislocation
sink, and the network is not being discretized — it stays. It would become a double count
only if a discrete network were introduced later, and that should be recorded at the key.
Likewise the grain boundary remains a Dirichlet condition and appears in neither sink
term, exactly as `ImmobileSinks.h` already documents.

**What is at risk of being *lost*, which is the mirror-image failure.** Both coarsening
channels are continuum terms keyed on `n_k`. Once ⟨c⟩ is zeroed they evaluate to nothing
— correctly, since there is no continuum ⟨c⟩ left — but unless the discrete side supplies
them the physics simply disappears:

- **loop–loop**: must become geometric contact and merging on the discrete side.
  `discrete_loops` already has a coalescence pass, and its diagnostic reports 2 passes for
  ⟨c⟩ at 1 and 10 dpa against 1 at lower dose — the same crossing the §4.1 detector fires
  on, found independently.
- **loop–network**: must become a stochastic removal law applied to discrete loops, using
  the same `kappaLN`, `ρ_N`. This is not a double count — it is the same physics on a
  different carrier — but it has no upstream precedent and will not appear by itself.

**The runtime audit.** Print invariants (I)–(III) per family immediately before and after
every transfer event, and at every fast solve thereafter:

- (I) and (II) moving at all is a bug — hard-fail with the per-node residual.
- (III) moving is reported as a ratio with the `loopSinkScale` decision named, so the
  discontinuity is visible in `report.md` rather than buried.

This is the same discipline as `post/boundary_flux.py`: measure the quantity a second,
independent way and let the two disagree out loud. It is cheap — three reductions over
nodes plus a sum over loops — and it is the only thing standing between a silent 3.4×
absorption error and a published number.

#### 4.6 The sub-steps

**4a. The `d_coarsen` detector.** ✅ **Implemented, reporting-only**, as
`dislocluster_code/post/coarsening.py` — `phi_LL`/`phi_LN` per interior node per family
from the CD-internal radius, thresholded on `phi_gate` with a hold, plus a `phi*`
sensitivity sweep. Run as

```
python -m dislocluster_code.post.coarsening <run> [--phi 0.2] [--frac 0.10] [--out …]
```

It switches nothing. Remaining work for this sub-step: move the same reduction inside the
immobile substep loop of `coupling/march.py` so the crossing is resolved to a substep
rather than interpolated across a decade of dose, and record `d_coarsen` with its `φ`
trajectory in `summary.json`.

**4b. A runtime continuum→discrete transition.** Port the logic of
`dislocluster_code/post/discrete_loops.py` — which already conserves defect count
exactly, places loops by density, and clips to the crystal — into a staging step that
emits the discrete population *at* `d_coarsen`: an `aLoops` microstructure with
`DD_SIDES = 12` for ⟨c⟩, and an ellipse table for ⟨a⟩. Doing it in Python first, between
march intervals, avoids any C++ change and keeps the transition inspectable.

**4c. The transfer ledger.** Implement §4.5: zero the transferred family's continuum
field through `FieldBridge.write_immobile_field`, which already owns that block, and add
the three-invariant audit. Resolve the `loopSinkScale` / `discreteDislocationBias`
discontinuity explicitly here — it is a decision, not a bug to be patched later. Add the
discrete-side loop–loop and loop–network coarsening laws so the physics zeroed out of the
continuum is not simply lost.

**4d. Nearest-neighbor infrastructure.** Cell list, `R_c = 3/k` from the existing sink
strength, self-term always retained, cutoff applied per periodic image. Build this
before turning the climb solver on, not after — with all-pairs assembly the 500 nm case
is not runnable long enough to debug.

**4e. The ⟨a⟩ ellipse ROM.** Two ODEs per loop per species in the slow step, driven by
`c̃` from the FEM plus the lagged monopole sum over neighbors within `R_c`.

**4f. ⟨c⟩ climb and the plastic strain.** `GalerkinClimbSolver` supplies the per-species
climb velocities for the nodal family; the continuum march continues to own whatever was
not transferred. Growth strain returns through the existing `F` output.

### Phase 5 — Calibration and validation

1. `p_m = 1` bit-identity (phase 1).
2. `simA`/`simB` against the paper's published curves (phase 3).
3. Sweep `p_m ∈ {0.9, 0.8, 0.7, 0.6, 0.5}` on the 200 nm case and check for the paper's
   two qualitative results: **elliptical ⟨a⟩ loops** (impossible in the continuum model —
   a genuine new prediction) and the **`p_m < 0.8` threshold** for simultaneous ⟨a⟩
   expansion and ⟨c⟩ contraction.
4. Boundary-flux closure (`post/boundary_flux.py`) must still hold with an anisotropic
   `D` — the divergence theorem does not care about anisotropy, so this is a real,
   independent check that the tensor entered the flux correctly and not merely the
   reaction terms.
5. **Cutoff convergence.** Sweep `R_c/L_s ∈ [2,4]` and require the loop kinetics to be
   flat. This is the honest test of §4.1: the truncation error of a bare kernel shows up
   as `R_c` sensitivity and nowhere else.
6. **The two representations must agree where both are valid, and this is also how the
   ROM's one free choice gets settled.** Run the ⟨a⟩ family nodally as well as through
   the ellipse ROM on one case. They should agree while the loop stays near-elliptical;
   where they diverge, the nodal answer is the reference and the divergence bounds the
   ROM's validity. Without this the split is an assumption rather than a verified
   reduction.

   Specifically, this measurement decides the capture-diffusivity closure of §4.3.1 —
   geometric mean versus normal projection — which is a **factor of 3 in the predicted
   ellipticity** at `p_m = 0.7` (`p_m⁻³ = 2.9` against `p_m⁻⁶ = 8.5`). Fit the closure
   to the nodal aspect ratio; do not pick it by argument. Until it is fitted, quote the
   ambiguity with any ellipticity the ROM reports.
7. **Defect-count conservation across the transition.** Total interstitials and vacancies
   stored must be continuous through the continuum→discrete handoff of 4b/4c. This is the
   direct test for the double-count failure, and the conservation channels of
   `post/volume_average.py` already measure it.
8. Refit the 28 parameters, once, against the anisotropic model at the corrected Ω.

---

## Sequencing and risk

| Phase | Effort | Risk | Blocking? |
|---|---|---|---|
| 1 — anisotropic tensor | low, data only | **conditioning of the fast solve** | no |
| 2 — derived DAD | low | changes the answer; needs refit | no |
| 3 — reproduce `simA` | low | model mismatch vs our 4-species set | no |
| 4a — `d_coarsen` detector | ✅ done (reporting) | **`phi*` sensitivity at 500 nm** | yes, for 4b |
| 4b — runtime transition | medium | — | yes, for 4c/4e/4f |
| 4c — transfer ledger | medium | **highest correctness risk**; `loopSinkScale` jump | yes |
| 4d — nearest-neighbor infrastructure | low–medium | cutoff convergence | **yes, for 4e/4f** |
| 4e — ⟨a⟩ ellipse ROM | medium | ROM validity vs nodal | — |
| 4f — ⟨c⟩ climb + strain | medium | conditioning at high `2r/d` | — |
| 5 — refit | high | — | — |

**Recommended order:** 1 → 3 → 2 → 4a → 4d → 4b → 4c → 4f → 4e → 5.

Three points about the ordering, all deliberate:

- **4a first, reporting-only.** The detector costs almost nothing, runs on runs that
  already exist, and answers the question everything else depends on — where
  `d_coarsen` actually is. It should be trusted before it is allowed to switch anything.
- **4d before the climb solver.** With all-pairs assembly the 500 nm case is not runnable
  long enough to debug the transition, so the neighbor infrastructure has to exist first.
  It is also the sub-step with no physics risk.
- **4f (⟨c⟩ nodal) before 4e (⟨a⟩ ellipse).** The nodal path already exists in the code
  and is the one the paper validates against; getting it right first gives the ROM a
  reference to be checked against (validation item 6). Doing the ROM first would leave
  nothing to measure it with.

Phase 3 is still placed before phase 2 so that the reference case is reproduced against a
code whose DAD still matches the paper's own convention.

Three things to decide before starting:

- **Which species carry the DAD.** The paper gives it to di-interstitials only. Our
  `dadAnisotropy` currently gives `p ≠ 1` to the vacancy as well (1.1788, i.e. faster
  along `c`). Deriving `p` from the tensor forces this to be stated explicitly in the
  migration energies rather than implied by a fit.
- **Whether the transition is one-way.** fullCD's is. A discrete loop that shrinks below
  `r_min` should either be deleted or returned to the continuum field; the second is
  more conservative of defect count and has no upstream precedent.
- **Whether ⟨a⟩ ever needs to leave the ROM.** The ellipse is exact for the DAD-driven
  shape response but cannot represent coalescence by contact, junction formation, or
  non-planar climb. `2r/d = 0.091` says none of those are active for ⟨a⟩ today — but that
  is a property of this dose and dose rate, not of the model, and it should be asserted
  at runtime rather than assumed.

## Open questions the paper does not settle for us

1. The paper's crystal has `c/a = √(8/3)`, "slightly deviating" from Zr's 1.59 and worth
   ~2.7% anisotropy along `c`. Our `Zr3d_ghoniem.txt` uses the true ratio. Whether that
   difference matters at `p_m = 0.7` has not been tested.
2. The paper's loops are seeded and evolved; ours are *nucleated by cluster dynamics* and
   then transferred. The transferred population inherits a mean size per family per node,
   not a size spectrum, so the discrete loops all start at the local mean.
3. SIPA stress weights `w_k(σ)` are still unwired on our side and are exact only at zero
   applied stress — which every case reported so far satisfies, but a creep case would not.
