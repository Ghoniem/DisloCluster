# The isolated-loop anneal, solved three ways

> **This is the working note. The manuscript is authoritative.** All of the
> material below is now written up as
> **§4.4, "The same anneal by the boundary-integral Green's function"**
> (`\label{sec:anneal-greens}`) in
> [`self_consistent_SRCD.tex`](self_consistent_SRCD.tex), with
> Eqs. (isoBVP), (pointsource), (dEdmlocal), (Zread), (kappavertex) and
> (Lcap-calib), Table (Ziso-refit), Table (anneal-shape) and
> Figs. (anneal-greens), (anneal-shape). This note is kept for the reproduction
> commands and for the derivations at more length than the manuscript wants.

Companion note to §4.3 of [`self_consistent_SRCD.tex`](self_consistent_SRCD.tex),
*"An isolated basal loop: the faulted and the perfect state"*. §4.3 integrates
one growth law, Eq. (isolatedR), in which the whole of the transport is
compressed into a closed-form capture efficiency and the loop is a circle by
assumption. This note solves the **same anneal** by the other two constructions
the manuscript carries — the boundary-integral Green's function method of
§3 (Eq. cDD / Gkernel / I01 / climbsystem) and the two-parameter ellipse ROM
(Eq. romA / romB) — and reports where they agree and where they do not.

Everything here is produced by
[`dislocluster_code/studies/loop_annealing_greens.py`](../../../dislocluster_code/studies/loop_annealing_greens.py),
which imports `loop_annealing` rather than reimplementing it, so §4.3's own
numbers are unchanged and reproduced as the baseline.

```
python -m dislocluster_code.studies.loop_annealing_greens verify
python -m dislocluster_code.studies.loop_annealing_greens ziso | lifetimes | lcap | closure | shape
python -m dislocluster_code.studies.loop_annealing_greens figure-rt    --out annealing_isolated_greens.png
python -m dislocluster_code.studies.loop_annealing_greens figure-shape --out annealing_loop_shape.png
```

---

## 1. The diffusion equation the Green's function method solves

One loop, alone, in an infinite crystal held at `c^v → c^{v,eq}_∞`, no
irradiation and no other sink. The mobile vacancy field is quasi-steady on the
climb time scale, so outside the loop core it obeys the **anisotropic steady
diffusion equation** with the line held at its own local equilibrium:

```
∇·( D_v ∇c^v(x) ) = 0                      x outside the core tube of radius r_0
c^v(x) → c^{v,eq}_∞                        |x| → ∞                            (I)
c^v(x)  = c^{v,eq}_sk( κ(x), σ )           x on the loop line

v_n(x) = ς_s(v) · Ω/(b·n)_k · Φ^v(x)                                          (II)
```

with `Φ^v` the vacancy flux collected per unit length of line. **(I) is a
Dirichlet problem on a closed line in an infinite anisotropic medium**, and (II)
is the kinematic statement Eq. (vn-common). Note what (I) is *not*: it is not a
flux boundary condition with an assumed capture efficiency. `Z^iso_k` does not
appear anywhere in it.

It is solved by superposition, `c^v = c^v_C + c^v_D`, with `c^v_C = c^{v,eq}_∞`
the (here uniform) continuum part and `c^v_D` the field the climbing line itself
emits. A line element climbing at `v_n` inserts material at the volumetric rate
`(b×ξ̂)·η̂` per unit length, so `c^v_D` is the line integral of the anisotropic
point-source Green's function

```
∇·( D_m ∇G_m ) = -δ(x)
G_m(x) = 1 / [ 4π √(det D_m) √( x·D_m⁻¹·x ) ]
```

against that source. Carrying the network's own linear shape functions along each
chord turns the line integral into the closed form Eq. (Gkernel) with the
elementary integrals Eq. (I01); imposing (I) on every segment in a Galerkin sense
closes the system, Eq. (climbsystem):

```
Σ_ν' K_{νν'} v_ν' = F_ν
K = ∫ N_ν (b×L)·η̂_ν 𝒢 ,     F = ∫ N_ν (b×L)·η̂_ν [ c_line − c_C ]
```

The kernel is implemented to the letter of
`DislocationSegment::concentrationMatrices` (MoDELib3), including the core term
`r_0²/(det D)^{1/3}` of Eq. (ABC), so it is the same operator the C++
`GalerkinClimbSolver` assembles — evaluated here on one loop instead of a
population.

**What is held fixed between the three methods.** All three are driven by the
*same* `c^{v,eq}_sk`, built from the *same* `dE_k/dm` of §4.1 — fault term plus
anisotropic capillarity — generalized from a circle to a local curvature by
`dEdm_local`, which reduces to `loop_annealing.dEdm` **exactly** at `κ = 1/R`
(checked, relative error 0). So every difference below is a difference of
**transport** and of nothing else.

### The kernel test

In an **isotropic** medium the boundary-integral solve of a circular loop must
return Eq. (Ziso) itself, because the self-field of a ring of uniform strength is
`Q ln(8R/r_0)/(2πR)` — the toroidal capacitance. Measured, at 128 sides:

| family | R | `Z` solved | `2π/ln(8R/|b|)` | error |
|---|---:|---:|---:|---:|
| `c_p` | 5 nm | 1.44469 | 1.44359 | 0.076% |
| `c_p` | 25 nm | 1.05416 | 1.05389 | 0.026% |
| `a_i` | 5 nm | 1.30443 | 1.30384 | 0.045% |
| `a_i` | 25 nm | 0.977648 | 0.977407 | 0.025% |

**This is what says the Green's function implementation is the same physics as
`Z^iso`, and not a different model.** Everything reported below is therefore the
effect of the real diffusion tensor and of shape freedom, not of a coding
difference.

---

## 2. `Z^iso_k` refitted so that the two coincide

With the kernel verified, `Z^iso_k` becomes an **output**: solve the boundary
integral on a circle of radius `R` with a unit drive, read the velocity, and
invert Eq. (vn-common),

```
Z^iso_k = v (b·n)_k / ( D̄ Δc ),        D̄ = (det D)^{1/3}
```

so the `Z` returned absorbs both the geometry and the tensor's departure from the
invariant `D̄` that Eq. (isolatedR) carries. At 873 K
(`D_a = 4.967e-14`, `D_c = 9.493e-14 m²/s`, `D_c/D_a = 1.911`, `p_v = 1.11402`):

| family | `α` fitted in `2π/ln(αR/r_0)` | factor `f` in `f·2π/ln(8R/r_0)` | err(`α` fit) | err(`f` fit) | err as published |
|---|---:|---:|---:|---:|---:|
| `c_f` | 4.911 | **1.09434** | 5.07% | **0.66%** | 9.07% |
| `c_p` | 5.159 | **1.09245** | 5.69% | **0.72%** | 8.98% |
| `a_v` | 9.520 | **0.97004** | 1.53% | **0.22%** | 3.18% |
| `a_i` | 9.520 | **0.97004** | 1.53% | **0.22%** | 3.18% |

fitted over `R = 2–60 nm`. Four things worth carrying away.

1. **A constant factor fits, a shifted log does not.** Rescaling the logarithm
   (`f`) leaves ≤ 0.7% over a factor of 30 in `R`; shifting its argument (`α`)
   leaves 5%. The discrepancy is nearly `R`-independent, which a multiplicative
   factor represents and a shift does not.
2. **The sign is opposite for the two habit planes.** Basal loops capture ~9%
   **more** than Eq. (Ziso) says, prismatic loops ~3% **less**. For vacancies
   `D_c > D_a`, so a basal loop — whose plane is perpendicular to the fast axis —
   is better fed than the isotropic average, and a prismatic loop, whose plane
   contains the fast axis, is worse fed. `D̄ = (det D)^{1/3}` cannot know which.
3. **`a_v` and `a_i` give identical `Z`**, as they must: `Z` is pure transport,
   and the two families differ only in `ς_s(v)` and in their thermodynamics.
   That is a free consistency check on the whole construction.
4. `f` is temperature-dependent, because `D_c/D_a` is. It is not a constant of
   the material.

### What that does to Fig. 3(a) and Table (anneal-lifetime)

`R₀ = 25 nm → r_min = 1 nm`, MF = Eq. (isolatedR) with Eq. (Ziso), GF = the same
law with the solved `Z`:

| T [K] | `c_f` MF | `c_f` GF | GF/MF | `c_p` MF | `c_p` GF | GF/MF |
|---:|---:|---:|---:|---:|---:|---:|
| 773 | 22.9 d | 20.6 d | 0.902 | 0.284 yr | 93.6 d | 0.903 |
| 823 | 49.7 h | 45.1 h | 0.907 | 9.05 d | 8.22 d | 0.908 |
| 873 | 5.95 h | 5.43 h | 0.912 | 25.2 h | 23.0 h | 0.913 |
| 923 | 53.9 min | 49.4 min | 0.917 | 3.69 h | 3.39 h | 0.918 |
| 973 | 9.9 min | 9.11 min | 0.921 | 39.7 min | 36.6 min | 0.922 |
| 1023 | 2.15 min | 1.99 min | 0.924 | 8.44 min | 7.81 min | 0.925 |

**Every conclusion of §4.3 survives.** The lifetimes fall by a uniform 8–10%,
and everything §4.3 actually asserts is a ratio or a shape:

| quantity | mean field | Green's function | change |
|---|---:|---:|---:|
| `t_cp/t_cf` at 773 K | 4.5309 | 4.5377 | +0.15% |
| `t_cp/t_cf` at 1023 K | 3.9240 | 3.9284 | +0.11% |
| `Q_eff` `c_f` | 2.6269 eV | 2.6200 eV | −0.26% |
| `Q_eff` `c_p` | 2.6661 eV | 2.6593 eV | −0.25% |
| `Q_eff` `a_i` | 2.7905 eV | 2.7922 eV | +0.06% |
| `Q_eff` `a_v` | 2.5541 eV | 2.5558 eV | +0.07% |

so Table (anneal-lifetime)'s last column stands at the precision it is quoted to,
`Q_eff` still sits below `E^f_v + E^m_v = 2.800 eV` by the same margin, and the
`a_i`/`a_v` bracketing is untouched. **Eq. (Ziso) was never the load-bearing
approximation in §4.3; the fault term is.**

Figure: [`annealing_isolated_greens.png`](annealing_isolated_greens.png).

---

## 3. The ROM's capture length is off by ~300×

Eq. (romA) writes the arrival rate as `Z_m D/L_cap`. The mean field and the
Green's function both write it as `Z^iso D̄/Ω` — a concentration is an atom
fraction, so `c/Ω` is a number density and no length appears. Matching the two
at the reference radius,

```
L_cap = Ω Z_m / ( b² Z^iso(R) )   ≈   Ω/b² = 0.690 |b| = 0.223 nm
```

| family | `R₀` | `Z^iso` | `L_cap` | `L_cap` [b] | `L_s/L_cap` |
|---|---:|---:|---:|---:|---:|
| `c_f` | 25 nm | 0.9750 | 0.2287 nm | 0.708 | 258 |
| `c_p` | 25 nm | 1.0539 | 0.2116 nm | 0.655 | 279 |
| `a_v` | 3.6 nm | 1.3992 | 0.1594 nm | 0.493 | 370 |
| `a_i` | 3.6 nm | 1.3992 | 0.1594 nm | 0.493 | 370 |

`ellipse_rom` defaults `L_cap` to the screening length, ~59 nm on the 500 nm
reference case. **At that default the ROM's isolated-loop climb rate is 260–370×
too slow.** The manuscript is right that `L_cap` "sets the time scale alone" and
cancels from the aspect ratio, so this calibration moves the ROM's lifetime and
leaves every shape statement untouched — but it means no ROM lifetime should be
quoted at the screening-length default.

---

## 4. Which capture-diffusivity closure — settled

§3 leaves Eq. (romattractor) open: *"this is not settled … only a nodal solve of
the same case can collapse it."* The isolated loop **is** a nodal solve of
exactly that case. Put a **circular** prismatic loop in the boundary-integral
solver with a **uniform** drive, so the only thing that can make the two axis
endpoints move at different speeds is the diffusion tensor, and read the ratio
off as an effective exponent `q = ln(v_A/v_B)/ln(p_v)`:

| T [K] | `p_v` | `q` at R = 2 nm | 3.6 nm | 10 nm | 25 nm |
|---:|---:|---:|---:|---:|---:|
| 773 | 1.12969 | 2.095 | 2.398 | 2.625 | 2.720 |
| 873 | 1.11402 | 2.116 | 2.408 | 2.630 | 2.724 |
| 1023 | 1.09652 | 2.137 | 2.419 | 2.635 | 2.727 |

against `q = 3` (geometric mean) and `q = 6` (normal projection).

**`q` is independent of temperature — hence of `p_v` — to three decimals, and
rises with `R` toward 3.** The geometric mean is the large-loop limit; the normal
projection is not a candidate at all. The shortfall at small `R` is the core
regularization blunting the anisotropy, and it is a *size* effect rather than a
material one, so it is a correction to the closure and not a competitor to it.

This removes the factor `p_m³` of ambiguity the manuscript requires to be quoted
with any ROM ellipticity.

---

## 5. The loop shape, ROM against Green's function

Figure: [`annealing_loop_shape.png`](annealing_loop_shape.png). `A_ℓ ∥ [0001]`,
`B_ℓ ∥ n̂_k × [0001]`, `ϱ_ℓ = B_ℓ/A_ℓ`, so `ϱ_ℓ > 1` is elongation **in** the
basal plane.

The three families answer different halves of the question:

- **`c_p` basal** is the control. Its habit plane is the basal plane, in which
  the tensor is isotropic, so the loop cannot become an ellipse whatever the
  anisotropy. Both constructions say so: the boundary-integral solve returns a
  velocity uniform around the line to `4e-15`, and the ROM's two capture
  diffusivities are identically equal. `ϱ = 1.0000` throughout.
- **`a_v`, `a_i` prismatic** have a habit plane containing `[0001]`, so their two
  in-plane directions sample different diffusivities and the shape genuinely
  moves.

### The curvature assignment in Eq. (romselfstress) is swapped

Parametrize the habit-plane ellipse as `(A cos t) e₁ + (B sin t) e₂`. Its
curvature is `κ(t) = AB/(A² sin²t + B² cos²t)^{3/2}`, so

```
at the A vertex (t = 0)      κ = A/B²      ← BLUNT when A is the minor axis
at the B vertex (t = π/2)    κ = B/A²      ← SHARP when B is the major axis
```

Eq. (romselfstress) and `ellipse_rom.curvature` both carry `κ_A = B/A²`,
`κ_B = A/B²` — **the pair swapped**. Verified numerically against the polygon
circumradius on a 1:3 ellipse: the true `κ` at the `A` vertex is `0.111111`
(`= A/B²`) and at the `B` vertex `3.00000` (`= B/A²`).

The swap is not cosmetic: it **reverses the sign of the capillary shape
feedback**. Eq. (romselfstress)'s own prose — *"as the loop elongates, κ rises at
the major-axis ends, the back stress there raises `c_eq` and growth throttles"* —
describes the geometric assignment; the equation beneath it puts the large
curvature on the minor axis instead, turning a restoring term into a runaway one.

`ellipse_rom.curvature` is deliberately **left alone**. It has no other caller in
the repository, so nothing published depends on either answer yet, and the fix is
a formulation decision rather than a code cleanup. `loop_annealing_greens` runs
both, selected by `axis_curvatures(..., convention="paper" | "geometric")`.

### What the three constructions give

`ϱ_ℓ = B_ℓ/A_ℓ` at five points down each anneal, at 873 K. `pape`/`geom` is the
curvature assignment, `geo`/`nor` the capture-diffusivity closure:

| family | `R/R₀` | Green's fn | `pape/geo` | `pape/nor` | `geom/geo` | `geom/nor` |
|---|---:|---:|---:|---:|---:|---:|
| `a_v` | 1.00 | 1.0000 | 1.0011 | 1.0019 | 1.0011 | 1.0019 |
| `a_v` | 0.83 | 1.0102 | 1.1259 | 1.2252 | 1.0393 | 1.0793 |
| `a_v` | 0.66 | 1.0081 | 1.6333 | 1.8795 | 1.0521 | 1.1068 |
| `a_v` | 0.49 | 1.0065 | 2.9558 | 3.4118 | 1.0495 | 1.1015 |
| `a_v` | 0.32 | **1.0002** | **4.7775** | 4.7817 | **1.0415** | 1.0848 |
| `a_i` | 0.83 | 1.0472 | 1.0703 | 1.1403 | 1.0654 | 1.1310 |
| `a_i` | 0.66 | 1.1072 | 1.1923 | 1.3949 | 1.1664 | 1.3452 |
| `a_i` | 0.49 | 1.1929 | 1.4341 | 1.9443 | 1.3551 | 1.7710 |
| `a_i` | 0.32 | **1.3264** | **2.0570** | 3.5426 | **1.8132** | 2.8847 |
| `c_p` | all | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

1. **The swapped curvature is the whole of the `a_v` disagreement.** With
   Eq. (romselfstress) as written the vacancy loop runs away to `ϱ = 4.78` — a
   sliver, visible in panel (a) of the figure. With the true curvature it settles
   at `ϱ = 1.04–1.05`, bounded and self-limiting, against the boundary integral's
   `1.00–1.01`. That is the "restoring term … growth throttles" behavior the
   manuscript's prose describes, and it only appears once the curvature is
   assigned to the right vertex.
2. **The two characters behave oppositely, and the reason is the exponential.**
   The drive is `c_∞(e^{μ/kT} − 1)` for a vacancy loop and `c_∞(1 − e^{−μ/kT})`
   for an interstitial one, so `d ln(drive)/dμ` is `1/(1−e^{−μ/kT})/kT ≈ 1.09/kT`
   against `1/(e^{μ/kT}−1)/kT ≈ 0.087/kT` at `μ/kT ≈ 2.5`. The vacancy loop's
   capillary shape control is **12.5× stiffer relative to its own drive**, which
   is why it stays round and `a_i` reaches `ϱ = 1.33`. It is the same exponent
   that makes `a_v` anneal 27× faster, read on the shape instead of the rate.
3. **The ROM over-reads the ellipticity even when it is corrected**, by 1.4× on
   `a_i` at `R/R₀ = 0.32`. Both remaining differences act the same way: the
   boundary integral couples every point of the line to every other point through
   the emitted field, so a point that runs ahead sits in the depletion its
   neighbors have already made, and the polygon carries the true curvature all
   the way round instead of only at two vertices.
4. **`c_p` is exactly circular in all five columns**, to `1.0000`, which is the
   control the whole comparison rests on.

---

## See also

The other half of §4.3's closing paragraph — that the loop is *alone* and its
ambient is *imposed* — is taken up in the new manuscript subsection
**"Annealing of loop populations"** (`\label{sec:anneal-population}`, inserted
between §4.3 and §4.4 of [`self_consistent_SRCD.tex`](self_consistent_SRCD.tex)),
built on
[`loop_annealing_population.py`](../../../dislocluster_code/studies/loop_annealing_population.py)
and its figures module. It solves for the remote vacancy concentration a mixed
microstructure holds, and finds the population coupling worth a factor of two in
each family's lifetime — in *opposite* directions.

---

## Scope and limits

- **Single species.** A pure anneal has no irradiation, so only vacancies appear.
  The kernel is per species and would carry all four unchanged.
- **Isolated.** No periodic images and no neighbors, so the screened cutoff
  Eq. (cutoff) never fires. That is the point of the comparison, not an omission:
  §4.3 assumes the same thing.
- **The core convention differs from MoDELib's.** `r_0 = |b_k|` is used
  throughout, to match Eq. (Ziso)'s own `ln(8R/|b|)`; the tutorials set
  `coreSize = 2.0 b`. Changing it shifts `Z` by roughly `ln 2/ln(8R/b)`.
- **Below ~3 nm the polygon segments approach the core radius** and the last
  snapshot of each shape march is discretization-limited. The `c_p` control shows
  the size of that: `ϱ = 0.9945` instead of 1 at the smallest radius drawn.
- **The 28-parameter set is not refitted here**, and the standing Ω, `loop_model`
  and `⟨c⟩`-Burgers staleness warnings in the root `CLAUDE.md` all still apply.
  Nothing in this note should be compared with experiment before that refit.
