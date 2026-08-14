# From continuum loop fields to discrete dislocation loops

How the cluster-dynamics loop fields — a number density and a stored defect
content per family per node — become an explicit set of dislocation loops with
positions, radii, habit-plane normals and Burgers vectors, in the form MoDELib3's
own microstructure generator reads.

Implementation: [`dislocluster_code/post/discrete_loops.py`](../../dislocluster_code/post/discrete_loops.py).

---

## What this replaces

The `loops_*` overlays produced by `modelib_fields._overlay_family` are a size
map drawn as an ensemble, not a microstructure. Platelet positions are CD node
coordinates, the platelet *count* is set by geometric packing rather than by the
density, and radii carry a fixed exaggeration factor (×7 for ⟨a⟩, ×1.5 for ⟨c⟩).
At 1e-4 dpa that figure draws 2500 platelets into a domain that physically holds
2.7 ⟨a⟩₁ loops. Nothing in it can be handed to dislocation dynamics.

---

## What MoDELib-fullCD does

`ClusterDynamics<dim>::initializeDiscreteClimbLoops`, in
`src/ClusterDynamics/ClusterDynamics.cpp` of `mlm335/MoDELib-fullCD`. It fires
once, when the simulation time reaches `clusterDiscretizationTime`, and for each
immobile species:

1. Rank finite elements by defect content `c_I · V_e / Ω`, descending.
2. Walk that list. For each element not yet consumed, seed a group and greedily
   absorb vertex-neighbors until the group holds
   `Σ n_I·V ≥ clusterDiscretizationFactor` **loops**.
3. Emit **one** 12-sided loop carrying the group's **entire** defect content:

   ```
   radius = sqrt(NiCiK.second * cdp.omega / (M_PI * cdp.b *
                 cdp.immobileSpeciesBurgersMagnitude(k)))
   ```

4. Center it on a random tetrahedron barycenter in the group, snapped to the
   closest crystallographic plane of normal `n = reciprocalLatticeDirection(b)`.
   The sign of `b` is chosen from the polygon's area normal and the species
   polarity (`immobileSpeciesVector`: negative vacancy, positive interstitial).

Two things are worth carrying over.

**The radius rule is defect-count conservation.** A loop of radius `r` on a
habit plane, one Burgers vector thick, stores `π r² |b| / Ω` point defects, so

```
r = sqrt( N_def · Ω / (π |b|) )
```

**`clusterDiscretizationFactor` is already an area-conserving coalescence**,
applied up front. One discrete loop stands in for a whole bundle of continuum
loops, and takes all their content — which is exactly `r_bundle = sqrt(Σ r_i²)`.
It is a fixed, user-chosen coarsening applied everywhere, whether or not the
loops at that location are actually crowded.

---

## What is done here instead

The population is **exact** — the real number of loops at the real size — and
coarsens only where the loops genuinely will not fit.

### 1. Count

The number of loops of family *k* in a region is not a parameter:

```
N_k = Σ_j  n_k(x_j) · V_j
```

over the CD nodes in that region, with `V_j` the node's volume share from the
Monte-Carlo Voronoi weights already used for volume averaging. Positions are an
inhomogeneous Poisson sample with intensity `n_k(x)`, so spatial structure
survives into the discrete population.

### 2. Size

Defects per loop, then the radius that stores them:

```
m(x) = c_k(x) / ( n_k(x) · Ω/b³ )
r(x) = sqrt( m(x) · Ω / (π |b|) )
```

### 3. Shape and crystallography

| family | habit plane | Burgers vector | \|b\| | sides | `planeID` | type | color |
|---|---|---|---:|---:|---:|---|---|
| ⟨c⟩ | basal (0001) | **½[0001]** | 0.8165 b | 6 (hexagon) | 0 | vacancy | blue |
| ⟨a⟩₁ | prismatic | **⅓[21̄1̄0]** | 1.0 b | 16 | 6 | interstitial | red |
| ⟨a⟩₂ | prismatic | **⅓[112̄0]** | 1.0 b | 16 | 8 | interstitial | green |
| ⟨a⟩₃ | prismatic | **⅓[1̄21̄0]** | 1.0 b | 16 | 10 | interstitial | yellow |

The three ⟨a⟩ entries are the members of ⅓⟨112̄0⟩. In the three-index lattice
basis the material file writes them as (1,0,0), (0,1,0) and (−1,1,0), which map
to the Miller–Bravais indices above and to Cartesian (1,0,0), (0.5,0.866,0) and
(−0.5,0.866,0) — all of unit length, hence crystallographically equivalent at
zero applied stress, as they must be.

These are pure climb loops, so the habit-plane normal is parallel to the Burgers
vector. Hexagons for ⟨c⟩ is not a drawing preference — basal vacancy loops in Zr
facet on ⟨10-10⟩-type edges. ⟨a⟩ loops are round enough that a 16-gon is
indistinguishable from a circle at these radii.

`planeID` indexes `singleCrystal->slipSystems()`. With
`enabledSlipSystems=fullBasal fullPrismatic` that list is six basal systems
(0–5), then the prismatic planes in pairs: (a1,c) → 6,7, (a3,c) → 8,9,
(−a2,c) → 10,11.

**The polygon carries the disc's area, not the disc's radius.** A regular
*n*-gon of circumradius *R* has area `(n/2)R² sin(2π/n)`, which for a hexagon is
only 82.7 % of `πR²`. Since the conserved quantity is the number of point
defects stored, the export scales the circumradius by
`sqrt(2π / (n sin(2π/n)))` — 1.0996 for the hexagon, 1.013 for the 16-gon.
fullCD does not do this and its 12-gons under-fill by 2.3 %.

---

## Two findings that came out of building it

### The ⟨c⟩ Burgers vector is not the same on the two sides

The cluster-dynamics material file gives the ⟨c⟩ family a **full** ⟨0001⟩
Burgers vector, `|b| = c/a = 1.632993 b`. MoDELib3's `aLoopGenerator` basal
branch builds a **half** ⟨0001⟩ loop, `|b| = 0.8165 b` — which is the physically
standard basal vacancy loop in Zr.

The invariant handed from CD to DD is the number of vacancies stored, not the
radius, so the discrete radius is computed from the **DD** Burgers magnitude.
The consequence is that ⟨c⟩ radii come out a factor √2 larger than the
CD-internal value: 47.7 nm rather than 33.7 nm at 10 dpa. This is a genuine
model inconsistency, not a conversion artifact, and it should be resolved on the
cluster-dynamics side.

### Area-conserving coalescence always makes overlap worse

With the total loop area `N π r²` held fixed and the mean spacing
`d = (V/N)^{1/3}`:

```
r ~ N^(-1/2),   d ~ N^(-1/3),   so   2r/d ~ N^(-1/6)
```

Every merge lowers `N`, so every merge **raises** `2r/d`. Coalescence cannot
repair overlap; iterating it to a fixed point percolates the instant the overlap
graph connects. Run unbounded on the ⟨c⟩ population at 1 dpa and 186 loops fuse
into a single loop of radius 650 nm — larger than the 500 nm box.

The pass therefore carries two bounds:

- **Coplanarity.** Two loops may merge only if their centers lie within one
  interplanar spacing of a common habit plane. This is the physical statement
  that loops on different parallel planes cannot become one loop without
  climbing. fullCD does not test this — it groups by finite-element adjacency.
- **Spacing guard.** A pass whose result would push `2⟨r⟩/d` above 1 is rejected
  and the population is reported `SATURATED`. Saturation is not a numerical
  failure: it says the continuum state has no non-overlapping discrete
  representation, which is the regime the model's own coalescence channel exists
  to describe.

---

## The numbers, 500 nm cube, interior

Whole-domain counts are dominated by the near-boundary artifact — at 10 dpa the
domain holds 19 940 ⟨a⟩₁ loops and the interior holds 205, a factor of 97.
Cascade nucleation is uniform but the only loop-removal channel needs the
absorbed mobile flux, which vanishes where Dirichlet pins the mobile
concentrations. The interior number is the physical one, and is what
`region='interior'` exports.

| dose | ⟨c⟩ | r⟨c⟩ | 2r/d ⟨c⟩ | ⟨a⟩ each | r⟨a⟩ | 2r/d ⟨a⟩ |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 13 | 2.41 nm | 0.023 | 4 | 1.53 nm | 0.010 |
| 0.0105 | 187 | 3.45 nm | 0.079 | 283 | 3.18 nm | 0.084 |
| 0.977 | 186 | 47.4 nm | **1.081** | 205 | 4.70 nm | 0.111 |
| 10 | 186 | 47.7 nm | **1.089** | 205 | 4.69 nm | 0.111 |

Total at 10 dpa: **801 loops**, entirely tractable for DD. Cross-check: the
⟨a⟩ total is 4.9e21 m⁻³, squarely in the literature range for Zr at 10 dpa.

⟨c⟩ crosses `2r/d = 1` between 0.1 and 1 dpa and is flagged saturated
thereafter: the mean ⟨c⟩ loop diameter exceeds the mean ⟨c⟩ spacing. In three
dimensions those loops appear to interpenetrate, but crystallographically they
do not — 950 basal planes are available to 186 loops, so coplanar overlap is
rare, and coalescence never fires in the interior.

### Where coalescence does fire

`region='domain'` at 10 dpa keeps the boundary artifact, and the ⟨a⟩ density
there is two orders of magnitude higher. That population is 56 290 loops, and
the pass is exercised:

| family | N before | N after | merged | largest cluster | ⟨r⟩ | 2r/d before → after |
|---|---:|---:|---:|---:|---:|---:|
| ⟨c⟩ | 186 | 186 | 0 | 1 | 45.3 nm | 1.034 → 1.034 (saturated) |
| ⟨a⟩₁ | 19 938 | 18 656 | 1 282 | 4 | 1.98 nm | 0.208 → 0.210 |
| ⟨a⟩₂ | 19 938 | 18 713 | 1 225 | 5 | 1.97 nm | 0.208 → 0.209 |
| ⟨a⟩₃ | 19 938 | 18 735 | 1 203 | 5 | 1.97 nm | 0.208 → 0.209 |

Total area is conserved to machine precision in every case, and `2r/d` **rises**
across the merge in all three ⟨a⟩ families — the `N^(-1/6)` result above,
measured rather than argued.

---

## Output

```
python -m dislocluster_code.post.discrete_loops <run_dir> [--doses 1e-4 0.01 1 10]
                                  [--region interior|domain]
                                  [--no-coalesce] [--coplanar-tol B]
```

| file | contents |
|---|---|
| `aLoops_<dose>.txt` | MoDELib3 `aLoop` individual-style microstructure file — `planeIDs`, `loopRadii_SI`, `loopSides`, `loopCenters`, `isVacancyLoop`, all per loop |
| `loops_<dose>.csv` | per-loop table: family, position (b and nm), radius, circumradius, Burgers vector, normal, merge count |
| `loops_<dose>.png` | all families together, drawn as tubular dislocation lines |
| `loops_<family>_<dose>.png` | one figure per family — the combined view is dominated by whichever family is largest, and at 10 dpa the ⟨c⟩ hexagons are ten times the ⟨a⟩ loops and hide them |
| `manifest.json` | counts, radii, spacings, `2r/d` before and after coalescence, area conservation and saturation flags |

`region='interior'` places a homogeneous population at the interior mean density
over the whole box — a bulk DD cell carrying the interior state without the
boundary layer attached. `region='domain'` keeps the actual spatial field,
boundary artifact included.

`--doses all` renders every snapshot in `march_state.npz`. The dose-0 snapshot
is pristine — every species sits at the 1e-20 floor — so it holds no loops and
is skipped rather than written as an unparseable empty microstructure.

### Line thickness

A dislocation line has no thickness, so the tube width is a drawing choice. A
single width across families makes the ⟨c⟩ hexagons read as wire next to the
⟨a⟩ loops they are ten times larger than, so the width is set per loop,

```
tube = clip(0.05 * r, 1.2 nm, 3.6 nm)
```

which leaves the ⟨a⟩ loops at the 1.2 nm floor and draws the 47 nm ⟨c⟩ loops at
2.4 nm.
