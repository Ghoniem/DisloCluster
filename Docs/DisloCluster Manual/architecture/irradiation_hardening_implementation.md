# Irradiation hardening — what the plan became, and what running it found

Companion to [`irradiation_hardening_dd_plan.md`](irradiation_hardening_dd_plan.md).
The plan is the design; this is the implementation, the corrections the
implementation forced, and the measurements that only appear once something is
actually run. Section numbers below refer to the plan.

**The code is [`dislocluster_code/studies/hardening.py`](../../../dislocluster_code/studies/hardening.py).**

```bash
python -m dislocluster_code.studies.hardening state    [<run>]          # section 2
python -m dislocluster_code.studies.hardening box      --L 200 500      # section 6
python -m dislocluster_code.studies.hardening routec   --alpha-c .4 --alpha-a .2 --taylor-M 3
python -m dislocluster_code.studies.hardening stage    <dir> --route A --L 500 --dose 10
python -m dislocluster_code.studies.hardening run      <dir>
python -m dislocluster_code.studies.hardening reduce   <dir> --control <dir>
python -m dislocluster_code.studies.hardening campaign <root> --doses 0.01 0.1 10
```

Every function the plan's §7 table names exists, under the names it gives, and
reuses what it says to reuse. `interior_state` is the §2 reduction,
`commensurate_box` the §6 lattice construction, `cube_population` the §7
`populate_cube`, `stage`/`run_case`/`reduce_run` the §7 staging, running and
reduction, `dispersed_barrier`/`fit_alpha` route C, and `campaign` the §10 step 4
sequence.

---

## 1. Verification against the plan's own numbers

Three of the plan's tables are arithmetic on measured quantities, so they are
regression tests. All three reproduce.

**§6, the commensurate box** — exactly, every row:

| target | n | m | p | edges [nm] |
|---:|---:|---:|---:|---|
| 200 nm | 619 | 357 | 388 | 199.94 × 199.72 × 199.82 |
| 300 nm | 929 | 536 | 583 | 300.07 × 299.87 × 300.24 |
| 500 nm | 1548 | 894 | 971 | 500.00 × 500.15 × 500.06 |
| 800 nm | 2477 | 1430 | 1553 | 800.07 × 800.02 × 799.79 |

**§5, route C** — every row, to the digit the plan prints (its 17.2 at 0.1 dpa
is 17.1 here, a rounding difference):

```
alpha_c = 0.4, alpha_a = 0.2, M = 3
     dpa    dtau_c    dtau_a      dtau   dsigma_y
  0.0001       2.5       1.0       2.7        8.1
    0.01      12.3       9.3      15.4       46.2
       1      39.6      14.2      42.1      126.2
      10      39.2      12.0      41.0      123.0
```

The per-variant root-sum-square this module computes is identical to treating
⟨a⟩ as one family with the summed density, which is what the plan did.

**§6, loop and segment counts at 500 nm** — within the rounding of the variant
split (the plan rounds the ⟨a⟩ total once, this rounds each of the three
variants):

| dpa | ⟨c⟩ | ⟨a⟩ | total segments | plan |
|---:|---:|---:|---:|---|
| 1e-4 | 6 | 6 | 168 | 6, 6, ~170 |
| 1e-2 | 96 | 471 | 8 688 | 96, 471, ~8 700 |
| 1 | 95 | 834 | 14 484 | 95, 835, ~14 500 |
| 10 | 95 | 525 | 9 540 | 95, 523, ~9 500 |

---

## 2. Seven corrections the implementation forced

These are places where following the plan exactly produces a case that does not
run, or runs and measures nothing. Each was found by running it.

### 2.1 `X0` is 0, not 0.5 — `unitCube24.msh` spans [−0.5, 0.5]³

§7 step 3 says `X0 = 0.5 0.5 0.5  # unitCube24 spans [0,1]^3`. It does not: its
14 nodes run from −0.5 to +0.5 on every axis, and the `periodic_density`
tutorial that the plan cites as the working template accordingly sets
`X0 = [0,0,0]`. With the plan's shift the whole crystal sits in the positive
octant and every loop centre drawn about the origin falls outside the mesh.

### 2.2 `periodicFaceIDs = -1`, not `0 1 2 3 4 5`

`SimplicialMesh::identifyParallelFaces` labels a face pair periodic when either
ID is listed **or when −1 is** (`SimplicialMesh.cpp:350`). The IDs are mesh
`sID`s, which for this mesh happen to run 0–5 but are not guaranteed to; −1 is
the form the tutorials use and cannot be wrong.

### 2.3 `F` needs 17 significant digits, not 10

The box edges have to be lattice vectors, and MoDELib checks it by mapping each
periodic shift into lattice coordinates and demanding an integer. Ten
significant digits truncate `971 × c/a = 1548.1888112…` to `1548.188811`, which
is 1.25e-7 lattice units short, and the run dies at startup with

```
d2contra, nd=… 9.709999996932040e+02
d2contra, rd=… 9.710000000000000e+02
terminating due to uncaught exception: Input vector is not a lattice vector
```

The 200 nm box survives only because `388 × c/a` happens to fit in ten digits.
`staging/inputs.py:_f_block` had the same limit and is fixed with it — no
coupled case ever noticed, because none of them is periodic.

### 2.4 The source dipole must sit on the loaded slip system

§7 step 3 suggests `periodicDipolesDensity`. That generator chooses the slip
system itself, and on the first run it chose a **basal** one — normal `[0001]`,
resolved shear under a prismatic load exactly zero. Measured: 3 000 steps to
41 MPa with `gamma_p = 8e-22`, a source that cannot move however hard it is
pulled, and nothing in the output says why. The module now writes
`periodicDipoleIndividual` on slip system 6 (`s = [1,0,0]`, `n = [0,-1,0]`,
taken from MoDELib's own printed list), exiting through the z face pair so the
line is an edge dipole gliding in x.

### 2.5 `glideSteps` must be non-zero or there is no mobile dislocation at all

`PeriodicDipoleGenerator::generateSingle` inserts the sessile prismatic loop
unconditionally, but the two **glissile** arms only
`if(std::fabs(glideStep)>FLT_EPSILON)`. With `glideSteps=0` the case contains
one sessile loop: `vMax = 0`, glissile density 0, `gamma_p` exactly 0 for 3 000
steps. It reads exactly like a perfectly pinned source, which is the wrong
conclusion to draw about an obstacle field.

### 2.6 The sign of `n` decides whether `gamma_p` is positive

Resolving the load on `(s, +y)` while the slip system is `(s, −y)` leaves `tau`
positive and `gamma_p` negative, so the offset criterion never fires however far
the ramp runs. Both now come from the same system.

### 2.7 The plan's ramp rate in SI is out by 1000

§3 gives the ramp as "1 MPa per 1e4 b/cs" and then as "~7e11 Pa/s". With
`b/cs = 1.4357e-13 s` the first is **6.97e14 Pa/s**. The DD-unit form is the one
implemented, and it is the one the quasi-static argument is about: a pinned line
relaxes in ~1e2 b/cs, over which the stress moves 0.01 MPa.

---

## 3. Two failure modes that are not the plan's fault

### 3.1 MoDELib refuses some loop realizations

A few loops per thousand end with a network node a few `b` outside the primary
cell — `insertJunctionLoop` maps the polygon's periodic patches back into the
cell and for a straddling loop the result can land just outside — and
`DislocationNode` refuses it at startup:

```
PlanarDislocationNode 4209 @ -7.766e+02 2.029e+02 -3.150e+02
DISLOCATION NODE OUTSIDE MESH.
```

Measured on the 500 nm cube at 1e-2 dpa: **seeds 0 and 1 fail, seeds 2 and 3
run**, with the same population statistics. It is a property of the draw, so
`run_case` raises `PlacementRejected` naming it, and `campaign` redraws with the
next seed and records which seed it used. It is not proximity to a face plane:
the closest vertex-to-face distance is 0.22 b in a failing draw and 0.016 b in a
passing one.

### 3.2 Zero-byte `F/F_0.txt` makes the NEXT run fail differently

A run that dies before writing leaves `F/F_0.txt` and `F/F_labels.txt` at zero
length, and the next DDomp **segfaults** reading them. So a first failure of any
kind turns every later attempt into an unrelated-looking crash.
`staging.inputs.clear_empty_F` now removes the stubs and both `case.bootstrap`
and `hardening.run_case` call it. `microstructureGenerator` has the mirror-image
problem — it segfaults (exit −11) if `evl/evl_0.txt` already exists — so
`run_case` clears `evl/` and `F/` before generating.

---

## 4. Measurements

### 4.1 Subcycling, measured rather than estimated

The plan's §6 estimates ~100× from `useSubCycling=1` and says to measure it.
Measured, 200 nm cube at 10 dpa, route D, 46 loops, 200 steps:

| | s/step |
|---|---:|
| `useSubCycling=1` | 0.0257 |
| `useSubCycling=0` | 0.1377 |
| **ratio** | **5.4×** |

Far short of 100×, and the reason is that the saving is the ratio of total
segments to *active* ones: this cell has ~600 segments where the 500 nm case at
0.1 dpa has ~25 000. The measurement should be repeated at 500 nm before a
campaign is sized on it; the direction is confirmed, the magnitude is not.

### 4.2 The ⟨a⟩ loops survive the remesh

§6 predicts they must, because `isGeometricallyRemovable` takes the
`loopAreaRate < 0` branch for a loop smaller than `Lmin²`, and a frozen loop has
zero area rate. Confirmed: **46 loops at step 0 and 46 at step 190**, through 20
remesh passes at `remeshFrequency=10`, with `r_⟨a⟩ = 3.77 nm = 11.7 b` against
`Lmin = 25`.

### 4.3 The loops are genuinely frozen, and the ramp is exactly what was asked

`sessile density` is constant to every printed digit across the run
(4.08895e14 m⁻²), and `tau` at step 200 is 32.369 MPa against 3.2369e5 b/cs of
elapsed time — 1 MPa per 1e4 b/cs, the ramp as configured.

### 4.4 Per-step cost

| cell | dose | loops | segments | s/step |
|---|---:|---:|---:|---:|
| 200 nm, route D | 10 dpa | 46 | ~600 | 0.026 |
| 200 nm, route A, control | — | 0 (1 dipole) | ~20 | 0.004 |
| 500 nm, route D | 10 dpa | ~620 | ~9 500 | 3.33 |
| 500 nm, route A | 1e-2 dpa | 567 | ~8 700 | 2.5 |

On 8 cores (4 performance), against the plan's 24-thread estimate of 0.2 s/step
at 500 nm. **The 500 nm campaign is a 24-core machine's job**; 200 nm is what
this one can do, at the cost of the realization spread §8 warns about.

---

## 5. The notebook, the figures and the movies

[`Simulations/hardening_simulation.ipynb`](../../../Simulations/hardening_simulation.ipynb)
drives all of this the way `run_simulation.ipynb` drives a march: six control
dicts (`MATERIAL`, `SOURCE`, `CELL`, `LOAD`, `OUTPUT`), then resolve, stage, run,
reduce, plot. Everything a run produces goes into ONE timestamped directory,
`Simulations/output/<stamp>_<hash>_<tag>/`:

```
cases/           the staged DD cases, control first
figures/         flow_curves.png, delta_tau_vs_dose.png, loops_<dose>.png
snapshots/       per dose and control: solved configurations, two panels each
movies/          motion_<dose>.gif (and .mp4 where ffmpeg is installed)
provenance.json  every control dict, every result, the fitted alpha
provenance.md    the same, readable
```

**`post/dd_frames.py` is the new piece.** `discrete_loops.py` draws the
population a continuum field IMPLIES; this draws what the solver has — the
`evl_<N>.txt` configuration written every `outputFrequency` steps, with the
dislocation where it got to and the shape it bowed into. Three things about it
are worth recording, because none is documented on the MoDELib side:

| | |
|---|---|
| `loopType` is **column 11** | the loop record is `sID B(3) N(3) P(3) grainID loopType …`, so column 10 is the GRAIN. Reading it as the type marks every loop sessile in a single-grain cell — including the dislocation whose motion is the whole measurement |
| `hasNetworkLink` (link column 3) separates dislocation from bookkeeping | a loop spanning the periodic cell is closed by links that carry no network link. 72 of one frame's 194 glissile links are of that kind, and drawing them puts straight segments across the cell that read as dislocation |
| periodic wrap has to be cut | following the link chain through network positions jumps across the cell wherever the loop wraps; those segments are split out, so a wrapped loop appears as the arcs it really is |
| a network link's dislocation is the **sum** of the loop links on it, and can be zero | `hasNetworkLink` is not the whole test: a loop boundary that runs out and back along the same node pair writes both records with `hasNetworkLink = 1` and opposite sense, so the link carries `b = 0`. `periodicDipoleIndividual` makes exactly that |
| the chain walk must include the node it ENDS on | the walk appends the nodes it enters, so its last link was never drawn — and with `visited` fragmenting one chain into several walks, that lost link falls in the MIDDLE of a line |

Each frame is **two panels**, because neither alone answers the question: the
cell in 3-D says where the line is, and the projection along the glide-plane
normal says what it is doing — pinning, bowing between obstacles, and the
released arc after breakaway are a few b of curvature seen edge-on in the 3-D
view and obvious in the projection.

Frames are solved configurations, never interpolated: the movie's cadence is the
run's own `outputFrequency`. MP4 needs ffmpeg (`pip install imageio-ffmpeg`);
without it the GIF is written and nothing fails.

### 5.1 The first notebook run, and two things it exposed

`20260819_095134_c64c7e4_hardening_200nm`, 200 nm cell, the three doses of the
plan's table, ~40 min end to end on 8 cores. Control (no loops)
`tau(1e-3) = 4.612 MPa`:

| dpa | seed | obstacles | tau_off [MPa] | dtau [MPa] |
|---:|---:|---:|---:|---:|
| 0.01 | 0 | 36 | 4.280 | **-0.331** |
| 0.1 | 1 | 99 | 13.402 | **8.790** |
| 10 | 0 | 39 | 13.057 | **8.446** |

fitted `alpha_c = 0.089`, `alpha_a = 0.031`, rms 2.40 MPa. Seed 0 at 0.1 dpa was
refused and seed 1 accepted, unattended, which is the `PlacementRejected` redraw
doing its job inside a notebook rather than ending the run.

**`n_loops` is not the obstacle count, and it read three too high at every
dose.** `evl_0` holds every loop the generator wrote, and the periodic dipole
source contributes three of its own — one sessile prismatic loop plus the two
glissile arms — so the first table published 39 / 102 / 42 where the obstacle
fields are 36 / 99 / 39. `stage` had recorded the true counts in `loop_counts`
all along; `run_case` now stores `n_obstacles` and `n_source_loops` beside
`n_loops`, and `obstacle_count(row)` recovers it for campaigns run before that
existed, from the case's own `hardening.json`, rather than subtracting a
constant that would be wrong the moment the source changes. Nothing measured
moves — `fit_alpha` takes `N_k` from the continuum state, never from this count.

**Delta-tau is NOT monotone in dose, and the reason is that the 200 nm cell
cannot hold the 10 dpa ⟨c⟩ population.** Measured from `evl_0` at 10 dpa, the
six basal loops come out at `r = 55.6 nm` — **111 nm across in a 200 nm cell** —
against ⟨a⟩ loops of 3.77 nm. So the obstacle field collapses from 99 loops at
0.1 dpa to 39 at 10 dpa, and Δτ falls with it (8.79 → 8.45 MPa) although the
stored defect content rises. This is the same geometric refusal
`transition.fits_in_crystal` reports for the continuum→discrete handoff, arriving
here through `cube_population`: **the 10 dpa point is a cell-size artifact, not a
saturation.** A ⟨c⟩ measurement at 10 dpa needs the 500 nm cell, which is a
24-core job (§4.4).

### 5.2 Two ways a correct configuration drew as broken line

Both were found by reading a frame and asking why it looked wrong, and neither
touches anything measured — Δτ, γ_p and the obstacle counts never come from
these polylines. But a figure is an argument, and both of these made it a false
one.

**Segments that are not dislocation.** `_null_pairs` sums the loop Burgers
vectors per network-node pair and `drop_null` (default **on**) cuts the polyline
wherever the sum is zero. On the 200 nm case at 0.1 dpa, step 1600, that is
**8 pairs of 1576**: the dipole's `x = 0` lines, its tie bar along `y`, the two
tie-backs, and one contact where the gliding line has zipped onto an ⟨a⟩ loop and
locally annihilated it — **121 nm of the 457 nm** the glissile loops appeared to
carry. The whole sessile prismatic source loop is null and now draws nothing, so
the panel's `obstacles` count is the irradiation loops alone and agrees with
`n_obstacles`. Because the two glissile arms' anchors differ only in `y`, the axis
the glide panel projects along, the two tie segments superimpose at the origin
and read as **one line joining the two dislocations** — which is exactly what a
reader reports.

**Dislocation that was there and was not drawn.** `loop_polylines` follows the
real-link chain with `while node in nxt and node not in visited` and appended only
the nodes it *entered*, never the sink it exited on, so every open run was one
segment short at its far end. That alone would clip a tip. What makes it a gap in
the MIDDLE of a line is `visited`: seeds are taken in file order, so one lands
mid-chain and the stretch behind it becomes its own walk, which then terminates on
the already-visited node without drawing the segment into it — and a run left with
one node is discarded outright. At 0.1 dpa step 7950 the gliding dislocation came
out as **four disjoint stubs with three gaps**, every second segment of it missing.

The check is geometric and needs no reference figure: each arm of the periodic
dipole must traverse the cell in `z`, so the pair has a hard floor of **400 nm** of
drawn line. Over every frame of every case:

| case | frames | drawn, before | drawn, after | frames under 400 nm |
|---|---:|---|---|---|
| control | 160 | 363.6–393.6 nm | 400.0–409.1 nm | 160 → 0 |
| 0.01 dpa | 160 | 175.2–398.4 nm | 399.3–497.5 nm | 160 → 2 |
| 0.1 dpa | 160 | 159.9–402.4 nm | 399.5–508.0 nm | 159 → 1 |
| 10 dpa | 160 | 178.4–421.6 nm | 397.2–511.4 nm | 158 → 6 |

so all 640 frames were affected and the worst drew **40%** of the line. Exactly
400.0 nm at step 0, where both arms are straight, is what says the terminal node
is now in and nothing else is: the piece count rises while the length lands on the
geometric minimum, so the recovered segments are line and not new spurious chords.
The frames still marginally under the floor are 397.2–399.6 nm — 0.1–0.7%, the
`drop_null` cut at a junction where the line has zipped onto a loop, and with
`drop_null` off those same frames draw 447–752 nm.

## 6. What is NOT done

- **The source run is not on this machine.** Run directories are git-ignored, so
  `20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent` lives only where it
  was produced. `interior_state` is written against `march_state.npz` and is
  verified on the local cubic run; the plan's §2 table is carried in
  `PLAN_STATE_TABLE` as a **cache**, clearly labelled, so route C and a campaign
  can run now. Point `--run` at the hexagonal run when it is available and
  nothing else changes.
- **Step 1 of §7, `discrete_loops --doses all`, has not been re-run** for the
  same reason.
- **Route B is not staged.** `stage(strain_rate_s=...)` writes the deck; §6's
  arithmetic is the argument for not paying for it.
- **`post/fields.py` still carries the pre-correction constants**
  (`B_SI = 3.233e-10`, `OMEGA_SI = 1.2e-29`), and `discrete_loops` imports them.
  This module reads `b_SI` and `atomicVolume_SI` from the material file instead.
  It cancels in a loop radius — `r = sqrt(c/(n*pi*b))` once `m = c/(n*Omega)` is
  substituted — but not in a number density (0.3%) or a defect count (1.94×).
