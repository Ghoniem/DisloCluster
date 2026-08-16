# Fast-solve architecture comparison

- reference : `20260815_175446_59a62ae_500nmHex_REF`
- optimized : `20260815_194727_ca4635d_500nmHex_OPT`
- CD nodes  : 90,617

## What changed

Both runs solve the same case with the same Python driver. They
differ only in the DDomp binary that takes the fast step. Every
change removes repeated work; none approximates anything, which is
why the physics section below is the one that decides whether the
timings mean anything.

| # | change | effect |
|---|---|---|
| 1 | `AcIR`, a sparse matrix assembled every Newton iteration and never read, removed | 1.64x |
| 2 | Dirichlet selection matrix `T` and its index map cached; `A1 = T'AT` replaced by a filtered copy, `T` being a selection matrix | ~1.00x alone |
| 3 | element assembly `BilinearWeakForm::globalTriplets` parallelized over explicit contiguous chunks | 1.33x |
| 4 | the three loop-invariant weak forms and `rSolver` hoisted out of the Newton iteration | 1.11x |

Cumulative on one isolated DDomp call at 200 nm: **96.6 s -> 37.8 s, 2.556x**. Change 2 measured as nothing on its own because `rSolver` was rebuilt each iteration and discarded the cache before it could pay; change 4 is what made it count.

Assembly order is preserved exactly at every step. `setFromTriplets` sums duplicate entries, so a different triplet order would re-associate those sums and the result would agree only to rounding -- which would have made bit-identity unavailable as a check.

## Physics

- dose grid identical: **True**
- state array `(9, 90617, 19)` bit-identical: **True**

## Wall clock

| quantity | reference | optimized | speedup |
|---|---:|---:|---:|
| march total | 1 h 18 m 05 s | 42 m 41 s | 1.829x |
| fast solves (8 x DDomp) | 1 h 04 m 53 s | 29 m 49 s | 2.176x |
| one fast solve | 8 m 06 s | 3 m 43 s | 2.176x |
| slow substeps | 12 m 18 s | 11 m 58 s | 1.028x |

The slow step is untouched by this work and acts as the control: its ratio is **1.028x**. A value far from 1.0 means the two runs were not measured under comparable machine load, and the fast-side number should not be quoted.

## Per interval

| dose from | dose to | ref fast | opt fast | speedup | ref slow | opt slow |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0001 | 0.0 s | 0.0 s | -- | 1 m 35 s | 1 m 30 s |
| 0.0001 | 0.001 | 8 m 17 s | 3 m 47 s | 2.182x | 1 m 35 s | 1 m 32 s |
| 0.001 | 0.01 | 8 m 19 s | 3 m 52 s | 2.152x | 1 m 34 s | 1 m 31 s |
| 0.01 | 0.1 | 8 m 06 s | 3 m 49 s | 2.117x | 1 m 34 s | 1 m 29 s |
| 0.1 | 1 | 8 m 24 s | 3 m 48 s | 2.209x | 1 m 33 s | 1 m 27 s |
| 1 | 2 | 8 m 08 s | 3 m 44 s | 2.174x | 1 m 32 s | 1 m 28 s |
| 2 | 5 | 8 m 07 s | 3 m 52 s | 2.099x | 1 m 25 s | 1 m 30 s |
| 5 | 10 | 7 m 58 s | 3 m 45 s | 2.119x | 1 m 26 s | 1 m 28 s |

