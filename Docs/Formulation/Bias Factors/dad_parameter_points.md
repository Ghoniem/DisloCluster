# DAD parameter points that were tried but not committed

A record of anisotropy parameter points that existed only as uncommitted edits to
`MoDELib3/Library/Materials/Zr3d_ghoniem.txt`. Each entry carries enough to
reconstruct the point exactly, because two of the four keys involved are **not**
written by `staging/anisotropy.py` and would not come back from re-running it.

---

## The 2026-09-05 point: anisotropic vacancies, isotropic interstitials

Held in a git stash from 2026-09-05 (`On main: local Zr3d_ghoniem params ... before ff
to 35fe0a0`), never committed, and dropped on 2026-09-29 after being recorded here.

| key | this point | what `main` carried on 2026-09-29 |
|---|---|---|
| `dadAnisotropy` ($p_m$) | `1.6  1  1  1` | `0.9166  0.7  0.7  0.7` |
| `dadZ0` | `1.3  1.3  1.3  1.3` | `1.3  0.8934793332  ×3` |
| `loopSinkScale` (entries 1 and 8) | `0.77256` | `2` |
| `mobileSpeciesEnergyMigration_eV`, v | `1.246415038 0 0 1.246415038 0 1.107169922` | `1.201955601 0 0 1.201955601 0 1.196088797` |
| `mobileSpeciesEnergyMigration_eV`, i/2i/3i | `0.759101` isotropic | `0.7238776946 … 0.8295476108` |

The remaining six `loopSinkScale` entries (the prismatic families) stay at `0.792317`
and the ninth stays at `1` in both.

### Why it is interesting

**It satisfies the co-growth criterion.** Simultaneous ⟨a⟩ and ⟨c⟩ growth requires
$p_I < p_v$, and here $1 < 1.6$ — so this is a deliberate probe of the co-growth window
from the opposite side of the measured set, which puts vacancies isotropic
($p_v = 1.000000$) and interstitials at the fitted $p_i = 0.913720$. Both satisfy the
criterion; they sit at opposite ends of it.

Note that $p_v = 1.6$ is far outside the measured tolerance $p_v \in [0.94, 1.07]$ for
the co-growth set, so this is an exploratory point and not a candidate calibration.

### Two internal checks that the point is self-consistent

The migration-energy split preserves $E_{\rm eff}$, as `anisotropy.py` intends:

- **vacancies.** At $T = 573$ K, $k_BT = 0.049377$ eV and $\ln 1.6 = 0.470004$, so
  $E^{\langle 11\rangle} = E_{\rm eff} + 2k_BT\ln p_v = E_{\rm eff} + 0.046415$ and
  $E^{\langle 33\rangle} = E_{\rm eff} - 4k_BT\ln p_v = E_{\rm eff} - 0.092830$.
  The recorded pair `1.246415038 / 1.107169922` gives $E_{\rm eff} = 1.200000$ from
  both — the workbook's own vacancy migration energy.
- **interstitials.** $p_i = 1$ makes the tensor isotropic, and all three components read
  `0.759101`, which is the material file's interstitial $E_{\rm eff}$.

So the tensor and $p_m$ were generated together rather than edited independently — the
failure mode `anisotropy.py` exists to prevent.

### Why it could not simply be restored

The stash's baseline is stale: it was taken when `dadAnisotropy` read `1.02 0.7 0.7 0.7`,
and by 2026-09-29 `main` carried `0.9166 0.7 0.7 0.7`. Popping it would have reverted
that intervening change to $p_v$ rather than applying a clean delta.

### Reproducing it exactly

`staging/anisotropy.py` rewrites **only** `mobileSpeciesEnergyMigration_eV` and
`dadAnisotropy`. `dadZ0` and `loopSinkScale` are read from the material file and never
written by it, so they must be set by hand:

```bash
python -m dislocluster_code.staging.anisotropy --p-m 1.6 1 1 1 --apply
```

then, in `MoDELib3/Library/Materials/Zr3d_ghoniem.txt`:

```
dadZ0=1.3 1.3 1.3 1.3
loopSinkScale=0.77256 0.792317 0.792317 0.792317 0.792317 0.792317 0.792317 0.77256 1
```

Setting `dadZ0` flat at 1.3 **discards the fitted interstitial value**
`0.8934793332`, which is what makes the four Woo efficiencies reproduce the workbook
constants. Anything measured at this point is therefore uncalibrated in two independent
ways — that, and the stale 28-parameter fit.

Editing the material file re-stages every case, because `cfg.domain_key` hashes the
material content.
