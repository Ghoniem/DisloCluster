# ZrClusterDynamics — the 0-D half of DisloCluster

This directory holds the **0-D reduced cluster-dynamics** code and the
formulation documents. It is one of the two codebases in the
[DisloCluster](../CLAUDE.md) repository; the 3-D code lives in `../MoDELib3/`.

> Historical note: this tree previously also carried `ClusterDynamics/`,
> `Creep/` and `ZrProps/`. Those were stripped when DisloCluster was assembled.
> Only `ZrMicro/` and `Docs/` remain.

---

## Contents

| Path | Purpose |
|---|---|
| `ZrMicro/` | Microstructure evolution in neutron-irradiated Zr — 19 ODEs |
| `ZrMicro/py_utils/paths.py` | **Single source of truth for every path in the repository** |
| `Docs/Formulation/` | Formulation LaTeX/PDF, `build_modelib_wsl.sh` |

See [`ZrMicro/CLAUDE.md`](ZrMicro/CLAUDE.md) for the model equations, the state
vector, the solver algorithm, and the file map.

---

## Architecture

```
Input (Excel)
  → InputData (parameter loading + derived quantities)
  → ReactionRates (pre-compute all rate constants)
  → RateEquations (ODE RHS)
  → Solver (C++ CVODE, or scipy LSODA fallback)
  → PostProcess (derived macroscopic quantities)
  → Visualization (PNG figures + provenance.md)
```

- **C++ primary:** SUNDIALS CVODE 7.1.1 (BDF, dense), invoked as a subprocess
  via `py_utils/cpp_bridge.py`. The `--batch_file` OpenMP mode integrates many
  parameter cases per subprocess — the coupling march relies on it.
- **Python fallback:** `scipy.integrate.solve_ivp` with LSODA.

Every run creates `ZrMicro/output/<YYYYMMDD_HHMMSS>_<git-hash>/` with figures
and `provenance.md`.

---

## Environment and build

Both are repository-wide; see [`../CLAUDE.md`](../CLAUDE.md) and
[`../README.md`](../README.md).

- Venv: `../.DisloClusterVenv/` (Python 3.14), kernel `dislocluster`
- Dependencies: `../requirements.txt` (a copy is kept here for reference)
- Build: `cmake -S ZrMicro/cpp_utils -B ZrMicro/build -DCMAKE_BUILD_TYPE=Release`

---

## Coupling to the 3-D code

The bridge modules live in `ZrMicro/py_utils/`:

| Module | Role |
|---|---|
| `paths.py` | resolves both codebases, binaries, and the shared material file |
| `modelib_coupling.py` | `pack_y0`, `run_immobile_step` — the pointwise slow march |
| `modelib_fem.py` | `MoDELibFEMSolver` — drives DDomp / pyMoDELib for the fast solve |
| `modelib_export.py` | dose-indexed closure table for the 3-D solver |
| `modelib_fields.py`, `modelib_gb.py` | field and grain-boundary plots from 3-D output |
| `run_zr3d_singlecrystal.py` | driver for the standalone 3-D single-cubic-crystal case |
| `transport_audit.py` | transport-term audit of the 0-D vs 3-D correspondence |

Drivers:

- [`ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb`](ZrMicro/code/coupled_0d_3d_ZrMicro.ipynb)
  — the operator-split 0-D ↔ 3-D dose march.
- [`ZrMicro/py_utils/run_zr3d_singlecrystal.py`](ZrMicro/py_utils/run_zr3d_singlecrystal.py)
  — post-processes `MoDELib3/tutorials/zrmicro_coupled` (1 µm cube) into a
  timestamped figure set. See [`../CLAUDE.md`](../CLAUDE.md) for the caveats.
