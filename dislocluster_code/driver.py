"""driver.py — one call per stage, for the notebooks under ``Simulations/``.

    cfg  = SimulationConfig.from_dicts(...)
    run  = prepare(cfg)          # mesh, staged case, bootstrap, seed, run dir
    res  = march(run)            # the operator split; resumable
    report(run, res)             # figures, movies, provenance

Each stage is idempotent and safe to re-run from a fresh kernel: ``prepare``
skips a staged domain that already matches, and ``march`` picks up from its
checkpoint. Re-executing the long cell after a kernel restart continues the run
rather than starting it over.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import march as _march
from dislocluster_code.coupling.config import MarchConfig
from dislocluster_code.coupling.progress import NotebookProgress
from dislocluster_code.staging import case as _case
from dislocluster_code.staging import seed as _seed

__all__ = ["PreparedRun", "MarchResult", "prepare", "march", "report",
           "march_config_from"]


@dataclass
class PreparedRun:
    cfg: object
    sim: object
    domain: dict
    sim_dir: Path
    qssa_dir: Path
    seed_evl: Path
    out_dir: Path
    y_seed: np.ndarray
    dose_reached: float

    @property
    def checkpoint_dir(self):
        return self.out_dir / "checkpoint"


@dataclass
class MarchResult:
    history: dict
    timing: list
    bridge: object
    diagnostics: dict
    wall_s: float
    out_dir: Path


def march_config_from(cfg):
    """The march's own settings, taken out of the simulation configuration."""
    c = cfg.coupling
    return MarchConfig(
        dose_seed=c.dose_seed, snaps=c.snaps, substeps=c.substeps_per_interval,
        fem_every=c.fem_every, dedup_rtol=c.dedup_rtol,
        variant_weights=c.variant_weights,
        max_failed_nodes=c.max_failed_nodes,
        on_unconverged=c.on_unconverged,
        phi_star=c.phi_star, frac_star=c.frac_star,
        coarsen_hold=c.coarsen_hold,
        loop_model=cfg.solver.loop_model,
        discrete_transition=c.discrete_transition,
        transition_units=c.transition_units,
        climb_cutoff_nL=c.climb_cutoff_nL,
        mobile_mode="qssa",
        rtol=cfg.solver.rtol, atol=cfg.solver.atol,
        analytic_jac=cfg.solver.analytic_jac)


def prepare(cfg, out_dir=None, force_stage=False, verbose=True):
    """Mesh, stage, bootstrap, seed, and open a run directory."""
    t0 = time.perf_counter()
    if verbose:
        print(cfg.summary())
        print()

    domain = _case.ensure_domain(cfg, force=force_stage, verbose=verbose)
    sim_dir = cfg.sim_dir
    scaffold = sim_dir / "evl" / domain["scaffold"]

    # The boundary is passed so the 0-D takes its applied load from BOUNDARY
    # rather than from the workbook's independent `sigma_n`; see
    # Material.build_sim.
    sim = cfg.material.build_sim(boundary=cfg.boundary)
    if verbose:
        print(f"\ncalibrated 0-D: {len(sim.overrides_applied)} fitted "
              f"parameters on top of {Path(sim.input_file).name}")

    d0 = cfg.coupling.dose_seed
    seed_evl = sim_dir / "evl" / (
        "evl_seed_pristine.txt" if d0 <= 0 else f"evl_seed_{d0:.3f}dpa.txt")
    y, hit = _seed.build_seed(sim, d0, seed_evl, scaffold,
                              variant_weights=cfg.coupling.variant_weights,
                              material_file=cfg.material.path,
                              loop_model=cfg.solver.loop_model)
    if verbose:
        L = _seed.lumped(y, cfg.solver.loop_model)
        what = "pristine" if d0 <= 0 else f"the 0-D at {d0:g} dpa"
        print(f"seed: {what} (solver reached {hit:.4g} dpa)")
        print(f"  N_a={L['N_a'][0]:.4e}  N_c={L['N_c'][0]:.4e}  "
              f"c_a={L['c_a'][0]:.4e}  c_c={L['c_c'][0]:.4e}")

    # The fast solves get their own directory: each one overwrites evl_0.txt,
    # so they must not run in the staged case and destroy its seed.
    qssa_dir = sim_dir.with_name(sim_dir.name + "_qssa")
    mcfg = march_config_from(cfg)
    qssa_dir, seed_used = _march.prepare_qssa_dir(sim_dir, qssa_dir,
                                                  seed_evl=seed_evl, cfg=mcfg)

    out = Path(out_dir) if out_dir else paths.run_dir(cfg.tag)
    (out / "evl_coupled").mkdir(parents=True, exist_ok=True)
    cfg.write(out / "config.json")
    if verbose:
        print(f"\nfast-solve dir: {qssa_dir}")
        print(f"output        : {out}")
        print(f"prepared in {time.perf_counter() - t0:.0f} s")

    return PreparedRun(cfg=cfg, sim=sim, domain=domain, sim_dir=sim_dir,
                       qssa_dir=qssa_dir, seed_evl=seed_used, out_dir=out,
                       y_seed=y, dose_reached=hit)


def march(run, progress=None, resume=None, verbose=False):
    """The operator-split march. Re-running this resumes it."""
    cfg = run.cfg
    mcfg = march_config_from(cfg)
    progress = NotebookProgress() if progress is None else progress
    resume = cfg.output.resume if resume is None else resume

    t0 = time.perf_counter()
    history, timing, br, diag = _march.run_coupled(
        run.sim, run.qssa_dir, run.seed_evl,
        np.asarray(cfg.coupling.snaps, dtype=float),
        run.out_dir / "evl_coupled", cfg=mcfg,
        checkpoint_dir=run.checkpoint_dir if cfg.output.checkpoint else None,
        resume=resume, progress=progress, verbose=verbose)
    wall = time.perf_counter() - t0

    doses = sorted(history)
    np.savez_compressed(run.out_dir / "march_state.npz",
                        doses=np.array(doses),
                        Y=np.array([history[d] for d in doses]),
                        nodes=br.nodes,
                        # Which of the two immobile formulations `Y` is in.
                        # Post-processing cannot tell from the array -- both
                        # are 19 columns of plausible numbers -- and reading it
                        # in the wrong one relabels <c> as <a> silently.
                        loop_model=np.array(cfg.solver.loop_model))
    (run.out_dir / "summary.json").write_text(json.dumps(dict(
        doses=doses, wall_s=wall, timing=timing, diagnostics=diag,
        config=cfg.to_dict()), indent=2, default=float), encoding="utf-8")

    return MarchResult(history=history, timing=timing, bridge=br,
                       diagnostics=diag, wall_s=wall, out_dir=run.out_dir)


def table(result):
    """The four lumped aggregates against dose, as printable rows."""
    rows = []
    lm = int(result.diagnostics.get("loop_model", 0))
    for d in sorted(result.history):
        L = {k: float(v.mean())
             for k, v in _seed.lumped(result.history[d], lm).items()}
        rows.append((d, L["N_a"], L["N_c"], L["c_a"], L["c_c"]))
    return rows


def print_table(result):
    print(f"{'dose':>9} {'N_a':>12} {'N_c':>12} {'c_a':>12} {'c_c':>12}")
    for d, na, nc, ca, cc in table(result):
        print(f"{d:9.4g} {na:12.4e} {nc:12.4e} {ca:12.4e} {cc:12.4e}")


def report(run, result=None, n_doses=6, verbose=True):
    """Figures, movies and the run report, from what the march wrote."""
    from dislocluster_code.post import march_report

    argv = [str(run.out_dir), "--n-doses", str(n_doses)]
    if run.cfg.output.movies:
        argv += ["--interp", str(run.cfg.output.movie_interp)]
    else:
        argv.append("--no-movies")
    if verbose:
        print(f"rendering {run.out_dir} ...")
    march_report.main(argv)

    if run.cfg.output.discrete_loops:
        from dislocluster_code.post import discrete_loops, loop_movie, tem_slices
        discrete_loops.main([str(run.out_dir)])
        if run.cfg.output.movies:
            loop_movie.main([str(run.out_dir)])
        # TOP-LEVEL tem_slices/, not the module default of
        # discrete_loops/tem_slices/. Nested two deep under a directory holding
        # hundreds of loop PNGs, the micrographs are effectively unfindable --
        # they were lost in exactly that way once already.
        tem_slices.main([str(run.out_dir),
                         "--out", str(run.out_dir / "tem_slices")])
    return run.out_dir
