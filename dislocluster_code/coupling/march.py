"""
run_coupled_vs_standalone.py — run the 1 um single-crystal reference through
BOTH routes from a shared 0-D seed, render both, and compare them.

Both routes start from the SAME microstructure: the 0-D ZrMicro state at
``DOSE_SEED`` written uniformly onto the CD nodes of the 1 um cube.

  standalone   MoDELib3 does the whole split itself. ``ClusterDynamicsFEM::
               solve()`` is exactly {solveMobileClusters(); solveImmobile-
               Clusters();} -- the steady mobile solve, then its own nodal
               semi-implicit immobile update (nSub = 20 substeps per 1 dpa
               step), first order, no error control. Run by DDomp; this
               module reads its evl snapshots.

  coupled      the same split, with the SLOW step replaced. MoDELib3 still
               solves the fast step -- DDomp is called with
               ``useImmobileSolver=0``, which runs solveMobileClusters() and
               passes the immobile field through untouched -- and the immobile
               state is then advanced at every node by CVODE BDF with
               acc_mode=2, the exact AD Jacobian, per-thread workspace reuse,
               the base+delta case protocol and deduplication.

THE FAST STEP IS IN THE LOOP
----------------------------
At every substep the coupled route re-solves C_M*(x) for the immobile state it
has built, freezes it, and integrates the immobile ODEs over the substep. That
is the operator split. An earlier version of this module did NOT do this: it
lifted the mobile field out of the standalone run's recorded snapshots. Under
that arrangement the mobile field could never respond to the immobile state the
march was producing, so the march had no mechanism to relax a seed that was not
already at quasi-steady state -- and a 0-D seed never is, because the 0-D
carries no boundary layer while the true C_M*(x) is pinned to thermal
equilibrium on all six faces. ``MOBILE_MODE`` retains that behaviour as
``"replay"`` purely so the effect of closing the loop can be measured.

WHAT THE COMPARISON DOES AND DOES NOT ISOLATE
---------------------------------------------
With the fast step closed, both routes now solve the SAME fast problem with the
SAME code, and each solves it against its own immobile state. Two differences
remain, and neither is the integrator alone:

  1. DIFFERENT IMMOBILE EQUATIONS. The coupled route integrates ZrMicro's
     immobile right-hand side; the standalone integrates MoDELib3's. These are
     different models, not two integrations of one model -- see the known
     non-correspondences in CLAUDE.md (DAD bias, the bi-pyramid vacancy family,
     size-dependent vacancy-loop emission).
  2. DIFFERENT INTEGRATORS, and different splitting frequency. CVODE BDF with
     error control against a first-order semi-implicit nodal update with none;
     and ``FEM_EVERY`` sets how often the coupled route refreshes C_M* against
     the standalone's once per internal substep.

Layout mirrors the reference case:
    <stamp>_<hash>_<tag>/
        0d/        the 0-D reference trajectory and the seed state
        3d/        field panels, standalone/ and coupled/ side by side
        gb/        grain-boundary profiles, both routes
        tables/    comparison tables (markdown + csv + json)
        evl_coupled/   the marched CD fields, restartable by MoDELib
        provenance.md
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np


from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.coupling import checkpoint as ckpt_mod          # noqa: E402
from dislocluster_code.coupling import field as mfield                 # noqa: E402
from dislocluster_code.coupling import progress as progress_mod        # noqa: E402
from dislocluster_code.coupling.config import MarchConfig, default_config  # noqa: E402
from dislocluster_code.coupling.progress import format_hms             # noqa: E402
from dislocluster_code.staging import seed as seed_mod                 # noqa: E402
from dislocluster_code.coupling import immobile as mc                  # noqa: E402
from dislocluster_code.coupling import qssa as mqssa                   # noqa: E402
from dislocluster_code.post import report as mreport               # noqa: E402
from dislocluster_code.zerod.calibration import build_sim                   # noqa: E402
from dislocluster_code.zerod.cpp_bridge import collect_solver_args          # noqa: E402

DOSE_SEED = 0.1
DOSE_MAX = 20.1
DOSE_INTERVAL = 5.0
SUBSTEPS_PER_INTERVAL = 20
DEDUP_RTOL = 1e-8          # exact-duplicate removal only; measured error 0.0

# How many immobile substeps run between two fast solves. 1 refreshes C_M*(x)
# at every substep, which is the split as written; larger values trade
# splitting accuracy for DDomp invocations. The standalone refreshes once per
# 1 dpa output step, so FEM_EVERY = SUBSTEPS_PER_INTERVAL * DOSE_INTERVAL
# matches its cadence.
FEM_EVERY = 1

# "qssa"   re-solve C_M*(x) from MoDELib for the current immobile state (the
#          split; the only physically defensible setting)
# "replay" read the mobile field out of the standalone snapshots (what this
#          module used to do -- kept to measure what closing the loop changed)
# "seed"   hold the 0-D seed's mobile field for the whole march (a control:
#          shows what the march does with no fast step at all)
MOBILE_MODE = "qssa"

# 0-D lumped quantities compared between the routes, in 0-D units.
COMPARE = [
    ("N_a", "<a> loop number density", ("CiL", "CaiL")),
    ("N_c", "<c> loop number density", ("CvL", "CavL")),
    ("c_a", "<a> loop content",        ("CiL_i", "CaiL_i")),
    ("c_c", "<c> loop content",        ("CvL_v", "CavL_v")),
]
IMM_ORDER = ["CiL", "CaiL", "CvL", "CavL", "CiL_i", "CaiL_i", "CvL_v", "CavL_v"]


def _cfg_seed(cfg):
    return (default_config() if cfg is None else cfg).dose_seed


def dose_snapshots(cfg=None):
    cfg = default_config() if cfg is None else cfg
    return np.asarray(cfg.dose_grid, dtype=float)


def evl_step_for(dose, dose_seed=None, cfg=None):
    """Output-step index holding `dose`, for a 1 dpa step.

    MoDELib writes after solve(), so evl_N is the state after (N+1) steps taken
    from the seed: dose = dose_seed + (N+1) and N = round(dose - dose_seed) - 1.
    This is exactly modelib_report.dose_steps, reused so the coupled snapshots
    are named on the same convention the standalone writes and the figure
    renderer reads.

    The seed dose is taken from ``DOSE_SEED`` rather than assumed to be 0.1.
    Hardcoding 0.1 was correct only while every case used that seed; a march
    seeded at 1 dpa would name its 6 dpa snapshot ``evl_5``, which the renderer
    and the standalone both read as 7 dpa.
    """
    if dose_seed is None:
        dose_seed = (default_config() if cfg is None else cfg).dose_seed
    return mreport.dose_steps([dose], 1.0, dose_seed)[0]


def standalone_dose_map(sim_dir, G, dose_per_step=1.0, cfg=None):
    """evl index -> the dose that snapshot holds.

    MoDELib keeps ONE appended F file (F_0.txt), not one per output step: each
    row is `runID time dt ...`, so row r carries the time at the START of
    runID r, i.e. r completed dose steps. Output is written after solve(), so
    evl_N holds the state after (N+1) steps.

    `dose_per_step` is 1 dpa by construction of DD.txt (dtMax was chosen as the
    time for exactly 1 dpa at this dose rate). That is asserted against the F
    time increment rather than trusted, since a changed dtMax would silently
    rescale every dose label in the comparison.
    """
    sim_dir = Path(sim_dir)
    dd = (sim_dir / "inputFiles" / "DD.txt").read_text(encoding="utf-8",
                                                       errors="replace")
    dtMax = float(re.search(r"^dtMax=([-\d.eE+]+)", dd, re.M).group(1))

    fpath = sim_dir / "F" / "F_0.txt"
    if not fpath.is_file():
        return {}
    rows = np.loadtxt(fpath, ndmin=2)
    if rows.size == 0:
        return {}
    if rows.shape[0] > 1:
        step = rows[1, 1] - rows[0, 1]
        if not np.isclose(step, dtMax, rtol=1e-9):
            raise ValueError(
                f"F time increment {step:.6e} does not match dtMax {dtMax:.6e}; "
                "the dose-per-step assumption behind every label is invalid.")

    completed = {int(r) for r in rows[:, 0]}
    out = {}
    for p in (sim_dir / "evl").glob("evl_*.txt"):
        m = re.fullmatch(r"evl_(\d+)\.txt", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n in completed or n + 1 in completed:
            out[n] = _cfg_seed(cfg) + (n + 1) * dose_per_step
    return out


def lumped_from_cd(cd, omega):
    """CD block -> the 0-D lumped quantities of COMPARE, per node."""
    imm = mfield.modelib_immobile_to_0d(cd[:, mfield.M_SIZE:], omega)
    d = {n: imm[:, i] for i, n in enumerate(IMM_ORDER)}
    return {key: sum(d[c] for c in cols) for key, _, cols in COMPARE}


def metrics(a, b):
    """Compare coupled (a) against standalone (b), per node."""
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    if not m.any():
        return dict(mean_rel=float("nan"), max_rel=float("nan"),
                    ratio=float("nan"), a_mean=float("nan"), b_mean=float("nan"))
    r = np.abs(a[m] - b[m]) / np.abs(b[m])
    return dict(mean_rel=float(r.mean()), max_rel=float(r.max()),
                ratio=float(a[m].mean() / b[m].mean()),
                a_mean=float(a[m].mean()), b_mean=float(b[m].mean()))


# ── setting up a simulation directory for the fast solves ────────────────────

def prepare_qssa_dir(standalone_sim, dest, seed_evl=None, cfg=None):
    """Clone a case into a directory used ONLY for fast solves.

    Every fast solve overwrites ``evl/evl_0.txt``, so it must not run in the
    standalone case's own directory: that would destroy the reference history
    the comparison is against.
    """
    standalone_sim, dest = Path(standalone_sim), Path(dest)
    (dest / "evl").mkdir(parents=True, exist_ok=True)
    (dest / "F").mkdir(exist_ok=True)

    # Build the new inputFiles/ beside the old one and swap it in, rather than
    # rmtree-then-copytree. An interrupt in that window used to leave the
    # fast-solve directory with NO inputFiles at all, and the next run died in
    # MobileQSSASolver with "no inputFiles/DD.txt in ..." -- two layers away
    # from the cause. os.replace of a directory is atomic on the same volume.
    staged = dest / "inputFiles.new"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(standalone_sim / "inputFiles", staged)
    old = dest / "inputFiles"
    if old.exists():
        doomed = dest / "inputFiles.old"
        if doomed.exists():
            shutil.rmtree(doomed)
        os.replace(old, doomed)
        os.replace(staged, old)
        shutil.rmtree(doomed, ignore_errors=True)
    else:
        os.replace(staged, old)

    shutil.copy2(standalone_sim / "evl" / "cdNodes.txt", dest / "evl" / "cdNodes.txt")

    src = Path(seed_evl) if seed_evl else (standalone_sim / "evl" /
                                           f"evl_seed_{_cfg_seed(cfg):.3f}dpa.txt")
    if not src.is_file():
        raise FileNotFoundError(f"no seed configuration at {src}")
    # A seed that lives inside the fast-solve directory's own evl/ is a trap:
    # MobileQSSASolver copies it aside as the template every configuration is
    # rebuilt from, so on a resume it would be re-copied from the LAST fast
    # solve's output and the non-CD records would drift for the rest of the run.
    if src.resolve().parent == (dest / "evl").resolve():
        raise ValueError(
            f"seed_evl {src} is inside the fast-solve directory's evl/, which "
            f"every fast solve overwrites. Point it at the stable seed in the "
            f"case directory instead.")
    shutil.copy2(src, dest / "evl" / "evl_0.txt")
    return dest, src


# ── the coupled march ────────────────────────────────────────────────────────

class MarchFailure(RuntimeError):
    """The slow step lost more nodes than the march is allowed to tolerate.

    ``run_immobile_step`` reports a failed point as ``None``, and the march
    substitutes that point's PREVIOUS state -- an identity step. That is a
    reasonable repair for one node out of 30 000 and a silent fabrication when
    it happens to all of them, which is exactly what a missing solver
    executable or a non-zero solver exit produces (``[None] * n``). Raising
    keeps a whole-batch failure from being recorded as a completed substep.
    """


def _march_fingerprint(cfg, qssa_sim, seed_evl, snaps, base_cli, N):
    """What a checkpoint may not be resumed across. See checkpoint.HARD_KEYS."""
    import os as _os
    return dict(
        # hard
        n_nodes=int(N), n_eq=int(mc.N_EQ),
        cd_nodes_sha=ckpt_mod.file_digest(Path(qssa_sim) / "evl" / "cdNodes.txt"),
        seed_evl_sha=ckpt_mod.file_digest(seed_evl),
        material_sha=ckpt_mod.file_digest(paths.MODELIB_MATERIAL),
        base_cli_sha=hashlib.blake2b("\n".join(sorted(base_cli)).encode(),
                                     digest_size=16).hexdigest(),
        march_cfg=cfg.fingerprint(),
        snaps_sha=ckpt_mod.array_digest(np.asarray(snaps, dtype=float)),
        # soft: worth reporting, not worth refusing over
        git_hash=paths.git_hash(),
        omp_num_threads=_os.environ.get("OMP_NUM_THREADS"),
        numpy=np.__version__,
    )


def run_coupled(sim, qssa_sim, seed_evl, snaps, evl_out, standalone_sim=None,
                dose_map=None, mobile_mode=None, fem_every=None, verbose=True,
                max_failed_nodes=None, cfg=None,
                checkpoint_dir=None, resume="auto", progress=None):
    """The operator split: fast FEM mobile solve, then the frozen-mobile march.

    ``max_failed_nodes`` is how many points may take an identity step in one
    substep before the march aborts. ``None`` (the default) tolerates any
    number, which is what every caller predating this argument expected; a
    WHOLE-batch failure aborts either way, because that is not a physics
    failure but a missing or crashed solver.

    ``checkpoint_dir`` turns on checkpointing; leaving it ``None`` reproduces
    the behaviour of every caller written before it existed. With it set,
    ``resume`` is ``"auto"`` (continue if a checkpoint is there), ``"never"``
    (start over, discarding it) or ``"require"`` (fail if there is none).

    ``progress`` is called with a :class:`~dislocluster_code.coupling.progress.SubstepEvent`
    after every substep; see that module for the two supplied renderers. Pass
    ``verbose=False`` alongside it, or the march's own prints will fight the
    single-line display.

    Note on reproducibility: a resumed march re-enters at exactly the state it
    left, but DDomp and the CVODE batch are both OpenMP-parallel, so the
    trajectory after a resume is not guaranteed bit-identical to an
    uninterrupted one. Pin ``OMP_NUM_THREADS`` if that matters.

    Returns ``(history, timing, bridge, diagnostics)``.
    """
    cfg = default_config() if cfg is None else cfg
    # An explicit keyword still wins over the config, so every existing caller
    # -- which passes fem_every/mobile_mode and monkey-patches the rest -- gets
    # exactly what it got before.
    mobile_mode = cfg.mobile_mode if mobile_mode is None else mobile_mode
    fem_every = cfg.fem_every if fem_every is None else int(fem_every)
    substeps = cfg.substeps
    if max_failed_nodes is None:
        max_failed_nodes = cfg.max_failed_nodes

    br = mfield.FieldBridge(qssa_sim, paths.MODELIB_MATERIAL)
    N = br.n_nodes

    # Coarsening detector: where the mean-field treatment of coalescence stops
    # being trustworthy, i.e. where a discrete-continuum handoff would belong.
    # Fed once per immobile substep so the crossing is resolved to a substep
    # rather than interpolated across whatever gap the dose list leaves. It is
    # REPORTING ONLY -- see post/coarsening.py -- and any failure to build it
    # disables it rather than costing the march.
    detector = None
    try:
        from dislocluster_code.post.coarsening import SubstepDetector
        detector = SubstepDetector(
            br.nodes, mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL),
            paths.MODELIB_MATERIAL,
            phi_star=getattr(cfg, "phi_star", 0.15),
            frac_star=getattr(cfg, "frac_star", 0.10),
            hold=getattr(cfg, "coarsen_hold", 2),
            variant_weights=getattr(cfg, "variant_weights",
                                    (1 / 3, 1 / 3, 1 / 3)))
    except Exception as exc:
        if verbose:
            print(f"  coarsening detector unavailable: {exc}")

    # ── the discrete transition (plan 4b/4c/4f) ─────────────────────────────
    # OFF unless COUPLING['discrete_transition'] asks for it, so nothing here
    # changes a run that predates it. When on, a family that has crossed phi*
    # is converted to discrete loops at the NEXT FAST-SOLVE BOUNDARY -- where a
    # DDomp call happens anyway and the state is already synchronized, which is
    # what plan 4.1 specifies -- and only what DD can actually take is removed
    # from the continuum.
    do_transition = bool(getattr(cfg, "discrete_transition", False))
    transition_units = tuple(getattr(cfg, "transition_units", ("c",)))
    transferred = set()
    transition_log = []
    trans_weights = trans_faces = None
    if do_transition and verbose:
        print(f"  discrete transition ARMED for {transition_units} "
              f"at phi* = {getattr(cfg, 'phi_star', 0.15)}")

    def _maybe_transition(Y, dose_now):
        """Convert any crossed unit to discrete loops. Returns the new ``Y``."""
        nonlocal trans_weights, trans_faces
        if not do_transition or detector is None:
            return Y
        crossed = {k for k, v in (detector.result().get("d_coarsen") or {}).items()
                   if v is not None}
        from dislocluster_code.coupling import transition as trans
        pending = [u for u in transition_units if u not in transferred
                   and crossed.intersection(trans.UNITS[u]["families"])]
        if not pending:
            return Y
        from dislocluster_code.coupling import neighbors as nbr
        omega = mfield.cluster_atomic_volume(paths.MODELIB_MATERIAL)
        if trans_weights is None:
            from dislocluster_code.post.discrete_loops import domain_weights
            trans_weights, trans_faces = domain_weights(br.nodes)
        for unit in pending:
            try:
                Y_new, pops, ledger = trans.transfer(
                    Y, br.nodes, omega, keys=(unit,),
                    weights=trans_weights, faces=trans_faces,
                    material=paths.MODELIB_MATERIAL, coalesce_pass=False)
                ok, msgs = trans.check(ledger)
                frac = ledger["rows"][0].get("frac_kept", 0.0)
                if frac <= 0.0:
                    # Nothing fits inside the crystal. Declining is the correct
                    # outcome -- see plan 4.6 -- and it must not be mistaken for
                    # a completed transfer, so the unit stays untransferred and
                    # will be retried at the next boundary.
                    if verbose:
                        print(f"      transition '{unit}' DECLINED at "
                              f"{dose_now:.4g} dpa: no loop fits the crystal")
                    transition_log.append(dict(unit=unit, dose=float(dose_now),
                                               declined=True, frac_kept=0.0))
                    continue
                n = trans.inject_discrete_loops(qssa_sim, pops,
                                                tag=f"{unit}_{dose_now:.4g}dpa",
                                                verbose=verbose)
                Rc = nbr.cutoff(nbr.screening_lengths(
                    trans.cd_block(Y, omega), paths.MODELIB_MATERIAL),
                    getattr(cfg, "climb_cutoff_nL", 4.0))
                trans.enable_discrete_climb(qssa_sim, Rc)
                transferred.add(unit)
                rec = dict(unit=unit, dose=float(dose_now), declined=False,
                           frac_kept=float(frac), R_c_b=float(Rc),
                           loops=n, ledger_ok=bool(ok), messages=msgs,
                           rows=ledger["rows"])
                transition_log.append(rec)
                diagnostics["transition"] = transition_log
                if verbose:
                    print(f"      transition '{unit}' at {dose_now:.4g} dpa: "
                          f"{n['realized']} loops, {100*frac:.1f}% of the "
                          f"family, R_c = {Rc:.4g} b, ledger "
                          f"{'OK' if ok else 'FAILED: ' + '; '.join(msgs)}")
                Y = Y_new
            except Exception as exc:                # never cost a march a probe
                if verbose:
                    print(f"      transition '{unit}' failed: {exc}")
                transition_log.append(dict(unit=unit, dose=float(dose_now),
                                           error=str(exc)))
        return Y

    evl_out = Path(evl_out)
    evl_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(qssa_sim) / "evl" / "cdNodes.txt", evl_out / "cdNodes.txt")

    fast = None
    if mobile_mode == "qssa":
        fast = mqssa.MobileQSSASolver(qssa_sim, paths.MODELIB_MATERIAL,
                                      seed_evl=seed_evl, verbose=verbose,
                                      on_unconverged=cfg.on_unconverged)
        if verbose:
            print(fast.describe())

    base_cli = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-6, atol=1e-20, stats=True))

    # ── checkpoint / resume ─────────────────────────────────────────────────
    fp = _march_fingerprint(cfg, qssa_sim, seed_evl, snaps, base_cli, N)
    saver = ckpt_mod.Checkpointer(checkpoint_dir or (evl_out.parent / "checkpoint"),
                                  fp, enabled=checkpoint_dir is not None)
    if saver.enabled and resume == "never" and saver.exists():
        shutil.rmtree(saver.dir, ignore_errors=True)
        saver.dir.mkdir(parents=True, exist_ok=True)
    resumed_Y = resumed_state = None
    resumed_history = {}
    if saver.enabled and resume in ("auto", "require"):
        resumed_Y, resumed_state, resumed_history = saver.load()
        if resumed_state is None and resume == "require":
            raise ckpt_mod.CheckpointMismatch(
                f"resume='require' but no usable checkpoint in {saver.dir}")
    resumed = resumed_state is not None

    ev0 = mfield.EvlFile(seed_evl)
    Y = np.zeros((N, mc.N_EQ))
    Y[:, 0:4] = ev0.mobile
    Y[:, 4:12] = mfield.modelib_immobile_to_0d(ev0.immobile, br.omega)
    Y[:, mc.IDX_RHO_N] = float(sim.input_data.material_params["rho"])
    Y_seed = Y.copy()

    diagnostics = {}
    G = float(sim.input_data.material_params["G"])
    mobile_trace = []
    wall_s_prior = 0.0

    if resumed:
        # The seed-relaxation block below must NOT run again. On a resumed Y it
        # would spend a fast solve (~245 s) measuring the wrong thing, overwrite
        # Y[:, 0:4] with a field solved against a stale immobile state, and file
        # the result as the seed diagnostic.
        Y = np.array(resumed_Y, dtype=float)
        st = resumed_state
        history = dict(resumed_history)
        timing = [dict(t) for t in st["timing"]]
        used_steps = set(st["used_steps"])
        k_global = int(st["k_global"])
        mobile_trace = [dict(t) for t in st["mobile_trace"]]
        diagnostics = dict(st["diagnostics"])
        wall_s_prior = float(st.get("wall_s_total", 0.0))
        if fast is not None:
            fast.n_calls = int(st.get("fast_n_calls", 0))
            fast.wall_s = float(st.get("fast_wall_s", 0.0))
            fast.n_converged = int(st.get("fast_n_converged", 0))
            fast.n_unconverged = int(st.get("fast_n_unconverged", 0))
        if verbose:
            print(f"\n  resuming at substep {k_global}/{cfg.n_substeps} "
                  f"({format_hms(wall_s_prior)} already spent)")
    else:
        history = {float(snaps[0]): Y.copy()}
        timing = []
        used_steps = set()
        k_global = 0
        # The measurement the split exists to make: is the seeded mobile field
        # the quasi-steady one? Solve for C_M* against the seed's own immobile
        # state BEFORE any immobile marching, and compare.
        if fast is not None:
            if verbose:
                print("\n  relaxing the seed to quasi-steady state:")
            C0 = fast.solve(Y)
            diagnostics["seed_relaxation"] = mqssa.relaxation_report(Y_seed, C0)
            Y[:, 0:4] = C0
            if verbose:
                for nm, d in diagnostics["seed_relaxation"].items():
                    print(f"      {nm:<4} seed {d['seed_mean']:.4e} -> QSSA "
                          f"{d['qssa_mean']:.4e}  (x{d['ratio_mean']:.3g}), "
                          f"spatial contrast {d['spatial_contrast']:.3g}")

    def _trace(dose, C_M):
        mobile_trace.append(dict(
            dose=float(dose),
            **{nm: dict(mean=float(C_M[:, j].mean()),
                        min=float(C_M[:, j].min()),
                        max=float(C_M[:, j].max()))
               for j, nm in enumerate(("Cv", "Ci", "C2i", "C3i"))}))

    if fast is not None and not resumed:
        _trace(snaps[0], Y[:, 0:4])

    # Where to re-enter. `substeps` is constant across the march, so the pair
    # (interval, substep-within-interval) is just k_global divided by it --
    # which is why k_global is the only counter that has to be stored.
    n_intervals = len(snaps) - 1
    i0, k0 = divmod(k_global, substeps) if substeps else (0, 0)
    eta = progress_mod.EtaEstimator(**(resumed_state or {}).get("eta", {}))
    t_session = time.perf_counter()
    prev_unconv = getattr(fast, "n_unconverged", 0)

    def _emit(**kw):
        if progress is None:
            return
        L = seed_mod.lumped(Y)
        progress(progress_mod.SubstepEvent(
            n_substeps_total=cfg.n_substeps, n_intervals=n_intervals,
            substeps_per_interval=substeps, resumed=resumed,
            dose_max=float(snaps[-1]),
            wall_s_session=time.perf_counter() - t_session,
            wall_s_total=wall_s_prior + (time.perf_counter() - t_session),
            mean_N_a=float(L["N_a"].mean()), mean_N_c=float(L["N_c"].mean()),
            mean_c_a=float(L["c_a"].mean()), mean_c_c=float(L["c_c"].mean()),
            mean_Cv=float(Y[:, 0].mean()), mean_Ci=float(Y[:, 1].mean()),
            get_Y=lambda: Y, **kw))

    _emit(phase="start", k_global=k_global, interval=i0)

    for i in range(i0, n_intervals):
        d0, d1 = float(snaps[i]), float(snaps[i + 1])
        # The accumulator reset is scoped to an interval. Re-firing it after a
        # mid-interval resume would zero the conservation quadratures this
        # interval had already banked.
        first_k = k0 if i == i0 else 0
        if first_k == 0:
            Y[:, 12:18] = 0.0

        edges = np.linspace(d0 / G, d1 / G, substeps + 1)
        t_start = time.perf_counter()
        n_int, n_fast, fast_s = 0, 0, 0.0
        if resumed and i == i0:
            prior = (resumed_state or {}).get("interval", {})
            n_int = int(prior.get("n_int", 0))
            n_fast = int(prior.get("n_fast", 0))
            fast_s = float(prior.get("fast_s", 0.0))
            t_start -= float(prior.get("wall_s", 0.0))
        for k, (a, b) in list(enumerate(zip(edges[:-1], edges[1:])))[first_k:]:
            # ── FAST STEP: steady mobile field for the CURRENT immobile state
            #
            # The cadence counts substeps over the WHOLE march, not within the
            # interval. Counting within the interval makes `fem_every` silently
            # inoperative whenever an interval holds fewer than `fem_every`
            # substeps -- with one substep per interval, k is always 0 and the
            # test always fires. For the earlier runs (20 substeps per interval,
            # fem_every=5) the two counters agree exactly, so this changes no
            # previous result.
            did_fast, this_fast_s = False, 0.0
            if k_global % fem_every == 0:
                # Act on the coarsening detector HERE, before the fast solve,
                # so the solve that follows already sees the discrete network
                # and the reduced continuum field. Doing it after would leave
                # one interval in which the loops are counted on both sides --
                # the double count plan 4.5 exists to prevent.
                Y = _maybe_transition(Y, d0 + (d1 - d0) * k / substeps)
                if fast is not None:
                    t_f = time.perf_counter()
                    if not (i == 0 and k == 0):      # step 0 solved above
                        Y[:, 0:4] = fast.solve(Y)
                        n_fast += 1
                        did_fast = True
                        _trace(d0 + (d1 - d0) * k / substeps,
                               Y[:, 0:4])
                    this_fast_s = time.perf_counter() - t_f
                    fast_s += this_fast_s
                elif mobile_mode == "replay" and dose_map:
                    dose_now = d0 + (d1 - d0) * k / substeps
                    n = min(dose_map, key=lambda kk: abs(dose_map[kk] - dose_now))
                    f = Path(standalone_sim) / "evl" / f"evl_{n}.txt"
                    if abs(dose_map[n] - dose_now) < 0.75 and f.is_file():
                        Y[:, 0:4] = mfield.EvlFile(f).mobile
                        n_fast += 1
                # mobile_mode == "seed": nothing to do, the field is held

            # ── SLOW STEP: immobile ODEs with the mobile species frozen
            st = {}
            t_slow = time.perf_counter()
            out = mc.run_immobile_step(base_cli, Y, a, b,
                                       base_dir=paths.ZRMICRO_DIR,
                                       dedup_rtol=cfg.dedup_rtol, stats=st)
            slow_s = time.perf_counter() - t_slow
            nfail = sum(o is None for o in out)
            if (N and nfail == N) or (max_failed_nodes is not None
                                      and nfail > max_failed_nodes):
                bad_idx = [q for q, o in enumerate(out) if o is None]
                # Dump the state that failed. Without this a failed march throws
                # away the one thing needed to diagnose it -- the exact input,
                # including the fast solve that preceded it, which is expensive
                # to reconstruct and easy to reconstruct WRONGLY: a solver built
                # without the same seed starts its Newton iteration elsewhere and
                # converges to a field that agrees to three printed digits and
                # not at the failing point.
                try:
                    dump = (saver.dir if saver.enabled else evl_out.parent)
                    np.savez_compressed(
                        Path(dump) / "failed_state.npz", Y=Y,
                        bad_idx=np.array(bad_idx, dtype=int),
                        t_begin=float(a), t_end=float(b),
                        dose_from=float(d0), dose_to=float(d1),
                        substep=int(k + 1), k_global=int(k_global))
                    print(f"      wrote {Path(dump) / 'failed_state.npz'}",
                          flush=True)
                except Exception as exc:
                    print(f"      could not dump failed state: {exc}",
                          flush=True)
                raise MarchFailure(
                    f"{nfail}/{N} points failed at substep {k + 1} of interval "
                    f"[{d0:g} -> {d1:g}] dpa (max_failed_nodes="
                    f"{max_failed_nodes}); node indices "
                    f"{bad_idx[:20]}{' ...' if len(bad_idx) > 20 else ''}. "
                    + ("The whole batch failed, which means the solver did not "
                       "run -- check that solver.exe exists and exits 0."
                       if nfail == N else
                       "Raise max_failed_nodes to let the march substitute an "
                       "identity step for these points."))
            if nfail and verbose:
                print(f"      warning: {nfail}/{N} points failed")
            Y = np.array([o if o is not None else Y[q] for q, o in enumerate(out)])
            n_int += st.get("n_integrated", N)
            k_global += 1

            # Coarsening detector -- REPORTING ONLY, nothing branches on it.
            # Fed here because Y has just been rebound wholesale, which is the
            # same consistent point the checkpoint uses.
            if detector is not None:
                try:
                    detector.update(Y, d0 + (d1 - d0) * (k + 1) / substeps)
                except Exception as exc:            # never cost a march a probe
                    if verbose:
                        print(f"      coarsening detector disabled: {exc}")
                    detector = None

            # ── CHECKPOINT ──────────────────────────────────────────────────
            # The only consistent point in the body: the fast solve is either
            # fully applied (above) or was not scheduled, Y has just been
            # rebound wholesale, and k_global now counts substeps COMPLETED --
            # so a resume is exactly "skip the first k_global substeps" and the
            # cadence test reproduces.
            eta.update(did_fast, this_fast_s, slow_s)
            wall_now = wall_s_prior + (time.perf_counter() - t_session)
            saver.save(Y, dict(
                k_global=k_global, timing=timing,
                used_steps=sorted(used_steps), mobile_trace=mobile_trace,
                diagnostics=diagnostics, wall_s_total=wall_now,
                eta=eta.state(),
                interval=dict(n_int=n_int, n_fast=n_fast, fast_s=fast_s,
                              wall_s=time.perf_counter() - t_start),
                fast_n_calls=getattr(fast, "n_calls", 0),
                fast_wall_s=getattr(fast, "wall_s", 0.0),
                fast_n_converged=getattr(fast, "n_converged", 0),
                fast_n_unconverged=getattr(fast, "n_unconverged", 0),
            ))

            _emit(phase="substep", k_global=k_global, interval=i,
                  k_in_interval=k + 1,
                  dose=d0 + (d1 - d0) * (k + 1) / substeps,
                  dose_from=d0, dose_to=d1, did_fast=did_fast,
                  fast_s=this_fast_s, slow_s=slow_s,
                  eta_s=eta.remaining(k_global, cfg.n_substeps, fem_every),
                  n_failed=nfail, n_integrated=st.get("n_integrated", N),
                  dedup_ratio=st.get("dedup_ratio", 1.0),
                  qssa_converged=None if fast is None else did_fast and
                  fast.n_unconverged == prev_unconv,
                  checkpoint_path=str(saver.dir / saver.CKPT)
                  if saver.enabled else None)
            prev_unconv = getattr(fast, "n_unconverged", 0)

            if verbose:
                print(f"      substep {k + 1:2d}/{substeps} "
                      f"[{d0:.1f}->{d1:.1f} dpa]  "
                      f"{time.perf_counter() - t_start:7.1f} s cumulative",
                      flush=True)

        dt = time.perf_counter() - t_start
        # Idempotent: a crash between here and the snapshot write leaves
        # k_global on an exact multiple of `substeps`, so this interval's tail
        # runs again on resume. history[d1] and write_immobile_field already
        # overwrite; timing.append would otherwise duplicate the row.
        if not any(t["dose_to"] == d1 for t in timing):
            timing.append(dict(dose_from=d0, dose_to=d1, wall_s=dt,
                               substeps=substeps, integrations=n_int,
                               dedup_ratio=(substeps * N) / max(n_int, 1),
                               fast_solves=n_fast, fast_s=fast_s,
                               slow_s=dt - fast_s, mobile_mode=mobile_mode,
                               resumed=resumed and i == i0))
        history[d1] = Y.copy()
        if verbose:
            print(f"    {d0:5.1f} -> {d1:5.1f} dpa  {dt:8.1f} s "
                  f"({fast_s:.0f} s fast / {dt - fast_s:.0f} s slow, "
                  f"{n_fast} FEM solves)   dedup x{timing[-1]['dedup_ratio']:.2f}")

        # 0-D -> 3-D: a complete, restartable configuration at the reference
        # step index, so both routes render through the same code path.
        #
        # The step index only exists for a snapshot grid that lies on the
        # dose-per-step lattice. A logarithmic grid through the nucleation
        # transient does not: 1e-3, 1e-2 and 0.1 dpa all round to the same
        # index, and a seed at dose 0 makes the first one negative. Falling
        # back to the snapshot ordinal keeps the files distinct instead of
        # silently overwriting one with the next.
        step = evl_step_for(d1, cfg=cfg)
        if step < 0 or step in used_steps:
            step = f"s{i + 1:02d}"
            if verbose:
                print(f"      note: {d1:g} dpa is off the 1 dpa lattice; "
                      f"snapshot filed as evl_{step}.txt")
        else:
            used_steps.add(step)
        br.write_immobile_field(Y, evl_src=seed_evl,
                                dest=evl_out / f"evl_{step}.txt")

        # The interval is complete: bank the history and the updated timing.
        saver.save(Y, dict(
            k_global=k_global, timing=timing,
            used_steps=sorted(used_steps), mobile_trace=mobile_trace,
            diagnostics=diagnostics,
            wall_s_total=wall_s_prior + (time.perf_counter() - t_session),
            eta=eta.state(), interval={},
            fast_n_calls=getattr(fast, "n_calls", 0),
            fast_wall_s=getattr(fast, "wall_s", 0.0),
            fast_n_converged=getattr(fast, "n_converged", 0),
            fast_n_unconverged=getattr(fast, "n_unconverged", 0),
        ), history=history)
        _emit(phase="interval", k_global=k_global, interval=i,
              dose=d1, dose_from=d0, dose_to=d1,
              eta_s=eta.remaining(k_global, cfg.n_substeps, fem_every))

    if fast is not None:
        diagnostics["fast_solver"] = dict(calls=fast.n_calls,
                                          wall_s=fast.wall_s,
                                          mean_s=fast.wall_s / max(fast.n_calls, 1))
        diagnostics["mobile_trace"] = mobile_trace
    if saver.enabled:
        diagnostics["checkpoint"] = dict(
            dir=str(saver.dir), writes=saver.n_writes,
            write_s=saver.write_s,
            mean_write_ms=1e3 * saver.write_s / max(saver.n_writes, 1),
            resumed=resumed)
    _emit(phase="done", k_global=k_global, interval=n_intervals,
          dose=float(snaps[-1]))
    if detector is not None and detector.history:
        diagnostics["coarsening"] = detector.result()

    return history, timing, br, diagnostics


# ── comparison ───────────────────────────────────────────────────────────────

def compare(standalone_sim, evl_coupled, snaps, dose_map, omega, out_dir,
            timing, ddomp_wall_s, diagnostics=None, mobile_mode="qssa",
            fem_every=1, cfg=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_dose = [], {}

    for d in snaps[1:]:
        n = evl_step_for(d)
        fs = Path(standalone_sim) / "evl" / f"evl_{n}.txt"
        fc = Path(evl_coupled) / f"evl_{n}.txt"
        if not (fs.is_file() and fc.is_file()):
            print(f"  skip {d:.1f} dpa (missing "
                  f"{fs.name if not fs.is_file() else fc.name})")
            continue
        cds = mfield.EvlFile(fs).cd
        cdc = mfield.EvlFile(fc).cd
        Ls, Lc = lumped_from_cd(cds, omega), lumped_from_cd(cdc, omega)
        per_dose[float(d)] = {}
        for key, label, _ in COMPARE:
            m = metrics(Lc[key], Ls[key])
            per_dose[float(d)][key] = m
            rows.append(dict(dose=float(d), quantity=key, label=label, **m))

    # ── markdown ────────────────────────────────────────────────────────────
    md = ["# Coupled march versus standalone MoDELib3", "",
          f"Seeded at {snaps[0]:g} dpa and advanced to {snaps[-1]:g} dpa "
          f"over {len(snaps) - 1} interval(s): "
          f"{', '.join(f'{d:g}' for d in snaps)} dpa.", "",
          "Both routes run the SAME operator split and the SAME fast step "
          "(MoDELib3 `solveMobileClusters`, a steady diffusion-reaction solve). "
          "They differ in the slow step: MoDELib3's nodal semi-implicit update "
          "against ZrMicro's CVODE BDF march. The immobile right-hand sides are "
          "themselves different models, so this is a model-plus-method "
          "comparison, not an integrator comparison.", "",
          f"Coupled route: mobile mode `{mobile_mode}`, fast solve every "
          f"{fem_every} immobile substep(s).", ""]

    if diagnostics and "seed_relaxation" in diagnostics:
        md += ["## Does the split repair a seed that is not quasi-steady?", "",
               f"The 0-D seed at {snaps[0]:g} dpa is spatially uniform and carries no "
               "boundary layer. Solving the fast step once against the seed's own "
               "immobile state, before any immobile marching, gives C_M*(x):", "",
               "| species | seeded (uniform) | QSSA mean | ratio | QSSA min | QSSA max | max/min |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for nm, d in diagnostics["seed_relaxation"].items():
            md.append(f"| {nm} | {d['seed_mean']:.4e} | {d['qssa_mean']:.4e} | "
                      f"{d['ratio_mean']:.4g} | {d['qssa_min']:.4e} | "
                      f"{d['qssa_max']:.4e} | {d['spatial_contrast']:.4g} |")
        md += ["", "A ratio far from 1, or a max/min far from 1, means the seeded "
                   "field was not the quasi-steady one and the fast step supplied "
                   "the difference. Under the old replay arrangement no such "
                   "correction could occur.", ""]

    trace = (diagnostics or {}).get("mobile_trace") or []
    if len(trace) > 1:
        md += ["### C_M*(x) over the march", "",
               "Each row is one fast solve: the mobile field re-established "
               "against the immobile state the march had reached.", "",
               "| dose [dpa] | Cv mean | Cv min | Cv max | Ci mean | Ci min | Ci max |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for t in trace:
            md.append(f"| {t['dose']:.2f} | {t['Cv']['mean']:.4e} | "
                      f"{t['Cv']['min']:.4e} | {t['Cv']['max']:.4e} | "
                      f"{t['Ci']['mean']:.4e} | {t['Ci']['min']:.4e} | "
                      f"{t['Ci']['max']:.4e} |")
        md.append("")

    md += ["## Accuracy — coupled (CVODE) against standalone (semi-implicit)", ""]
    md += ["| dose [dpa] | quantity | standalone mean | coupled mean | ratio | mean rel. diff | max rel. diff |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['dose']:.1f} | {r['label']} | {r['b_mean']:.4e} | "
                  f"{r['a_mean']:.4e} | {r['ratio']:.3f} | {r['mean_rel']:.3e} | "
                  f"{r['max_rel']:.3e} |")

    tot_coupled = sum(t["wall_s"] for t in timing)
    tot_fast = sum(t["fast_s"] for t in timing)
    n_int = sum(t["integrations"] for t in timing)
    n_pts = sum(t["substeps"] for t in timing)
    n_fast = sum(t["fast_solves"] for t in timing)
    span = float(snaps[-1] - snaps[0])
    md += ["", "## Speed", "",
           "| route | fast step | slow step | wall clock [s] | per dpa [s] |",
           "|---|---|---|---:|---:|"]
    md.append(f"| standalone | MoDELib3 FEM, in-process | MoDELib3 semi-implicit, "
              f"20 substeps/dpa | {ddomp_wall_s:.0f} | {ddomp_wall_s / span:.1f} |")
    md.append(f"| coupled | MoDELib3 FEM, {n_fast} DDomp calls, {tot_fast:.0f} s | "
              f"CVODE BDF, acc_mode=2, AD Jacobian, {tot_coupled - tot_fast:.0f} s | "
              f"{tot_coupled:.0f} | {tot_coupled / span:.1f} |")
    md += ["", f"Coupled march: {n_pts} substeps, {n_int} integrations actually "
               f"dispatched after deduplication, {n_fast} fast solves.", "",
           "The coupled route pays a full DDomp start-up per fast solve — mesh "
           "load, FE assembly and factorization are redone every call — where "
           "the standalone amortizes them over the whole run. That overhead is "
           "an artifact of driving MoDELib as a subprocess, not of the split.",
           ""]
    md += ["| dose interval [dpa] | wall [s] | fast [s] | slow [s] | FEM solves | integrations | dedup |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for t in timing:
        md.append(f"| {t['dose_from']:.1f} -> {t['dose_to']:.1f} | {t['wall_s']:.1f} | "
                  f"{t['fast_s']:.1f} | {t['slow_s']:.1f} | {t['fast_solves']} | "
                  f"{t['integrations']} | x{t['dedup_ratio']:.2f} |")

    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import csv
    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "comparison.json").write_text(
        json.dumps(dict(rows=rows, timing=timing, diagnostics=diagnostics or {},
                        mobile_mode=mobile_mode, fem_every=fem_every,
                        ddomp_wall_s=ddomp_wall_s,
                        coupled_wall_s=tot_coupled,
                        coupled_fast_s=tot_fast), indent=2, default=float),
        encoding="utf-8")
    return rows, per_dose, tot_coupled


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--standalone-sim",
                    default=str(paths.MODELIB_ROOT / "tutorials" / "zrmicro_seeded"))
    ap.add_argument("--qssa-sim", default=None,
                    help="directory for the fast solves (default: "
                         "<standalone>_qssa; NEVER the standalone dir)")
    ap.add_argument("--seed-evl", default=None)
    ap.add_argument("--tag", default="coupled_vs_standalone")
    ap.add_argument("--mobile-mode", default=MOBILE_MODE,
                    choices=("qssa", "replay", "seed"))
    ap.add_argument("--fem-every", type=int, default=FEM_EVERY)
    ap.add_argument("--dose-interval", type=float, default=None)
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--dose-seed", type=float, default=None)
    ap.add_argument("--dose-max", type=float, default=None)
    ap.add_argument("--dedup-rtol", type=float, default=None)
    ap.add_argument("--ddomp-wall", type=float, default=float("nan"),
                    help="measured DDomp wall time [s], for the speed table")
    ap.add_argument("--figures", action="store_true", help="render 3d/ and gb/")
    args = ap.parse_args(argv)

    over = {}
    if args.dose_interval is not None:
        over["dose_interval"] = args.dose_interval
    if args.substeps is not None:
        over["substeps"] = args.substeps
    if args.dose_seed is not None:
        over["dose_seed"] = args.dose_seed
    if args.dose_max is not None:
        over["dose_max"] = args.dose_max
    if args.dedup_rtol is not None:
        over["dedup_rtol"] = args.dedup_rtol
    cfg = default_config(mobile_mode=args.mobile_mode,
                         fem_every=args.fem_every, **over)

    standalone_sim = Path(args.standalone_sim)
    qssa_sim = Path(args.qssa_sim) if args.qssa_sim else \
        standalone_sim.with_name(standalone_sim.name + "_qssa")
    if qssa_sim.resolve() == standalone_sim.resolve():
        raise SystemExit("--qssa-sim must differ from --standalone-sim: every "
                         "fast solve overwrites evl_0.txt in it.")
    snaps = dose_snapshots(cfg)

    # The workbook alone is NOT the calibrated model -- 28 parameters have
    # drifted and 11 are missing from it entirely. Building straight from it,
    # as this driver used to, integrates a different model: N_a comes out
    # 2.3e-2 against the calibrated 8.1e-8, and the loop content exceeds unit
    # atom fraction. See py_utils/calibration.py.
    sim = build_sim()
    G = float(sim.input_data.material_params["G"])
    T = float(sim.input_data.material_params["T"])
    print(f"calibrated 0-D: {len(sim.overrides_applied)} fitted parameters "
          f"applied on top of {Path(sim.input_file).name}")

    out = paths.run_dir(args.tag)
    for sub in ("0d", "3d", "gb", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"output : {out}")
    print(f"T = {T} K, G = {G} dpa/s")
    print(f"doses  : {', '.join(f'{d:.1f}' for d in snaps)} dpa")
    print(f"mobile : {args.mobile_mode}, fast solve every {args.fem_every} substep(s)")

    dose_map = standalone_dose_map(standalone_sim, G, cfg=cfg)
    if dose_map:
        print(f"standalone: {len(dose_map)} snapshots, "
              f"{min(dose_map.values()):.2f} .. {max(dose_map.values()):.2f} dpa")

    qssa_sim, seed_evl = prepare_qssa_dir(standalone_sim, qssa_sim,
                                          args.seed_evl, cfg=cfg)
    print(f"fast-solve dir: {qssa_sim}")

    print("\ncoupled march:")
    history, timing, br, diag = run_coupled(
        sim, qssa_sim, seed_evl, snaps, out / "evl_coupled",
        standalone_sim=standalone_sim, dose_map=dose_map,
        mobile_mode=args.mobile_mode, fem_every=args.fem_every, cfg=cfg)

    np.savez_compressed(out / "march_state.npz",
                        doses=np.array(sorted(history)),
                        Y=np.array([history[d] for d in sorted(history)]),
                        nodes=br.nodes)

    print("\ncomparison:")
    rows, per_dose, tot = compare(standalone_sim, out / "evl_coupled", snaps,
                                  dose_map, br.omega, out / "tables",
                                  timing, args.ddomp_wall, diagnostics=diag,
                                  mobile_mode=args.mobile_mode,
                                  fem_every=args.fem_every, cfg=cfg)
    for r in rows:
        print(f"  {r['dose']:5.1f} dpa  {r['quantity']:<5} ratio {r['ratio']:7.3f}  "
              f"mean rel {r['mean_rel']:.3e}  max rel {r['max_rel']:.3e}")

    if args.figures:
        doses_fig = [float(d) for d in snaps[1:]]
        for route, evl in (("standalone", standalone_sim / "evl"),
                           ("coupled", out / "evl_coupled")):
            print(f"\nrendering {route} ...")
            try:
                mreport.write_3d_figures(evl, doses_fig, out / "3d" / route,
                                         dose_per_step=1.0, verbose=True)
                mreport.write_gb_figures(evl, doses_fig, out / "gb" / route,
                                         dose_per_step=1.0, verbose=True)
            except Exception as e:                      # noqa: BLE001
                print(f"  {route} rendering failed: {e}")

    n_fast = sum(t["fast_solves"] for t in timing)
    tot_fast = sum(t["fast_s"] for t in timing)
    (out / "provenance.md").write_text("\n".join([
        "# Coupled march versus standalone MoDELib3", "",
        f"- generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- git hash  : {paths.git_hash()}",
        f"- input     : {sim.input_file}",
        f"- T, G      : {T} K, {G} dpa/s",
        f"- doses     : {', '.join(f'{d:.1f}' for d in snaps)} dpa",
        f"- seed      : 0-D state at {cfg.dose_seed} dpa, uniform on {br.n_nodes} CD nodes",
        f"- standalone: {standalone_sim}",
        f"- fast solve: {args.mobile_mode}, every {args.fem_every} substep(s), "
        f"{n_fast} DDomp calls, {tot_fast:.1f} s",
        f"- coupled   : CVODE BDF, acc_mode=2, analytic AD Jacobian, "
        f"dedup_rtol={cfg.dedup_rtol:g}, {cfg.substeps} substeps/interval",
        f"- coupled wall clock: {tot:.1f} s",
        f"- standalone wall clock: {args.ddomp_wall:.1f} s",
    ]) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    main()
