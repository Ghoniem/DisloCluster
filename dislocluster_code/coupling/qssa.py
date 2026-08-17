"""
modelib_qssa.py — the FAST STEP of the operator split, as a callable.

WHAT THIS IS FOR
----------------
The two-time-scale scheme alternates two solves over each dose step:

  fast   C_M*(x)  — the QUASI-STEADY mobile field for the immobile state
                    currently held. No time derivative: MoDELib3's
                    ``ClusterDynamicsFEM::solveMobileClusters`` is a
                    steady diffusion-reaction solve, with the immobile
                    population entering only through ``ImmobileSinks``
                    evaluated at every quadrature point.
  slow   the immobile ODEs integrated over the step with C_M* frozen.

The slow step already had a driver (``modelib_coupling.run_immobile_step``,
CVODE at every node). The fast step did NOT: the march used to lift its mobile
field out of a previously recorded standalone run. That is a replay, not a
split. The mobile field never responded to the immobile state the march was
building, so the loop had no mechanism to relax a seed that started away from
quasi-steady state -- and a seed taken from the 0-D at low dose is exactly
that, because the 0-D has no boundary layer at all while the true C_M*(x) is
pinned to thermal equilibrium on every face.

This module closes that loop. ``MobileQSSASolver.solve(Y)`` writes the current
per-node immobile state into the CD block, runs DDomp for a single step with
``useImmobileSolver=0`` so the immobile field is passed through untouched, and
reads the relaxed mobile field back. It is the callback the march was missing.

HOW THE SINGLE STEP IS ARRANGED
-------------------------------
``DefectiveCrystal::runSingleStep`` writes its output BEFORE incrementing
runID, so a run with ``startAtTimeStep=0`` and ``Nsteps=1`` reads ``evl_0.txt``
and overwrites the same file with the result. That in-place round trip is what
this class uses: the seed is regenerated from ``Y`` on every call, so nothing
is lost by overwriting it.

Nothing about the fast solve depends on dt or on the accumulated dose --
``cascadeGlobalProduction`` is built from the constant generation rate, and
``solveMobileClusters`` has no time derivative -- so the run's time bookkeeping
is irrelevant here and is left alone.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


from dislocluster_code import paths                          # noqa: E402
from dislocluster_code.coupling import field as mfield        # noqa: E402


class QSSASolveError(RuntimeError):
    """DDomp failed, or produced no usable mobile field."""


# These live with the other input-file writers now. Re-exported because
# staging.domain, staging.standalone and the notebooks import them from here.
from dislocluster_code.staging.inputs import (  # noqa: E402,F401
    set_dd_scalar, get_dd_scalar, FAST_STEP_SETTINGS, elastic_is_trivial)


class MobileQSSASolver:
    """Run MoDELib3's steady mobile solve on demand, for a given immobile field.

    Parameters
    ----------
    sim_dir : path
        A MoDELib simulation directory carrying the mesh and boundary
        conditions of the case. It is reconfigured on construction
        (``useImmobileSolver=0``, ``Nsteps=1``, ``startAtTimeStep=0``) and used
        only for fast solves, so it must NOT be the directory of a standalone
        run whose history you want to keep.
    material_file : path
        MoDELib material file, for the atom-fraction/per-b^3 conversion.
    seed_evl : path | None
        Configuration whose non-CD records (mesh nodes, loops, inclusions) are
        copied into every solve. Defaults to ``sim_dir/evl/evl_0.txt`` as it
        stands at construction, saved aside so later overwrites cannot destroy
        it.

    Notes
    -----
    The Newton loop in ``solveMobileClusters`` is started from whatever mobile
    field is in the CD block, so passing the previous C_M* as a warm start is
    worth doing and ``solve`` does it by default.
    """

    def __init__(self, sim_dir, material_file=None, seed_evl=None,
                 ddomp=None, use_wsl=None, verbose=True,
                 on_unconverged="warn", elastic=None, loop_model=0):
        if on_unconverged not in ("warn", "raise"):
            raise ValueError("on_unconverged must be 'warn' or 'raise'")
        self.on_unconverged = on_unconverged
        # Which immobile layout Y[:, 4:12] arrives in. The fast solve itself is
        # unaffected -- it only ever sees the CD block -- but the map that fills
        # that block from Y is a different one in each mode, and getting it
        # wrong is silent: both produce a well-formed block, one of them for the
        # wrong four families.
        self.loop_model = int(loop_model)
        # None -> decide from the staged case (see `_configure`); True or
        # False forces the elastic solve on or off.
        self.elastic = elastic
        self.sim_dir = Path(sim_dir)
        self.evl_dir = self.sim_dir / "evl"
        self.dd_file = self.sim_dir / "inputFiles" / "DD.txt"
        if not self.dd_file.is_file():
            raise FileNotFoundError(f"no inputFiles/DD.txt in {self.sim_dir}")
        self.material_file = Path(material_file or paths.MODELIB_MATERIAL)
        self.omega = mfield.cluster_atomic_volume(self.material_file)
        self.verbose = verbose

        self.ddomp = Path(ddomp) if ddomp else paths.modelib_ddomp()
        if self.ddomp is None or not Path(self.ddomp).exists():
            raise QSSASolveError(
                f"DDomp not found ({self.ddomp}); build MoDELib3 first "
                "(Docs/Formulation/build_modelib.sh).")
        self.use_wsl = paths.use_wsl() if use_wsl is None else bool(use_wsl)

        # Preserve the seed configuration: every solve overwrites evl_0.txt.
        src = Path(seed_evl) if seed_evl else (self.evl_dir / "evl_0.txt")
        if not src.is_file():
            raise FileNotFoundError(f"no seed configuration at {src}")
        self._seed = self.sim_dir / "evl_qssa_seed.txt"
        if not self._seed.is_file() or src != self._seed:
            shutil.copy2(src, self._seed)

        self.nodes = mfield.read_cd_nodes(self.evl_dir)
        self.n_nodes = self.nodes.shape[0]

        self._configure()
        self.n_calls = 0
        self.wall_s = 0.0
        self.n_converged = 0
        self.n_unconverged = 0

    # -- one-time reconfiguration --------------------------------------------
    def _configure(self):
        """Put DD.txt into fast-solve mode, and record what was changed.

        ``mobileSolverClampInLoop=0`` is not cosmetic. With the historical
        setting (1) MoDELib applies its positivity floor to the STATE between
        the Newton update and the error test, which makes the mobile solve a
        projected Newton iteration. On a mobile field solved against an
        immobile state supplied by this march, that iteration does not
        converge: it reaches ~1e-4 in nine iterations and then cycles
        indefinitely, and because the upstream loop has no iteration cap, DDomp
        never returns. Measured on the 1 um case:

            clamp in loop            30 iterations, no convergence, 392 s
            clamp deferred (this)     5 iterations, 6.8e-7,         112 s
            clamp in loop, w=0.7     20 iterations, 8.2e-6,         283 s

        The two converged variants agree to 0.19% and their means to five
        figures, so the clamp was corrupting the iteration and not the answer.
        Deferring it also leaves the result physical: zero negative entries and
        the same 7090 floored dofs as the clamped runs.

        The floor itself is not discarded -- MoDELib applies it once after the
        loop. That is where ZrMicro has it too: ZrMicro floors before every
        RATE EVALUATION, it does not project the state mid-iteration.
        """
        self.previous = {k: get_dd_scalar(self.dd_file, k)
                         for k in ("useImmobileSolver", "Nsteps",
                                   "startAtTimeStep", "outputFrequency",
                                   "mobileSolverClampInLoop",
                                   "mobileSolverMaxIterations",
                                   "useElasticDeformation",
                                   "useElasticDeformationFEM")}
        set_dd_scalar(self.dd_file, "useImmobileSolver", "0")
        set_dd_scalar(self.dd_file, "Nsteps", "1")
        set_dd_scalar(self.dd_file, "startAtTimeStep", "0")
        set_dd_scalar(self.dd_file, "outputFrequency", "1")
        set_dd_scalar(self.dd_file, "mobileSolverClampInLoop", "0")
        set_dd_scalar(self.dd_file, "mobileSolverMaxIterations", "50")

        # Skip an elastic solve that can only return zero. `elastic` defaults to
        # None = decide from the staged case; True or False forces it, so a
        # comparison against a run made before this existed can be reproduced.
        if self.elastic is None:
            trivial, reason = elastic_is_trivial(self.sim_dir)
        else:
            trivial = not self.elastic
            reason = f"elastic={self.elastic} passed explicitly"
        self.elastic_skipped, self.elastic_reason = trivial, reason
        if trivial:
            set_dd_scalar(self.dd_file, "useElasticDeformation", "0")
            set_dd_scalar(self.dd_file, "useElasticDeformationFEM", "0")

    # -- the fast solve -------------------------------------------------------
    def solve(self, Y, variant_weights=(1 / 3, 1 / 3, 1 / 3), warm_start=True):
        """Steady mobile field for the immobile state carried by ``Y``.

        Parameters
        ----------
        Y : (N,19) array
            Native ZrMicro state per CD node. Columns 4..11 supply the immobile
            field; columns 0..3 are the warm start for the Newton loop.

        Returns
        -------
        (N,4) array — C_M*(x) = [Cv, Ci, C2i, C3i] at the CD nodes.
        """
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] != self.n_nodes:
            raise ValueError(
                f"state has {Y.shape[0]} rows but the case has {self.n_nodes} "
                "CD nodes")

        ev = mfield.EvlFile(self._seed)
        ev.cd[:, mfield.M_SIZE:] = mfield.immobile_0d_to_modelib(
            Y, self.omega, variant_weights, loop_model=self.loop_model)
        if warm_start:
            ev.cd[:, :mfield.M_SIZE] = Y[:, 0:mfield.M_SIZE]
        ev.write(self.evl_dir / "evl_0.txt")

        t0 = time.perf_counter()
        proc = self._run_ddomp()
        dt = time.perf_counter() - t0
        self.n_calls += 1
        self.wall_s += dt

        out = self.evl_dir / "evl_0.txt"
        try:
            C_M = mfield.EvlFile(out).mobile.copy()
        except Exception as e:                       # noqa: BLE001
            raise QSSASolveError(
                f"could not read the mobile field back from {out}: {e}\n"
                f"DDomp stdout tail:\n{proc.stdout[-2000:]}")
        if C_M.shape != (self.n_nodes, mfield.M_SIZE):
            raise QSSASolveError(
                f"mobile field has shape {C_M.shape}, expected "
                f"({self.n_nodes},{mfield.M_SIZE})")
        if not np.isfinite(C_M).all():
            raise QSSASolveError("the mobile field contains non-finite values")

        if self.verbose:
            rng = "  ".join(f"{nm} {C_M[:, j].min():.2e}..{C_M[:, j].max():.2e}"
                            for j, nm in enumerate(("Cv", "Ci", "C2i", "C3i")))
            print(f"      QSSA solve {dt:6.1f} s   {rng}")
        return C_M

    def _run_ddomp(self):
        cmd, cwd = paths.ddomp_cmd(self.sim_dir, exe=self.ddomp)
        if self.use_wsl != paths.use_wsl():
            # An explicit use_wsl= overrides the platform default; keep honoring
            # it rather than silently ignoring the argument.
            cmd = (["wsl.exe", "-e", paths.windows_to_wsl(self.ddomp),
                    paths.windows_to_wsl(self.sim_dir)] if self.use_wsl
                   else [str(self.ddomp), str(self.sim_dir)])
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="replace", cwd=cwd)
        if proc.returncode != 0:
            raise QSSASolveError(
                f"DDomp exited {proc.returncode}\n"
                f"stdout tail:\n{proc.stdout[-3000:]}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}")
        if "immobile solver SKIPPED" not in proc.stdout:
            raise QSSASolveError(
                "DDomp ran but did NOT skip the immobile solver -- the binary "
                "predates the useImmobileSolver flag, or DD.txt was not "
                "reconfigured. The immobile field would have been advanced "
                "twice (once here, once by the march), so this is fatal "
                "rather than a warning.")

        # A capped solve returns its best iterate rather than a converged one.
        # That is far better than the hang it replaced, but it must not pass
        # unnoticed: the difference is not uniformly small. On the case that
        # first exposed this, the capped field matched the converged one in the
        # mean to four figures while one node was off by 54% in Ci.
        if "mobile solver STOPPED" in proc.stdout:
            line = next((l.strip() for l in proc.stdout.splitlines()
                         if "mobile solver STOPPED" in l), "")
            self.n_unconverged += 1
            if self.on_unconverged == "raise":
                raise QSSASolveError(
                    f"fast solve did not converge and on_unconverged='raise' "
                    f"-- {line}. The capped iterate would have been frozen as "
                    f"C_M* for this substep.")
            print(f"      WARNING: fast solve did not converge -- {line}")
        elif "mobile solver converged" in proc.stdout:
            self.n_converged += 1
        return proc

    # -- diagnostics ----------------------------------------------------------
    def describe(self):
        return "\n".join([
            f"sim_dir   : {self.sim_dir}",
            f"DDomp     : {self.ddomp}  (wsl={self.use_wsl})",
            f"CD nodes  : {self.n_nodes}",
            f"omega     : {self.omega:.6g} b^3",
            f"seed      : {self._seed.name}",
            f"DD.txt was: {self.previous}",
            f"elastic   : " + ("SKIPPED — " if self.elastic_skipped
                               else "solved — ") + self.elastic_reason,
            f"calls     : {self.n_calls}, {self.wall_s:.1f} s total "
            f"({self.n_converged} converged, {self.n_unconverged} capped)",
        ])


def relaxation_report(Y_seed, C_M_star, nodes=None):
    """Compare a seeded mobile field against the QSSA-relaxed one.

    Returns a dict of per-species diagnostics. This is the measurement that
    answers whether the split repairs a seed which is not at quasi-steady
    state: if the seed were already C_M*, every ratio would be 1.
    """
    Y_seed = np.atleast_2d(np.asarray(Y_seed, dtype=float))
    seed = Y_seed[:, 0:mfield.M_SIZE]
    out = {}
    for j, nm in enumerate(("Cv", "Ci", "C2i", "C3i")):
        a, b = seed[:, j], C_M_star[:, j]
        m = np.isfinite(a) & np.isfinite(b) & (a > 0)
        out[nm] = dict(
            seed_mean=float(a[m].mean()) if m.any() else float("nan"),
            qssa_mean=float(b[m].mean()) if m.any() else float("nan"),
            qssa_min=float(b.min()), qssa_max=float(b.max()),
            ratio_mean=float(b[m].mean() / a[m].mean()) if m.any() else float("nan"),
            max_rel_change=float(np.abs(b[m] - a[m]).max() / np.abs(a[m]).max())
            if m.any() else float("nan"),
            spatial_contrast=float(b.max() / max(b.min(), 1e-300)),
        )
    return out
