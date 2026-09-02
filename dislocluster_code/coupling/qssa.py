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
import os
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


class _Result:
    """Just enough of subprocess.CompletedProcess for the checks below."""
    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


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
        self.n_loops = mfield.EvlFile(self._seed).n_loops

        self._configure()
        # Resident DDomp: build the mesh, trial functions and the diffusion
        # operator's Cholesky ONCE instead of once per fast solve. 53% of an
        # invocation at 189 533 nodes is that repeated setup, the
        # factorization alone being 44%. Bit-identical by construction --
        # each cycle re-reads the CD fields and nothing else. Set
        # DISLOCLUSTER_DDOMP_SERVER=0 to go back to one process per solve.
        self.server = os.environ.get('DISLOCLUSTER_DDOMP_SERVER', '1') != '0'
        self._proc = None
        self._server_started = 0
        self.n_calls = 0
        self.wall_s = 0.0
        self.last_superposed = None
        self.last_evl = None
        self.n_converged = 0
        self.n_unconverged = 0

    # -- adopting a discrete network ------------------------------------------
    def adopt_network(self, evl_with_network, verbose=True):
        """Re-seed from an evl that carries dislocation records.

        THIS IS WHAT MAKES A RUNTIME TRANSITION REACH THE FAST SOLVE, and the
        bug it fixes is one step further along than it looks.
        ``transition.inject_discrete_loops`` does run
        ``microstructureGenerator`` and does merge the network into the staged
        case's ``evl/evl_0.txt``, preserving the CD block. But ``solve`` writes
        ``evl_0.txt`` from ``self._seed`` on EVERY call -- that is the whole
        reason the seed is kept, see ``__init__`` -- so the very next fast solve
        overwrote the merged network with the loop-free seed. Every solve then
        ran with an empty dislocation network, before and after the transfer,
        and nothing in the logs said so.

        Only the network records are adopted; the CD block is supplied by
        ``solve`` from ``Y`` as always, so no field is taken from the file.
        """
        src = Path(evl_with_network)
        ev = mfield.EvlFile(src)
        if ev.n_cd != self.n_nodes:
            raise ValueError(
                f"{src} has {ev.n_cd} CD rows but the case has "
                f"{self.n_nodes} CD nodes")
        shutil.copy2(src, self._seed)
        before, self.n_loops = self.n_loops, ev.n_loops

        # A SECOND STEP IS REQUIRED, and the reason is an ordering fact rather
        # than an accuracy argument. DefectiveCrystal emplaces its
        # microstructures in a fixed order (DefectiveCrystal.cpp:38-44):
        # ClusterDynamics BEFORE DislocationNetwork. MicrostructureContainer::
        # solve() then walks that order, so within one step the CD mobile solve
        # runs first and the climb velocities are computed after it.
        #
        # The discrete field enters the CD solve as c_DD in its Dirichlet
        # values, and c_DD is LINEAR in the nodal climbVelocityScalar
        # (DislocationSegment::clusterConcentration). On step 0 those are still
        # zero, so a one-step fast solve sees c_DD = 0 no matter how many loops
        # are present -- the superposition would be armed and inert. Step 1's CD
        # solve sees the velocities step 0 computed.
        if self.n_loops:
            self.n_steps = 2
            set_dd_scalar(self.dd_file, "Nsteps", str(self.n_steps))
        if verbose:
            print(f"      fast solve adopted the discrete network: "
                  f"{before} -> {ev.n_loops} loops, {ev.n_nodes} nodes"
                  + (f"; Nsteps -> {self.n_steps} so the CD solve sees nonzero "
                     f"climb velocities" if self.n_loops else ""))
        return ev.n_loops

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
        self.n_steps = 1          # raised to 2 by adopt_network; see there
        set_dd_scalar(self.dd_file, "useImmobileSolver", "0")
        set_dd_scalar(self.dd_file, "Nsteps", str(self.n_steps))
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
        # Clear any evl a previous multi-step solve left, so "the highest
        # numbered file present" below cannot pick up a stale one.
        for stale in self.evl_dir.glob("evl_*.txt"):
            if re.fullmatch(r"evl_(\d+)\.txt", stale.name):
                stale.unlink()
        ev.write(self.evl_dir / "evl_0.txt")

        t0 = time.perf_counter()
        proc = self._run_ddomp()
        dt = time.perf_counter() - t0
        self.n_calls += 1
        self.wall_s += dt

        # THE LAST STEP'S OUTPUT, not always evl_0. With one step DDomp writes
        # evl_0 and this is what it always was; with the two steps a discrete
        # network requires (see adopt_network) the converged field is in the
        # highest-numbered file. Selecting by what is on disk rather than by an
        # assumed naming convention keeps this correct either way.
        written = sorted(
            (int(m.group(1)), p) for p in self.evl_dir.glob("evl_*.txt")
            if (m := re.fullmatch(r"evl_(\d+)\.txt", p.name)))
        if not written:
            raise QSSASolveError(
                f"DDomp wrote no evl_<n>.txt in {self.evl_dir}\n"
                f"stdout tail:\n{proc.stdout[-2000:]}")
        out = written[-1][1]
        self.last_evl = out
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

        # The PHYSICAL field c_FEM + c_DD, when MoDELib published it. C_M above
        # stays the corrective FEM field: it is what the slow step must freeze,
        # since the immobile equations are written against the same field the
        # sinks were assembled from. The superposed one is for figures and
        # diagnostics only, and it is None whenever there is no discrete
        # network -- in which case it would equal C_M anyway.
        self.last_superposed = mfield.read_superposed_mobile(
            self.evl_dir, self.n_nodes)

        if self.verbose:
            rng = "  ".join(f"{nm} {C_M[:, j].min():.2e}..{C_M[:, j].max():.2e}"
                            for j, nm in enumerate(("Cv", "Ci", "C2i", "C3i")))
            print(f"      QSSA solve {dt:6.1f} s   {rng}")
        return C_M

    # ── resident DDomp ("server mode") ───────────────────────────────────────
    #: Sentinel DDomp prints after each solve cycle. Must match DDomp.cpp.
    _DONE = "@@DDOMP_DONE@@"

    def _server_cycle(self, cmd, cwd, env):
        """One solve in a RESIDENT DDomp, starting it on first use.

        WHY. DDomp is a one-shot batch program the march calls 64 times per
        40 dpa run, and every invocation rebuilds state identical across all of
        them. Profiled at 189 533 CD nodes, of a 398.5 s invocation: the
        Cholesky factorization of the diffusion operator 175.6 s (44.1%),
        process startup 22.8 s, mesh 12.0 s, trial functions 2.1 s -- 53%
        repeated. The Cholesky alone exceeds the entire iterative solve and
        scales as N^1.86 against its N^1.33, so the waste grows with domain
        size. Keeping the process alive amortises all of it, and does so
        WITHOUT touching the numerics: `initializeConfiguration` re-reads only
        the CD fields and `initializeSolver` is guarded by `solverInitialized`,
        so every cycle solves the same problem it would have solved alone.

        stdout and stderr are merged deliberately. Two pipes would need two
        readers to stay drained, and a full stderr pipe deadlocks a process
        that is mid-solve with no one reading it.
        """
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                list(cmd) + ["--server"], cwd=cwd, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
                bufsize=1)
            self._server_started += 1
            return self._read_until_done(first=True)
        self._proc.stdin.write("SOLVE\n")
        self._proc.stdin.flush()
        return self._read_until_done()

    def _read_until_done(self, first=False):
        out = []
        while True:
            line = self._proc.stdout.readline()
            if not line:                      # EOF: the child died mid-solve
                rc = self._proc.poll()
                tail = "".join(out[-40:])
                self._proc = None
                raise QSSASolveError(
                    f"resident DDomp exited{'' if rc is None else f' ({rc})'} "
                    f"without finishing a solve.\noutput tail:\n{tail}")
            if line.startswith(self._DONE):
                return "".join(out)
            out.append(line)

    def close(self):
        """Stop the resident solver, if one is running. Safe to call twice."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _run_ddomp(self):
        cmd, cwd = paths.ddomp_cmd(self.sim_dir, exe=self.ddomp)
        if self.use_wsl != paths.use_wsl():
            # An explicit use_wsl= overrides the platform default; keep honoring
            # it rather than silently ignoring the argument.
            cmd = (["wsl.exe", "-e", paths.windows_to_wsl(self.ddomp),
                    paths.windows_to_wsl(self.sim_dir)] if self.use_wsl
                   else [str(self.ddomp), str(self.sim_dir)])
        # DDomp inherited the machine's default thread count, which on a
        # 2x20-core Xeon is 80 -- MEASURED as the WORST setting at every size:
        # 22.1 s against 17.0 s at 40 on the 200 nm case, and 510.8 s against
        # 476.0 s at 20 on the 1000 nm one. The mobile solve barely scales at
        # all (1.68x from 1 to 40 threads at 200 nm, 7% spread at 1000 nm),
        # because half of it is serial sparse-matrix construction and Eigen's
        # sparse mat-vec is not threaded; oversubscribing 80 threads onto the
        # 40 logical processors one Windows processor group exposes only makes
        # it worse. DISLOCLUSTER_DDOMP_THREADS overrides.
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS",
                       os.environ.get("DISLOCLUSTER_DDOMP_THREADS", "20"))
        if self.use_wsl:
            # WSL does not inherit the Windows environment; ask it to pass this
            # one through explicitly.
            env["WSLENV"] = ((env.get("WSLENV", "") + ":") if env.get("WSLENV")
                             else "") + "OMP_NUM_THREADS"
        if self.server:
            proc = _Result(0, self._server_cycle(cmd, cwd, env), "")
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  errors="replace", cwd=cwd, env=env)
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
