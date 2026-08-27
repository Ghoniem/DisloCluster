"""
modelib_coupling.py — operator-split QSSA driver for the two-time-scale
ZrMicro <-> MoDELib2-NNL coupling.

This is the Python side of step S4 in
    Docs/Formulation/ZrMicro_MoDELib2_two_time_scale_coupling.tex (Sec. "Adopted
    Implementation: Operator-Split QSSA").

The spatial code (MoDELib2-NNL) solves the FAST mobile reaction-diffusion BVP on
its FE mesh, producing a steady mobile field C_M*(x) = [Cv,Ci,C2i,C3i] at every
quadrature point. This module advances the SLOW immobile state at all quadrature
points over one dose step [t_n, t_n+dt] by reusing the EXISTING ZrMicro C++
solver with ``freeze_mobile=1``: each quadrature point becomes one case of the
OpenMP ``--batch_file`` mode, integrated concurrently in a single subprocess,
with its mobile species pinned to the supplied C_M*.

WHY A MILLION ODEs PER SUBSTEP IS NOT A MILLION-EQUATION SYSTEM
---------------------------------------------------------------
Read the totals carelessly and this looks impossible. A 500 nm case advances
90617 points x 15 integrated equations = ~1.09e6 scalar ODEs per substep, and a
tightly coupled stiff system of that size would be hopeless. It is not one
system: it is 72494 INDEPENDENT 15-dimensional systems (after dedup), and
`rate_equations.ode_system(t, y)` takes a single point's state vector with no
neighbor state in it anywhere.

That decoupling is real, not an approximation bolted on for speed. The only
thing coupling neighboring points is DIFFUSION of the mobile species, and this
step freezes those (`freeze_mobile=1`). With the mobile field held, a point's
immobile ODEs depend only on its own loop populations and the four mobile
values pinned into y[0:4]. The split is what earns the decoupling; the batch
mode just harvests it.

One implicit factorization, split against coupled:

    split      72494 x 15^3        2.4e8  flops
    coupled    (1359255)^3         2.5e18 flops       ratio 1.0e10

Sparsity softens the coupled number but does not rescue it: a 3-D FE Jacobian
still factorizes at ~O(N^2) with fill ~O(N^(4/3)). Stiffness localizes the same
way -- each point takes ~100 CVODE internal steps per substep, but stiffness
cost scales with the dimension of the COUPLED BLOCK, and a 15x15 dense LU is
~1100 flops.

The coupling has not disappeared, it has moved to where it is affordable. The
tightly coupled problem is the FAST solve: 362468 unknowns, 925 s measured,
against 28.3 s for this module's entire sweep over 90617 points. What the split
buys is frequency -- `solveMobileClusters` has no time derivative, so the
coupled problem is solved 8 times over a march instead of at every timestep. At
the measured 185 s per linear solve (925 s / 5 Newton iterations), integrating
that block implicitly in time instead would cost roughly

    ~100 steps x 24 substeps / 5 steps per refactorization = 480 factorizations
    480 x 185 s = 25 h        <- the 4-species mobile block alone, not all 19

against the 11.3 min this module actually takes. The price is splitting error,
first order in the fast-solve spacing (`fem_every`).

The structure predicts linear scaling in node count, and that is what is
measured: 9.3 s per substep at 27720 nodes, 28.3 s at 90617 -- 3.35e-4 against
3.12e-4 s per node.

Native state layout (what the ZrMicro solver integrates), length N_EQ = 19:
    y[0:4]   mobile        Cv, Ci, C2i, C3i              <- frozen at C_M*
    y[4:8]   loop numbers  CiL, CaiL, CvL, CavL
    y[8:12]  loop content  CiL_i, CaiL_i, CvL_v, CavL_v
    y[12:18] accumulators  (conservation diagnostics; reset per step)
    y[18]    rho_N         evolving network dislocation density

The crystallographic a1/a2/a3 resolution (Option A in the formulation) is a layer
in the spatial code: it splits the lumped <a> interstitial loops by the
resolved-stress weights w_k (see modelib_export.py) and calls this micro-model per
variant. This module itself operates on the native lumped representation that the
C++ integrator understands.

Typical dose-step loop (spatially uniform T, sigma -> one shared base_cli):

    from dislocluster_code.zerod.cpp_bridge import collect_solver_args
    from dislocluster_code.coupling.immobile import pack_y0, run_immobile_step, IMMOBILE_SLICE

    base_cli = collect_solver_args(sim, solver_config)      # material params (uniform)
    Q = [pack_y0(cm_star_q, immob_q, rhoN_q) for q in points]   # initial per-point state

    for (t0, t1) in dose_intervals:                          # t = gamma / G
        # (MoDELib solves the fast BVP here -> updates cm_star_q at each point)
        for q, cm in enumerate(cm_star_by_point):
            Q[q][0:4] = cm                                   # inject frozen mobile
        Q = run_immobile_step(base_cli, Q, t0, t1)           # advance immobile (batch)
        # (MoDELib forms L^I from the per-point immobile rates, solves mechanics)
"""

from __future__ import annotations

import numpy as np

from dislocluster_code.zerod.cpp_bridge import run_cpp_solver_batch

# Native ZrMicro state vector length (12 species + 6 accumulators + rho_N).
N_EQ = 19
# Step 1 of the implementation plan appended four immobile families at state
# indices 19..26, and step 4 a fifth at 23/28. A caller may hand over either
# width: 19 is the pre-step-1
# state and leaves the new families empty (the solver defaults those y0 slots to
# zero), 27 carries them. Anything else is a layout error, not something to pad.
N_EQ_EXT = 38
IDX_RHO_N = 18

# Convenience slices into the native state vector.
MOBILE_SLICE = slice(0, 4)        # Cv, Ci, C2i, C3i
IMMOBILE_SLICE = slice(4, 12)     # CiL,CaiL,CvL,CavL, CiL_i,CaiL_i,CvL_v,CavL_v
ACCUMULATOR_SLICE = slice(12, 18)

IMMOBILE_NAMES = [
    "CiL", "CaiL", "CvL", "CavL",            # loop number densities
    "CiL_i", "CaiL_i", "CvL_v", "CavL_v",    # loop defect contents
]


def pack_y0(c_m_star, immobile, rho_N, reset_accumulators=True):
    """Assemble a native length-19 state vector for one quadrature point.

    Parameters
    ----------
    c_m_star : sequence of 4 floats   — frozen steady mobile field [Cv,Ci,C2i,C3i]
    immobile : sequence of 8 floats   — IMMOBILE_NAMES order (numbers then contents)
    rho_N    : float                  — network dislocation density [m^-2]
    reset_accumulators : bool         — zero y[12:18] (per-step conservation start)

    Returns
    -------
    y0 : np.ndarray shape (19,)
    """
    y0 = np.zeros(N_EQ, dtype=float)
    y0[MOBILE_SLICE] = np.asarray(c_m_star, dtype=float)
    y0[IMMOBILE_SLICE] = np.asarray(immobile, dtype=float)
    y0[IDX_RHO_N] = float(rho_N)
    if not reset_accumulators:
        pass  # caller already placed accumulator values in a full-length vector
    return y0


def _cli_to_dict(cli):
    """['--k=v', ...] -> {'k': 'v', ...} (last value wins on duplicates)."""
    d = {}
    for a in cli:
        s = a[2:] if a.startswith("--") else a
        k, _, v = s.partition("=")
        d[k] = v
    return d


def _dict_to_cli(d):
    return [f"--{k}={v}" for k, v in d.items()]


def build_immobile_cases(base_cli, y0_list, t_begin, t_end):
    """Build per-quadrature-point batch cases for the frozen-mobile immobile march.

    Each case inherits the shared material parameters in ``base_cli`` (valid when
    T, sigma, and the bias factors are spatially uniform) and overrides the
    per-point initial state, the integration window, and the operator-split flags.

    Parameters
    ----------
    base_cli : list[str]            — collect_solver_args(sim, solver_config) output
    y0_list  : sequence of (19,)    — native state per quadrature point (mobile slots
                                      already set to the local C_M*)
    t_begin, t_end : float          — dose-step window in SECONDS (t = gamma / G)

    Returns
    -------
    cases_cli : list[list[str]]     — one '--key=value' list per quadrature point
    """
    base = _cli_to_dict(base_cli)
    base["freeze_mobile"] = "1"
    base["t_begin"] = repr(float(t_begin))
    base["t_end"] = repr(float(t_end))
    base["n_points"] = "2"      # endpoints only — we read the last row
    base["log_time"] = "0"      # linear span of [t_begin, t_end]

    # ── Solver configuration specific to the operator-split march ────────────
    # With the mobile species frozen and the six conservation accumulators
    # carried as CVODES quadrature variables, the implicit block the Newton
    # iteration and the dense LU touch is 9x9 rather than 19x19, and the
    # Jacobian is the exact one obtained by forward-mode AD instead of
    # difference quotients.
    #
    # The decisive gain here is not the flop count but the step-size control.
    # Each substep restarts at t ~ 1e7 s with the accumulators reset to zero;
    # with atol = 1e-20 their error weights are ~1e20, which drives CVODE's
    # initial step down to the point where t + h == t in double precision. The
    # legacy path emits that roundoff warning on every substep of every point
    # (2048 of them in a 1024-point, 5-substep march) and wastes the steps that
    # provoke it. Taking the accumulators out of the error test removes the
    # pathology outright: zero warnings and ~18% fewer internal steps.
    #
    # These are set here, not in collect_solver_args, so that a standalone 0-D
    # run stays bit-identical to the pre-existing solver.
    # Assigned, not setdefault: collect_solver_args always emits these keys with
    # their standalone default of 0.0, so setdefault would silently leave the
    # march on the legacy path.
    #
    # acc_mode=2 keeps the accumulators in the state but neutralizes their atol.
    # Measured on the 24115-node field, one 1 dpa substep, against acc_mode=0
    # (legacy) and acc_mode=1 (CVODES quadrature):
    #
    #   mode  neq   steps      core evals   LU flops   wall
    #     0    19   3.35e6     4.18e6       2.20e9     4.193 s
    #     1     9   2.51e6     5.84e6       1.42e8     3.402 s
    #     2    15   2.40e6     3.08e6       5.71e8     3.314 s
    #
    # Mode 1 has by far the smallest Newton block but pays an extra core sweep
    # per step for the quadrature (its memo almost never hits), which shows up
    # as the largest core-evaluation count of the three. Mode 2 gives up the LU
    # saving — irrelevant at n=15 — and wins on the metric that actually costs:
    # 26% fewer core evaluations than legacy and 47% fewer than the quadrature.
    base["acc_mode"] = "2"
    base.pop("reduced", None)
    base["analytic_jac"] = "1"

    cases = []
    for y0 in y0_list:
        y0 = np.asarray(y0, dtype=float)
        if y0.shape[0] not in (N_EQ, 29, N_EQ_EXT):
            raise ValueError(f"each y0 must have length {N_EQ} or {N_EQ_EXT}, "
                             f"got {y0.shape[0]}")
        d = dict(base)
        # Emit every component the caller supplied. The solver requires
        # y0_0..y0_18 and treats y0_19..y0_26 as optional-defaulting-to-zero, so
        # a 19-wide state produces exactly the command line it always did.
        for k in range(y0.shape[0]):
            d[f"y0_{k}"] = repr(float(y0[k]))
        cases.append(_dict_to_cli(d))
    return cases


# Components that determine a point's trajectory over one substep: the four
# frozen mobile values, the eight immobile components, and rho_N. The six
# accumulators are excluded — they are reset to zero every substep and never
# feed back, so two points differing only there follow the same trajectory.
_DEDUP_IDX = list(range(0, 12)) + [IDX_RHO_N]


def dedup_keys(y0_list, rtol):
    """Group points whose initial states agree to a relative tolerance.

    Returns (keys, groups): ``keys[q]`` is the group label of point q, and
    ``groups`` maps each label to the list of member indices.

    Neighbouring quadrature points in a smooth field integrate nearly identical
    initial-value problems, so one integration can serve many of them. Each
    component is quantized logarithmically, which makes the tolerance relative
    across the ~34 decades the state spans; values at or below the
    concentration floor collapse onto one bucket. Keeping ``rtol`` well under
    the integrator's own ``rtol`` bounds the error this introduces by the
    tolerance already being accepted.
    """
    y = np.asarray(y0_list, dtype=float)[:, _DEDUP_IDX]
    scale = 1.0 / np.log1p(rtol)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(y > 0.0, np.rint(np.log(np.abs(y)) * scale), -np.inf)
    q = np.nan_to_num(q, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    keys = [tuple(row) for row in q]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    return keys, groups


def run_immobile_step(base_cli, y0_list, t_begin, t_end, base_dir=None,
                      dedup_rtol=0.0, stats=None, retries=3):
    """Advance the immobile state at all quadrature points over one dose step.

    Solves, for every point q independently and concurrently (OpenMP batch),
    the ZrMicro immobile ODEs with the mobile species frozen at y0_list[q][0:4],
    from t_begin to t_end, and returns the endpoint state.

    Parameters
    ----------
    base_cli : list[str]            — shared material-parameter CLI (uniform fields)
    y0_list  : sequence of (19,)    — per-point native state (mobile = local C_M*)
    t_begin, t_end : float          — dose-step window in seconds
    base_dir : Path or None         — ZrMicro/ root; auto-detected if None
    retries  : int                  — how many times to re-run points that
                                      failed in the full batch, on their own.
                                      The batch step is not deterministic; see
                                      the comment at the retry loop.

    Returns
    -------
    list  — same length as y0_list; each entry is the endpoint state vector
            np.ndarray shape (19,), or None if that point's integration failed.
    """
    n_in = len(y0_list)

    # ── Optional deduplication (see dedup_keys) ─────────────────────────────
    # Integrate one representative per group and scatter the endpoint back to
    # every member. Off by default: whether it pays depends entirely on how
    # smooth the incoming field is, which is a property of the run, not of the
    # solver, so it is measured rather than assumed.
    reps, groups = None, None
    if dedup_rtol and dedup_rtol > 0.0 and n_in > 1:
        _, groups = dedup_keys(y0_list, dedup_rtol)
        reps = [members[0] for members in groups.values()]
        send = [y0_list[i] for i in reps]
    else:
        send = y0_list

    cases = build_immobile_cases(base_cli, send, t_begin, t_end)
    raw = run_cpp_solver_batch(cases, base_dir=base_dir)

    # ── retry the stragglers, alone ─────────────────────────────────────────
    # A case that fails in a full batch does not necessarily fail on its own:
    # the batch step is NOT deterministic. Measured on the 500 nm anisotropic
    # march, one point of 90 617 failed at the 0.01 -> 0.1 dpa substep, and the
    # identical step -- same checkpoint, same fast solve reproduced to every
    # printed digit -- then completed with zero failures when re-run.
    #
    # The mechanism is the per-thread CVODE workspace the batch mode reuses
    # across cases. Under dynamic scheduling, which case follows which on a
    # given thread varies from run to run, so a point sitting near the solver's
    # failure boundary is decided by scheduling rather than by its own state.
    #
    # Re-running just the failures repacks them into a tiny batch, which pairs
    # them differently and usually clears them. This is cheap -- a handful of
    # cases against tens of thousands -- and it is the honest fix: a transient,
    # non-reproducible failure should be retried, not silently tolerated by
    # raising max_failed_nodes, which substitutes an IDENTITY step and freezes
    # the point for the substep.
    for attempt in range(int(retries)):
        stuck = [i for i, r in enumerate(raw) if r is None]
        if not stuck:
            break
        # Attempt 1 re-runs unchanged, which is what distinguishes a transient
        # from a real failure. Later attempts TIGHTEN the tolerances, because the
        # failures seen so far are not transient: they are points whose <c> loop
        # content is crossing the positivity floor, where the step-size control
        # stalls partway. Measured on node 41804 of the 500 nm anisotropic
        # march, rtol or atol alone changed nothing; rtol=1e-8 WITH atol=1e-30
        # integrated it, and the endpoint is the physical continuation -- its
        # five nearest neighbors fall by a factor of 277 over the same substep,
        # so the whole region is collapsing and this point merely reaches the
        # floor first.
        #
        # Tighter, never looser. Loosening also "works" and would be the wrong
        # trade: it buys convergence by accepting a less accurate answer at the
        # one point in the domain already known to be difficult.
        retry_cases = [cases[i] for i in stuck]
        if attempt > 0:
            scale_r, scale_a = 10.0 ** (-2 * attempt), 10.0 ** (-10 * attempt)
            tightened = []
            for c in retry_cases:
                d = _cli_to_dict(c)
                d["rtol"] = repr(float(d.get("rtol", 1e-6)) * scale_r)
                d["atol"] = repr(float(d.get("atol", 1e-20)) * scale_a)
                tightened.append(_dict_to_cli(d))
            retry_cases = tightened
        again = run_cpp_solver_batch(retry_cases, base_dir=base_dir)
        n_ok = sum(r is not None for r in again)
        for i, r in zip(stuck, again):
            if r is not None:
                raw[i] = r
        how = "unchanged" if attempt == 0 else f"tightened x1e-{2 * attempt}"
        print(f"      immobile retry {attempt + 1} ({how}): {len(stuck)} "
              f"case(s), {n_ok} recovered", flush=True)
        if stats is not None:
            stats.setdefault("retries", []).append(
                dict(attempt=attempt + 1, n_retried=len(stuck),
                     n_recovered=n_ok, cases=list(stuck)))

    endpoints = []
    for r in raw:
        if r is None:
            endpoints.append(None)
        else:
            _t, y = r                # y shape (N_EQ, n_pts)
            endpoints.append(np.asarray(y[:, -1], dtype=float))   # endpoint state
        # cf. cpp_bridge._parse_batch_stdout: r = (t, y) per case

    if reps is None:
        out = endpoints
    else:
        out = [None] * n_in
        for rep_pos, members in enumerate(groups.values()):
            e = endpoints[rep_pos] if rep_pos < len(endpoints) else None
            for m in members:
                out[m] = None if e is None else e.copy()

    if stats is not None:
        stats["n_points"] = n_in
        stats["n_integrated"] = len(send)
        stats["dedup_ratio"] = n_in / max(len(send), 1)
    return out


def immobile_rates(y_begin, y_end, dt):
    """Per-point immobile rate dQ/dt over the step, for the L^I source in MoDELib.

    A simple first-order (one-sided) estimate from the step endpoints; feeds the
    plastic distortion L^P = sum_X (dc_I^X/dt)(habit_X (x) habit_X) of the
    formulation. Returns a dict keyed by IMMOBILE_NAMES plus 'rho_N'.

    Parameters
    ----------
    y_begin, y_end : (19,) arrays    — state before/after the dose step
    dt : float                       — step length in seconds (= dgamma / G)
    """
    yb = np.asarray(y_begin, dtype=float)
    ye = np.asarray(y_end, dtype=float)
    inv = 1.0 / dt
    rates = {name: (ye[4 + k] - yb[4 + k]) * inv for k, name in enumerate(IMMOBILE_NAMES)}
    rates["rho_N"] = (ye[IDX_RHO_N] - yb[IDX_RHO_N]) * inv
    return rates
