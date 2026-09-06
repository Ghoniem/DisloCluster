"""Step 0 of the code implementation plan: lock and hash a reference trajectory.

**What this is.** Every regression test in the plan (`app:plan` of the
self-consistent formulation) is a comparison against a recorded artifact of the
*present* model, so that artifact has to exist before anything changes and has to
be reproducible on a second machine. This module builds it, hashes it, and
re-runs it.

**What the artifact is a trajectory OF, precisely.** Not the coupled march --
that is 44 minutes and alternates the fast solve with the slow one, so it cannot
be a routine regression check. It is the **immobile operator alone**: for each
dose interval of the reference run, the reference state at a fixed set of nodes
is advanced from ``dose[n]`` to ``dose[n+1]`` with the mobile species frozen at
their reference values, through the same ``coupling.immobile.run_immobile_step``
the march calls. That is the operator steps 1-5 of the plan modify, and it is
what their "reproduces step N bit-for-bit" clauses are about.

Each interval starts from the **reference** state rather than from the previous
interval's endpoint. That is deliberate: it makes the eight intervals eight
independent probes, so a regression localizes to an interval instead of
contaminating every later one.

**Bit-for-bit is achievable, and that was not obvious.**
``run_immobile_step`` documents the OpenMP batch as *not* deterministic --
per-thread CVODE workspaces are reused across cases and dynamic scheduling
decides which case follows which. Measured here on 512 states over the
0.1 -> 1 dpa interval, two runs agreed **exactly**, every component of every
row. So the documented non-determinism governs *which marginal points fail*,
not the values of points that succeed. Failures are therefore recorded as
failures (``retries=0``, no tightening) rather than retried: a retry is the one
part of the path that genuinely is scheduling-dependent, and a baseline must not
contain it.

**The full solver CLI is stored, and verify checks it first.** All 104
parameters are resolved explicitly -- including ``dad_p_*``, which
``staging/anisotropy.py`` writes into the shared material file. A concurrent run
rewriting that file is a documented way for a diagnostic to silently measure
something else, so a moved parameter is reported as a moved parameter, by name,
before any trajectory comparison is attempted.

Usage
-----
    python -m dislocluster_code.studies.baseline lock [<run>] [--n 512]
    python -m dislocluster_code.studies.baseline verify [--baseline <path>]
    python -m dislocluster_code.studies.baseline show   [--baseline <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from dislocluster_code import paths, config as dcfg
from dislocluster_code.coupling import immobile as imm
from dislocluster_code.zerod.cpp_bridge import collect_solver_args

# The run named as the reference for the plan's implementation: the 500 nm
# hexagonal matched leg at loop_model = 1, whose interior means are the numbers
# the formulation quotes for the self-consistent formulation.
REFERENCE_RUN = "20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent"

#: Where the artifact lives. Beside the plan it serves, not in an output run --
#: it outlives any one run and is compared against by every later step.
BASELINE_PATH = paths.REPO_ROOT / "dislocluster_code" / "studies" / \
    "baseline_step0.npz"

#: Number of node states sampled from each reference snapshot.
N_SAMPLE = 512

#: Artifact format version. Bump when the LAYOUT changes; a bump invalidates
#: comparison against older artifacts, which is the point.
FORMAT = 1


# ─────────────────────────────────────────────────────────────────────────────
# Reference run resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_run(run=None):
    """The reference run directory.

    Accepts a full path, a bare run NAME, or None for the module default. The
    name form matters: `verify` reads `reference_run` back out of the manifest,
    which stores the name rather than the path so that an artifact locked on one
    machine resolves on another with a different output root.
    """
    name = REFERENCE_RUN if run is None else str(run)
    p = Path(name)
    if p.is_dir():
        return p
    for root in paths.OUTPUT_DIRS:
        q = Path(root) / p.name
        if q.is_dir():
            return q
    raise SystemExit(
        f"reference run {p.name!r} not found under any of:\n  "
        + "\n  ".join(str(r) for r in paths.OUTPUT_DIRS))


def sample_indices(n_nodes, n_sample):
    """A deterministic, host-independent spread of node indices.

    Pure integer arithmetic on purpose. Any RNG -- even a seeded one -- ties the
    artifact to a particular NumPy version's bit stream, and the plan's test for
    this step is that two *hosts* agree.
    """
    n_sample = min(int(n_sample), int(n_nodes))
    return np.unique(np.linspace(0, n_nodes - 1, n_sample).astype(np.int64))


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def _canon(a):
    """Canonical bytes for a float array: C-order, little-endian float64.

    Byte order is forced rather than inherited so the digest is the same on a
    big-endian host.
    """
    return np.ascontiguousarray(a, dtype="<f8").tobytes()


def trajectory_digest(traj, ok):
    """SHA-256 over the trajectory and its integration mask.

    The mask is part of the digest because "this point failed" is a result of
    the model, not an absence of one -- a change that makes a previously failing
    point integrate has changed the model and must not hash the same.
    """
    h = hashlib.sha256()
    h.update(b"dislocluster-baseline-v%d\n" % FORMAT)
    h.update(np.ascontiguousarray(ok, dtype=np.uint8).tobytes())
    h.update(_canon(np.nan_to_num(traj, nan=0.0)))
    return h.hexdigest()


def file_digest(path):
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# The trajectory itself
# ─────────────────────────────────────────────────────────────────────────────

def build_cli(run_dir, loop_model, variant_weights=(1 / 3, 1 / 3, 1 / 3)):
    """The slow step's CLI, built through the same path the march uses.

    `config.sim_for_run` rather than `calibration.build_sim()`: the latter takes
    the applied load from the workbook, where `sigma_n` stood at 100 MPa
    until 2026-09-04, and
    that sets `f_a` -- so a zero-stress run post-processed the other way is a
    different model. Measured: f_a = 0.4015 against the 1/3 the march ran with.
    """
    sim = dcfg.sim_for_run(run_dir)
    return collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=1e9, n_points=2, log_time=False,
        rtol=1e-6, atol=1e-20, stats=True,
        loop_model=int(loop_model),
        material_file=paths.MODELIB_MATERIAL,
        variant_weights=tuple(variant_weights)))


def run_trajectory(run_dir, n_sample=N_SAMPLE, verbose=True):
    """Advance the reference state one immobile step per dose interval.

    Returns ``(traj, ok, meta)`` with ``traj`` of shape
    ``(n_intervals, n_sample, 19)`` and ``ok`` the boolean integration mask.
    """
    run_dir = Path(run_dir)
    st = np.load(run_dir / "march_state.npz", allow_pickle=True)
    Y, doses = st["Y"], np.asarray(st["doses"], dtype=float)
    loop_model = int(st["loop_model"]) if "loop_model" in st.files else 0

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    G = float(cfg["material"]["dose_rate_dpa_s"])
    vw = tuple(cfg["coupling"].get(
        "variant_weights", (1 / 3, 1 / 3, 1 / 3)))

    idx = sample_indices(Y.shape[1], n_sample)
    cli = build_cli(run_dir, loop_model, vw)
    n_int = len(doses) - 1

    # Width comes from what the SOLVER returns, not from the reference run's
    # stored width. Step 1 of the plan appends eight family slots, so the two
    # differ from that step onward and sizing from the reference would truncate
    # -- or, as it first did, raise a broadcast error that says nothing about
    # what actually changed.
    traj = None
    ok = np.zeros((n_int, len(idx)), dtype=bool)

    if verbose:
        print(f"reference : {run_dir.name}")
        print(f"            Y{Y.shape}, loop_model={loop_model}, G={G:g} dpa/s")
        print(f"sampling  : {len(idx)} nodes, {n_int} dose intervals "
              f"-> {len(idx) * n_int} integrations")

    t_start = time.perf_counter()
    for n in range(n_int):
        t0, t1 = doses[n] / G, doses[n + 1] / G
        y0 = [Y[n, i, :].copy() for i in idx]
        # retries=0: a retry repacks the failures into a differently-scheduled
        # batch and (from attempt 2) tightens tolerances. Both are exactly the
        # scheduling-dependent part of the path, so neither belongs in an
        # artifact whose whole purpose is to be reproduced elsewhere.
        out = imm.run_immobile_step(cli, y0, t0, t1, retries=0)
        if traj is None:
            width = next((len(r) for r in out if r is not None), Y.shape[2])
            traj = np.full((n_int, len(idx), int(width)), np.nan)
        for i, r in enumerate(out):
            if r is not None:
                traj[n, i, :] = r
                ok[n, i] = True
        if verbose:
            print(f"  [{n + 1}/{n_int}] {doses[n]:g} -> {doses[n + 1]:g} dpa : "
                  f"{int(ok[n].sum())}/{len(idx)} integrated", flush=True)
    wall = time.perf_counter() - t_start

    meta = dict(
        format=FORMAT,
        reference_run=run_dir.name,
        loop_model=loop_model,
        dose_rate_dpa_s=G,
        doses=[float(x) for x in doses],
        node_indices=[int(i) for i in idx],
        n_nodes_total=int(Y.shape[1]),
        n_state=int(Y.shape[2]),
        variant_weights=[float(x) for x in vw],
        solver_cli=list(cli),
        material_file=str(paths.MODELIB_MATERIAL),
        material_sha256=file_digest(paths.MODELIB_MATERIAL),
        reference_Y_sha256=hashlib.sha256(_canon(Y)).hexdigest(),
        git_hash=paths.git_hash(),
        wall_s=round(wall, 3),
        # Recorded, never compared: the plan's test is that two HOSTS agree, so
        # a host difference must not be reported as a regression.
        host=dict(platform=platform.platform(), python=sys.version.split()[0],
                  numpy=np.__version__),
    )
    return traj, ok, meta


# ─────────────────────────────────────────────────────────────────────────────
# lock / verify / show
# ─────────────────────────────────────────────────────────────────────────────

def lock(run=None, n_sample=N_SAMPLE, out=BASELINE_PATH, verbose=True):
    run_dir = resolve_run(run)
    traj, ok, meta = run_trajectory(run_dir, n_sample, verbose=verbose)
    meta["trajectory_sha256"] = trajectory_digest(traj, ok)
    meta["n_failed"] = int((~ok).sum())

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, traj=traj, ok=ok,
                        manifest=np.array(json.dumps(meta, indent=2)))
    if verbose:
        print(f"\nbaseline written : {out}")
        print(f"  trajectory sha256 : {meta['trajectory_sha256']}")
        print(f"  reference Y sha256: {meta['reference_Y_sha256']}")
        print(f"  material  sha256  : {meta['material_sha256']}")
        print(f"  failed integrations: {meta['n_failed']} of {ok.size}")
        print(f"  wall              : {meta['wall_s']:.1f} s")
    return meta


def load(path=BASELINE_PATH):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"no baseline at {p} -- run `baseline lock` first")
    d = np.load(p, allow_pickle=True)
    return d["traj"], d["ok"], json.loads(str(d["manifest"]))


def verify(path=BASELINE_PATH, run=None, verbose=True):
    """Re-run the baseline and compare bit-for-bit. Returns True on a match."""
    traj0, ok0, meta = load(path)
    run_dir = resolve_run(run or meta.get("reference_run"))

    # ── the CLI first ───────────────────────────────────────────────────────
    # A moved parameter explains a moved trajectory, and reporting it the other
    # way round -- "the model changed" when in fact the material file was
    # rewritten underneath -- is the failure this ordering exists to prevent.
    cli_now = build_cli(run_dir, meta["loop_model"],
                        tuple(meta["variant_weights"]))
    moved = _cli_diff(meta["solver_cli"], cli_now)
    if moved:
        print("SOLVER PARAMETERS MOVED since the baseline was locked:")
        for k, a, b in moved:
            print(f"  {k}: {a}  ->  {b}")
        mat_now = file_digest(paths.MODELIB_MATERIAL)
        if mat_now != meta.get("material_sha256"):
            print(f"  (the material file itself changed: "
                  f"{meta.get('material_sha256', '?')[:12]} -> "
                  f"{(mat_now or '?')[:12]})")
        print("\nThe trajectory below is NOT a like-for-like comparison.")

    traj1, ok1, _ = run_trajectory(run_dir, len(meta["node_indices"]),
                                   verbose=verbose)
    digest = trajectory_digest(traj1, ok1)
    match = digest == meta["trajectory_sha256"]

    print()
    print(f"baseline digest : {meta['trajectory_sha256']}")
    print(f"re-run   digest : {digest}")
    print(f"BIT-FOR-BIT     : {'MATCH' if match else 'DIFFER'}")

    if not match:
        _report_difference(traj0, ok0, traj1, ok1)
    return match and not moved


def _cli_diff(old, new):
    def as_dict(cli):
        d = {}
        for a in cli:
            if a.startswith("--") and "=" in a:
                k, v = a[2:].split("=", 1)
                d[k] = v
        return d

    a, b = as_dict(old), as_dict(new)
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append((k, a.get(k, "<absent>"), b.get(k, "<absent>")))
    return out


def _report_difference(t0, ok0, t1, ok1):
    if t0.shape[:-1] != t1.shape[:-1]:
        print(f"  shape changed: {t0.shape} -> {t1.shape}")
        return
    if t0.shape[-1] != t1.shape[-1]:
        # A state that GREW is the expected signature of a plan step that adds
        # families. The question that matters is then not "did the digest move"
        # -- it must have -- but whether the pre-existing components are
        # untouched and the new ones are inert. Answer both, in that order.
        n0, n1 = t0.shape[-1], t1.shape[-1]
        print(f"  state width  : {n0} -> {n1}  "
              f"({'grew' if n1 > n0 else 'shrank'} by {abs(n1 - n0)})")
        k = min(n0, n1)
        both = ok0 & ok1
        shared_same = np.array_equal(t0[..., :k][both], t1[..., :k][both])
        print(f"  components 0..{k - 1} bit-for-bit: "
              f"{'MATCH' if shared_same else 'DIFFER'}")
        if not shared_same:
            _component_report(t0[..., :k], t1[..., :k], both)
        if n1 > n0:
            new = t1[..., k:][both]
            allzero = bool(np.all(new == 0.0))
            print(f"  components {k}..{n1 - 1} identically zero: {allzero}"
                  + ("" if allzero else
                     f"  (max |.| = {np.abs(new).max():.3e})"))
        return
    flipped = ok0 != ok1
    if flipped.any():
        print(f"  integration mask changed at {int(flipped.sum())} point(s): "
              f"{int((ok1 & ~ok0).sum())} newly integrated, "
              f"{int((ok0 & ~ok1).sum())} newly failed")
    both = ok0 & ok1
    if not both.any():
        return
    d = np.abs(t0 - t1)
    rel = np.where(d > 0, d / np.maximum(np.abs(t0), 1e-300), 0.0)
    rel[~both] = 0.0
    print(f"  differing rows : {int((d[both] > 0).any(-1).sum())} "
          f"of {int(both.sum())}")
    print(f"  max abs diff   : {d[both].max():.6e}")
    print(f"  max rel diff   : {rel.max():.6e}")
    _component_report(t0, t1, both)


def _component_report(t0, t1, both):
    """Which state components moved -- usually the whole diagnosis.

    Each plan step touches a known set of components, so naming them says
    immediately whether the change was the intended one.
    """
    d = np.abs(t0 - t1)
    rel = np.where(d > 0, d / np.maximum(np.abs(t0), 1e-300), 0.0)
    rel[~both] = 0.0
    per = rel.reshape(-1, rel.shape[-1]).max(0)
    hit = [(i, per[i]) for i in range(len(per)) if per[i] > 0]
    if hit:
        print("  components moved (index: max rel):")
        for i, v in sorted(hit, key=lambda x: -x[1])[:8]:
            print(f"    y[{i:2d}]: {v:.3e}")


def selftest(path=BASELINE_PATH, run=None):
    """Prove the harness has teeth: a deliberate perturbation must be caught.

    A regression test that cannot fail is worse than none, because it reports
    success either way. This integrates the same states with ``rtol`` loosened by
    one decade -- a change small enough to leave every physical conclusion intact
    and large enough that the digest must move -- and fails if it does not.
    """
    _, _, meta = load(path)
    run_dir = resolve_run(run or meta.get("reference_run"))
    st = np.load(run_dir / "march_state.npz", allow_pickle=True)
    Y = st["Y"]
    doses = np.asarray(st["doses"], dtype=float)
    G = float(meta["dose_rate_dpa_s"])
    idx = np.asarray(meta["node_indices"], dtype=np.int64)

    cli = [a if not a.startswith("--rtol=") else "--rtol=1e-5"
           for a in meta["solver_cli"]]
    assert "--rtol=1e-5" in cli, "rtol not present in the recorded CLI"

    # Width from the solver, not from the reference run -- the same reason
    # run_trajectory does it that way.
    traj = None
    ok = np.zeros((len(doses) - 1, len(idx)), dtype=bool)
    for n in range(len(doses) - 1):
        out = imm.run_immobile_step(
            cli, [Y[n, i, :].copy() for i in idx],
            doses[n] / G, doses[n + 1] / G, retries=0)
        if traj is None:
            width = next((len(r) for r in out if r is not None), Y.shape[2])
            traj = np.full((len(doses) - 1, len(idx), int(width)), np.nan)
        for i, r in enumerate(out):
            if r is not None:
                traj[n, i, :] = r
                ok[n, i] = True

    digest = trajectory_digest(traj, ok)
    caught = digest != meta["trajectory_sha256"]
    print(f"baseline digest      : {meta['trajectory_sha256']}")
    print(f"rtol 1e-6 -> 1e-5    : {digest}")
    print(f"perturbation CAUGHT  : {caught}")
    if not caught:
        print("\nFAIL: the harness did not notice a loosened tolerance, so it "
              "cannot be relied on to notice a model change either.")
    return caught


def show(path=BASELINE_PATH):
    traj, ok, meta = load(path)
    print(f"baseline : {Path(path)}")
    for k in ("format", "reference_run", "loop_model", "dose_rate_dpa_s",
              "git_hash", "wall_s", "n_failed", "trajectory_sha256",
              "reference_Y_sha256", "material_sha256"):
        if k in meta:
            print(f"  {k:20s} {meta[k]}")
    print(f"  {'doses':20s} {meta['doses']}")
    print(f"  {'shape':20s} {traj.shape}  ok={int(ok.sum())}/{ok.size}")
    print(f"  {'host (recorded)':20s} {meta['host']}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Step 0: lock and hash a reference trajectory.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lock", help="build and hash the baseline artifact")
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("--n", type=int, default=N_SAMPLE)
    p.add_argument("--out", default=str(BASELINE_PATH))

    p = sub.add_parser("verify", help="re-run and compare bit-for-bit")
    p.add_argument("--baseline", default=str(BASELINE_PATH))
    p.add_argument("--run", default=None)

    p = sub.add_parser("selftest",
                       help="prove a deliberate perturbation is caught")
    p.add_argument("--baseline", default=str(BASELINE_PATH))
    p.add_argument("--run", default=None)

    p = sub.add_parser("show", help="print the manifest")
    p.add_argument("--baseline", default=str(BASELINE_PATH))

    a = ap.parse_args(argv)
    if a.cmd == "lock":
        lock(a.run, a.n, a.out)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.baseline, a.run) else 1)
    elif a.cmd == "selftest":
        sys.exit(0 if selftest(a.baseline, a.run) else 1)
    elif a.cmd == "show":
        show(a.baseline)


if __name__ == "__main__":
    main()
