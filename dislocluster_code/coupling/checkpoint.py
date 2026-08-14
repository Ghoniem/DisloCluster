"""checkpoint.py — make a multi-hour march survive an interruption.

WHAT HAS TO BE SAVED
--------------------
``Y`` (N x 19) is the whole answer and cannot be recovered from anything else
on disk. The snapshots the march writes into ``evl_out/`` carry only 12 of the
19 columns -- the six conservation accumulators and ``rho_N`` are not in the CD
block at all -- and ``modelib_immobile_to_0d`` splits the aligned/non-aligned
pair 50/50 when it has no previous state to go on, so four more columns come
back wrong. Text evl files are also written at ``%.15e``, sixteen significant
digits where float64 needs seventeen; since ``dedup_keys`` buckets on
``rint(log(y)*scale)``, a last-bit change can move a node into a different
deduplication group and alter the trajectory. So: raw float64, nothing else.

Alongside ``Y`` the counters matter. ``k_global`` drives the fast-solve cadence
``k_global % fem_every``; restoring it wrong shifts every remaining fast solve.
``used_steps`` decides whether a snapshot is filed as ``evl_4.txt`` or
``evl_s02.txt``, and the comparison looks for the former by name.

WHAT IS NOT SAVED, BECAUSE IT IS CHEAPER TO REBUILD
---------------------------------------------------
``FieldBridge`` is stateless. ``MobileQSSASolver`` holds no warm-start state
either -- ``solve()`` regenerates ``evl/evl_0.txt`` from the stored seed and
``Y`` on every call, so the warm start IS ``Y[:, 0:4]``, and the fast-solve
directory's ``evl_0.txt`` is a scratch buffer rather than state. That single
fact is what makes resume tractable here. ``base_cli``, ``G``, ``edges``,
``Y_seed`` and the loop indices are all pure functions of things already held.

WHERE THE CHECKPOINT IS WRITTEN
-------------------------------
Immediately after ``k_global`` is incremented, which is the only consistent
point in the substep body: the fast solve is either fully applied or was not
scheduled, and ``Y`` has just been rebound wholesale. Anywhere between the fast
solve and the immobile step would store a half-applied substep.

ATOMICITY ON WINDOWS
--------------------
``os.replace``, never ``Path.rename`` -- the latter raises ``FileExistsError``
on Windows when the destination exists, which is the classic Windows-only
checkpoint bug. The live file is never memory-mapped, because a mapping keeps a
handle open and makes the next replace fail with ``PermissionError``; Defender
and OneDrive can do the same, so the replace is retried with backoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

__all__ = ["SCHEMA_VERSION", "CheckpointMismatch", "Checkpointer",
           "file_digest", "array_digest"]

SCHEMA_VERSION = 1


class CheckpointMismatch(RuntimeError):
    """The checkpoint on disk was not written by this configuration.

    Raised rather than silently starting over: quietly discarding an
    eight-hour march because a hash moved is worse than stopping.
    """


def array_digest(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(),
                           digest_size=16).hexdigest()


def file_digest(p, missing_ok=False):
    p = Path(p)
    if not p.is_file():
        if missing_ok:
            return None
        raise FileNotFoundError(p)
    h = hashlib.blake2b(digest_size=16)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_savez(path, arrays, meta, retries=5):
    """Write `arrays` + JSON `meta` into one npz, replacing `path` atomically.

    The metadata goes INSIDE the npz as UTF-8 bytes so there is exactly one
    file to swap and no `allow_pickle` anywhere.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    prev = path.with_suffix(path.suffix + ".prev")

    payload = dict(arrays)
    payload["_meta"] = np.frombuffer(
        json.dumps(meta, default=str).encode("utf-8"), dtype=np.uint8)
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
        fh.flush()
        os.fsync(fh.fileno())

    delay = 0.05
    for attempt in range(retries):
        try:
            if path.exists():
                os.replace(path, prev)
            os.replace(tmp, path)
            return path
        except PermissionError:
            # A virus scanner, a sync client or a stray np.load(mmap_mode=...)
            # is holding the destination. Back off rather than lose the step.
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return path


def _load_npz(path):
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["_meta"]).decode("utf-8"))
            arrays = {k: z[k] for k in z.files if k != "_meta"}
        return arrays, meta
    except (OSError, ValueError, KeyError, EOFError, UnicodeDecodeError,
            zipfile.BadZipFile, json.JSONDecodeError):
        # A torn write. npz is a zip, and a truncated one raises BadZipFile,
        # which is a plain Exception rather than an OSError -- catching only
        # the obvious errors here let a half-written checkpoint crash the
        # resume instead of falling back to .prev.
        return None, None


class Checkpointer:
    """Rolling checkpoint for one march.

    ``ckpt.npz`` is rewritten every substep (~5 MB at 32 k nodes, tens of
    milliseconds, against a fast solve measured at ~245 s). ``hist.npz`` holds
    the per-snapshot history and only changes at an interval boundary, so it is
    written there rather than paying ~29 MB every substep.
    """

    CKPT = "ckpt.npz"
    HIST = "hist.npz"

    def __init__(self, directory, fingerprint, enabled=True):
        self.dir = Path(directory)
        self.fingerprint = dict(fingerprint)
        self.enabled = bool(enabled)
        self.n_writes = 0
        self.write_s = 0.0
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ── writing ──────────────────────────────────────────────────────────────

    def save(self, Y, state, history=None):
        """Persist one substep. `history` triggers the interval-scale write."""
        if not self.enabled:
            return
        t0 = time.perf_counter()
        meta = dict(schema_version=SCHEMA_VERSION,
                    fingerprint=self.fingerprint,
                    state=state,
                    Y_sha=array_digest(Y),
                    written=time.strftime("%Y-%m-%d %H:%M:%S"))
        _atomic_savez(self.dir / self.CKPT, {"Y": np.ascontiguousarray(Y)}, meta)
        if history:
            doses = sorted(history)
            _atomic_savez(
                self.dir / self.HIST,
                {"doses": np.asarray(doses, dtype=float),
                 "Y": np.asarray([history[d] for d in doses])},
                dict(schema_version=SCHEMA_VERSION,
                     fingerprint=self.fingerprint, n=len(doses)))
        self.n_writes += 1
        self.write_s += time.perf_counter() - t0

    # ── reading ──────────────────────────────────────────────────────────────

    def exists(self):
        return (self.dir / self.CKPT).is_file()

    def load(self, strict=True):
        """Return ``(Y, state, history)``, or ``(None, None, None)``.

        Falls back to the previous generation when the live file is torn, which
        costs at most one substep of rework.
        """
        for name in (self.CKPT, self.CKPT + ".prev"):
            arrays, meta = _load_npz(self.dir / name)
            if arrays is None:
                continue
            Y = arrays["Y"]
            if meta.get("Y_sha") and array_digest(Y) != meta["Y_sha"]:
                continue                      # torn payload, try the previous
            if strict:
                self._check(meta)
            return Y, meta["state"], self._load_history()
        return None, None, None

    def _load_history(self):
        arrays, meta = _load_npz(self.dir / self.HIST)
        if arrays is None:
            return {}
        return {float(d): arrays["Y"][j]
                for j, d in enumerate(arrays["doses"])}

    def _check(self, meta):
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointMismatch(
                f"checkpoint schema {meta.get('schema_version')} != "
                f"{SCHEMA_VERSION}; delete {self.dir} to start over")
        have, want = meta.get("fingerprint", {}), self.fingerprint
        bad = [k for k in want
               if k in HARD_KEYS and have.get(k) != want.get(k)]
        if bad:
            lines = "\n".join(
                f"    {k}: checkpoint {have.get(k)!r} != now {want.get(k)!r}"
                for k in bad)
            raise CheckpointMismatch(
                f"this checkpoint was written for a different run:\n{lines}\n"
                f"  Resuming would mix two configurations. Either restore the "
                f"old settings, or pass resume='never' to start over "
                f"(which discards {self.dir}).")
        soft = [k for k in want
                if k not in HARD_KEYS and have.get(k) != want.get(k)]
        if soft:
            print("  warning: resuming across a changed environment — " +
                  ", ".join(f"{k}: {have.get(k)!r} -> {want.get(k)!r}"
                            for k in soft))


# Any difference in these makes the stored Y meaningless for the current run.
# `cd_nodes_sha` rather than a node COUNT: the row order is what binds Y[q] to a
# position in space, and two meshes can agree on the count. `base_cli_sha`
# catches a changed workbook, calibration override, temperature or dose rate,
# none of which the march config would show.
HARD_KEYS = frozenset({
    "cd_nodes_sha", "seed_evl_sha", "material_sha", "base_cli_sha",
    "march_cfg", "snaps_sha", "n_nodes", "n_eq",
})
