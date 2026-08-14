"""progress.py — live reporting for a march that runs for hours.

The march emits one :class:`SubstepEvent` per substep. A callback turns it into
whatever the caller wants; ``ConsoleProgress`` and ``NotebookProgress`` are
provided.

WHY THE ETA IS BIMODAL
----------------------
A substep costs either ``fast + slow`` or just ``slow``, and the two differ by
two orders of magnitude -- a fast solve on the 500 nm case measures ~245 s
against ~10 s for the immobile batch. Averaging over both gives an estimate
that is wrong by the duty cycle for any ``fem_every > 1``, and it swings every
time the cadence fires. Two exponentially-weighted averages, applied to the
count of each kind of substep still to come, do not.

WHY THE EVENT CARRIES NO ARRAY
------------------------------
A notebook that appends events to a list is the obvious thing to write. At
32 k nodes ``Y`` is 4.8 MB, so a hundred-substep march would pin ~500 MB in the
kernel. The event carries six scalars summarizing ``Y`` and a ``get_Y``
accessor that is only valid during the callback.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

__all__ = ["SubstepEvent", "EtaEstimator", "ConsoleProgress",
           "NotebookProgress", "format_hms"]


def format_hms(s):
    if s is None or s != s or s < 0:          # None or NaN
        return "--:--"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


@dataclass(frozen=True)
class SubstepEvent:
    """One completed substep. All scalars; see the module docstring."""

    phase: str                 # "start" | "substep" | "interval" | "done"
    # position
    k_global: int = 0
    n_substeps_total: int = 0
    interval: int = 0
    n_intervals: int = 0
    k_in_interval: int = 0
    substeps_per_interval: int = 0
    resumed: bool = False
    # dose
    dose: float = 0.0
    dose_from: float = 0.0
    dose_to: float = 0.0
    dose_max: float = 0.0
    # cost
    did_fast: bool = False
    fast_s: float = 0.0
    slow_s: float = 0.0
    wall_s_session: float = 0.0
    wall_s_total: float = 0.0
    eta_s: float = float("nan")
    # health
    n_failed: int = 0
    n_integrated: int = 0
    dedup_ratio: float = 1.0
    qssa_converged: bool | None = None
    # a summary of Y, never Y itself
    mean_N_a: float = 0.0
    mean_N_c: float = 0.0
    mean_c_a: float = 0.0
    mean_c_c: float = 0.0
    mean_Cv: float = 0.0
    mean_Ci: float = 0.0
    # escape hatches
    checkpoint_path: str | None = None
    get_Y: object = None

    @property
    def fraction(self):
        return (self.k_global / self.n_substeps_total
                if self.n_substeps_total else 0.0)


class EtaEstimator:
    """Two EWMAs: one for substeps that carry a fast solve, one for the rest."""

    def __init__(self, alpha=0.3, fast_s=None, slow_s=None):
        self.alpha = alpha
        self.fast = fast_s
        self.slow = slow_s

    def update(self, did_fast, fast_s, slow_s):
        a = self.alpha
        if did_fast and fast_s > 0:
            self.fast = fast_s if self.fast is None \
                else (1 - a) * self.fast + a * fast_s
        self.slow = slow_s if self.slow is None \
            else (1 - a) * self.slow + a * slow_s

    def remaining(self, k_global, n_total, fem_every):
        if self.slow is None or k_global >= n_total:
            return float("nan")
        left = n_total - k_global
        n_fast = sum(1 for j in range(k_global, n_total) if j % fem_every == 0)
        return n_fast * (self.fast or 0.0) + left * self.slow

    def state(self):
        return dict(fast_s=self.fast, slow_s=self.slow)


class ConsoleProgress:
    """A single rewriting line. Pass ``verbose=False`` to the march."""

    def __init__(self, stream=None, every=1):
        import sys
        self.stream = stream or sys.stdout
        self.every = every

    def __call__(self, ev):
        if ev.phase == "start":
            self.stream.write(
                f"march: {ev.n_intervals} interval(s), "
                f"{ev.n_substeps_total} substeps"
                + (" (resumed)" if ev.resumed else "") + "\n")
            return
        if ev.phase == "done":
            self.stream.write(
                f"\ndone: {ev.k_global} substeps in "
                f"{format_hms(ev.wall_s_total)}\n")
            return
        if ev.phase != "substep" or ev.k_global % self.every:
            return
        bar_n = 24
        filled = int(bar_n * ev.fraction)
        self.stream.write(
            f"\r  [{'#' * filled}{'.' * (bar_n - filled)}] "
            f"{100 * ev.fraction:5.1f}%  {ev.dose:8.4g} dpa  "
            f"{'FEM' if ev.did_fast else '   '}  "
            f"N_a {ev.mean_N_a:.3e}  "
            f"{format_hms(ev.wall_s_total)} elapsed, "
            f"ETA {format_hms(ev.eta_s)}   ")
        if ev.n_failed:
            self.stream.write(f"\n  warning: {ev.n_failed} node(s) failed at "
                              f"{ev.dose:g} dpa\n")
        self.stream.flush()


class NotebookProgress:
    """An in-place HTML line for Jupyter, falling back to the console."""

    def __init__(self, every=1):
        self.every = every
        self._h = None
        try:
            from IPython.display import display, HTML
            self._display, self._HTML = display, HTML
        except ImportError:
            self._display = None
            self._console = ConsoleProgress(every=every)

    def _html(self, ev):
        pct = 100 * ev.fraction
        warn = (f" &nbsp; <span style='color:#c62828'>{ev.n_failed} node(s) "
                f"failed</span>" if ev.n_failed else "")
        return self._HTML(
            "<div style='font-family:monospace;font-size:12px'>"
            f"<div style='background:#eee;border-radius:3px;height:14px;"
            f"width:420px;overflow:hidden'>"
            f"<div style='background:#1f4fbf;height:14px;width:{pct:.1f}%'>"
            f"</div></div>"
            f"dose <b>{ev.dose:.4g}</b> / {ev.dose_max:g} dpa &nbsp; "
            f"substep {ev.k_global}/{ev.n_substeps_total} "
            f"({pct:.1f}%){' &nbsp;<b>FEM</b>' if ev.did_fast else ''}<br>"
            f"N_a {ev.mean_N_a:.4e} &nbsp; N_c {ev.mean_N_c:.4e} &nbsp; "
            f"Cv {ev.mean_Cv:.4e}<br>"
            f"elapsed {format_hms(ev.wall_s_total)} &nbsp; "
            f"ETA {format_hms(ev.eta_s)}{warn}"
            "</div>")

    def __call__(self, ev):
        if self._display is None:
            return self._console(ev)
        if ev.phase == "start":
            self._h = self._display(self._html(ev), display_id=True)
            return
        if ev.phase != "substep" or (ev.k_global % self.every and
                                     ev.phase != "done"):
            return
        if self._h is None:
            self._h = self._display(self._html(ev), display_id=True)
        else:
            self._h.update(self._html(ev))
