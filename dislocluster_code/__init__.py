"""DisloCluster — coupled cluster dynamics and dislocation
dynamics for irradiated zirconium.

Import the pieces you need::

    from dislocluster_code import paths
    from dislocluster_code.zerod.calibration import build_sim
    from dislocluster_code.coupling import march
"""
import sys as _sys

__version__ = "0.1.0"


def _make_stdio_unicode_safe():
    """Stop a decorative glyph from killing a multi-hour run on Windows.

    49 `print` lines across 12 modules carry non-ASCII characters -- mostly
    U+2713 CHECK MARK and U+274C CROSS MARK as status markers, plus em dashes.
    On Windows an interactive console and a REDIRECTED stream get cp1252, which
    cannot encode any of them, so the print raises UnicodeEncodeError and takes
    the process with it.

    That is not hypothetical and it is not cheap: the run (b) march died at
    `input_data.py:38` -- `print(f"\\u2713 File exists: ...")` -- AFTER 914 s of
    CD-node bootstrapping, before a single dose step. It survives interactively
    only because a resumed march skips `driver.prepare`, which is the call that
    reaches `InputData`.

    Fixing the 49 call sites would not hold: the next status marker someone
    types reintroduces it. Fixing the STREAM does hold, so this runs once at
    package import.

    UTF-8 is tried first -- files and modern terminals take it -- and if the
    stream refuses to be reconfigured at all, the errors handler alone is
    relaxed to `backslashreplace`, which degrades a check mark to `\\u2713`
    rather than to an exception. Streams that already encode the glyph (any
    UTF-8 locale, i.e. Linux and macOS) are left completely alone, so this is a
    no-op everywhere except the platform that needs it.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(_sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:              # redirected to a non-TextIO object
            continue
        enc = getattr(stream, "encoding", None) or ""
        try:                                 # already fine? leave it untouched
            "✓❌—".encode(enc)
            continue
        except (LookupError, UnicodeEncodeError):
            pass
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:                    # noqa: BLE001 -- last resort
            try:
                reconfigure(errors="backslashreplace")
            except Exception:                # noqa: BLE001
                pass


_make_stdio_unicode_safe()
