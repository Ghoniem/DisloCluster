"""build.py — make sure both C++ codes are built.

This lived only inside ``code/coupled_0d_3d_ZrMicro.ipynb``, so nothing that
was not that notebook could check or trigger a build.

Both functions are no-ops when the binary is already on disk.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dislocluster_code import paths

__all__ = ["BuildError", "ensure_zrmicro_solver", "ensure_modelib",
           "solver_runs", "ddomp_runs", "preflight", "stale_cache_home",
           "describe"]

DEFAULT_BUILD_TYPE = "Release"      # the batch mode is called thousands of times


class BuildError(RuntimeError):
    pass


def _run(cmd, cwd=None, timeout=3600, echo=True, env=None):
    """Run a command, echoing a trimmed transcript. Returns (rc, output)."""
    if echo:
        print(f"  $ {' '.join(str(c) for c in cmd)}")
    p = subprocess.run([str(c) for c in cmd], cwd=cwd and str(cwd),
                       capture_output=True, text=True, timeout=timeout,
                       env=env)
    out = (p.stdout or "") + (p.stderr or "")
    if echo:
        for line in [l for l in out.splitlines() if l.strip()][-12:]:
            print("    " + line[:150])
    return p.returncode, out


def stale_cache_home():
    """The source dir a stale CMake cache was configured for, else None.

    A CMake cache records the absolute paths it was configured with, so moving
    or renaming the repository makes any later reconfigure hard-error with
    "the current CMakeCache.txt directory ... is different". The binary already
    built keeps working -- only reconfiguring breaks -- which is what makes
    this easy to miss for months.

    Both sides are resolved before comparing: the cache stores forward slashes
    and may disagree on drive-letter case, so a raw string or unresolved Path
    comparison can call an identical directory stale (or, worse, a moved one
    current).
    """
    cache = paths.BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        return None
    home = next((l.split("=", 1)[1].strip()
                 for l in cache.read_text(errors="ignore").splitlines()
                 if l.startswith("CMAKE_HOME_DIRECTORY:")), "")
    if not home:
        return None
    try:
        same = Path(home).resolve() == Path(paths.CPP_UTILS).resolve()
    except OSError:                       # the old path may no longer exist
        same = False
    return None if same else home


def ensure_zrmicro_solver(force=False, build_type=DEFAULT_BUILD_TYPE,
                          verbose=True):
    """Configure and build the 0-D SUNDIALS solver if it is missing or stale."""
    # Check the cache BEFORE the "already built" shortcut. This used to sit
    # after it, so the one situation the discard logic exists for -- the tree
    # was moved, the binary still runs, the cache points at the old path --
    # returned early and never reached it. The next reconfigure then failed
    # with a CMake error that looked unrelated to the move.
    stale = stale_cache_home()
    exe = paths.zrmicro_solver_exe()
    if exe and not force and not stale:
        if verbose:
            print(f"[ZrMicro] solver found : {exe}")
        return exe

    if stale:
        if verbose:
            print(f"[ZrMicro] cache was configured for {stale}")
            print(f"          the tree has moved to {paths.CPP_UTILS}; "
                  f"discarding and rebuilding")
        shutil.rmtree(paths.BUILD_DIR, ignore_errors=True)
    elif verbose:
        print("[ZrMicro] solver not found -- configuring and building")

    rc, _ = _run(["cmake", "-S", paths.CPP_UTILS, "-B", paths.BUILD_DIR,
                  f"-DCMAKE_BUILD_TYPE={build_type}"], echo=verbose)
    if rc != 0:
        raise BuildError("CMake configure failed for ZrMicro (see log above)")
    rc, _ = _run(["cmake", "--build", paths.BUILD_DIR, "--config", build_type],
                 echo=verbose)
    if rc != 0:
        raise BuildError("CMake build failed for ZrMicro (see log above)")

    exe = paths.zrmicro_solver_exe()
    if not exe:
        raise BuildError(
            f"the build reported success but no solver binary is under "
            f"{paths.BUILD_DIR}")
    if verbose:
        print(f"[ZrMicro] built : {exe}")
    return exe


def solver_runs():
    """Can solver.exe actually start? ``(ok, message)``.

    Existing on disk is not the same as being runnable. A binary can be there
    and still fail to load -- a missing SUNDIALS DLL on Windows gives exit
    0xC0000135 before `main` is reached, and a Linux ELF cloned onto a machine
    with a different glibc fails the same way. Ask it for its usage line.
    """
    exe = paths.zrmicro_solver_exe()
    if exe is None:
        return False, f"no solver binary under {paths.BUILD_DIR}"
    try:
        p = subprocess.run([str(exe), "--help"], capture_output=True,
                           text=True, timeout=60)
    except OSError as e:
        return False, f"{exe} will not start: {e}"
    if p.returncode in (0, 1, 2) and (p.stdout or p.stderr):
        return True, str(exe)
    return False, (f"{exe} exited {p.returncode} "
                   f"(0x{p.returncode & 0xFFFFFFFF:08X}) with no output -- "
                   f"most likely a missing runtime library. Rebuild with "
                   f"ensure_zrmicro_solver(force=True).")


def ddomp_runs():
    """Can DDomp actually start? ``(ok, message)``.

    Same reasoning as :func:`solver_runs`, and it matters more here: DDomp is a
    Linux ELF invoked through WSL, and a build tree committed to git by one
    machine used to be pulled by others that could not run it.
    """
    exe = paths.modelib_ddomp()
    if exe is None:
        return False, f"no DDomp under {paths.MODELIB_BUILD}"
    cmd = (["wsl.exe", "-e", paths.windows_to_wsl(exe), "--version"]
           if paths.use_wsl() else [str(exe), "--version"])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except OSError as e:
        return False, f"{exe} will not start: {e}"
    blob = (p.stdout or "") + (p.stderr or "")
    # DDomp has no --version; reaching its own argument handling (rather than
    # the loader failing) is the signal we want.
    if "error while loading shared libraries" in blob or "No such file" in blob:
        return False, f"{exe} fails to load: {blob.strip()[:200]}"
    return True, str(exe)


def ensure_modelib(force=False, verbose=True):
    """Build MoDELib3 (DDomp) if it is missing. Returns the DDomp path or None.

    Exit status is NOT a reliable signal: the build script ends with
    ``ls ... || echo WARN`` and so returns 0 even when it produced nothing.
    The artifact is verified on disk instead.
    """
    exe = paths.modelib_ddomp()
    if exe and not force:
        if verbose:
            print(f"[MoDELib] DDomp found : {exe}")
        return exe

    if not paths.MODELIB_ROOT.is_dir():
        print(f"[MoDELib] no checkout at {paths.MODELIB_ROOT} -- the 3-D "
              f"backend is unavailable. Expected DisloCluster/MoDELib3/, or "
              f"set MODELIB_ROOT.")
        return None
    if not paths.MODELIB_BUILD_SCRIPT.is_file():
        print(f"[MoDELib] build script missing: {paths.MODELIB_BUILD_SCRIPT}")
        return None

    if verbose:
        print("[MoDELib] binaries absent -- building (10-30 min)")
    if paths.use_wsl():
        # Run as root: the script's `sudo apt-get` needs no password there, and
        # MSYS_NO_PATHCONV stops Git Bash rewriting /mnt/... into a Windows path.
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        _run(["wsl.exe", "-u", "root", "-e", "bash", "-lc",
              f"bash {paths.windows_to_wsl(paths.MODELIB_BUILD_SCRIPT)} "
              f"{paths.windows_to_wsl(paths.MODELIB_ROOT)}"],
             timeout=7200, echo=verbose, env=env)
    else:
        _run(["bash", paths.MODELIB_BUILD_SCRIPT, paths.MODELIB_ROOT],
             timeout=7200, echo=verbose)

    exe = paths.modelib_ddomp()
    if exe:
        if verbose:
            print(f"[MoDELib] built : {exe}")
    else:
        print("[MoDELib] the build produced no DDomp -- the 3-D backend stays "
              "unavailable.")
        print("          Common causes: WSL first-run setup not completed; "
              "libfftw3-dev missing; CMake >= 4 rejecting "
              "cmake_minimum_required(3.1).")
    return exe


def preflight(build_missing=True, require_modelib=True, verbose=True):
    """Make this machine ready to run a simulation, or say exactly why not.

    Everything a run needs is resolved from the repository, built if absent and
    then *executed* once to prove it works. Nothing here depends on the machine
    the repository was last used on:

      * paths come from the ``.dislocluster_root`` marker, so the checkout can
        live anywhere;
      * neither build tree is in git, so a fresh clone builds its own;
      * a CMake cache left pointing at another location is discarded;
      * the binaries are run, not merely found on disk.

    Raises :class:`BuildError` with actionable text when something cannot be
    fixed automatically -- SUNDIALS absent, WSL absent, no compiler. Returns a
    dict of what was found.
    """
    out = {}
    say = print if verbose else (lambda *a, **k: None)
    say("=" * 68)
    say("DisloCluster preflight")
    say("=" * 68)
    say(paths.describe())
    say("")

    # ── 0-D solver ──────────────────────────────────────────────────────────
    stale = stale_cache_home()
    if stale:
        say(f"[ZrMicro] CMake cache belongs to {stale} -- rebuilding")
    if (paths.zrmicro_solver_exe() is None or stale) and build_missing:
        ensure_zrmicro_solver(verbose=verbose)
    ok, msg = solver_runs()
    if not ok and build_missing:
        say(f"[ZrMicro] {msg}\n[ZrMicro] rebuilding")
        ensure_zrmicro_solver(force=True, verbose=verbose)
        ok, msg = solver_runs()
    if not ok:
        raise BuildError(
            f"the 0-D solver is not usable: {msg}\n"
            f"  Build it with:\n"
            f"    cmake -S {paths.CPP_UTILS} -B {paths.BUILD_DIR} "
            f"-DCMAKE_BUILD_TYPE=Release\n"
            f"    cmake --build {paths.BUILD_DIR} --config Release\n"
            f"  It needs SUNDIALS 7.1.1 (README.md, Setup step 2) and a C++17 "
            f"compiler.")
    out["solver"] = msg
    say(f"[ZrMicro] solver runs : {msg}")

    # ── 3-D solver ──────────────────────────────────────────────────────────
    if paths.modelib_ddomp() is None and build_missing:
        ensure_modelib(verbose=verbose)
    ok, msg = ddomp_runs()
    if not ok and build_missing and paths.modelib_ddomp() is not None:
        say(f"[MoDELib] {msg}\n[MoDELib] rebuilding")
        ensure_modelib(force=True, verbose=verbose)
        ok, msg = ddomp_runs()
    if not ok:
        text = (f"MoDELib3 is not usable: {msg}\n"
                f"  Build it with:\n"
                f"    wsl -u root -e bash "
                f"{paths.MODELIB_BUILD_SCRIPT.relative_to(paths.REPO_ROOT).as_posix()}\n"
                f"  It takes 10-30 min the first time and needs WSL with a "
                f"C++20 compiler, Eigen, SuiteSparse and libfftw3-dev.\n"
                f"  The 3-D solve, and therefore the coupled march, cannot run "
                f"without it.")
        if require_modelib:
            raise BuildError(text)
        say("WARNING  " + text)
    out["ddomp"] = msg
    say(f"[MoDELib] DDomp runs  : {msg}")

    # ── the inputs a run reads ──────────────────────────────────────────────
    missing = [str(p) for p in (paths.INPUT_DIR, paths.MODELIB_MATERIAL,
                                paths.GMSH_DIR / "generate_mesh.py")
               if not p.exists()]
    if missing:
        raise BuildError("missing repository inputs: " + ", ".join(missing))
    drift = paths.workbook_drift()
    if drift:
        say(f"WARNING  workbooks differ between {paths.INPUT_DIR} and "
            f"{paths.LEGACY_INPUT_DIR}: {', '.join(drift)}")
    out["drift"] = drift

    say("")
    say("ready.")
    return out


def describe():
    solver, ddomp = paths.zrmicro_solver_exe(), paths.modelib_ddomp()
    lines = [
        f"{'OK ' if solver else 'MISS'}  0-D solver   {solver or paths.BUILD_DIR}",
        f"{'OK ' if ddomp else 'MISS'}  MoDELib DDomp {ddomp or paths.MODELIB_BUILD}",
    ]
    stale = stale_cache_home()
    if stale:
        lines += ["",
                  f"STALE the CMake cache in {paths.BUILD_DIR}",
                  f"      was configured for {stale}.",
                  f"      The binary still runs, but any reconfigure will fail "
                  f"until it is discarded.",
                  f"      Fix: ensure_zrmicro_solver(force=True)"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
