"""paths.py — canonical path resolution for the DisloCluster repository.

Every element of DisloCluster (the 0-D ZrMicro solver, the 3-D MoDELib code, the
coupling notebook, the docs build scripts) needs to agree on where the other
elements live. Before this module those locations were hard-coded per file, so
moving or copying the repository broke the coupling. Resolve everything here
instead, and import it everywhere.

Repository layout
-----------------
    DisloCluster/                      <- REPO_ROOT   (holds .dislocluster_root)
      .DisloClusterVenv/               <- VENV_DIR    (Python 3.14 environment)
      requirements.txt
      Simulations/                     <- SIMULATIONS_DIR (run notebooks)
        input/                         <- INPUT_DIR   (the Excel workbooks)
        output/                        <- OUTPUT_DIR  (new run directories)
      Gmsh/                            <- GMSH_DIR    (mesh generator + meshes/)
      Docs/                            <- DOCS_DIR    (ALL documents)
        Formulation/                   <- DOCS_FORMULATION (+ build_modelib.sh)
        DisloCluster Manual/           <- DOCS_MANUAL
      ZrMicro/                         <- ZRMICRO_DIR (0-D code)
          code/  cpp_utils/  py_utils/  build/
          input/                       <- LEGACY_INPUT_DIR  (copies; see below)
          output/                      <- LEGACY_OUTPUT_DIR (runs made before
                                          Simulations/output existed; still
                                          searched by find_runs)
      MoDELib3/                        <- MODELIB_ROOT (3-D DD / SR-CD)
        build/                         <- MODELIB_BUILD (built in WSL on Windows)
        Library/Materials/
          Zr3d_ghoniem.txt             <- MODELIB_MATERIAL (the coupled material)
          Zr4.txt                      <- MODELIB_MATERIAL_STANDALONE
        tutorials/                     <- MODELIB_TUTORIALS, and SIM_ROOT: where
                                          staged simulation cases are written

Environment overrides (all optional)
------------------------------------
    DISLOCLUSTER_ROOT     absolute path of the repository root
    DISLOCLUSTER_SIM_ROOT where staged simulation cases are written
    MODELIB_ROOT          absolute path of the MoDELib checkout
    MODELIB_BUILD         absolute path of the MoDELib build directory

Typical use
-----------
    from dislocluster_code import paths
    print(paths.REPO_ROOT, paths.MODELIB_ROOT)
    exe = paths.zrmicro_solver_exe()          # None if not built yet
"""

from __future__ import annotations

import os
import subprocess
import sys
import time as _time
from pathlib import Path

__all__ = [
    "REPO_ROOT", "ZR_ROOT", "ZRMICRO_DIR", "DOCS_DIR", "DOCS_FORMULATION",
    "DOCS_MANUAL", "MODELIB_ROOT",
    "MODELIB_BUILD", "MODELIB_MATERIAL", "MODELIB_MATERIAL_COUPLED",
    "MODELIB_MATERIAL_STANDALONE", "MODELIB_TUTORIALS", "COUPLED_SIM_TUTORIAL",
    "MODELIB_BUILD_SCRIPT", "VENV_DIR", "GMSH_DIR", "GMSH_MESHES",
    "INPUT_DIR", "OUTPUT_DIR", "OUTPUT_DIRS", "LEGACY_INPUT_DIR",
    "LEGACY_OUTPUT_DIR", "CPP_UTILS", "BUILD_DIR", "PY_UTILS", "CODE_DIR",
    "SIMULATIONS_DIR", "SIM_ROOT",
    "ROOT_MARKER", "VENV_NAME", "KERNEL_NAME",
    "find_repo_root", "zrmicro_solver_exe", "modelib_ddomp", "venv_python",
    "ddomp_cmd", "git_hash", "windows_to_wsl", "use_wsl", "run_dir", "find_runs",
    "latest_run", "workbook_drift", "describe",
]

# ── Names that identify this project ─────────────────────────────────────────
ROOT_MARKER = ".dislocluster_root"
VENV_NAME   = ".DisloClusterVenv"
KERNEL_NAME = "dislocluster"

# Candidate directory names for the two sub-codes, in preference order. The
# second entry of each is a legacy name kept only so an older checkout still
# resolves: `Fluor_Zr` is what the 0-D tree was called before the rename, and
# `MoDELib2-NNL` is the upstream name of the 3-D fork.
_ZR_DIR_NAMES      = ("ZrClusterDynamics", "Fluor_Zr")
_MODELIB_DIR_NAMES = ("MoDELib3", "MoDELib2-NNL", "MoDELib")


def find_repo_root(start=None):
    """Return the DisloCluster repository root.

    Resolution order:
      1. ``$DISLOCLUSTER_ROOT`` if set and it exists.
      2. The nearest ancestor of *start* (default: this file) containing the
         ``.dislocluster_root`` marker.
      3. The nearest ancestor holding both a ZrMicro-side and a MoDELib-side
         directory — a fallback for checkouts where the marker was dropped.

    Raises RuntimeError if none of these succeed, rather than silently
    returning a wrong root that would send the coupling to stale locations.
    """
    env = os.environ.get("DISLOCLUSTER_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    here = Path(start).resolve() if start else Path(__file__).resolve()
    chain = [here, *here.parents] if here.is_dir() else list(here.parents)

    for parent in chain:
        if (parent / ROOT_MARKER).is_file():
            return parent

    for parent in chain:
        # ZrMicro sits at the repository root; older trees nested it one level
        # down inside ZrClusterDynamics/.
        has_zr = ((parent / "ZrMicro" / "cpp_utils").is_dir()
                  or any((parent / n / "ZrMicro" / "cpp_utils").is_dir()
                         for n in _ZR_DIR_NAMES))
        has_md = any((parent / n / "CMakeLists.txt").is_file() for n in _MODELIB_DIR_NAMES)
        if has_zr and has_md:
            return parent

    raise RuntimeError(
        f"Cannot locate the DisloCluster repository root from {here}. "
        f"Expected a '{ROOT_MARKER}' marker file in an ancestor directory, or "
        f"set DISLOCLUSTER_ROOT."
    )


def _first_existing(root, names, must_contain=None):
    """First ``root/name`` that exists (and contains *must_contain*, if given)."""
    for n in names:
        p = root / n
        if p.is_dir() and (must_contain is None or (p / must_contain).exists()):
            return p
    return root / names[0]          # non-existent, but the canonical location


# ── Resolved locations ───────────────────────────────────────────────────────
REPO_ROOT   = find_repo_root()
VENV_DIR    = REPO_ROOT / VENV_NAME

# The 0-D code. It now sits at the repository root as ZrMicro/; before the
# flattening it was ZrClusterDynamics/ZrMicro/, and that layout is still
# resolved so an older checkout keeps working. ZR_ROOT is retained as the
# parent of ZRMICRO_DIR -- at the root that is simply REPO_ROOT.
ZRMICRO_DIR = (REPO_ROOT / "ZrMicro"
               if (REPO_ROOT / "ZrMicro" / "cpp_utils").is_dir()
               else _first_existing(REPO_ROOT, _ZR_DIR_NAMES, "ZrMicro") / "ZrMicro")
ZR_ROOT     = ZRMICRO_DIR.parent
# Documents live in the REPOSITORY-ROOT Docs/ tree, not under ZrClusterDynamics.
# DOCS_DIR pointed at ZR_ROOT/"Docs"/"Formulation", which does not exist; that
# silently broke MODELIB_BUILD_SCRIPT below and with it the notebook's automatic
# MoDELib build, which checks `is_file()` and returns quietly when it fails.
DOCS_DIR         = REPO_ROOT / "Docs"
DOCS_FORMULATION = DOCS_DIR / "Formulation"
DOCS_MANUAL      = DOCS_DIR / "DisloCluster Manual"

# Mesh generation. GMSH_DIR holds generate_mesh.py; GMSH_MESHES caches the
# generated .msh files (gitignored — they are reproducible from the spec).
# At the repository root, not under ZrClusterDynamics/: meshing serves the 3-D
# domain rather than the 0-D model, so it sits beside Simulations/ with the
# other repository-wide pieces. A checkout that predates the move is still
# found, so an old tree keeps working.
GMSH_DIR    = (REPO_ROOT / "Gmsh" if (REPO_ROOT / "Gmsh").is_dir()
               else ZR_ROOT / "Gmsh")
GMSH_MESHES = GMSH_DIR / "meshes"

CODE_DIR    = ZRMICRO_DIR / "code"
PY_UTILS    = ZRMICRO_DIR / "py_utils"
CPP_UTILS   = ZRMICRO_DIR / "cpp_utils"
BUILD_DIR   = ZRMICRO_DIR / "build"

# ── Where simulations read and write ─────────────────────────────────────────
# `Simulations/` is the working home: input/ beside output/, the way ZrMicro/
# was used. Both fall back to ZrMicro's copies so a checkout that predates the
# move, and the older drivers under ZrMicro/code/, keep working.
SIMULATIONS_DIR = REPO_ROOT / "Simulations"

LEGACY_INPUT_DIR  = ZRMICRO_DIR / "input"
LEGACY_OUTPUT_DIR = ZRMICRO_DIR / "output"

INPUT_DIR = (SIMULATIONS_DIR / "input"
             if (SIMULATIONS_DIR / "input").is_dir() else LEGACY_INPUT_DIR)

# OUTPUT_DIR is where a new run is WRITTEN. OUTPUT_DIRS is every place a run
# may be FOUND, newest home first -- post-processing has to keep reaching the
# 1.4 GB of runs already under ZrMicro/output/, which are not being moved.
OUTPUT_DIR = (SIMULATIONS_DIR / "output"
              if (SIMULATIONS_DIR / "output").is_dir() else LEGACY_OUTPUT_DIR)
OUTPUT_DIRS = tuple(dict.fromkeys(
    d for d in (OUTPUT_DIR, LEGACY_OUTPUT_DIR) if d.is_dir()))

_md_env     = os.environ.get("MODELIB_ROOT")
MODELIB_ROOT = (Path(_md_env).expanduser().resolve() if _md_env
                else _first_existing(REPO_ROOT, _MODELIB_DIR_NAMES, "CMakeLists.txt"))

_MODELIB_BUILD_NAMES = ("build", "build_dc")


def _first_build_dir(root):
    """The MoDELib build tree that actually holds a DDomp, else ``build``.

    ``build/`` came into DisloCluster as a verbatim copy of the upstream
    ``MoDELib2-NNL`` build tree, absolute paths and all, so CMake refuses to
    reconfigure it ("the current CMakeCache.txt directory ... is different")
    and it can only ever be discarded. ``build_dc`` is this repository's own
    tree. Preferring whichever one carries the binary means a checkout that
    still has the inherited tree, and one that has been rebuilt cleanly, both
    resolve without an environment variable.
    """
    for name in _MODELIB_BUILD_NAMES:
        if (root / name / "tools" / "DDomp" / "DDomp").exists() or \
           (root / name / "tools" / "DDomp" / "DDomp.exe").exists():
            return root / name
    return root / _MODELIB_BUILD_NAMES[0]


_mb_env      = os.environ.get("MODELIB_BUILD")
MODELIB_BUILD = (Path(_mb_env).expanduser().resolve() if _mb_env
                 else _first_build_dir(MODELIB_ROOT))

# Material file shared by the two codes. `Zr3d_ghoniem.txt` is the coupled
# definition: its cluster-dynamics parameters are taken from the 0-D code, and
# it is the only one carrying the fitted `dadAnisotropy` / `dadZ0` that
# reconcile the 3-D DAD capture efficiencies with the 0-D symmetric forms. It is
# also what tutorials/zrmicro_coupled actually loads. `Zr4.txt` is Po's
# standalone 3-D calibration and lacks those keys — comparing against it made
# the notebook's 0-D/3-D DAD check fail to parse. Prefer the coupled file.
MODELIB_MATERIAL_COUPLED  = MODELIB_ROOT / "Library" / "Materials" / "Zr3d_ghoniem.txt"
MODELIB_MATERIAL_STANDALONE = MODELIB_ROOT / "Library" / "Materials" / "Zr4.txt"
MODELIB_MATERIAL = (MODELIB_MATERIAL_COUPLED if MODELIB_MATERIAL_COUPLED.is_file()
                    else MODELIB_MATERIAL_STANDALONE)

MODELIB_TUTORIALS    = MODELIB_ROOT / "tutorials"
COUPLED_SIM_TUTORIAL = MODELIB_TUTORIALS / "zrmicro_coupled"
# build_modelib.sh builds on Linux, WSL and macOS alike. The old WSL-only name
# is still accepted so a checkout that predates the rename keeps building.
MODELIB_BUILD_SCRIPT = next(
    (DOCS_FORMULATION / n for n in ("build_modelib.sh", "build_modelib_wsl.sh")
     if (DOCS_FORMULATION / n).is_file()),
    DOCS_FORMULATION / "build_modelib.sh")

# ── Simulation cases ─────────────────────────────────────────────────────────
# (SIMULATIONS_DIR, INPUT_DIR and OUTPUT_DIR are defined above, with the rest
# of the read/write locations.)
#
# Where a staged MoDELib case is written. Historically every builder hard-coded
# `MODELIB_ROOT / "tutorials" / name`, which is why upstream tutorials and this
# project's generated cases now share one directory. The default is unchanged,
# so nothing moves; naming it once means relocating the cases later is an edit
# here (or a DISLOCLUSTER_SIM_ROOT export) rather than a hunt through
# setup_domain, setup_standalone and run_seeded_march.
_sr_env = os.environ.get("DISLOCLUSTER_SIM_ROOT")
SIM_ROOT = (Path(_sr_env).expanduser().resolve() if _sr_env
            else MODELIB_TUTORIALS)


# ── Helpers ──────────────────────────────────────────────────────────────────
def use_wsl():
    """True when MoDELib's Linux binaries must be invoked through wsl.exe.

    On Windows the MoDELib build is produced inside WSL (it needs GCC/Clang +
    Eigen + SuiteSparse), so DDomp is an ELF binary that Windows cannot exec
    directly. On Linux/macOS the native binary is used as-is.
    """
    return sys.platform == "win32"


def windows_to_wsl(p):
    """``d:/GitHub/DisloCluster`` -> ``/mnt/d/GitHub/DisloCluster``."""
    p = Path(p).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = p.as_posix()[len(p.drive):].lstrip("/")
    return f"/mnt/{drive}/{rest}" if drive else p.as_posix()


def venv_python():
    """Path to the repository interpreter, or None if the venv is absent."""
    for rel in ("Scripts/python.exe", "bin/python3", "bin/python"):
        p = VENV_DIR / rel
        if p.exists():
            return p
    return None


def zrmicro_solver_exe(base_dir=None):
    """Locate the compiled 0-D solver binary, or None.

    Mirrors ``cpp_bridge._find_solver_exe``: multi-config generators (MSVC) put
    the binary under Debug/ or Release/, single-config ones directly in build/.
    """
    exe = "solver.exe" if sys.platform == "win32" else "solver"
    bld = Path(base_dir) / "build" if base_dir else BUILD_DIR
    for c in (bld / "Debug" / exe, bld / "Release" / exe, bld / exe):
        if c.exists():
            return c
    return None


def modelib_ddomp():
    """Locate MoDELib's DDomp runner, or None. Prefers the WSL/ELF build."""
    names = ("DDomp",) if use_wsl() else (
        ("DDomp.exe",) if sys.platform == "win32" else ("DDomp",))
    ddir = MODELIB_BUILD / "tools" / "DDomp"
    for sub in ("", "Release", "Debug"):
        for name in names:
            c = (ddir / sub / name) if sub else (ddir / name)
            if c.exists():
                return c
    return None


def ddomp_cmd(sim_dir, exe=None, in_case_dir=False):
    """``(argv, cwd)`` that runs DDomp on *sim_dir* on THIS machine.

    Windows reaches the ELF binary through ``wsl.exe`` and has to hand it
    ``/mnt/...`` paths; Linux and macOS exec it directly. Four call sites had
    grown their own copy of that decision (case.bootstrap, domain.bootstrap,
    standalone.run_ddomp, qssa._run_ddomp) and only one of them actually
    branched -- the other three built a ``wsl.exe`` command line unconditionally
    and so could not run anywhere but Windows.

    *in_case_dir* makes DDomp run with the case as its working directory, which
    the bootstrap needs: DDomp writes ``evl/cdNodes.txt`` to that relative path,
    so from anywhere else the file lands outside the case.
    """
    exe = Path(exe).resolve() if exe else modelib_ddomp()
    if exe is None:
        raise FileNotFoundError(f"no DDomp under {MODELIB_BUILD}")
    # Absolute, always: *in_case_dir* moves the working directory, so a relative
    # case path handed to DDomp would resolve against the case itself and it
    # would report "inputFiles/polycrystal.txt cannot be opened".
    sim_dir = Path(sim_dir).resolve()
    if use_wsl():
        wsl_dir = windows_to_wsl(sim_dir)
        wsl_exe = windows_to_wsl(exe)
        if in_case_dir:
            return (["wsl.exe", "-e", "bash", "-c",
                     f"cd '{wsl_dir}' && '{wsl_exe}' '{wsl_dir}'"], None)
        return (["wsl.exe", "-e", wsl_exe, wsl_dir], None)
    return ([str(exe), str(sim_dir)], str(sim_dir) if in_case_dir else None)


def modelib_generator():
    """Locate MoDELib's microstructureGenerator, or None.

    Same build tree and same WSL/ELF preference as :func:`modelib_ddomp`. It is
    a separate binary because it does a separate job: it turns a microstructure
    SPECIFICATION into the node/loop/link topology an `evl` configuration needs,
    which is what the runtime continuum->discrete transition uses rather than
    reproducing that topology in Python.
    """
    names = ("microstructureGenerator",) if use_wsl() else (
        ("microstructureGenerator.exe",) if sys.platform == "win32"
        else ("microstructureGenerator",))
    gdir = MODELIB_BUILD / "tools" / "MicrostructureGenerator"
    for sub in ("", "Release", "Debug"):
        for name in names:
            c = (gdir / sub / name) if sub else (gdir / name)
            if c.exists():
                return c
    return None


def generator_cmd(sim_dir, exe=None):
    """``(argv, cwd)`` that runs microstructureGenerator on *sim_dir*.

    The generator writes ``evl/evl_0.txt`` under the case, so like the
    bootstrap's DDomp call it must run WITH THE CASE AS ITS WORKING DIRECTORY;
    from anywhere else the file lands outside the case.
    """
    exe = Path(exe).resolve() if exe else modelib_generator()
    if exe is None:
        raise FileNotFoundError(
            f"no microstructureGenerator under {MODELIB_BUILD}")
    sim_dir = Path(sim_dir).resolve()
    if use_wsl():
        wsl_dir = windows_to_wsl(sim_dir)
        wsl_exe = windows_to_wsl(exe)
        return (["wsl.exe", "-e", "bash", "-c",
                 f"cd '{wsl_dir}' && '{wsl_exe}' '{wsl_dir}'"], None)
    return ([str(exe), str(sim_dir)], str(sim_dir))


def git_hash(path=None):
    """Short git hash for *path*, falling back through the nested checkouts.

    DisloCluster was assembled by copying two independent repositories. It is
    a single checkout now, but a split one -- where ``ZrMicro/`` or
    ``MoDELib3/`` carries its own ``.git`` -- still resolves: try the root
    first, then the 0-D code, so run directories always carry a meaningful
    provenance tag.
    """
    for cand in ([Path(path)] if path else []) + [REPO_ROOT, ZR_ROOT, MODELIB_ROOT]:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(cand),
                stderr=subprocess.DEVNULL, timeout=5)
            h = out.decode().strip()
            if h:
                return h
        except Exception:
            continue
    return "nogit"


def run_dir(tag=None, base=None, create=True):
    """``<output>/<YYYYmmdd_HHMMSS>_<git-hash>[_<tag>]`` — one run's directory.

    Four drivers had grown their own verbatim copy of this idiom
    (run_coupled_vs_standalone, run_seeded_march, compare_seeding,
    run_zr3d_singlecrystal), so a change to the naming convention had to be
    made four times and the 0-D helper (visualization.create_run_directory,
    which predates tags) disagreed with all of them.
    """
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{git_hash()}" + (f"_{tag}" if tag else "")
    d = Path(base or OUTPUT_DIR) / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def find_runs(pattern="*", marker="march_state.npz", dirs=None):
    """Every run directory, across all output roots, oldest first.

    New runs go to ``OUTPUT_DIR``; the ones made before ``Simulations/output/``
    existed are still under ``ZrMicro/output/``, so anything that goes looking
    for a run has to search both.
    """
    out = []
    for root in (dirs or OUTPUT_DIRS):
        if not Path(root).is_dir():
            continue
        for d in Path(root).glob(pattern):
            if d.is_dir() and (not marker or (d / marker).is_file()):
                out.append(d)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def latest_run(pattern="*", marker="march_state.npz"):
    """The most recent run directory, or None."""
    runs = find_runs(pattern, marker)
    return runs[-1] if runs else None


def workbook_drift():
    """Names of workbooks that differ between INPUT_DIR and the ZrMicro copy.

    The workbooks were COPIED into ``Simulations/input/`` rather than moved, so
    two copies exist and can be edited independently. That is exactly how the
    0-D calibration drifted from its workbook once before, five orders of
    magnitude in N_a, so the divergence is reported rather than left to be
    discovered. Empty list means the copies agree (or there is only one).
    """
    if INPUT_DIR == LEGACY_INPUT_DIR or not LEGACY_INPUT_DIR.is_dir():
        return []
    import hashlib

    def digest(p):
        return hashlib.blake2b(p.read_bytes(), digest_size=16).hexdigest()

    drift = []
    for new in sorted(INPUT_DIR.glob("*.xlsx")):
        old = LEGACY_INPUT_DIR / new.name
        if old.is_file() and digest(old) != digest(new):
            drift.append(new.name)
    return drift


def describe():
    """Human-readable resolution report — print this from notebooks."""
    rows = [
        ("repo root",       REPO_ROOT,        REPO_ROOT.is_dir()),
        ("venv",            VENV_DIR,         venv_python() is not None),
        ("0-D code",        ZRMICRO_DIR,      ZRMICRO_DIR.is_dir()),
        ("0-D solver",      zrmicro_solver_exe() or BUILD_DIR, zrmicro_solver_exe() is not None),
        ("input",           INPUT_DIR,        INPUT_DIR.is_dir()),
        ("output (writes)", OUTPUT_DIR,       OUTPUT_DIR.is_dir()),
        ("MoDELib root",    MODELIB_ROOT,     MODELIB_ROOT.is_dir()),
        ("MoDELib build",   MODELIB_BUILD,    MODELIB_BUILD.is_dir()),
        ("MoDELib DDomp",   modelib_ddomp() or MODELIB_BUILD / "tools/DDomp", modelib_ddomp() is not None),
        ("Zr material",     MODELIB_MATERIAL, MODELIB_MATERIAL.is_file()),
        ("coupled sim case", COUPLED_SIM_TUTORIAL, COUPLED_SIM_TUTORIAL.is_dir()),
        ("sim case root",   SIM_ROOT,         SIM_ROOT.is_dir()),
        ("Simulations",     SIMULATIONS_DIR,  SIMULATIONS_DIR.is_dir()),
        ("Gmsh generator",  GMSH_DIR / "generate_mesh.py", (GMSH_DIR / "generate_mesh.py").is_file()),
        ("build script",    MODELIB_BUILD_SCRIPT, MODELIB_BUILD_SCRIPT.is_file()),
    ]
    for extra in OUTPUT_DIRS[1:]:
        rows.insert(6, ("output (also read)", extra, extra.is_dir()))
    w = max(len(r[0]) for r in rows)
    out = "\n".join(f"{'OK ' if ok else 'MISS'}  {name:<{w}}  {p}"
                    for name, p, ok in rows)
    drift = workbook_drift()
    if drift:
        out += (f"\n\nWARNING  these workbooks differ between\n"
                f"         {INPUT_DIR}\n"
                f"         {LEGACY_INPUT_DIR}\n"
                f"         " + ", ".join(drift) +
                f"\n         Only the first is read. Reconcile them, or the "
                f"two trees are running different models.")
    return out


def _main():
    """Console entry point (``dislocluster-paths``)."""
    print(describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
