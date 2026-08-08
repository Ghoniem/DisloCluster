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
      Docs/                            <- DOCS_DIR    (ALL documents)
        Formulation/                   <- DOCS_FORMULATION (+ build_modelib_wsl.sh)
        DisloCluster Manual/           <- DOCS_MANUAL
      ZrClusterDynamics/               <- ZR_ROOT     (0-D code)
        Gmsh/                          <- GMSH_DIR    (mesh generator + meshes/)
        ZrMicro/                       <- ZRMICRO_DIR
          code/  cpp_utils/  py_utils/  input/  build/  output/
      MoDELib3/                        <- MODELIB_ROOT (3-D DD / SR-CD)
        build/                         <- MODELIB_BUILD (built in WSL on Windows)
        Library/Materials/
          Zr3d_ghoniem.txt             <- MODELIB_MATERIAL (the coupled material)
          Zr4.txt                      <- MODELIB_MATERIAL_STANDALONE

Environment overrides (all optional)
------------------------------------
    DISLOCLUSTER_ROOT     absolute path of the repository root
    MODELIB_ROOT          absolute path of the MoDELib checkout
    MODELIB_BUILD         absolute path of the MoDELib build directory

Typical use
-----------
    from py_utils import paths
    print(paths.REPO_ROOT, paths.MODELIB_ROOT)
    exe = paths.zrmicro_solver_exe()          # None if not built yet
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "REPO_ROOT", "ZR_ROOT", "ZRMICRO_DIR", "DOCS_DIR", "DOCS_FORMULATION",
    "DOCS_MANUAL", "MODELIB_ROOT",
    "MODELIB_BUILD", "MODELIB_MATERIAL", "MODELIB_MATERIAL_COUPLED",
    "MODELIB_MATERIAL_STANDALONE", "MODELIB_TUTORIALS", "COUPLED_SIM_TUTORIAL",
    "MODELIB_BUILD_SCRIPT", "VENV_DIR", "GMSH_DIR", "GMSH_MESHES",
    "INPUT_DIR", "OUTPUT_DIR", "CPP_UTILS", "BUILD_DIR", "PY_UTILS", "CODE_DIR",
    "ROOT_MARKER", "VENV_NAME", "KERNEL_NAME",
    "find_repo_root", "zrmicro_solver_exe", "modelib_ddomp", "venv_python",
    "git_hash", "windows_to_wsl", "use_wsl", "describe",
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
        has_zr = any((parent / n / "ZrMicro" / "py_utils").is_dir() for n in _ZR_DIR_NAMES)
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

ZR_ROOT     = _first_existing(REPO_ROOT, _ZR_DIR_NAMES, "ZrMicro")
ZRMICRO_DIR = ZR_ROOT / "ZrMicro"
# Documents live in the REPOSITORY-ROOT Docs/ tree, not under ZrClusterDynamics.
# DOCS_DIR pointed at ZR_ROOT/"Docs"/"Formulation", which does not exist; that
# silently broke MODELIB_BUILD_SCRIPT below and with it the notebook's automatic
# MoDELib build, which checks `is_file()` and returns quietly when it fails.
DOCS_DIR         = REPO_ROOT / "Docs"
DOCS_FORMULATION = DOCS_DIR / "Formulation"
DOCS_MANUAL      = DOCS_DIR / "DisloCluster Manual"

# Mesh generation. GMSH_DIR holds generate_mesh.py; GMSH_MESHES caches the
# generated .msh files (gitignored — they are reproducible from the spec).
GMSH_DIR    = ZR_ROOT / "Gmsh"
GMSH_MESHES = GMSH_DIR / "meshes"

CODE_DIR    = ZRMICRO_DIR / "code"
PY_UTILS    = ZRMICRO_DIR / "py_utils"
CPP_UTILS   = ZRMICRO_DIR / "cpp_utils"
BUILD_DIR   = ZRMICRO_DIR / "build"
INPUT_DIR   = ZRMICRO_DIR / "input"
OUTPUT_DIR  = ZRMICRO_DIR / "output"

_md_env     = os.environ.get("MODELIB_ROOT")
MODELIB_ROOT = (Path(_md_env).expanduser().resolve() if _md_env
                else _first_existing(REPO_ROOT, _MODELIB_DIR_NAMES, "CMakeLists.txt"))

_mb_env      = os.environ.get("MODELIB_BUILD")
MODELIB_BUILD = (Path(_mb_env).expanduser().resolve() if _mb_env
                 else MODELIB_ROOT / "build")

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
MODELIB_BUILD_SCRIPT = DOCS_FORMULATION / "build_modelib_wsl.sh"


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


def git_hash(path=None):
    """Short git hash for *path*, falling back through the nested checkouts.

    DisloCluster was assembled by copying two independent repositories, so the
    root itself may not be a git repository while ``ZrClusterDynamics/`` and
    ``MoDELib3/`` still are. Try the root first, then the 0-D code, so run
    directories always carry a meaningful provenance tag.
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


def describe():
    """Human-readable resolution report — print this from notebooks."""
    rows = [
        ("repo root",       REPO_ROOT,        REPO_ROOT.is_dir()),
        ("venv",            VENV_DIR,         venv_python() is not None),
        ("0-D code",        ZRMICRO_DIR,      ZRMICRO_DIR.is_dir()),
        ("0-D solver",      zrmicro_solver_exe() or BUILD_DIR, zrmicro_solver_exe() is not None),
        ("0-D input",       INPUT_DIR,        INPUT_DIR.is_dir()),
        ("0-D output",      OUTPUT_DIR,       OUTPUT_DIR.is_dir()),
        ("MoDELib root",    MODELIB_ROOT,     MODELIB_ROOT.is_dir()),
        ("MoDELib build",   MODELIB_BUILD,    MODELIB_BUILD.is_dir()),
        ("MoDELib DDomp",   modelib_ddomp() or MODELIB_BUILD / "tools/DDomp", modelib_ddomp() is not None),
        ("Zr material",     MODELIB_MATERIAL, MODELIB_MATERIAL.is_file()),
        ("coupled sim case", COUPLED_SIM_TUTORIAL, COUPLED_SIM_TUTORIAL.is_dir()),
        ("Gmsh generator",  GMSH_DIR / "generate_mesh.py", (GMSH_DIR / "generate_mesh.py").is_file()),
        ("build script",    MODELIB_BUILD_SCRIPT, MODELIB_BUILD_SCRIPT.is_file()),
    ]
    w = max(len(r[0]) for r in rows)
    return "\n".join(f"{'OK ' if ok else 'MISS'}  {name:<{w}}  {p}"
                     for name, p, ok in rows)


if __name__ == "__main__":
    print(describe())
