"""
modelib_fem.py — bridge from the ZrMicro operator-split coupling to the
spatially-resolved FEM mobile solve in MoDELib2-NNL
(https://github.com/mlm335/MoDELib2-NNL).

This is the "fast solve" of the two-time-scale scheme (see
Docs/Formulation/ZrMicro_MoDELib2_two_time_scale_coupling.tex, the section
"Adopted Implementation: Operator-Split QSSA"). It replaces the analytic
boundary-layer PLACEHOLDER in coupling_demo.ipynb with a real finite-element
reaction-diffusion solve: given the current immobile loop sink field Q(x), it
returns the steady mobile concentrations C_M*(x) on the mesh quadrature points.

------------------------------------------------------------------------------
STATUS / ENVIRONMENT
------------------------------------------------------------------------------
MoDELib2-NNL is a C++20 library built with GCC/Clang (`-Ofast -march=native
-fopenmp`) and depends on Eigen, pybind11, and (optionally) Boost / SuiteSparse
/ FFTW. Its Python interface is the `pyMoDELib` pybind11 module
(tools/pyMoDELib/). On a machine where `pyMoDELib` is built and importable this
module drives it directly; otherwise `MoDELibFEMSolver.available()` returns
False and the notebook falls back to the analytic placeholder. Nothing here
requires MoDELib at import time.

------------------------------------------------------------------------------
COUPLING INTERFACE (verified against the cloned repo)
------------------------------------------------------------------------------
A spatial cluster-dynamics run is a *simulation directory* containing
`inputFiles/` (see tutorials/irradiation_singlecrystal/generateInputFiles.py):

  DD.txt                 master traits: useClusterDynamics=1, useFEM=1,
                         climbSolverType=Galerkin, Nsteps, timeSteppingMethod,
                         dtMax, outputQuadraturePoints=1, ...
  Zr4.txt                material (Library/Materials/Zr4.txt)
  ElasticDeformation.txt applied stress (ExternalStress0)
  polycrystal.txt        mesh + temperature + box scaling + grain orientation
  <mesh>.msh             Library/Meshes/unitCube_15K.msh
  aLoopsDensity.txt      <a> prismatic loops (slipSystemIDs, targetDensity,
                         loopRadiusMean, areVacancyLoops)  <- a1/a2/a3 sink field
  frankLoopsDensity.txt  <c> Frank/vacancy loops (targetDensity, radius, ...)
  initialMicrostructure.txt   lists the microstructure files

It is driven from Python via `pyMoDELib`
(see python/modelibPy11.py):

  import pyMoDELib
  ddBase  = pyMoDELib.DislocationDynamicsBase(simulationDir)
  crystal = pyMoDELib.DefectiveCrystal(ddBase)     # builds the CD/FEM system
  crystal.runSteps()                               # advances the FEM solve

The per-quadrature-point loop sink field is set from the ZrMicro slow state Q by
writing the density/radius of each loop family into aLoopsDensity.txt (a1/a2/a3)
and frankLoopsDensity.txt (c). The steady mobile field C_M*(x) is read back from
MoDELib's quadrature output (outputQuadraturePoints=1).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


class MoDELibNotBuilt(RuntimeError):
    """Raised when a real FEM solve is requested but MoDELib is not built."""


_BUILD_HINT = (
    "pyMoDELib is not importable. MoDELib2-NNL must be built first (C++20 + Eigen "
    "+ pybind11; GCC/Clang, not MSVC). See "
    "Docs/Formulation/MoDELib_build_and_coupling_notes.md. After building, add "
    "<MoDELib2-NNL>/build/tools/pyMoDELib to sys.path or pass modelib_build_dir."
)


def pymodelib_available(modelib_build_dir=None):
    """True if the pyMoDELib pybind11 module can be imported.

    NOTE: pyMoDELib built in WSL/Linux is an ELF .so and CANNOT be imported by
    Windows Python. For a WSL build, use the DDomp executable path with
    wsl_exec=True instead (the in-process pyMoDELib path is Linux-Python only).
    """
    if modelib_build_dir is not None:
        p = str(Path(modelib_build_dir) / "tools" / "pyMoDELib")
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.util.find_spec("pyMoDELib") is not None


def windows_to_wsl_path(p):
    """Translate a Windows path (d:/x or d:\\x) to a WSL path (/mnt/d/x)."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


class MoDELibFEMSolver:
    """Drive the MoDELib2-NNL FEM cluster-dynamics mobile solve as the fast step.

    Parameters
    ----------
    sim_dir : str | Path
        A MoDELib simulation directory (must contain, or will be populated with,
        an inputFiles/ tree). Use a copy of tutorials/irradiation_singlecrystal.
    modelib_root : str | Path | None
        Path to the cloned MoDELib2-NNL repo (for Library templates). Optional if
        sim_dir already has a complete inputFiles/.
    modelib_build_dir : str | Path | None
        Path to <MoDELib2-NNL>/build so pyMoDELib can be located on sys.path.

    Notes
    -----
    This object is inert until `solve()` is called. `available()` lets the caller
    (e.g. the coupling notebook) decide between this and the analytic placeholder.
    """

    # Map ZrMicro's four crystallographic loop families to MoDELib microstructure
    # inputs. a1/a2/a3 are <a> prismatic loops (aLoopsDensity.txt), c is the basal
    # Frank/vacancy family (frankLoopsDensity.txt). slipSystemIDs for the three
    # prism variants follow the irradiation_singlecrystal tutorial (6,8,10).
    A_VARIANT_SLIP_IDS = (6, 8, 10)

    def __init__(self, sim_dir, modelib_root=None, modelib_build_dir=None,
                 wsl_exec=False, wsl_distro=None):
        self.sim_dir = Path(sim_dir)
        self.modelib_root = Path(modelib_root) if modelib_root else None
        self.modelib_build_dir = Path(modelib_build_dir) if modelib_build_dir else None
        # wsl_exec: DDomp was built in WSL (Linux ELF); invoke it via wsl.exe and
        # translate Windows paths to /mnt/<drive>/... . Use this when MoDELib is
        # built in WSL but the coupling driver runs in Windows Python.
        self.wsl_exec = bool(wsl_exec)
        self.wsl_distro = wsl_distro
        self._pm = None  # lazily imported pyMoDELib

    # -- capability -----------------------------------------------------------
    def available(self):
        """True if EITHER the DDomp executable or pyMoDELib is built."""
        return (self.find_ddomp() is not None) or pymodelib_available(self.modelib_build_dir)

    def find_ddomp(self):
        """Locate the built DDomp executable (tools/DDomp), or None.

        DDomp is MoDELib's standalone runner: `DDomp <simDir>` reads inputFiles/
        and advances the FEM cluster-dynamics solve, writing evl/ and F/ output.
        This is the file-based driver, analogous to ZrMicro's solver.exe.
        """
        if self.modelib_build_dir is None:
            return None
        # A WSL/Linux build produces an extension-less ELF "DDomp"; a native
        # Windows build produces "DDomp.exe".
        names = ("DDomp",) if self.wsl_exec else (
            ("DDomp.exe",) if sys.platform == "win32" else ("DDomp",))
        ddir = Path(self.modelib_build_dir) / "tools" / "DDomp"
        for sub in ("", "Release", "Debug"):
            for name in names:
                c = (ddir / sub / name) if sub else (ddir / name)
                if c.exists():
                    return c
        return None

    def _import_pymodelib(self):
        if self._pm is None:
            if not self.available():
                raise MoDELibNotBuilt(_BUILD_HINT)
            import pyMoDELib  # noqa: E402  (import deferred until a real solve)
            self._pm = pyMoDELib
        return self._pm

    # -- write the immobile sink field into MoDELib inputs --------------------
    def write_sink_field(self, Q_per_variant):
        """Write the ZrMicro loop state into the MoDELib microstructure inputs.

        Parameters
        ----------
        Q_per_variant : dict
            {'a1': (N_m3, r_m), 'a2': (N_m3, r_m), 'a3': (N_m3, r_m),
             'c':  (N_m3, r_m)}  number density [m^-3] and mean radius [m] per
            loop family (e.g. from modelib_export.build_modelib_fields).

        Updates inputFiles/aLoopsDensity.txt and inputFiles/frankLoopsDensity.txt
        in sim_dir using the same key=value format the tutorial uses
        (python/modlibUtils.setInputVector / setInputVariable).
        """
        sys_path_modlib = None
        if self.modelib_root is not None:
            sys_path_modlib = str(self.modelib_root / "python")
            import sys
            if sys_path_modlib not in sys.path:
                sys.path.insert(0, sys_path_modlib)
        try:
            from modlibUtils import setInputVector, setInputVariable
        except ImportError as e:
            raise MoDELibNotBuilt(
                "Could not import modlibUtils from MoDELib2-NNL/python "
                "(pass modelib_root). " + str(e)
            )

        inp = self.sim_dir / "inputFiles"
        aL = inp / "aLoopsDensity.txt"
        cL = inp / "frankLoopsDensity.txt"
        if not aL.exists() or not cL.exists():
            raise FileNotFoundError(
                f"Expected microstructure templates in {inp}. Seed sim_dir from "
                "tutorials/irradiation_singlecrystal (run its generateInputFiles.py "
                "once) before coupling."
            )

        Na = np.array([Q_per_variant[k][0] for k in ("a1", "a2", "a3")], float)
        ra = np.array([Q_per_variant[k][1] for k in ("a1", "a2", "a3")], float)
        # three prism variants (interstitial a-loops). areVacancyLoops=0 for <a>.
        setInputVector(str(aL), "slipSystemIDs", np.array(self.A_VARIANT_SLIP_IDS), "")
        setInputVector(str(aL), "targetDensity", Na, "a1,a2,a3 number density [m^-3]")
        setInputVector(str(aL), "loopRadiusMean", ra, "a1,a2,a3 mean radius [m]")
        setInputVector(str(aL), "loopRadiusStd", np.zeros(3), "")
        setInputVector(str(aL), "areVacancyLoops", np.zeros(3, int), "")

        Nc, rc = Q_per_variant["c"]
        setInputVariable(str(cL), "targetDensity", repr(float(Nc)))
        setInputVariable(str(cL), "radiusDistributionMean", repr(float(rc)))
        setInputVariable(str(cL), "areVacancyLoops", "1")

    # -- run the FEM solve and read back the mobile field ---------------------
    def solve(self, Q_per_variant, nsteps=1):
        """Run the MoDELib FEM cluster-dynamics solve for the current sink field.

        Returns
        -------
        dict with 'points' (M,3) quadrature coordinates and 'C_M' (M,4) mobile
        concentrations [Cv,Ci,C2i,C3i] at those points.

        Raises MoDELibNotBuilt if pyMoDELib is unavailable.
        """
        self.write_sink_field(Q_per_variant)

        ddomp = self.find_ddomp()
        if ddomp is not None:
            # Primary path: the standalone DDomp executable (file-based, like
            # ZrMicro's solver.exe). `DDomp <simDir>` advances the FEM CD solve.
            if self.wsl_exec:
                # Linux binary built in WSL: run it through wsl.exe with
                # /mnt/<drive>/... paths.
                cmd = ["wsl.exe"]
                if self.wsl_distro:
                    cmd += ["-d", self.wsl_distro]
                cmd += ["-e", windows_to_wsl_path(ddomp),
                        windows_to_wsl_path(self.sim_dir)]
            else:
                cmd = [str(ddomp), str(self.sim_dir)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"DDomp failed (exit {proc.returncode}):\n{proc.stderr}")
        else:
            # Alternative path: drive pyMoDELib in-process.
            pm = self._import_pymodelib()
            ddBase = pm.DislocationDynamicsBase(str(self.sim_dir))
            crystal = pm.DefectiveCrystal(ddBase)
            crystal.runSteps() if nsteps != 1 else crystal.runSingleStep()

        return self._read_quadrature_mobile_field()

    def _read_quadrature_mobile_field(self):
        """Parse MoDELib's quadrature output for the mobile concentration field.

        With outputQuadraturePoints=1 MoDELib writes per-quadrature-point data to
        the F/ output folder. The exact column layout for the ClusterDynamics
        concentrations is confirmed on the first real run (the file is small and
        self-describing); this reader locates the latest F/ output and returns
        (points, C_M). Until a built MoDELib produces such a file, this raises a
        clear error so the caller falls back to the placeholder.
        """
        fdir = self.sim_dir / "F"
        candidates = sorted(fdir.glob("quadrature_*.txt")) if fdir.exists() else []
        if not candidates:
            raise MoDELibNotBuilt(
                "No quadrature output found in "
                f"{fdir} (run a built MoDELib once with outputQuadraturePoints=1 "
                "to produce it). The column map for [x y z Cv Ci C2i C3i] is then "
                "fixed in _read_quadrature_mobile_field()."
            )
        data = np.loadtxt(candidates[-1])
        # Column convention (confirm against the header on first real run):
        #   0:3 = x,y,z ; then the mobile concentrations in ZrMicro order.
        points = data[:, 0:3]
        C_M = data[:, 3:7]
        return {"points": points, "C_M": C_M}


def seed_sim_dir_from_tutorial(modelib_root, dest, tutorial="zrmicro_coupled",
                               copy_input_files=False):
    """Seed a MoDELib simulation directory for coupling from a tutorial case.

    ``zrmicro_coupled`` is the DisloCluster verification case: its
    ``inputFiles/`` already reference ``Zr3d_ghoniem.txt``, the material file
    whose cluster-dynamics parameters come from the 0-D code.

    Parameters
    ----------
    modelib_root : str | Path      — the MoDELib checkout (``paths.MODELIB_ROOT``)
    dest : str | Path              — simulation directory to create
    tutorial : str                 — case name under ``tutorials/``
    copy_input_files : bool
        False (default) copies only ``generateInputFiles.py``; the caller then
        runs it once with MoDELib's MicrostructureGenerator to populate
        ``inputFiles/``. True copies an existing ``inputFiles/`` tree verbatim,
        which is faster but inherits that case's mesh and dose schedule.

    After seeding, ``MoDELibFEMSolver.write_sink_field`` overrides the loop
    densities on every dose step.
    """
    src = Path(modelib_root) / "tutorials" / tutorial
    if not src.is_dir():
        raise FileNotFoundError(f"No such MoDELib tutorial: {src}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    gen = src / "generateInputFiles.py"
    if gen.is_file():
        shutil.copy2(gen, dest / "generateInputFiles.py")

    if copy_input_files:
        src_inputs = src / "inputFiles"
        if not src_inputs.is_dir():
            raise FileNotFoundError(f"{src} has no inputFiles/ to copy")
        shutil.copytree(src_inputs, dest / "inputFiles", dirs_exist_ok=True)
        for sub in ("evl", "F"):
            (dest / sub).mkdir(exist_ok=True)
    return dest
