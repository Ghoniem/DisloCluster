"""
setup_domain.py — stage a MoDELib3 cluster-dynamics case on a fresh mesh.

``setup_standalone`` stages a case on the *reference* mesh, where the CD node
set is already known and a seed configuration already exists. A new mesh has
neither: the CD trial functions live on second-order elements, so the node set
is not the ``.msh`` vertex list and cannot be derived from the mesh file. Only
MoDELib knows it, and it writes it out as ``evl/cdNodes.txt`` on its first run.

The bootstrap that resolves the circularity:

  1. write ``evl/evl_0.txt`` with all ten header counts zero — a valid, empty
     configuration, which is what the microstructure generator produces for a
     case with ``useDislocations=0``;
  2. run DDomp once with ``useImmobileSolver=0``. That runs
     ``solveMobileClusters`` and passes the immobile field through untouched, so
     it writes ``cdNodes.txt`` and an ``evl_0.txt`` carrying a **pristine**
     immobile population and the correct steady mobile field for it;
  3. that file is the scaffold every later configuration is built from.

Because the immobile solver is off, no dose is accumulated by the bootstrap
whatever ``dtMax`` says — nothing evolves. The result is exactly the pristine
starting state, which Section "Removing the 0-D seed entirely" of the
architecture note shows is a legitimate place to start a march.

USAGE
-----
    python -m py_utils.setup_domain --size-nm 500 --layer-nm 150 \
        --layers-across 5 --interior-lc-nm 120 --name zr3d_500nm
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


from dislocluster_code import paths                                   # noqa: E402
from dislocluster_code.coupling import field as mfield                 # noqa: E402
from dislocluster_code.coupling.qssa import get_dd_scalar, set_dd_scalar  # noqa: E402
from dislocluster_code.staging.standalone import TEMPLATE, FAST_STEP_SETTINGS  # noqa: E402

EMPTY_EVL = "\n".join(["0"] * mfield.N_HEADER) + "\n"


def build_mesh(size_nm, layer_nm, layers_across, interior_lc_nm,
               element_order=1, out_dir=None, verbose=True):
    """Generate the mesh and return ``(path, F_b, n_tets, n_nodes)``."""
    if str(paths.GMSH_DIR) not in sys.path:
        sys.path.insert(0, str(paths.GMSH_DIR))
    import generate_mesh as gm

    spec = gm.MeshSpec(
        geometry="cubic", size_nm=size_nm, lc_nm=interior_lc_nm,
        element_order=element_order, boundary_layer=True,
        boundary_thickness_nm=layer_nm, boundary_layers_across=layers_across,
        optimize=True, verbosity=0)
    res = gm.generate(spec, out_dir=out_dir, force=False)
    if verbose:
        print(f"mesh: {res.path.name}")
        print(f"  {size_nm:g} nm cube, boundary layer {layer_nm:g} nm at "
              f"{layer_nm / layers_across:g} nm, interior {interior_lc_nm:g} nm")
        print(f"  {res.n_tets} tets, {res.n_nodes} mesh nodes")
    return res


def stage(sim_dir, mesh_res, template=None, verbose=True):
    """Copy the template case and point it at the new mesh."""
    sim_dir, template = Path(sim_dir), Path(template or TEMPLATE)
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    (sim_dir / "evl").mkdir(parents=True)
    (sim_dir / "F").mkdir()
    shutil.copytree(template / "inputFiles", sim_dir / "inputFiles")

    lib = paths.MODELIB_ROOT / "Library"
    (sim_dir / "inputFiles" / paths.MODELIB_MATERIAL.name).write_bytes(
        paths.MODELIB_MATERIAL.read_bytes().replace(b"\r\n", b"\n"))
    # Drop the reference mesh and install the new one, LF-converted.
    for old in (sim_dir / "inputFiles").glob("*.msh"):
        old.unlink()
    mesh_name = mesh_res.path.name
    (sim_dir / "inputFiles" / mesh_name).write_bytes(
        mesh_res.path.read_bytes().replace(b"\r\n", b"\n"))

    # polycrystal.txt: mesh name and the deformation gradient that restores the
    # physical size. The mesh is written normalised to the unit box, so F is the
    # only place the physical dimension appears.
    pc = sim_dir / "inputFiles" / "polycrystal.txt"
    txt = pc.read_text(encoding="utf-8", errors="replace")
    txt = re.sub(r"^meshFile\s*=.*$", f"meshFile={mesh_name}; # mesh file",
                 txt, count=1, flags=re.M)
    F = np.asarray(mesh_res.F_b, dtype=float)
    fblock = (f"F={F[0, 0]:.10g} {F[0, 1]:.10g} {F[0, 2]:.10g}\n"
              f"  {F[1, 0]:.10g} {F[1, 1]:.10g} {F[1, 2]:.10g}\n"
              f"  {F[2, 0]:.10g} {F[2, 1]:.10g} {F[2, 2]:.10g}; "
              f"# mesh deformation gradient, x = F*(X-X0)")
    # `.*?` spans the three lines of the F block up to its first ';'; the
    # trailing comment must then be matched with [^\n]* rather than `.*`, or
    # DOTALL lets it run greedily to the last ';' in the file and delete
    # periodicFaceIDs and C2G1 along with it.
    txt, n = re.subn(r"^F=.*?;[^\n]*$", fblock, txt, count=1,
                     flags=re.M | re.S)
    if n != 1:
        raise KeyError("polycrystal.txt has no F=...; block to replace")
    for key in ("periodicFaceIDs", "C2G1", "X0"):
        if not re.search(rf"^{key}\s*=", txt, re.M):
            raise KeyError(f"polycrystal.txt lost its {key} line")
    pc.write_text(txt, encoding="utf-8")

    (sim_dir / "evl" / "evl_0.txt").write_text(EMPTY_EVL, encoding="utf-8")
    if verbose:
        print(f"staged {sim_dir}")
        print(f"  meshFile={mesh_name}, F diag = "
              f"{F[0, 0]:.1f} {F[1, 1]:.1f} {F[2, 2]:.1f} b")
    return sim_dir


def bootstrap(sim_dir, verbose=True):
    """One DDomp call with the immobile solver off: writes cdNodes + scaffold."""
    sim_dir = Path(sim_dir)
    dd = sim_dir / "inputFiles" / "DD.txt"
    for k, v in (("useImmobileSolver", "0"), ("Nsteps", "1"),
                 ("startAtTimeStep", "0"), ("outputFrequency", "1"),
                 *FAST_STEP_SETTINGS.items()):
        set_dd_scalar(dd, k, v)

    # DDomp writes cdNodes.txt to the RELATIVE path "evl/cdNodes.txt", i.e.
    # against its working directory rather than against the simulation folder
    # it was handed. Launched from anywhere else the write lands outside the
    # case, or fails silently when no evl/ exists there -- which is why every
    # earlier case had its cdNodes.txt copied in by hand instead. Running from
    # inside the case makes the relative path resolve where it should.
    cmd, cwd = paths.ddomp_cmd(sim_dir, in_case_dir=True)
    if verbose:
        print(f"bootstrapping the CD node set in {sim_dir}", flush=True)
    t0 = time.perf_counter()
    log = sim_dir / "bootstrap.log"
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                           cwd=cwd)
    wall = time.perf_counter() - t0
    out = log.read_text(encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"DDomp exited {r.returncode}; tail:\n{out[-3000:]}")
    if "immobile solver SKIPPED" not in out:
        raise RuntimeError("DDomp did not skip the immobile solver — the "
                           "binary predates useImmobileSolver, or DD.txt was "
                           "not reconfigured. The bootstrap would have "
                           "advanced the state by a full dose step.")

    nodes = mfield.read_cd_nodes(sim_dir / "evl")
    ev = mfield.EvlFile(sim_dir / "evl" / "evl_0.txt")
    scaffold = sim_dir / "evl" / "evl_scaffold_pristine.txt"
    shutil.copy2(sim_dir / "evl" / "evl_0.txt", scaffold)
    if verbose:
        print(f"  {wall:.0f} s, {nodes.shape[0]} CD nodes, "
              f"CD block {ev.cd.shape}")
        imm = ev.immobile
        print(f"  immobile at start: max {imm.max():.3e} (pristine if ~0)")
        print(f"  mobile Cv {ev.mobile[:, 0].min():.3e} .. "
              f"{ev.mobile[:, 0].max():.3e}")
        print(f"  scaffold -> {scaffold.name}")
    return nodes, scaffold


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-nm", type=float, default=500.0)
    ap.add_argument("--layer-nm", type=float, default=150.0)
    ap.add_argument("--layers-across", type=int, default=5)
    ap.add_argument("--interior-lc-nm", type=float, default=120.0)
    ap.add_argument("--element-order", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args(argv)

    name = args.name or f"zr3d_{args.size_nm:g}nm"
    sim_dir = paths.MODELIB_ROOT / "tutorials" / name
    res = build_mesh(args.size_nm, args.layer_nm, args.layers_across,
                     args.interior_lc_nm, args.element_order)
    stage(sim_dir, res)
    info = dict(name=name, size_nm=args.size_nm, layer_nm=args.layer_nm,
                layers_across=args.layers_across,
                layer_lc_nm=args.layer_nm / args.layers_across,
                interior_lc_nm=args.interior_lc_nm,
                mesh=res.path.name, n_tets=res.n_tets,
                n_mesh_nodes=res.n_nodes,
                generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                git_hash=paths.git_hash())
    if not args.no_bootstrap:
        nodes, scaffold = bootstrap(sim_dir)
        info.update(n_cd_nodes=int(nodes.shape[0]), scaffold=scaffold.name)
    (sim_dir / "domain.json").write_text(json.dumps(info, indent=2),
                                         encoding="utf-8")
    print(f"\nwrote {sim_dir / 'domain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
