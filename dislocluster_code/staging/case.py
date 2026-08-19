"""case.py — build a runnable MoDELib case from a :class:`~dislocluster_code.config.SimulationConfig`.

Three steps, in order, all idempotent:

1. :func:`build_mesh`     — Gmsh writes (or reuses) the ``.msh``.
2. :func:`stage_case`     — copy the template deck, install the mesh, the
                            material and the boundary conditions.
3. :func:`bootstrap`      — one DDomp call with the immobile solver OFF, which
                            is the only way to learn the CD node set.

:func:`ensure_domain` runs all three and skips the work when the staged case on
disk already matches the configuration, which it records in ``domain.json``.

WHY THE BOOTSTRAP EXISTS
------------------------
The cluster-dynamics trial functions live on second-order elements, so the CD
node set is NOT the ``.msh`` vertex list and cannot be derived from the mesh
file. Only MoDELib knows it, and it writes it out as ``evl/cdNodes.txt`` on its
first run. The circularity -- a configuration needs the node set, the node set
needs a run -- is broken by running once with ``useImmobileSolver=0``: that
solves the mobile field and passes the immobile field through untouched, so no
dose is accumulated whatever ``dtMax`` says, and the result is exactly the
pristine starting state.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dislocluster_code import paths
from dislocluster_code.coupling import field as mfield
from dislocluster_code.staging import inputs as minputs

__all__ = ["DEFAULT_TEMPLATE", "build_mesh", "stage_case", "bootstrap",
           "ensure_domain", "read_domain", "StagingError"]

# The deck every case is cloned from: a complete, parseable inputFiles/ with the
# CD keys already present. Only the mesh, the material, the temperature, the
# boundary conditions and the dose schedule are then overwritten, so a change
# upstream in MoDELib's Library does not silently alter a staged case.
DEFAULT_TEMPLATE = paths.MODELIB_TUTORIALS / "zrmicro_seeded"

EMPTY_EVL = "\n".join(["0"] * mfield.N_HEADER) + "\n"


class StagingError(RuntimeError):
    """The case could not be built, or the bootstrap did not do what it claims."""


def build_mesh(cfg, out_dir=None, verbose=True):
    """Generate (or reuse) the mesh. Returns ``generate_mesh.MeshResult``."""
    if str(paths.GMSH_DIR) not in sys.path:
        sys.path.insert(0, str(paths.GMSH_DIR))
    import generate_mesh as gm

    spec = cfg.mesh.spec(cfg.geometry)
    res = gm.generate(spec, out_dir=out_dir, force=cfg.mesh.regenerate)
    if verbose:
        lay = (f", boundary layer {cfg.mesh.boundary_thickness_nm:g} nm at "
               f"{spec.layer_lc_nm():g} nm" if cfg.mesh.boundary_layer else "")
        print(f"mesh: {res.path.name}")
        print(f"  {cfg.geometry.type}, {cfg.geometry.size_nm:g} nm, "
              f"interior {spec.interior_lc_nm():g} nm{lay}")
        print(f"  {res.n_tets} tets, {res.n_nodes} mesh nodes")
    return res


def stage_case(cfg, mesh_res, sim_dir=None, template=None, verbose=True):
    """Copy the template deck and point it at this configuration.

    Writes the mesh, the material, the temperature, ``F`` (which is the only
    place the physical size appears -- the mesh is normalized to the unit box),
    the periodic-face list and the mechanical load.
    """
    sim_dir = Path(sim_dir or cfg.sim_dir)
    template = Path(template or DEFAULT_TEMPLATE)
    if not (template / "inputFiles").is_dir():
        raise StagingError(f"template {template} has no inputFiles/")

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    (sim_dir / "evl").mkdir(parents=True)
    (sim_dir / "F").mkdir()
    shutil.copytree(template / "inputFiles", sim_dir / "inputFiles")

    # Material and mesh come from the Library with LF endings: MoDELib's parser
    # treats a trailing CR as part of the value and mis-reads the last field.
    mat = cfg.material.path
    minputs.copy_lf(mat, sim_dir / "inputFiles" / mat.name)
    for old in (sim_dir / "inputFiles").glob("*.msh"):
        old.unlink()
    minputs.copy_lf(mesh_res.path, sim_dir / "inputFiles" / mesh_res.path.name)

    minputs.write_polycrystal(
        sim_dir, mesh_name=mesh_res.path.name, F=mesh_res.F_b,
        temperature_K=cfg.material.temperature_K, material_file=mat.name,
        periodic_face_ids=cfg.boundary.periodic_face_ids)

    mu_SI = minputs.material_mu_SI(mat, cfg.material.temperature_K)
    minputs.write_elastic_deformation(sim_dir, cfg.boundary, mu_SI)

    (sim_dir / "evl" / "evl_0.txt").write_text(EMPTY_EVL, encoding="utf-8")

    if verbose:
        F = mesh_res.F_b
        bc = ("Dirichlet on every face" if cfg.boundary.all_dirichlet
              else f"periodic faces {list(cfg.boundary.periodic_face_ids)}")
        sig = cfg.boundary.applied_stress_MPa
        print(f"staged {sim_dir}")
        print(f"  meshFile={mesh_res.path.name}, F diag = "
              f"{F[0][0]:.1f} {F[1][1]:.1f} {F[2][2]:.1f} b")
        print(f"  {mat.name} at {cfg.material.temperature_K:g} K, {bc}")
        if any(sig):
            print(f"  applied stress [MPa] " +
                  " ".join(f"{k}={v:g}" for k, v in zip(minputs.VOIGT, sig) if v)
                  + f"  (mu = {mu_SI / 1e9:.3g} GPa)")
    return sim_dir


def bootstrap(sim_dir, verbose=True):
    """One DDomp call with the immobile solver off: writes cdNodes + scaffold."""
    sim_dir = Path(sim_dir)
    dd = sim_dir / "inputFiles" / "DD.txt"
    for k, v in (("useImmobileSolver", "0"), ("Nsteps", "1"),
                 ("startAtTimeStep", "0"), ("outputFrequency", "1"),
                 *minputs.FAST_STEP_SETTINGS.items()):
        minputs.set_dd_scalar(dd, k, v)

    # A run that died before writing anything leaves a ZERO-BYTE F/F_0.txt
    # behind, and the next DDomp SEGFAULTS reading it: the loader reports
    # "(0 entries)" and the deformation-gradient row is then indexed anyway. The
    # first attempt's failure would therefore look like a different, much more
    # confusing failure on the second. Clear the empty stubs; a real F_0.txt is
    # never empty.
    minputs.clear_empty_F(sim_dir)

    # in_case_dir: DDomp writes cdNodes.txt to the RELATIVE path
    # "evl/cdNodes.txt", i.e. against its working directory rather than against
    # the simulation folder it was handed. Launched from anywhere else the write
    # lands outside the case, or fails silently when no evl/ exists there.
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
        raise StagingError(f"DDomp exited {r.returncode}; tail:\n{out[-3000:]}")
    # DDomp reports a solver failure by printing the exception and exiting 0, so
    # the returncode above proves nothing. Say what actually happened: the
    # "did not skip the immobile solver" message below is right only when DDomp
    # ran to completion, and reading it after a linear-solver breakdown sends
    # you looking at the binary's age instead of at the solve.
    solver_failure = next((l.strip() for l in out.splitlines()
                           if "FixedDirichletSolver failed" in l
                           or "terminate called" in l), None)
    if solver_failure:
        raise StagingError(
            f"the fast (mobile) solve failed during the bootstrap: "
            f"{solver_failure}\n"
            f"  This is the linear solve inside MoDELib's mobile Newton "
            f"iteration, not the 0-D side.\n"
            f"  Check first that MoDELib3 was built against Eigen 3: against "
            f"Eigen 5 this breaks down on the first Newton step "
            f"(the configure step now refuses it).\n"
            f"  Otherwise it is the cold start of ZR3D_GHONIEM_CHANGES.md issue "
            f"12 -- the mobile field must cross nine orders from thermal "
            f"equilibrium in one solve.\n"
            f"  Log: {log}")
    if "immobile solver SKIPPED" not in out:
        raise StagingError(
            "DDomp did not skip the immobile solver — the binary predates "
            "useImmobileSolver, or DD.txt was not reconfigured. The bootstrap "
            "would have advanced the state by a full dose step.")

    nodes = mfield.read_cd_nodes(sim_dir / "evl")
    ev = mfield.EvlFile(sim_dir / "evl" / "evl_0.txt")
    scaffold = sim_dir / "evl" / "evl_scaffold_pristine.txt"
    shutil.copy2(sim_dir / "evl" / "evl_0.txt", scaffold)
    if verbose:
        print(f"  {wall:.0f} s, {nodes.shape[0]} CD nodes, "
              f"CD block {ev.cd.shape}")
        print(f"  immobile at start: max {ev.immobile.max():.3e} "
              f"(pristine if ~0)")
        print(f"  scaffold -> {scaffold.name}")
    return nodes, scaffold


def read_domain(sim_dir):
    """The ``domain.json`` of a staged case, or None."""
    f = Path(sim_dir) / "domain.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def ensure_domain(cfg, force=False, verbose=True):
    """Mesh + stage + bootstrap, skipped when the case on disk already matches.

    The check is on ``cfg.domain_key``, which covers the geometry, the mesh, the
    boundary conditions, the material and the temperature -- but NOT the dose
    grid, so two runs over different doses share one staged domain and one
    bootstrap.

    Returns the ``domain.json`` contents.
    """
    sim_dir = cfg.sim_dir
    have = read_domain(sim_dir)
    if (not force and have and have.get("domain_key") == cfg.domain_key
            and (sim_dir / "evl" / have.get("scaffold", "")).is_file()
            and (sim_dir / "evl" / "cdNodes.txt").is_file()):
        if verbose:
            print(f"domain up to date: {sim_dir.name} "
                  f"({have['n_cd_nodes']} CD nodes) — staging skipped")
        return have

    res = build_mesh(cfg, verbose=verbose)
    stage_case(cfg, res, sim_dir=sim_dir, verbose=verbose)
    nodes, scaffold = bootstrap(sim_dir, verbose=verbose)

    info = dict(
        case_name=cfg.case_name, domain_key=cfg.domain_key,
        geometry=cfg.geometry.type, size_nm=cfg.geometry.size_nm,
        element_order=cfg.mesh.element_order,
        boundary_layer=cfg.mesh.boundary_layer,
        layer_nm=cfg.mesh.boundary_thickness_nm,
        layers_across=cfg.mesh.boundary_layers_across,
        mesh=res.path.name, n_tets=res.n_tets, n_mesh_nodes=res.n_nodes,
        n_cd_nodes=int(nodes.shape[0]), scaffold=scaffold.name,
        material=cfg.material.file, material_sha=cfg.material_digest,
        temperature_K=cfg.material.temperature_K,
        periodic_face_ids=list(cfg.boundary.periodic_face_ids),
        applied_stress_MPa=list(cfg.boundary.applied_stress_MPa),
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_hash=paths.git_hash())
    (sim_dir / "domain.json").write_text(json.dumps(info, indent=2),
                                         encoding="utf-8")
    if verbose:
        print(f"wrote {sim_dir / 'domain.json'}")
    return info
