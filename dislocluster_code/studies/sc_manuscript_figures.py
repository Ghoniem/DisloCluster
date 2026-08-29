"""Every figure the self-consistent SRCD manuscript includes, built from a run.

    python -m dislocluster_code.studies.sc_manuscript_figures            # all
    python -m dislocluster_code.studies.sc_manuscript_figures --only mesh

The manuscript is `Docs/Formulation/self-consistent/SC_manuscript`. This module
exists so that every figure in it is reproducible from one command against one
named run, rather than assembled by hand from whatever was on disk -- which is
how a paper ends up carrying a figure from a superseded run.

WHAT IS BUILT HERE AND WHAT IS COPIED. The field grids are RENDERED, because
`plot_field_panels` lays out one row per species and one column per dose
natively -- shared colorbar per row, one orientation triad on the bottom-left
panel -- and compositing the run's own single-quantity panels instead leaves a
band of white between every drawing and its colorbar. The calibration figures
and the grain-boundary profiles are COPIED unchanged: they are the artifacts the
refit and the march themselves produced, and redrawing them here would create a
second implementation to keep correct.

THE RUN IS NAMED, NOT DISCOVERED. `paths.latest_run()` would silently repoint
the manuscript at whatever ran last.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

from dislocluster_code import paths
from dislocluster_code.post import movies as movies_mod
from dislocluster_code.post.fields import plot_field_panels
from dislocluster_code.post.report import write_mesh_figure

#: The march the manuscript reports. Refitted parameter set, grain-boundary
#: loop absorption on, 500 nm hexagonal single crystal, 0 -> 10 dpa.
RUN = "20260828_190015_431eb18_500nmHex_Refit_GBabsorb"

#: The 0-D refit whose parameter set that march carries.
REFIT = "20260828_101638_387c368_Refit_Physical_Bias"

#: The same case WITHOUT the loop absorption channel -- the twin that makes the
#: denuded zone a measurement rather than a picture.
TWIN = "20260828_104231_387c368_500nmHex_Refit_PhysicalBias"

MANUSCRIPT = (paths.REPO_ROOT / "Docs" / "Formulation" / "self-consistent"
              / "SC_manuscript")
FIGDIR = MANUSCRIPT / "figures"

#: Early and late. 1e-4 dpa is inside the nucleation transient and 10 dpa is the
#: end of the march, so the pair brackets every field the paper discusses.
DOSES = (1e-4, 10.0)


def _run_dir(name):
    for root in paths.OUTPUT_DIRS:
        p = Path(root) / name
        if p.exists():
            return p
    raise SystemExit(f"run {name!r} not found under {list(paths.OUTPUT_DIRS)}")


def _blocks(run):
    doses, nodes, frames = movies_mod.cd_blocks(run)
    idx = [int(np.argmin(np.abs(np.asarray(doses) - d))) for d in DOSES]
    return doses, nodes, frames, idx


def fig_mesh(run, verbose=True):
    """The FE mesh, cut so the boundary-layer refinement is visible (Sec 8.1)."""
    import json
    case = json.loads((run / "config.json").read_text())["derived"]["sim_dir"]
    dom = json.loads((Path(case) / "domain.json").read_text())
    msh = paths.GMSH_DIR / "meshes" / dom["mesh"]
    if not msh.exists():
        msh = Path(case) / "inputFiles" / dom["mesh"]
    out = FIGDIR / "fe_mesh.png"
    write_mesh_figure(msh, out, cut="y", title=None)
    if verbose:
        print(f"  mesh   {dom['n_tets']} tets, {dom['n_mesh_nodes']} mesh nodes, "
              f"{dom['n_cd_nodes']} CD nodes -> {out.name}")
    return out


def _grid(run, species, out_name, verbose=True):
    doses, nodes, frames, idx = _blocks(run)
    out = FIGDIR / out_name
    plot_field_panels(None, idx, [float(doses[i]) for i in idx],
                      species=tuple(species), plane=("y", "z"),
                      fields={i: frames[i] for i in idx},
                      out_file=out, column_titles=True, title=None)
    if verbose:
        print(f"  grid   {len(species)}x{len(idx)}  {', '.join(species)} "
              f"-> {out.name}")
    return out


def fig_mobile(run, verbose=True):
    """Four mobile species against dose (Sec 8.2)."""
    return _grid(run, ("Cv", "Ci", "C2i", "C3i"), "mobile_early_late.png",
                 verbose)


def fig_moments(run, verbose=True):
    """The three moments of the basal and the prismatic family (Sec 8.3).

    ONE FAMILY PER FIGURE, not one moment per figure. The three moments of a
    family are what the closure relates -- `Delta = q n / c^2` -- so the row
    that a reader has to compare across is the family's own, and putting `N` of
    all nine families in one figure would put the comparison in the wrong place.
    """
    return [_grid(run, ("n_vL", "c_vL", "q_vL"), "moments_cf.png", verbose),
            _grid(run, ("n_a1", "c_a1", "q_a1"), "moments_a1.png", verbose)]


def fig_loops_discrete(run, verbose=True):
    """The discrete population at four doses (Sec 8.7).

    Built by `post.loop_montage`, which renders each panel and composites --
    correct here, where the panels are the `_discrete` overlays and the figure
    wants four doses of ONE family rather than a species grid.
    """
    from dislocluster_code.post import loop_montage
    out = []
    for fam in ("c", "a1"):
        f, used = loop_montage.montage(
            run, family=fam, doses=(1e-4, 1e-2, 1.0, 10.0),
            out_file=FIGDIR / f"loops_{fam}_4dose.png",
            panel_labels=True, verbose=False, loop_source="discrete")
        out.append(f)
        if verbose:
            print(f"  loops  {fam}: {', '.join(f'{u:g}' for u in used)} dpa "
                  f"-> {f.name}")
    return out


#: (source run, relative path, destination name). Copied, not redrawn.
COPIES = [
    (REFIT, "figures/fit_A_density.png", "fit_A_density.png"),
    (REFIT, "figures/fit_A_diameter.png", "fit_A_diameter.png"),
    (REFIT, "figures/fit_C_density.png", "fit_C_density.png"),
    (REFIT, "figures/fit_C_diameter.png", "fit_C_diameter.png"),
    (REFIT, "figures/fit_parity.png", "fit_parity.png"),
    (RUN, "gb/denuded_zone.png", "denuded_zone.png"),
    (RUN, "gb/gb_Cv.png", "gb_Cv.png"),
    (RUN, "gb/gb_Ci.png", "gb_Ci.png"),
    (RUN, "gb/gb_N_c.png", "gb_N_c.png"),
    (RUN, "gb/gb_N_a1.png", "gb_N_a1.png"),
    (RUN, "volume_average/conservation_channels_i.png", "conservation_i.png"),
    (RUN, "volume_average/conservation_channels_v.png", "conservation_v.png"),
    (RUN, "volume_average/interstitial_fractions.png", "fractions_i.png"),
    (RUN, "volume_average/vacancy_fractions.png", "fractions_v.png"),
    (RUN, "volume_average/loop_density.png", "loop_density.png"),
    (RUN, "volume_average/loop_sizes_vs_experiment.png", "loop_sizes_vs_exp.png"),
    (RUN, "volume_average/point_defects.png", "point_defects.png"),
]


def copies(verbose=True):
    out = []
    for src_run, rel, dest in COPIES:
        src = _run_dir(src_run) / rel
        if not src.exists():
            print(f"  MISSING {src}")
            continue
        shutil.copy2(src, FIGDIR / dest)
        out.append(FIGDIR / dest)
    if verbose:
        print(f"  copied {len(out)} of {len(COPIES)} artifact figures")
    return out


BUILDERS = {
    "mesh": fig_mesh,
    "mobile": fig_mobile,
    "moments": fig_moments,
    "loops": fig_loops_discrete,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=sorted(BUILDERS) + ["copies"])
    a = ap.parse_args(argv)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    run = _run_dir(RUN)
    print(f"{run.name}\n  -> {FIGDIR}\n")
    want = a.only or (sorted(BUILDERS) + ["copies"])
    for key in want:
        if key == "copies":
            copies()
        else:
            BUILDERS[key](run)
    n = len(list(FIGDIR.glob("*.png")))
    print(f"\n  {n} figures in {FIGDIR}")


if __name__ == "__main__":
    main()
