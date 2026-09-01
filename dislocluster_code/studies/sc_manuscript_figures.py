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
#: loop absorption on, 500 nm hexagonal single crystal, 0 -> 40 dpa on the
#: 13-point grid 1e-4 .. 40, six substeps per interval and TWO fast solves per
#: interval (the earlier 10 dpa leg had one). 40 dpa rather than 10 so that the
#: <c> measurements, which reach 35 dpa, are inside the simulated range instead
#: of being compared against an extrapolation.
RUN = "20260831_072855_d4fbc06_sec7_calibrated_40dpa"

#: The 0-D refit whose parameter set that march carries.
REFIT = "20260830_180216_d4fbc06_calibration_final"

#: The same case WITHOUT the loop absorption channel -- the twin that makes the
#: denuded zone a measurement rather than a picture.
TWIN = "20260831_091311_d4fbc06_sec7_noGBloop_40dpa"

MANUSCRIPT = (paths.REPO_ROOT / "Docs" / "Formulation" / "self-consistent"
              / "SC_manuscript")
FIGDIR = MANUSCRIPT / "figures"

#: Early and late. 1e-4 dpa is inside the nucleation transient and 40 dpa is the
#: end of the march, so the pair brackets every field the paper discusses.
DOSES = (1e-4, 40.0)


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
            run, family=fam, doses=(1e-4, 1e-2, 1.0, 40.0),
            out_file=FIGDIR / f"loops_{fam}_4dose.png",
            panel_labels=True, verbose=False, loop_source="discrete")
        out.append(f)
        if verbose:
            print(f"  loops  {fam}: {', '.join(f'{u:g}' for u in used)} dpa "
                  f"-> {f.name}")
    return out


#: (source run, relative path, destination name). Copied, not redrawn.
#:
#: THE VOLUME-AVERAGED ARTIFACTS CARRY A DOSE-AXIS CROP, and it is set where
#: they are written rather than here, because redrawing them here is exactly
#: the second implementation the module header refuses. Regenerate them with
#:
#:     python -m dislocluster_code.post.volume_average <run> \
#:         --x-floor conservation_channels_v=1e-3 \
#:         --x-floor vacancy_fractions=1e-3
#:
#: Fig. 10 (the vacancy balance, both panels) opens at 1e-3 dpa rather than at
#: the march's first dose of 1e-4: across that first decade the channels have
#: not yet separated, so it spends a fifth of the panel on curves lying on top
#: of one another. `--x-floor` crops the view only -- the fractions panel is a
#: change since the first sample, so masking the data would move every curve on
#: it. Its interstitial counterpart, Fig. 11, is deliberately NOT cropped: the
#: point-defect panel beside it is where the transient is the subject.
COPIES = [
    (REFIT, "figures/fit_A_density.png", "fit_A_density.png"),
    (REFIT, "figures/fit_A_diameter.png", "fit_A_diameter.png"),
    (REFIT, "figures/fit_C_density.png", "fit_C_density.png"),
    (REFIT, "figures/fit_C_diameter.png", "fit_C_diameter.png"),
    (REFIT, "figures/fit_parity.png", "fit_parity.png"),
    (RUN, "gb/denuded_zone.png", "denuded_zone.png"),
    (RUN, "gb/gb_Cv.png", "gb_Cv.png"),
    (RUN, "gb/gb_N_c.png", "gb_N_c.png"),
    (RUN, "volume_average/conservation_channels_i.png", "conservation_i.png"),
    (RUN, "volume_average/conservation_channels_v.png", "conservation_v.png"),
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


def fig_loops_all(run, doses=(1e-4, 1e-2, 1.0, 40.0), verbose=True):
    """Every family superimposed, one panel per dose (Sec 7.8).

    `post.discrete_loops.render` already draws all eight populations into one
    axes, and the per-dose `loops_<tag>.png` it writes IS that combined view --
    the per-family montages exist because the combined one is dominated by
    whichever family is largest. That domination is the point here: the figure
    is meant to show the <c> population overtaking the <a> variants as dose
    accumulates, which no single-family panel can show.

    The panels are RENDERED here rather than composited from the run's own
    `discrete_loops/` output, because those are written only when
    OUTPUT['discrete_loops'] was on and at whatever doses that pass used.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from dislocluster_code.post import discrete_loops as DL

    tmp = FIGDIR / "_loops_all_panels"
    tmp.mkdir(parents=True, exist_ok=True)
    panels, used, counts = [], [], {}
    for dose in doses:
        d, pops, stats, box, P = DL.build(run, dose, region="interior",
                                          coalesce_pass=True,
                                          coplanar_tol="plane", seed=0,
                                          verbose=False)
        if sum(len(q) for q in pops) == 0:
            continue
        # Build the STEM first and append the extension after. Spelling this
        # as f"panel_{d:.4g}.png".replace(".", "p", 1) puts the substitution on
        # whichever dot comes first, which for a dose that formats without a
        # decimal point (1 dpa -> "1") is the extension separator: "panel_1ppng".
        tag = f"{d:.4g}".replace(".", "p").replace("-", "m")
        f = tmp / f"panel_{tag}.png"
        DL.render(pops, box[0], box[1], f, title="", domain_pts=P,
                  clip_to_domain=True, verbose=False, orientation=(not panels),
                  legend=False)
        counts[d] = {fam["key"]: len(q) for fam, q in zip(DL.FAMILIES, pops)
                     if len(q)}
        panels.append(f); used.append(d)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6), squeeze=False)
    for ax, f, d in zip(axes.ravel(), panels, used):
        ax.imshow(mpimg.imread(f)); ax.axis("off")
        ax.set_title(f"{d:g} dpa", fontsize=10)
    handles = [plt.Line2D([], [], color=fam["color"], lw=3,
                          label=fam["label"].replace("<c>", r"$\langle c\rangle$")
                                            .replace("<a>", r"$\langle a\rangle$"))
               for fam in DL.FAMILIES if fam["key"] in
               ("c", "a1", "a2", "a3", "a1v", "a2v", "a3v")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIGDIR / "loops_all_4dose.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  loops  all variants: {', '.join(f'{u:g}' for u in used)} dpa "
              f"-> {out.name}")
    return [out]


def fig_spectrum(run, verbose=True):
    """The reconstructed loop size distributions (Sec 8.8).

    Built here rather than copied, because nothing in the march writes them:
    the spectrum is the closure of `sec:moments` evaluated on the three carried
    moments, which is a post-processing step and not an output.
    """
    from dislocluster_code.post import size_spectrum as SS
    fams = ("c", "a1", "a1v")
    d, data = SS.build(run, doses=(1e-4, 1e-2, 1.0, 40.0), slugs=fams,
                       verbose=verbose)
    used = sorted({k[1] for k in data})
    a = SS.render(run, d, data, fams, used, FIGDIR / "size_spectrum_dose.png")
    b, _deltas = SS.render_regions(run, used, fams,
                                   FIGDIR / "size_spectrum_regions.png",
                                   verbose=False)
    return [a, b]


BUILDERS = {
    "mesh": fig_mesh,
    "spectrum": fig_spectrum,
    "mobile": fig_mobile,
    "moments": fig_moments,
    "loops": fig_loops_discrete,
    "loops_all": fig_loops_all,
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
