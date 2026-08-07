"""run_zr3d_singlecrystal.py — post-process the 3-D single-cubic-crystal run.

Renders the MoDELib (`Zr3d_ghoniem`) spatially-resolved cluster-dynamics output
for the single cubic crystal case `MoDELib3/tutorials/zrmicro_coupled` into a
timestamped run directory: interior field panels, per-family loop overlays,
grain-boundary profiles, and a `provenance.md`.

This replaces the ad-hoc script that produced the reference run
`20260804_114520_8a8439e_zr3d_ghoniem`; that script was never committed, so the
figure set was not reproducible. Everything here resolves through
`py_utils.paths`, so it runs wherever the repository is checked out.

The 3-D solve itself is NOT run here — it takes hours. This reads the `evl/`
output already present in the tutorial directory. To regenerate that output
first:

    wsl -e bash MoDELib3/tutorials/zrmicro_coupled/clean_run.sh

Usage
-----
    .DisloClusterVenv/Scripts/python.exe \\
        ZrClusterDynamics/ZrMicro/py_utils/run_zr3d_singlecrystal.py [options]

    --sim-dir DIR     simulation directory (default: the zrmicro_coupled case)
    --doses A B C D   doses [dpa] to render (default: 1 5 10 30)
    --tag NAME        run-directory suffix (default: zr3d_ghoniem)
    --max-nm X        extent of the gb/ profiles in nm (default: 150)
    --no-overlays     skip the four loop-overlay figures (they are the slow part)
    --n-loops N       platelets drawn per overlay panel (decorative; default 26)
"""

from __future__ import annotations

import argparse
import datetime
import platform
import sys
from pathlib import Path

import numpy as np

# Allow both `python py_utils/run_zr3d_singlecrystal.py` and `-m py_utils....`
_ZRMICRO = Path(__file__).resolve().parent.parent
if str(_ZRMICRO) not in sys.path:
    sys.path.insert(0, str(_ZRMICRO))

from py_utils import paths                                          # noqa: E402
from py_utils.modelib_fields import (                               # noqa: E402
    load_cd_fields, plot_field_panels, gb_distance, domain_faces,
    FAMILIES, B_SI, OMEGA_B3,
)
from py_utils import modelib_gb                                     # noqa: E402

# Output is written AFTER solve(), so evl_N holds the state at (N+1) dose steps
# of 1 dpa each: dose D dpa is step D-1.
DEFAULT_DOSES = (1.0, 5.0, 10.0, 30.0)

# Loop-radius exaggeration per family, matching the reference figures. <c> loops
# are ~6x larger than <a>, so a single factor would leave <a> invisible; the
# consequence is that sizes are NOT comparable between the two figures.
LOOP_SCALE = {0: 3.0, 1: 14.0, 2: 14.0, 3: 14.0}
FAMILY_SLUG = {0: "c", 1: "a1", 2: "a2", 3: "a3"}
FAMILY_MATH = {0: r"$\langle c\rangle$", 1: r"$\langle a\rangle_1$",
               2: r"$\langle a\rangle_2$", 3: r"$\langle a\rangle_3$"}

# The overlay background is the loop CONTENT field, not the density. Content is
# what sets the platelet radius that is being drawn on top of it, so the two
# carry the same information and the figure reads as one quantity; the density
# field is nearly flat in the interior (it spans <1% there) and would give the
# colour bar almost no range to work with.
FAMILY_BG = {0: "c_vL", 1: "c_a1", 2: "c_a2", 3: "c_a3"}


def dose_to_step(dose):
    """Dose [dpa] -> output step index, for a 1 dpa/step schedule."""
    return int(round(dose)) - 1


def interior_mask(P, interior_frac=0.5):
    """Nodes deeper than `interior_frac` of the domain's inradius.

    Defined on distance-from-the-boundary rather than on an axis-aligned box, so
    it is correct for a hexagonal prism as well as a cube. For a cube the two
    definitions coincide exactly: the deepest point is the centre at half the
    edge, so `d > 0.5 * d_max` is the same box the old form selected.
    """
    d = gb_distance(P)
    return d > interior_frac * d.max()


def interior_table(evl_dir, step, interior_frac=0.5):
    """Per-family (density, content, defects/loop, diameter) in the interior.

    The depletion shell at the Dirichlet faces is excluded (see interior_mask).
    """
    P, F = load_cd_fields(evl_dir, step)
    sel = interior_mask(P, interior_frac)

    rows = []
    for k, (label, ncol, ccol, bmag, _n, _c) in enumerate(FAMILIES):
        n_b3 = float(np.mean(F[sel, ncol]))          # loops per b^3
        c_at = float(np.mean(F[sel, ccol]))          # content, per atom
        n_m3 = n_b3 / B_SI ** 3
        stored_m3 = c_at / (OMEGA_B3 * B_SI ** 3)
        m = c_at / max(n_b3 * OMEGA_B3, 1e-300)      # defects per loop
        r_b = np.sqrt(max(m, 0.0) * OMEGA_B3 / (np.pi * bmag))
        rows.append((label, n_m3, stored_m3, m, 2.0 * r_b * B_SI * 1e9))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    ap.add_argument("--tag", default="zr3d_ghoniem")
    ap.add_argument("--max-nm", type=float, default=150.0)
    ap.add_argument("--no-overlays", action="store_true")
    # Purely decorative: how many sample sites are drawn as platelets. The
    # reference figure set used a larger, unrecorded value (roughly twice this),
    # so its overlays look denser. Carries no information either way -- the
    # radii come from the real local field and the physics tables are unaffected.
    ap.add_argument("--n-loops", type=int, default=26)
    a = ap.parse_args(argv)

    sim_dir = Path(a.sim_dir) if a.sim_dir else paths.COUPLED_SIM_TUTORIAL
    evl = sim_dir / "evl"
    if not (evl / "cdNodes.txt").is_file():
        raise SystemExit(
            f"No cluster-dynamics output in {evl}.\n"
            "cdNodes.txt is written by ClusterDynamicsFEM::writeNodePositions();\n"
            f"regenerate with:  wsl -e bash {sim_dir / 'clean_run.sh'}")

    doses = list(a.doses)
    steps = [dose_to_step(d) for d in doses]
    missing = [s for s in steps if not (evl / f"evl_{s}.txt").is_file()]
    if missing:
        raise SystemExit(f"Missing output steps {missing} in {evl}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.OUTPUT_DIR / f"{stamp}_{paths.git_hash()}_{a.tag}"
    d3, dgb = run_dir / "3d", run_dir / "gb"
    d3.mkdir(parents=True, exist_ok=True)
    dgb.mkdir(parents=True, exist_ok=True)

    P0, _ = load_cd_fields(evl, steps[0])
    lo, hi = P0.min(0), P0.max(0)
    span_b = hi - lo
    span_nm = span_b * B_SI * 1e9
    n_faces = len(domain_faces(P0)[0])
    inradius_nm = gb_distance(P0).max() * B_SI * 1e9
    print(f"simulation   : {sim_dir}")
    print(f"nodes        : {P0.shape[0]}")
    print(f"faces        : {n_faces}   (inradius {inradius_nm:.1f} nm)")
    print(f"domain       : {span_b[0]:.0f} x {span_b[1]:.0f} x {span_b[2]:.0f} b"
          f"  =  {span_nm[0]:.0f} x {span_nm[1]:.0f} x {span_nm[2]:.0f} nm")
    print(f"doses        : {doses}  (steps {steps})")
    print(f"run directory: {run_dir}\n")

    # ── interior fields, one column per dose ────────────────────────────────
    print("3d/ field panels")
    for name, species, title in (
        ("mobile_fields", ("Cv", "Ci", "C2i", "C3i"),
         "Mobile species, interior mid-plane cut"),
        ("immobile_density_fields", ("n_vL", "n_a1", "n_a2", "n_a3"),
         "Immobile loop number density"),
        ("immobile_content_fields", ("c_vL", "c_a1", "c_a2", "c_a3"),
         "Immobile loop defect content"),
    ):
        plot_field_panels(evl, steps, doses, species=species,
                          out_file=d3 / f"{name}.png", title=title)
        print(f"  {name}.png")

    if not a.no_overlays:
        for k, slug in FAMILY_SLUG.items():
            plot_field_panels(
                evl, steps, doses, species=(FAMILY_BG[k],),
                loop_family=k, loop_scale=LOOP_SCALE[k], n_loops=a.n_loops,
                out_file=d3 / f"loops_{slug}_overlay.png",
                title=(f"Zr3d_ghoniem: {FAMILY_MATH[k]} loop population, whole "
                       f"domain (platelet radii x{LOOP_SCALE[k]:g}; sizes not "
                       f"comparable between the <c> and <a> figures)"))
            print(f"  loops_{slug}_overlay.png")

    # ── grain-boundary profiles ─────────────────────────────────────────────
    print("gb/ profiles")
    for fn, name in ((modelib_gb.plot_density_profiles, "gb_density_profiles"),
                     (modelib_gb.plot_content_profiles, "gb_content_profiles"),
                     (modelib_gb.plot_size_profiles,    "gb_size_profiles"),
                     (modelib_gb.plot_mobile_profiles,  "gb_mobile_profiles")):
        fn(evl, steps, doses, out_file=dgb / f"{name}.png", max_nm=a.max_nm)
        print(f"  {name}.png")

    # ── provenance ──────────────────────────────────────────────────────────
    rows = interior_table(evl, steps[-1])
    size_hist = [(d, interior_table(evl, s)) for d, s in zip(doses, steps)]

    L = [
        f"# Zr3d_ghoniem 3-D output — {a.tag}", "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S} on "
        f"{platform.node()}",
        f"({platform.platform()}), DisloCluster at `{paths.git_hash()}`.", "",
        "Reproduce with:", "",
        "```",
        f"{paths.VENV_NAME}/Scripts/python.exe \\",
        "    ZrClusterDynamics/ZrMicro/py_utils/run_zr3d_singlecrystal.py \\",
        f"    --doses {' '.join(f'{d:g}' for d in doses)} --max-nm {a.max_nm:g}",
        "```", "",
        "## Source", "",
        f"MoDELib (`Zr3d_ghoniem`), simulation `{sim_dir.relative_to(paths.REPO_ROOT).as_posix()}`,",
        "1 dpa per output step at G = 1e-7 dpa/s, T = 573 K, zero applied stress.",
        "Loop nucleation is cascade PLUS homogeneous SIA clustering (i+3i, 2i+2i,",
        "2i+3i); see MoDELib3/ZR3D_GHONIEM_CHANGES.md issue 17.",
        f"Doses shown: {', '.join(f'{d:g} dpa' for d in doses)} "
        f"(output steps {steps}).",
        "Output is written AFTER solve(), so `evl_N` holds the state at N+1 dose",
        "steps.", "",
        "## Domain", "",
        f"{span_b[0]:.0f} x {span_b[1]:.0f} x {span_b[2]:.0f} b",
        f"= {span_nm[0]:.0f} x {span_nm[1]:.0f} x {span_nm[2]:.0f} nm bounding box,",
        f"{P0.shape[0]} finite-element nodes, {n_faces} planar faces, inradius",
        f"{inradius_nm:.1f} nm. Every outer face carries the Dirichlet condition, so",
        "the whole surface is grain boundary and the distance used in the `gb/`",
        "profiles is the minimum over all faces — taken from the convex hull of the",
        "node cloud, so a hexagonal prism's slanted faces are measured correctly.", "",
        f"## Interior state at {doses[-1]:g} dpa", "",
        "| family | density [m^-3] | stored defects [m^-3] | defects/loop | diameter [nm] |",
        "|---|---|---|---|---|",
    ]
    for label, n_m3, stored, m, dia in rows:
        L.append(f"| {label} | {n_m3:.4e} | {stored:.4e} | {m:.1f} | {dia:.2f} |")

    L += ["", "## Loop size vs dose (interior)", "",
          "| dpa | <c> | <a> |", "|---|---|---|"]
    for d, rr in size_hist:
        L.append(f"| {d:g} | {rr[0][4]:.2f} | {rr[1][4]:.2f} |")
    L += ["",
          "Both families saturate early: density and content saturate together and",
          "the mean size is their ratio, so the interior platelet overlays look",
          "nearly identical across doses. That is the model result, not a plotting",
          "artefact — the real size variation is SPATIAL, resolved in",
          "`gb/gb_size_profiles.png`.", "",
          "## Figures", "",
          "`3d/` — one interior mid-plane cut normal to y, viewed almost frontally",
          "(elev 10, azim -80), one column per dose, log colour scale anchored below",
          "the 99.9th percentile so the colour range falls across the depletion",
          "shell rather than the interior plateau.", "",
          "- `mobile_fields.png`, `immobile_density_fields.png`,",
          "  `immobile_content_fields.png`",
          "- `loops_{c,a1,a2,a3}_overlay.png` — one family per figure, platelets at",
          "  the true local radius times the factor in the title (3x for <c>, 14x for",
          "  <a>). No upper clamp, so relative sizes are faithful WITHIN a figure;",
          "  the factors differ because <c> loops are ~6x larger, so sizes are not",
          "  comparable BETWEEN figures.", "",
          f"`gb/` — radially averaged profiles over the first {a.max_nm:g} nm from the",
          "grain boundary, which contains essentially all of the spatial structure",
          "(the point-defect depletion length is about 73 nm).", "",
          "Loop diameters are formed from the binned mean content and density, not by",
          "averaging nodal diameters: size is a ratio of two fields, and the mean of a",
          "ratio is not the ratio of the means.", "",
          "## Known behaviour worth noting", "",
          "The <a> loop density rises steeply toward the grain boundary and keeps",
          "growing linearly in dose there. Cascade nucleation is spatially uniform,",
          "but the only removal channel is coalescence, driven by the absorbed mobile",
          "flux — and that flux vanishes where the Dirichlet condition pins the mobile",
          "concentrations. The grain boundary is a sink for mobile defects in this",
          "formulation but NOT for loops. The interior comparison against ZrMicro is",
          "unaffected, being evaluated far from the boundary, but the near-boundary",
          "loop density should not be used quantitatively.", "",
          "## Files", ""]
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            L.append(f"- `{p.relative_to(run_dir).as_posix()}` "
                     f"({p.stat().st_size / 1024:.1f} KB)")

    (run_dir / "provenance.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n{'=' * 66}")
    print(f"RUN COMPLETE -> {run_dir}")
    print(f"  3d/ : {len(list(d3.glob('*.png')))} figures")
    print(f"  gb/ : {len(list(dgb.glob('*.png')))} figures")
    print(f"{'=' * 66}")
    print(f"\nInterior at {doses[-1]:g} dpa:")
    for label, n_m3, stored, m, dia in rows:
        print(f"  {label:<22} N = {n_m3:.4e} m^-3   d = {dia:.2f} nm")
    return run_dir


if __name__ == "__main__":
    main()
