"""compare_zr3d_runs.py — put two 3-D cluster-dynamics runs side by side.

Renders a direct comparison between two MoDELib (`Zr3d_ghoniem`) simulations
that differ only in domain geometry — by default the hexagonal Zr single crystal
against the reference 1 um cube. Both must have been run with the same material,
temperature, dose rate and dose schedule; only the mesh and `F` differ.

Two families of figure are written:

`cmp/` — grain-boundary profiles, one file per quantity, both runs overlaid at
         each dose (solid = run A, dashed = run B).
`cmp/` — interior value against dose, one file per quantity, one curve per run.

The interior is defined on distance from the boundary, not on an axis-aligned
box, so the hexagonal prism and the cube are treated consistently: nodes deeper
than half the domain inradius. For a cube that is exactly the old box criterion.

Usage
-----
    .DisloClusterVenv/Scripts/python.exe \\
        -m dislocluster_code.post.compare_runs \\
        --sim-a <hexagonal sim dir> [--sim-b <cube sim dir>] \\
        [--doses 1 5 10 30] [--max-nm 150] [--tag cmp_hex_vs_cube]
"""

from __future__ import annotations

import argparse
import datetime
import platform
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib import colormaps                                    # noqa: E402

from dislocluster_code import paths                                          # noqa: E402
from dislocluster_code.post.fields import (                               # noqa: E402
    load_cd_fields, gb_distance, domain_faces, FAMILIES, SPECIES,
    B_SI, OMEGA_SI, OMEGA_B3,
)
from dislocluster_code.post.gb import profile                             # noqa: E402

FAMILY_SLUG = {0: "c", 1: "a1", 2: "a2", 3: "a3"}
MOBILE = (("Cv", 0, r"$C_v$"), ("Ci", 1, r"$C_i$"),
          ("C2i", 2, r"$C_{2i}$"), ("C3i", 3, r"$C_{3i}$"))


def dose_to_step(dose):
    """Dose [dpa] -> output step index, for a 1 dpa/step schedule."""
    return int(round(dose)) - 1


def geometry_of(P):
    """(n_faces, inradius [nm], bounding-box span [nm]) for a node cloud."""
    span = (P.max(0) - P.min(0)) * B_SI * 1e9
    return len(domain_faces(P)[0]), gb_distance(P).max() * B_SI * 1e9, span


def interior_stats(P, F, interior_frac=0.5):
    """Per-family (density [m^-3], content [m^-3], defects/loop, diameter [nm]).

    Averaged over nodes deeper than `interior_frac` of the inradius.
    """
    d = gb_distance(P)
    sel = d > interior_frac * d.max()
    out = []
    for label, ncol, ccol, bmag, _n, _c in FAMILIES:
        n_b3 = float(np.mean(F[sel, ncol]))
        c_at = float(np.mean(F[sel, ccol]))
        m = c_at / max(n_b3 * OMEGA_B3, 1e-300)
        r_b = np.sqrt(max(m, 0.0) * OMEGA_B3 / (np.pi * bmag))
        out.append((label, n_b3 / B_SI ** 3, c_at / OMEGA_SI, m,
                    2.0 * r_b * B_SI * 1e9))
    mob = {k: float(np.mean(F[sel, i]) / OMEGA_SI) for k, i, _ in MOBILE}
    return out, mob


# ── profile comparison ──────────────────────────────────────────────────────

def _dose_colors(n):
    cmap = colormaps["coolwarm"]
    return [cmap(t) for t in np.linspace(0.0, 1.0, n)]


def _profile_figure(runs, doses, values_of, ylabel, title, out_file,
                    logy=True, max_nm=150.0, n_bins=60):
    """One quantity, both runs, all doses. Solid = run A, dashed = run B."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = _dose_colors(len(doses))
    styles = ("-", "--", ":")
    for (name, data), ls in zip(runs.items(), styles):
        for (dose, col) in zip(doses, colors):
            P, F = data[dose_to_step(dose)]
            x, y = values_of(P, F, n_bins, max_nm)
            ax.plot(x, y, ls, color=col, lw=1.7,
                    label=f"{name}, {dose:g} dpa")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("distance from grain boundary, $x$ [nm]", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _size_values(ncol, ccol, bmag):
    def f(P, F, n_bins, max_nm):
        x, nbar = profile(P, F[:, ncol], n_bins, max_nm)
        _, cbar = profile(P, F[:, ccol], n_bins, max_nm)
        m = cbar / np.maximum(nbar * OMEGA_B3, 1e-300)
        r_b = np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * bmag))
        return x, 2.0 * r_b * B_SI * 1e9
    return f


def _scaled_values(col, scale):
    def f(P, F, n_bins, max_nm):
        return profile(P, F[:, col] / scale, n_bins, max_nm)
    return f


# ── interior-vs-dose comparison ─────────────────────────────────────────────

def _interior_figure(series, doses, key, ylabel, title, out_file, logy=True):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    marks = ("o", "s", "^")
    for (name, rows), mk in zip(series.items(), marks):
        for k, (label, _n, _c, _b, _nn, colr) in enumerate(FAMILIES):
            y = [rows[d][0][k][key] for d in doses]
            ax.plot(doses, y, color=colr, lw=1.6, ms=4, marker=mk,
                    alpha=1.0 if mk == "o" else 0.65,
                    ls="-" if mk == "o" else "--",
                    label=f"{label} — {name}")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("dose [dpa]", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mobile_interior_figure(series, doses, out_file):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = ("#1f4fbf", "#c62828", "#2e7d32", "#6a1b9a")
    for (name, rows), ls in zip(series.items(), ("-", "--")):
        for (k, _i, lab), c in zip(MOBILE, cols):
            ax.plot(doses, [rows[d][1][k] for d in doses], ls, color=c,
                    marker="o" if ls == "-" else "s", ms=4, lw=1.6,
                    label=f"{lab} — {name}")
    ax.set_yscale("log")
    ax.set_xlabel("dose [dpa]", fontsize=11)
    ax.set_ylabel("interior concentration [m$^{-3}$]", fontsize=11)
    ax.set_title("Mobile species in the interior", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-a", required=True, help="first simulation directory")
    ap.add_argument("--name-a", default="hexagonal")
    ap.add_argument("--sim-b", default=None,
                    help="second simulation directory (default: the cube case)")
    ap.add_argument("--name-b", default="cube (reference)")
    ap.add_argument("--doses", type=float, nargs="+", default=[1, 5, 10, 30])
    ap.add_argument("--max-nm", type=float, default=150.0)
    ap.add_argument("--tag", default="cmp_hex_vs_cube")
    ap.add_argument("--out-dir", default=None,
                    help="write into this existing run directory instead of a "
                         "new timestamped one")
    a = ap.parse_args(argv)

    sims = {a.name_a: Path(a.sim_a),
            a.name_b: Path(a.sim_b) if a.sim_b else paths.COUPLED_SIM_TUTORIAL}
    doses = list(a.doses)
    steps = [dose_to_step(d) for d in doses]

    runs, geom = {}, {}
    for name, sim in sims.items():
        evl = sim / "evl"
        missing = [s for s in steps if not (evl / f"evl_{s}.txt").is_file()]
        if missing:
            raise SystemExit(f"{name}: missing output steps {missing} in {evl}")
        runs[name] = {s: load_cd_fields(evl, s) for s in steps}
        geom[name] = geometry_of(runs[name][steps[0]][0])

    if a.out_dir:
        run_dir = Path(a.out_dir)
    else:
        run_dir = paths.run_dir(a.tag)
    out = run_dir / "cmp"
    out.mkdir(parents=True, exist_ok=True)

    for name, (nf, inr, span) in geom.items():
        print(f"{name:<18} {runs[name][steps[0]][0].shape[0]:>6} nodes  "
              f"{nf} faces  inradius {inr:6.1f} nm  bbox "
              f"{span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f} nm")

    # ── grain-boundary profiles ─────────────────────────────────────────────
    print("\ncmp/ grain-boundary profiles")
    for key, col, lab in MOBILE:
        _profile_figure(runs, doses, _scaled_values(col, OMEGA_SI),
                        "concentration [m$^{-3}$]",
                        f"{lab} against distance from the grain boundary",
                        out / f"cmp_gb_{key}.png", max_nm=a.max_nm)
        print(f"  cmp_gb_{key}.png")

    for k, (label, ncol, ccol, bmag, _n, _c) in enumerate(FAMILIES):
        s = FAMILY_SLUG[k]
        _profile_figure(runs, doses, _scaled_values(ncol, B_SI ** 3),
                        "loop number density [m$^{-3}$]",
                        f"{label} loop density near the grain boundary",
                        out / f"cmp_gb_N_{s}.png", max_nm=a.max_nm)
        _profile_figure(runs, doses, _scaled_values(ccol, OMEGA_SI),
                        "stored defects [m$^{-3}$]",
                        f"{label} loop content near the grain boundary",
                        out / f"cmp_gb_C_{s}.png", max_nm=a.max_nm)
        _profile_figure(runs, doses, _size_values(ncol, ccol, bmag),
                        "mean loop diameter [nm]",
                        f"{label} loop size near the grain boundary",
                        out / f"cmp_gb_d_{s}.png", logy=False, max_nm=a.max_nm)
        print(f"  cmp_gb_{{N,C,d}}_{s}.png")

    # ── interior against dose ───────────────────────────────────────────────
    print("cmp/ interior against dose")
    series = {name: {d: interior_stats(*runs[name][dose_to_step(d)])
                     for d in doses} for name in runs}
    _interior_figure(series, doses, 1, "loop number density [m$^{-3}$]",
                     "Interior loop density", out / "cmp_interior_density.png")
    _interior_figure(series, doses, 2, "stored defects [m$^{-3}$]",
                     "Interior loop content", out / "cmp_interior_content.png")
    _interior_figure(series, doses, 4, "mean loop diameter [nm]",
                     "Interior loop size", out / "cmp_interior_diameter.png",
                     logy=False)
    _mobile_interior_figure(series, doses, out / "cmp_interior_mobile.png")
    print("  cmp_interior_{density,content,diameter,mobile}.png")

    # ── provenance ──────────────────────────────────────────────────────────
    A, B = a.name_a, a.name_b
    L = [f"# 3-D cluster dynamics — {A} vs {B}", "",
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S} on "
         f"{platform.node()} ({platform.platform()}),",
         f"DisloCluster at `{paths.git_hash()}`.", "",
         "## Runs", "",
         "| | domain | nodes | faces | inradius | bounding box |",
         "|---|---|---|---|---|---|"]
    for name, sim in sims.items():
        nf, inr, span = geom[name]
        L.append(f"| {name} | `{sim.name}` | "
                 f"{runs[name][steps[0]][0].shape[0]:,} | {nf} | {inr:.1f} nm | "
                 f"{span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f} nm |")
    L += ["",
          "Both runs use the same material file, temperature, dose rate, dose",
          "schedule, nucleation channels and zero applied stress; only the mesh and",
          "the deformation gradient `F` differ.", "",
          f"Doses compared: {', '.join(f'{d:g} dpa' for d in doses)} "
          f"(output steps {steps}). Output is written after `solve()`, so `evl_N`",
          "holds the state at N+1 dose steps.", ""]

    for d in doses:
        L += [f"## Interior at {d:g} dpa", "",
              f"| family | density [m^-3] {A} | {B} | ratio | "
              f"diameter [nm] {A} | {B} |",
              "|---|---|---|---|---|---|"]
        ra, rb = series[A][d][0], series[B][d][0]
        for (la, na, _ca, _ma, da), (_lb, nb, _cb, _mb, db) in zip(ra, rb):
            L.append(f"| {la} | {na:.4e} | {nb:.4e} | {na / max(nb, 1e-300):.3f} | "
                     f"{da:.2f} | {db:.2f} |")
        L.append("")
        ma, mb = series[A][d][1], series[B][d][1]
        L += ["| mobile species | " + f"{A} [m^-3] | {B} [m^-3] | ratio |",
              "|---|---|---|---|"]
        for k, _i, lab in MOBILE:
            L.append(f"| {k} | {ma[k]:.4e} | {mb[k]:.4e} | "
                     f"{ma[k] / max(mb[k], 1e-300):.3f} |")
        L.append("")

    L += ["## Reading the figures", "",
          f"In every `cmp_gb_*` figure the solid curves are **{A}** and the dashed",
          f"curves are **{B}**; colour is dose (blue = lowest, red = highest).",
          "The interior figures use circles/solid for the first run and",
          "squares/dashed for the second, with the family colours of the field",
          "figures.", "",
          "The distance used on the x axis is the minimum distance to any domain",
          "face, taken from the convex hull of the node cloud. That matters here:",
          "the hexagonal prism has eight faces, six of them slanted, and an",
          "axis-aligned bounding-box distance would overstate the depth of every",
          "node near a slanted face.", "",
          "Interior averages are over nodes deeper than half the inradius. The two",
          "domains have different inradii, so the interior regions are not the same",
          "size — the cube's is 500 nm deep at the centre and the prism's only",
          "173 nm (its apothem). Where the interior has genuinely saturated this",
          "does not matter; where it has not, the prism's interior average is taken",
          "closer to the boundary and will sit below the cube's.", "",
          "## Files", ""]
    for p in sorted(out.rglob("*")):
        if p.is_file():
            L.append(f"- `cmp/{p.name}` ({p.stat().st_size / 1024:.1f} KB)")
    (run_dir / "provenance_comparison.md").write_text("\n".join(L) + "\n",
                                                      encoding="utf-8")

    print(f"\n{'=' * 66}")
    print(f"COMPARISON -> {run_dir}")
    print(f"  cmp/ : {len(list(out.glob('*.png')))} figures")
    print(f"{'=' * 66}")
    for d in (doses[-1],):
        print(f"\nInterior at {d:g} dpa   ({A} / {B})")
        for (la, na, _ca, _ma, da), (_lb, nb, _cb, _mb, db) in zip(
                series[A][d][0], series[B][d][0]):
            print(f"  {la:<22} N = {na:.3e} / {nb:.3e} m^-3    "
                  f"d = {da:6.2f} / {db:6.2f} nm")
    return run_dir


if __name__ == "__main__":
    main()
