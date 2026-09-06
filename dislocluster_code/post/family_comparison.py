"""Per-family loop density and size against experiment, all nine families.

`post.visualization.plot_loop_density` / `plot_loop_sizes_vs_experiment` draw
the same two comparisons, but from the LEGACY four-slot state: under
`loop_model = 1` with `n_fam = 9` they see what `field.to_legacy_layout` lumps
into those four names, so the three prismatic variants arrive as one curve, the
three prismatic VACANCY variants as another, and the basal pair as `CvL`/`CavL`.
That is lossless for the totals and shows none of the structure.

This draws the structure. Every family gets a DOTTED line; the two totals that
experiment actually measures get a SOLID one:

    <c> total = c_f + c_p          the faulted and perfect basal loops
    <a> total = a1..a3 + a1v..a3v  all six prismatic variants

THE PYRAMID IS NOT IN EITHER TOTAL. `c_0` is a compact cluster, not a loop --
no perimeter, no lambda sqrt(m) radius -- so it has no diameter to compare with
a micrograph and cannot be added to a loop count. It is drawn on the density
axes in grey, labelled, because it is part of the basal chain's population and
its size relative to the loops is the whole question of whether it is
observable; it is absent from the size axes entirely.

Sizes use the SOLVER'S OWN radius scale, `size_spectrum.BEDGE`, which is
`l_c` for both basal families and `l_a` for the prismatic ones -- matching
`rate_equations_core.h:356`, `f_lscale[k] = is_prism[k] ? l_a : l_c`. Note this
disagrees with `studies.loop_annealing`, which declares `c_p` a full [0001]
loop (`b.n = c`) where the radius scale treats it as `c/2`: the emission
channel and the radius channel do not use the same Burgers vector for `c_p`.
The figures follow the integrator, since that is what produced the numbers,
and the discrepancy is a factor 1.414 on the `c_p` diameter alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.post import size_spectrum as SS
from dislocluster_code.post.coarsening import interior_mask

#: Drawn order, and which total each family belongs to. The pyramid's total is
#: None, which is what keeps it out of both sums.
FAMILIES = (
    ("c",   "c", "lightcoral", (0, (1, 1))),
    ("cp",  "c", "darkred",    (0, (3, 1, 1, 1))),
    ("a1",  "a", "cornflowerblue", (0, (1, 1))),
    ("a2",  "a", "tab:cyan",   (0, (1, 1))),
    ("a3",  "a", "steelblue",  (0, (1, 1))),
    ("a1v", "a", "tab:purple", (0, (3, 1, 1, 1))),
    ("a2v", "a", "orchid",     (0, (3, 1, 1, 1))),
    ("a3v", "a", "slateblue",  (0, (3, 1, 1, 1))),
    ("pyr", None, "0.55",      (0, (5, 2))),
)
TOTAL_STYLE = {"c": ("tab:red", r"$\langle c\rangle$ total ($c_f + c_p$)"),
               "a": ("tab:blue", r"$\langle a\rangle$ total (6 variants)")}


def _omega_SI():
    """Atomic volume in m^3 -- the state's densities are per atom."""
    from dislocluster_code.coupling import field as mfield
    return float(mfield.read_material_scalar(paths.MODELIB_MATERIAL,
                                             "atomicVolume_SI"))


def family_series(run, region="interior"):
    """Per-family (N [m^-3], d [nm]) at every snapshot dose.

    The mean is taken over NODES first and the size formed from the means,
    which is the reduction `region_table` and the manuscript's interior numbers
    use: d = 2 lambda sqrt(<c>/<n>), not <2 lambda sqrt(c/n)>.
    """
    doses, Y, nodes = SS.load(run)
    mask = (interior_mask(nodes) if region == "interior"
            else np.ones(len(nodes), bool))
    om = _omega_SI()
    out = {}
    for slug, _tot, _colour, _ls in FAMILIES:
        k = SS.FAMILY_INDEX[slug]
        lam = SS._lambda_nm(slug)
        N = np.empty(len(doses))
        d = np.empty(len(doses))
        for j in range(len(doses)):
            n = (Y[j][:, 4 + k] if k < 4
                 else Y[j][:, 19 + (k - 4)])[mask].mean()
            c = (Y[j][:, 8 + k] if k < 4
                 else Y[j][:, 24 + (k - 4)])[mask].mean()
            N[j] = n / om
            # A family with no members has no size. Reporting 0 would draw a
            # line down to the axis; NaN leaves a gap, which is the truthful
            # mark for "this family is not populated at this dose".
            d[j] = (2.0 * lam * np.sqrt(c / n)
                    if (n > 1e-25 and c > 0) else np.nan)
        out[slug] = (N, d)
    return doses, out


def totals(series):
    """(N, d) for the two populations experiment resolves.

    d is NUMBER-weighted over the member families, which is the observation
    operator the fit uses (`fit_cloops`: d_C = 2 sum(C_k r_k)/sum(C_k)).
    """
    res = {}
    for tot in ("c", "a"):
        members = [s for s, t, _c, _l in FAMILIES if t == tot]
        N = sum(series[s][0] for s in members)
        num = sum(series[s][0] * np.nan_to_num(series[s][1]) for s in members)
        res[tot] = (N, num / np.maximum(N, 1e-30))
    return res


def _exp_targets(sheet, T0, tol=50.0):
    import pandas as pd
    try:
        df = pd.read_excel(paths.INPUT_DIR / "Zr_input_parameters.xlsx",
                           sheet_name=sheet).iloc[:, :5]
    except Exception:
        return None
    df.columns = ["T", "dpa", "N", "d", "Source"]
    for c in ("T", "dpa", "N", "d"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["T"] = df["T"].ffill()
    df["Source"] = df["Source"].ffill()
    df = df.dropna(subset=["T", "dpa"])
    return df[(df["T"] - T0).abs() <= tol]


def _run_temperature(run):
    import json
    try:
        cfg = json.loads((Path(run) / "config.json").read_text())
        return float(cfg["material"]["temperature_K"])
    except Exception:
        return 573.0


def _draw_exp(ax, T0, col):
    marks = ["o", "s", "^", "D", "v", "P", "*", "X"]
    i = n = 0
    for sheet, colour, lt in (("Targets_A", "tab:blue", "a"),
                              ("Targets_C", "tab:red", "c")):
        ed = _exp_targets(sheet, T0)
        if ed is None:
            continue
        for (T, src), g in ed.dropna(subset=[col]).groupby(["T", "Source"]):
            ax.errorbar(g["dpa"], g[col], yerr=0.2 * g[col],
                        fmt=marks[i % len(marks)], color=colour, ms=7,
                        capsize=3, lw=1, mec="k", mew=0.5, zorder=5,
                        label=f"{lt}-exp {T:.0f} K, {src}")
            i += 1
            n += len(g)
    return n


def render(run, region="interior", out=None, verbose=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    out = Path(out) if out else run / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)
    doses, series = family_series(run, region)
    tot = totals(series)
    T0 = _run_temperature(run)
    written = []

    for what, ylab, title, logy in (
            ("N", "Number density (m$^{-3}$)", "Loop density by family", True),
            ("d", "Mean loop diameter (nm)", "Loop size by family", True)):
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        for slug, belongs, colour, ls in FAMILIES:
            if what == "d" and belongs is None:
                continue            # a compact cluster has no loop diameter
            N, d = series[slug]
            y = N if what == "N" else d
            if not np.isfinite(y).any():
                continue
            if what == "N" and np.nanmax(y) <= 0:
                continue
            lab = SS.LABEL[slug]
            if belongs is None:
                lab += " (compact; not in total)"
            ax.plot(doses, y, ls=ls, color=colour, lw=1.5, label=lab, zorder=2)
        for key, (colour, lab) in TOTAL_STYLE.items():
            N, d = tot[key]
            ax.plot(doses, N if what == "N" else d, "-", color=colour, lw=2.8,
                    label=lab, zorder=4)
        n_exp = _draw_exp(ax, T0, "N" if what == "N" else "d")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
            # Autoscale on a log axis reaches for whatever the earliest,
            # smallest family happens to be, which on this model is four empty
            # decades below anything anyone reads. Clamp to the data.
            _i = 0 if what == "N" else 1
            _cand = ([series[s2][_i] for s2, b2, *_ in FAMILIES
                      if not (what == "d" and b2 is None)]
                     + [tot[k2][_i] for k2 in tot])
            lo = [np.nanmin(v[np.isfinite(v) & (v > 0)]) for v in _cand
                  if np.any(np.isfinite(v) & (v > 0))]
            if lo:
                ax.set_ylim(bottom=10 ** np.floor(np.log10(min(lo))))
        ax.set_xlabel("Dose (dpa)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title} — {region} mean, {T0:.0f} K\n"
                     "dotted: individual families; solid: totals; "
                     "experiment ±20%", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7.5, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14))
        if n_exp == 0:
            ax.text(0.02, 0.96, f"no data within 50 K of {T0:.0f} K",
                    transform=ax.transAxes, va="top", fontsize=8, color="0.35")
        p = out / ("loop_density_families.png" if what == "N"
                   else "loop_sizes_families.png")
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
        if verbose:
            print(f"  {p.name}")
    _write_readme(out, doses, series, tot, region, T0)
    return written


def _write_readme(out, doses, series, tot, region, T0):
    j = int(np.argmax(doses))
    L = ["# Loop families against experiment\n",
         f"`{region}` mean at {T0:.0f} K. Dotted lines are the individual "
         "families; solid lines are the two totals experiment resolves, "
         "`<c> = c_f + c_p` and `<a>` over all six prismatic variants. The "
         "pyramid `c_0` is a compact cluster and is in NEITHER total: it has "
         "no loop diameter and is absent from the size figure.\n",
         f"At {doses[j]:g} dpa:\n",
         "| family | N (m^-3) | d (nm) | counted in |",
         "|---|---:|---:|---|"]
    for slug, belongs, _c, _l in FAMILIES:
        N, d = series[slug]
        ds = "--" if not np.isfinite(d[j]) else f"{d[j]:.1f}"
        L.append(f"| `{slug}` | {N[j]:.3e} | {ds} | {belongs or '--'} |")
    for key in ("c", "a"):
        N, d = tot[key]
        L.append(f"| **{key} total** | **{N[j]:.3e}** | **{d[j]:.1f}** | |")
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--region", default="interior",
                    choices=("interior", "domain"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run) if a.run else paths.latest_run()
    print(f"family comparison: {run}")
    render(run, region=a.region, out=a.out)


if __name__ == "__main__":
    main()
