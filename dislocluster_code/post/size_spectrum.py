"""The loop size distribution the three carried moments imply, per family.

A model that carries only $n$ and $c$ has one size per family per node and no
distribution at all: any "size histogram" it produces is a histogram of local
MEAN sizes across space, which is what `report.py`'s `size_dist_*` figures are
and which they say so. With the second content $q$ the closure of
`sec:moments` supplies a genuine per-loop spectrum, and this module evaluates
it.

    python -m dislocluster_code.post.size_spectrum <run> --doses 1e-4 1e-2 1 10
    python -m dislocluster_code.post.size_spectrum <run> --region interior

TWO DISTRIBUTIONS ARE IN PLAY AND THEY ARE NOT THE SAME. At one node the
closure is log-normal in the defect count `m`, with `mbar = c/n` and
`Delta = q n / c^2 = exp(s^2)`. Over the crystal the population is the
VOLUME-WEIGHTED MIXTURE of those per-node log-normals, and a mixture of
log-normals is not log-normal: it is broader than any of its components and can
be multimodal when the field varies strongly, which near a Dirichlet surface it
does. The mixture is the observable one -- it is what a micrograph of the whole
specimen samples -- so it is what this module reports, with the per-node width
recoverable as the `Delta` column of the summary.

WORKING IN DIAMETER, EXACTLY. If `m` is log-normal with parameters
`(mu, s)` then `d = 2 lambda sqrt(m)` is log-normal with `(ln(2 lambda) + mu/2,
s/2)`, so no numerical change of variables is needed and the mixture is
accumulated directly on a log-spaced diameter grid.

WHAT THE RECONSTRUCTION INHERITS. The carried moments are inverted for
`(mu, s)` as if the distribution were untruncated, while the solver's
grain-boundary channel removes its large end against a support capped at the
crystal (`sec:gb-loops`). The reconstruction therefore reproduces what the model
integrated, including that inconsistency, and not a separately truncated
distribution -- which is the right choice for a figure whose purpose is to show
what the model believes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import field as mfield
from dislocluster_code.post.fields import domain_faces, domain_volume, gb_distance
from dislocluster_code.post.volume_average import voronoi_weights

#: Families in the 40-wide state, by the slug the rest of the post-processing
#: uses. Value is the family index k: number at 4+k (k<4) or 19+(k-4), content
#: at 8+k or 24+(k-4), second content at 29+k throughout.
FAMILY_INDEX = {"c": 0, "a1": 1, "a2": 2, "a3": 3,
                "a1v": 4, "a2v": 5, "a3v": 6, "cp": 7, "pyr": 8}

#: Burgers edge component per habit, in units of b: <a> loops store on a full
#: a, basal loops on c/2. These are the solver's own `f_lscale` inputs, so the
#: reconstruction uses the same lambda the march used.
BEDGE = {"a": 1.0, "c": 0.7972136}

LABEL = {"c": r"$\langle c\rangle_f$", "a1": r"$\langle a\rangle_1$",
         "a2": r"$\langle a\rangle_2$", "a3": r"$\langle a\rangle_3$",
         "a1v": r"$\langle a\rangle_1^{v}$", "a2v": r"$\langle a\rangle_2^{v}$",
         "a3v": r"$\langle a\rangle_3^{v}$", "cp": r"$\langle c\rangle_p$",
         "pyr": "pyramid"}

DEFAULT_DOSES = (1e-4, 1e-2, 1.0, 10.0)


def _slots(k):
    return (4 + k, 8 + k) if k < 4 else (19 + (k - 4), 24 + (k - 4))


def _lambda_nm(slug):
    """The solver's own length scale, in nm: lambda = sqrt(Omega / pi b_edge)."""
    om = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL,
                                           "atomicVolume_SI"))
    b = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL, "b_SI"))
    bedge = BEDGE["c"] if slug in ("c", "cp") else BEDGE["a"]
    return float(np.sqrt(om / (np.pi * bedge * b ** 3)) * b * 1e9)


def _size_map(slug):
    """(ln-prefactor, exponent) carrying ln m to ln(diameter in nm).

    A LOOP is planar, d = 2 lambda sqrt(m), so the exponent is 1/2. The PYRAMID
    is COMPACT: `parameters.h` gives it R = (m Omega/sqrt8)^(1/3), Eq. (Rsfp),
    so D = 2R has exponent 1/3 and a different prefactor entirely -- there is no
    lambda and no Burgers vector in it.

    Sizing it as a loop, which this module used to do, is wrong twice over,
    because `_lambda_nm` also falls through to the PRISMATIC BEDGE for any slug
    that is not basal: a pyramid of 400 vacancies came out 6.1 nm across where
    it is 3.0. The mapping is a power law either way, so a log-normal in m stays
    log-normal in D -- only the exponent changes, and at 1/2 the expressions
    below are exactly what they were.
    """
    if slug == "pyr":
        om = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL,
                                               "atomicVolume_SI"))
        return np.log(2.0 * (om / np.sqrt(8.0)) ** (1.0 / 3.0) * 1e9), 1.0 / 3.0
    return np.log(2.0 * _lambda_nm(slug)), 0.5


def load(run):
    z = np.load(Path(run) / "march_state.npz")
    return (np.asarray(z["doses"], float), z["Y"],
            np.asarray(z["nodes"], float))


def nodal_volumes(nodes, n_samples=400_000, verbose=False):
    """Volume per node, in nm^3, restricted to the crystal the nodes fill."""
    b = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL, "b_SI"))
    w = voronoi_weights(nodes, n_samples=n_samples, verbose=verbose,
                        faces=domain_faces(nodes))
    return w * domain_volume(nodes) * (b * 1e9) ** 3


def spectrum(Y_dose, nodes, vol, slug, d_grid, mask=None, floor=1e-19,
             delta_cap=None):
    """Volume-weighted mixture of the per-node log-normals, on `d_grid` (nm).

    Returns `(dN_dlnd, stats)`: the population per unit ln(d) per cubic metre,
    and a dict of the aggregate the mixture was built from.
    """
    k = FAMILY_INDEX[slug]
    ncol, ccol = _slots(k)
    n = Y_dose[:, ncol].astype(float)
    c = Y_dose[:, ccol].astype(float)
    q = Y_dose[:, 29 + k].astype(float)

    ok = (n > floor) & (c > floor) & (q > 0.0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if not ok.any():
        return np.zeros_like(d_grid), dict(count=0.0, nodes=0)

    n, c, q, v = n[ok], c[ok], q[ok], vol[ok]
    mbar = c / n
    delta = np.clip(q * n / (c * c), 1.0 + 1e-12, None)   # Cauchy-Schwarz
    # A DIAGNOSTIC, NOT A CORRECTION. `delta_cap` limits how wide a per-node
    # log-normal is allowed to be, so that the effect of the handful of nodes
    # whose dispersion has run away can be seen by difference. The default is
    # None: the published spectrum carries every node the march produced, and
    # the outliers are reported rather than removed.
    n_over = int((delta > (delta_cap or np.inf)).sum())
    if delta_cap is not None:
        delta = np.minimum(delta, float(delta_cap))
    s2 = np.log(delta)
    # log-normal in m: mean = exp(mu + s2/2); in d = 2 lambda sqrt(m) the
    # parameters halve, exactly.
    ln_pref, expo = _size_map(slug)
    mu_d = ln_pref + expo * (np.log(mbar) - 0.5 * s2)
    s_d = expo * np.sqrt(s2)

    om = float(mfield.read_material_scalar(paths.MODELIB_MATERIAL,
                                           "atomicVolume_SI"))
    # loops contributed by each node, as an absolute count
    counts = (n / om) * (v * 1e-27)

    lnd = np.log(d_grid)[None, :]
    z = (lnd - mu_d[:, None]) / s_d[:, None]
    pdf = np.exp(-0.5 * z * z) / (s_d[:, None] * np.sqrt(2.0 * np.pi))
    dN_dlnd = (counts[:, None] * pdf).sum(axis=0)

    tot = float(counts.sum())
    V = float(v.sum()) * 1e-27
    raw = q * n / (c * c)
    hi = raw > 3.0
    return dN_dlnd, dict(
        count=tot, nodes=int(ok.sum()), volume_m3=V,
        density=tot / V if V > 0 else np.nan,
        # THE SAME MAPPING AS THE SPECTRUM, not a second copy of the loop
        # formula: at exponent 1/2 this is exactly 2 lambda sqrt(c/n) as before,
        # and for the compact pyramid it is 2 (mbar Omega/sqrt8)^(1/3).
        d_mean=float(np.exp(ln_pref + expo * np.log(c.sum() / n.sum()))),
        delta_med=float(np.median(raw)), delta_max=float(raw.max()),
        n_delta_gt3=int(hi.sum()),
        # what share of the POPULATION those wide nodes carry: the number that
        # decides whether an outlier matters to a figure
        frac_delta_gt3=float(counts[hi].sum() / tot) if tot > 0 else 0.0,
        n_capped=n_over)


def build(run, doses=DEFAULT_DOSES, slugs=("c", "a1", "a1v"),
          region="domain", n_d=260, d_lim=(0.3, 400.0), verbose=True,
          delta_cap=None):
    all_doses, Y, nodes = load(run)
    vol = nodal_volumes(nodes, verbose=verbose)
    d = np.geomspace(*d_lim, n_d)
    mask = None
    if region == "interior":
        x = gb_distance(nodes)
        mask = x >= np.quantile(x, 0.75)
    elif region == "shell":
        x = gb_distance(nodes)
        mask = x < np.quantile(x, 0.25)
    elif region != "domain":
        raise SystemExit(f"region must be domain, interior or shell")

    out = {}
    for slug in slugs:
        for want in doses:
            i = int(np.argmin(np.abs(all_doses - want)))
            y, st = spectrum(Y[i], nodes, vol, slug, d, mask=mask,
                             delta_cap=delta_cap)
            out[(slug, float(all_doses[i]))] = (y, st)
            if verbose:
                print(f"  {slug:<4} {all_doses[i]:>8.4g} dpa  "
                      f"N {st['count']:>12.3e} loops  "
                      f"d_mean {st.get('d_mean', float('nan')):>7.2f} nm  "
                      f"Delta med {st.get('delta_med', float('nan')):>6.3f}"
                      f"  max {st.get('delta_max', float('nan')):>9.3g}"
                      f"  >3: {st.get('n_delta_gt3', 0):>3d} nodes,"
                      f" {100*st.get('frac_delta_gt3', 0):>6.3f}% of loops")
    return d, out


def render(run, d, data, slugs, doses, out_file, region="domain", dpi=200):
    """One panel per family, one curve per dose, plus a cross-family panel."""
    n_p = len(slugs) + 1
    ncol = 2
    nrow = int(np.ceil(n_p / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.2, 3.5 * nrow),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    cols = [cmap(t) for t in np.linspace(0.06, 0.86, len(doses))]

    # LINEAR IN DIAMETER, AS A MEASURED SIZE DISTRIBUTION IS PLOTTED, and the
    # ordinate is dN/dd rather than dN/dln(d) to match: a TEM size histogram is
    # counts per nanometre of diameter against diameter, and plotting the
    # log-density against a linear axis would misstate the area under every
    # peak.
    #
    # THREE DECADES OF ORDINATE, not the data's full range. At the first
    # snapshot every family is monodisperse to three decimal places, so its
    # log-normal is nearly a delta function whose tails fall to 1e-270; on an
    # unbounded log axis that spans 300 decades and flattens every curve in the
    # panel into a horizontal line.
    DECADES = 3.0

    # A RESOLUTION LIMIT, and it has to be applied before the axes are scaled.
    # dN/dd carries a factor 1/d relative to dN/dln(d), so it amplifies the
    # small-diameter tail of every log-normal; below about a nanometre that
    # tail is both unresolvable in a micrograph and, here, dominated by the few
    # nodes whose dispersion has run away. Plotting from D_MIN is what an
    # experimental size distribution does, and the peak and the upper limit are
    # taken over the visible window so that an invisible spike cannot set them.
    D_MIN = 1.0

    vis = d >= D_MIN

    def _window(curves):
        """(peak, largest visible diameter) over the plotted range."""
        pk, hi = 0.0, D_MIN
        for yv in curves:
            v = yv[vis]
            if v.size == 0 or v.max() <= 0:
                continue
            pk = max(pk, float(v.max()))
        for yv in curves:
            v = yv[vis]
            if v.size == 0 or v.max() <= 0:
                continue
            above = np.flatnonzero(v >= pk * 10 ** (-DECADES))
            if above.size:
                hi = max(hi, float(d[vis][above[-1]]))
        return pk, hi

    for j, slug in enumerate(slugs):
        ax = axes.ravel()[j]
        peak, curves = 0.0, []
        for cdx, dose in enumerate(doses):
            y, st = data[(slug, dose)]
            if st["count"] <= 0:
                continue
            yv = y / max(st["volume_m3"], 1e-300) / d      # dN/dd
            curves.append(yv)
            peak = max(peak, float(yv.max()))
            ax.plot(d[vis], yv[vis], color=cols[cdx], lw=1.9,
                    label=f"{dose:g} dpa")
        ax.set_yscale("log")
        peak, hi = _window(curves)
        if peak > 0:
            ax.set_ylim(peak * 10 ** (-DECADES), peak * 3)
            ax.set_xlim(0.0, 1.08 * hi)
        ax.set_xlabel("loop diameter $d$ [nm]")
        ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}d$  [m$^{-3}$nm$^{-1}$]")
        ax.set_title(LABEL[slug], fontsize=11)
        ax.grid(alpha=0.3, which="both")

    # the cross-family panel, at the last dose
    ax = axes.ravel()[len(slugs)]
    fam_col = {"c": "#1f4fbf", "a1": "#c62828", "a1v": "#00838f",
               "cp": "#4527a0"}
    pk, curves = 0.0, []
    for slug in slugs:
        y, st = data[(slug, doses[-1])]
        if st["count"] <= 0:
            continue
        yv = y / max(st["volume_m3"], 1e-300) / d
        curves.append(yv)
        pk = max(pk, float(yv.max()))
        ax.plot(d[vis], yv[vis], color=fam_col.get(slug, "k"), lw=2.0,
                label=LABEL[slug])
    ax.set_yscale("log")
    pk, hi = _window(curves)
    if pk > 0:
        ax.set_ylim(pk * 10 ** (-DECADES), pk * 3)
        ax.set_xlim(0.0, 1.08 * hi)
    ax.set_xlabel("loop diameter $d$ [nm]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}d$  [m$^{-3}$nm$^{-1}$]")
    ax.set_title(f"all families at {doses[-1]:g} dpa", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    for ax in axes.ravel()[n_p:]:
        ax.axis("off")
    fig.tight_layout()

    # LEGENDS OUTSIDE THE AXES. Placed inside at `loc="best"` they sat on the
    # curves rather than beside them: every panel rises steeply from the left,
    # so the upper-left corner matplotlib scores as emptiest is exactly where
    # the leading edge is, and the box covered the first snapshot's near-delta
    # peak in three panels of four. There is no free corner to move them to --
    # a distribution fills its panel by construction -- so they go below the
    # figure, where `bbox_inches="tight"` picks them up. Two legends and not
    # one: the final panel keys families where the others key doses, and
    # merging them would imply a colour means both.
    h_d, l_d = axes.ravel()[0].get_legend_handles_labels()
    h_f, l_f = ax.get_legend_handles_labels()
    if h_d:
        fig.legend(h_d, l_d, loc="upper center", bbox_to_anchor=(0.5, 0.004),
                   ncol=len(l_d), frameon=False, fontsize=9,
                   title="panels 1-3: one curve per dose")
    if h_f:
        fig.legend(h_f, l_f, loc="upper center", bbox_to_anchor=(0.5, -0.055),
                   ncol=len(l_f), frameon=False, fontsize=9,
                   title="final panel: one curve per family")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_file}")
    return out_file


def render_regions(run, doses, slugs, out_file, dose=None, dpi=200,
                   verbose=True):
    """One panel per family: the whole crystal against its interior and shell.

    This is the figure that says whether the width of the domain spectrum is
    the width of a LOCAL distribution or the spread of local means across
    space. It cannot be answered from the domain spectrum alone, and the two
    answers have different consequences: a locally broad population coarsens,
    a spatially sorted one does not.
    """
    dose = doses[-1] if dose is None else dose
    sets = {}
    for region in ("domain", "interior", "shell"):
        d, data = build(run, doses=(dose,), slugs=slugs, region=region,
                        verbose=verbose)
        sets[region] = (d, data)

    style = {"domain": ("k", "-", 2.2), "interior": ("#c62828", "--", 1.9),
             "shell": ("#00838f", ":", 2.2)}
    fig, axes = plt.subplots(1, len(slugs), figsize=(4.3 * len(slugs), 3.6),
                             squeeze=False)
    deltas = {}
    for j, slug in enumerate(slugs):
        ax = axes[0, j]
        pk, curves = 0.0, []
        for region in ("domain", "interior", "shell"):
            d, data = sets[region]
            key = [k for k in data if k[0] == slug][0]
            y, st = data[key]
            if st["count"] <= 0:
                continue
            yv = y / max(st["volume_m3"], 1e-300) / d
            curves.append(yv)
            pk = max(pk, float(yv.max()))
            col, ls, lw = style[region]
            ax.plot(d[d >= 1.0], yv[d >= 1.0], color=col, ls=ls, lw=lw,
                    label=region)
            deltas.setdefault(slug, {})[region] = float(st["delta_med"])
        ax.set_yscale("log")
        vis = d >= 1.0
        pk = max((float(yv[vis].max()) for yv in curves if yv[vis].size), default=0.0)
        if pk > 0:
            ax.set_ylim(pk * 1e-3, pk * 3)
            hi = 1.0
            for yv in curves:
                a = np.flatnonzero(yv[vis] >= pk * 1e-3)
                if a.size:
                    hi = max(hi, float(d[vis][a[-1]]))
            ax.set_xlim(0.0, 1.08 * hi)
        ax.set_xlabel("loop diameter $d$ [nm]")
        if j == 0:
            ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}d$  [m$^{-3}$nm$^{-1}$]")
        ax.set_title(f"{LABEL[slug]} at {dose:g} dpa", fontsize=11)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    # ONE SHARED LEGEND, BELOW. The three curves mean the same thing in every
    # panel, so three boxes were redundant as well as obstructive -- each sat
    # over the peak of the panel it was in. The per-panel dispersions the
    # labels used to carry cannot go into a shared key, because they differ by
    # panel; they are returned instead, for the caption to report.
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.004),
                   ncol=len(l), frameon=False, fontsize=9)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_file}")
    # The dispersions the legend labels used to carry, for the caption. Printed
    # as well as returned: the caption is written by hand and this is the only
    # place these numbers exist.
    for slug, per in deltas.items():
        print(f"  Delta med  {slug:4s} "
              + "  ".join(f"{r} {v:.3f}" for r, v in per.items()))
    return out_file, deltas


#: The nine carried families, in the order the state lays them out.
ALL_FAMILIES = ("c", "a1", "a2", "a3", "a1v", "a2v", "a3v", "cp", "pyr")

#: Doses for the per-family sweep. Wider than DEFAULT_DOSES because these
#: figures exist to show the WHOLE evolution of one family rather than to
#: compare families at a few points, so the nucleation transient and the
#: saturated end both have to be on the same axes.
PER_FAMILY_DOSES = (1e-4, 1e-2, 1.0, 10.0, 40.0)


def render_family(d, data, slug, doses, out_file, region="domain", dpi=200):
    """ONE family, one curve per dose -- the per-panel style of `render`.

    Same construction as the multi-panel figure and deliberately so: linear in
    diameter with dN/dd on a log ordinate, because that is the form a measured
    TEM size distribution takes and plotting a log density against a linear
    axis would misstate the area under every peak. Three decades of ordinate
    below the visible peak, and nothing below D_MIN = 1 nm, for the reasons
    `render` gives -- at the first snapshot a family is monodisperse to three
    decimals and its tails reach 1e-270, which on an unbounded axis flattens
    every curve in the figure into a horizontal line.

    A family with no population at any dose still gets a figure, annotated.
    `c_p` and the pyramid are fed only by the basal chain, so with that switch
    off they are legitimately empty -- and an absent file would read as a
    failure of this script rather than as a statement about the model.
    """
    DECADES, D_MIN = 3.0, 1.0
    vis = d >= D_MIN
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = plt.get_cmap("viridis")
    cols = [cmap(t) for t in np.linspace(0.06, 0.86, len(doses))]

    peak, hi, drawn = 0.0, D_MIN, 0
    for cdx, dose in enumerate(doses):
        entry = data.get((slug, dose))
        if entry is None:
            continue
        y, st = entry
        if st["count"] <= 0:
            continue
        yv = y / max(st["volume_m3"], 1e-300) / d          # dN/dd
        v = yv[vis]
        if v.size == 0 or v.max() <= 0:
            continue
        ax.plot(d[vis], v, color=cols[cdx], lw=1.9, label=f"{dose:g} dpa")
        peak = max(peak, float(v.max()))
        drawn += 1
    if peak > 0:
        for cdx, dose in enumerate(doses):
            entry = data.get((slug, dose))
            if entry is None or entry[1]["count"] <= 0:
                continue
            v = (entry[0] / max(entry[1]["volume_m3"], 1e-300) / d)[vis]
            above = np.flatnonzero(v >= peak * 10 ** (-DECADES))
            if above.size:
                hi = max(hi, float(d[vis][above[-1]]))
        ax.set_ylim(peak * 10 ** (-DECADES), peak * 3)
        ax.set_xlim(0.0, 1.08 * hi)
    ax.set_yscale("log")
    ax.set_xlabel("loop diameter $d$ [nm]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}d$  [m$^{-3}$nm$^{-1}$]")
    ax.set_title(f"{LABEL[slug]}   ({region})", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    if drawn:
        ax.legend(frameon=False, fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), title="dose")
    else:
        ax.text(0.5, 0.5, "no population at any dose",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="0.35")
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(1e-3, 1e3)
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file, drawn


def write_per_family(run, doses=PER_FAMILY_DOSES, out_dir=None,
                     region="domain", families=ALL_FAMILIES, verbose=True):
    """`<run>/size_distributions/`: one size-distribution figure per family.

    Requested doses are SNAPPED to the march's own snapshots and deduplicated,
    so a run that stops at 10 dpa does not silently plot its 10 dpa state twice
    under two labels. What is actually used is reported and written into the
    directory's own README.
    """
    run = Path(run)
    out_dir = Path(out_dir) if out_dir else (run / "size_distributions")
    all_doses = np.asarray(load(run)[0], float)

    used, skipped = [], []
    for want in doses:
        i = int(np.argmin(np.abs(all_doses - float(want))))
        got = float(all_doses[i])
        # A snapshot more than 25% away in log-dose is not the dose asked for.
        if want > 0 and got > 0 and abs(np.log10(got / want)) > 0.12:
            skipped.append((float(want), got))
            continue
        if got not in used:
            used.append(got)
    used.sort()
    if verbose:
        print(f"  size_distributions: {len(families)} families at "
              + ", ".join(f"{u:g}" for u in used) + " dpa")
        for want, got in skipped:
            print(f"    requested {want:g} dpa: nearest snapshot is {got:g}, "
                  f"too far -- omitted")
    if not used:
        print("  size_distributions: no usable snapshot, nothing written")
        return []

    # A WIDER GRID THAN THE MULTI-PANEL FIGURE'S. `build` defaults to 400 nm,
    # which the basal family outgrows: at a mean of ~155 nm its distribution is
    # still well above the three-decade window at the grid edge, so the curve
    # ended where the array ended rather than where the population did, and the
    # axis limit -- taken from the largest visible diameter -- inherited that.
    # 2000 nm is past the crystal's own dimension, so nothing physical is cut.
    d, data = build(run, doses=used, slugs=families, region=region,
                    n_d=420, d_lim=(0.3, 2000.0), verbose=False)
    written, lines = [], []
    for slug in families:
        f, drawn = render_family(d, data, slug, used,
                                 out_dir / f"size_dist_{slug}.png",
                                 region=region)
        written.append(f)
        tot = sum(data[(slug, u)][1]["count"] for u in used
                  if (slug, u) in data)
        lines.append(f"| `{slug}` | {LABEL[slug]} | {drawn} of {len(used)} | "
                     f"{tot:.3e} |")
        if verbose:
            print(f"    {slug:<4} {drawn} of {len(used)} doses populated "
                  f"-> {f.name}")

    readme = out_dir / "README.md"
    readme.write_text(
        "# Loop size distributions\n\n"
        f"Reconstructed from the three carried moments per family, over the "
        f"**{region}**, at {', '.join(f'{u:g}' for u in used)} dpa. One figure "
        "per family; `dN/dd` against a linear diameter axis, which is the form "
        "a measured TEM size distribution takes.\n\n"
        "The per-node closure is log-normal in the defect count; what is "
        "plotted is the VOLUME-WEIGHTED MIXTURE of those per-node "
        "distributions over the crystal, which is not itself log-normal and "
        "is what a micrograph of a whole specimen samples.\n\n"
        "| family | symbol | doses populated | total loops |\n"
        "|---|---|---|---:|\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    if verbose:
        print(f"  wrote {out_dir}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--doses", nargs="*", type=float, default=list(DEFAULT_DOSES))
    ap.add_argument("--families", nargs="*", default=["c", "a1", "a1v"])
    ap.add_argument("--region", default="domain",
                    choices=("domain", "interior", "shell"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--regions", action="store_true",
                    help="instead of the dose sweep, compare the whole "
                         "crystal with its interior and its boundary shell")
    ap.add_argument("--per-family", action="store_true",
                    help="write <run>/size_distributions/: one figure per "
                         "family, one curve per dose, over all nine families")
    ap.add_argument("--delta-cap", type=float, default=None,
                    help="diagnostic: clip the per-node dispersion, to see "
                         "by difference what the widest nodes contribute")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    if a.per_family:
        doses = a.doses if argv and "--doses" in argv else PER_FAMILY_DOSES
        write_per_family(run, doses=doses, out_dir=a.out, region=a.region,
                         families=(a.families if argv and "--families" in argv
                                   else ALL_FAMILIES))
        return
    if a.regions:
        out = a.out or (run / "size_spectrum" / "spectrum_regions.png")
        render_regions(run, sorted(a.doses), a.families, out)
        return
    d, data = build(run, doses=a.doses, slugs=a.families, region=a.region,
                    delta_cap=a.delta_cap)
    out = a.out or (run / "size_spectrum" / f"spectrum_{a.region}.png")
    used = sorted({k[1] for k in data})
    render(run, d, data, a.families, used, out, region=a.region)


if __name__ == "__main__":
    main()
