"""dad_window.py -- the analytic co-growth window, and what it requires.

THE RESULT
----------
Write the arrival rate of each mobile species at a loop as ``Dbar_m c_m |m|``.
Split it into the vacancy part and the interstitial part,

    A = Dbar_v c_v                          (vacancy arrival)
    B = sum_{m interstitial} Dbar_m c_m |m| (interstitial arrival)

A vacancy <c> loop grows when it absorbs more vacancies than interstitials, an
interstitial <a> loop when it absorbs more interstitials than vacancies. With
the Eq. (15) capture efficiencies

    Z_c(m) = Z0_m p_m                       <c>, basal, vacancy-type
    Z_a(m) = Z0_m (p_m + p_m^-2)/2          <a>, prismatic, interstitial-type

and one anisotropy ``p_I`` shared by the interstitial species (Z0 is already
common to them in `Zr3d_ghoniem.txt`), those two conditions are

    <c> grows:   Z0_v p_v A  >  Z0_I p_I B         =>  A/B > (Z0_I/Z0_v) p_I/p_v
    <a> grows:   Z0_I f(p_I) B > Z0_v f(p_v) A     =>  A/B < (Z0_I/Z0_v) f(p_I)/f(p_v)

so BOTH families grow only for arrival ratios inside a window whose width is

    W = [f(p_I)/f(p_v)] / [p_I/p_v] = g(p_v)/g(p_I),   g(p) = 2/(1 + p^-3)

`g` is strictly increasing, so:

    ** the window is non-empty if and only if  p_I < p_v **

That is the whole DAD argument in one line -- interstitials must be the species
biased INTO the basal plane relative to vacancies -- and it is a statement about
the capture efficiencies alone, independent of the microstructure, the dose and
the mobile field.

WHAT IT DOES NOT SETTLE
-----------------------
Whether the window is non-empty is not whether the material sits IN it. That
depends on the realized `A/B`, which the fast solve sets and which the anisotropy
itself perturbs. So this module does two things:

  `window`   the bounds and width, closed form, free
  `measure`  the realized A/B from one fast solve, and the p_I that would be
             needed to bring the measured ratio inside the window

The inversion is the useful direction. Given a measured `R = (A/B)(Z0_v/Z0_I)`,
holding `p_v = 1`, <a> growth needs `f(p_I) > R`, i.e.

    p_I  <  the root of (p + p^-2)/2 = R

which for R >> 1 approaches ``p_I < 1/sqrt(2R)``. A vacancy-rich interior
therefore demands a STRONGLY basal-biased interstitial cluster, and the required
p_I falls only as the square root -- doubling the vacancy excess costs a factor
1.41 in the anisotropy, not 2.

USAGE
-----
    python -m dislocluster_code.studies.dad_window --sim <staged qssa dir> \\
        --state <march_state.npz> --doses 0.01 0.1 1 10 [--out report.md]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from dislocluster_code.coupling import field as mf, qssa as mq
from dislocluster_code.post.fields import gb_distance
from dislocluster_code.staging import anisotropy as ani
from dislocluster_code.studies.dad_sweep import (MOBILE, _material_of,
                                                 _read_d0, _read_dadz0)


def f_a(p):
    """<a> capture shape ``(p + p^-2)/2`` -- minimum 1 at p = 1."""
    p = np.asarray(p, float)
    return 0.5 * (p + p ** -2.0)


def g(p):
    """``p / f_a(p) = 2/(1 + p^-3)`` -- strictly increasing."""
    p = np.asarray(p, float)
    return 2.0 / (1.0 + p ** -3.0)


def window(p_v, p_I, z0_v, z0_I):
    """``(lower, upper, width)`` bounds on A/B for simultaneous growth."""
    lo = (z0_I / z0_v) * (p_I / p_v)
    hi = (z0_I / z0_v) * (f_a(p_I) / f_a(p_v))
    return float(lo), float(hi), float(hi / lo)


def window_mixed(p_v, p_int, weights, z0_v, z0_I):
    """The window when the interstitial species do NOT share one anisotropy.

    `Li et al.`'s structure gives the DAD to the interstitial CLUSTERS and
    leaves the single interstitial isotropic, so ``p_int`` differs across the
    species and the single-`p_I` reduction no longer applies. Each condition
    still collapses to one number, now an ARRIVAL-WEIGHTED mean:

        p_bar = sum_m w_m p_m          w_m = Dbar_m c_m |m| / B
        f_bar = sum_m w_m f(p_m)

        <c> grows  <=>  A/B > (Z0_I/Z0_v) p_bar / p_v
        <a> grows  <=>  A/B < (Z0_I/Z0_v) f_bar / f(p_v)

    so the window is non-empty iff ``p_v f_bar > f(p_v) p_bar``. Since

        f(p) - p = (1 - p^3) / (2 p^2)

    is positive exactly when ``p < 1``, an isotropic species contributes
    NOTHING to opening the window: it enters `p_bar` and `f_bar` with the same
    value. **Only the anisotropic species help, and only in proportion to the
    share of the interstitial arrival they carry.** Giving the DAD to the
    clusters alone therefore buys a window only to the extent that the clusters
    -- not the single interstitial -- deliver the interstitial flux.
    """
    p_int = np.asarray(p_int, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    p_bar = float((w * p_int).sum())
    f_bar = float((w * f_a(p_int)).sum())
    lo = (z0_I / z0_v) * p_bar / p_v
    hi = (z0_I / z0_v) * f_bar / float(f_a(p_v))
    return lo, hi, hi / lo, p_bar, f_bar


def required_p_I(ratio, p_v, z0_v, z0_I):
    """Largest ``p_I`` for which <a> still grows at the given ``A/B``.

    Solves ``f_a(p_I) = (A/B)(Z0_v/Z0_I) f_a(p_v)`` on ``(1e-4, 1]``. Returns
    ``None`` when even ``p_I -> 0`` cannot do it (impossible only if the target
    is below 1, i.e. the <a> condition already holds at p_I = 1).
    """
    target = float(ratio) * (z0_v / z0_I) * float(f_a(p_v))
    if target <= 1.0:
        return None                       # already satisfied at p_I = 1
    try:
        return float(brentq(lambda p: f_a(p) - target, 1e-4, 1.0))
    except ValueError:
        return None


def measure(sim_dir, Y, p_m, T=573.0):
    """Realized ``A/B`` per node from one fast solve at ``p_m``."""
    mat = _material_of(sim_dir)
    ani.apply(p_m, material_file=mat, T=T)
    t0 = time.perf_counter()
    cm = mq.MobileQSSASolver(sim_dir, mat, verbose=False).solve(Y)
    wall = time.perf_counter() - t0

    E = ani.read_migration(mat)
    d0 = _read_d0(mat)
    dbar = np.array([(d0[k, 0] * np.exp(-E[k, 0] / (ani.KB_EV * T))) ** (2 / 3)
                     * (d0[k, 5] * np.exp(-E[k, 5] / (ani.KB_EV * T))) ** (1 / 3)
                     for k in range(E.shape[0])])
    arrival = np.column_stack([dbar[m] * cm[:, m] * abs(size)
                               for m, (_n, size) in enumerate(MOBILE)])
    A = arrival[:, 0]
    B = arrival[:, 1:].sum(axis=1)
    return A / np.maximum(B, 1e-300), arrival, cm, wall


def run(sim_dir, state_npz, doses, p_m=(1.0, 1.0, 1.0, 1.0), T=573.0,
        verbose=True):
    sim_dir = Path(sim_dir)
    mat = _material_of(sim_dir)
    backup = mat.read_text(encoding="utf-8")
    z0 = _read_dadz0(mat)
    z0_v, z0_I = float(z0[0]), float(z0[1])

    z = np.load(state_npz)
    all_doses = np.asarray(z["doses"], float)
    nodes = mf.FieldBridge(sim_dir, mat).nodes
    d = gb_distance(nodes)
    interior = d > 0.5 * float(d.max())

    out = []
    try:
        for dose in doses:
            i = int(np.argmin(np.abs(all_doses - dose)))
            Y = np.array(z["Y"][i], dtype=float)
            ratio, arrival, cm, wall = measure(sim_dir, Y, p_m, T)
            r = ratio[interior]
            # Share of the INTERSTITIAL arrival carried by each species. This
            # is what decides whether giving the DAD to the clusters alone can
            # open a window at all -- an isotropic species is inert in the
            # mixed criterion, so the anisotropic ones only help in proportion
            # to the flux they deliver.
            ai = arrival[interior, 1:]
            w = ai / np.maximum(ai.sum(axis=1, keepdims=True), 1e-300)
            wmed = np.median(w, axis=0)
            rec = dict(dose=float(all_doses[i]),
                       ratio_med=float(np.median(r)),
                       ratio_q1=float(np.quantile(r, 0.25)),
                       ratio_q3=float(np.quantile(r, 0.75)),
                       Cv=float(np.median(cm[interior, 0])),
                       Ci=float(np.median(cm[interior, 1])),
                       w_i=float(wmed[0]), w_2i=float(wmed[1]),
                       w_3i=float(wmed[2]),
                       wall_s=wall)
            rec["p_I_needed"] = required_p_I(rec["ratio_med"], p_m[0],
                                             z0_v, z0_I)
            out.append(rec)
            if verbose:
                need = rec["p_I_needed"]
                print(f"  {rec['dose']:8.4g} dpa  A/B = {rec['ratio_med']:.4g} "
                      f"[{rec['ratio_q1']:.3g}, {rec['ratio_q3']:.3g}]  "
                      f"w(i,2i,3i) = {wmed[0]:.3f},{wmed[1]:.3f},{wmed[2]:.3f}"
                      f"  p_I needed <= "
                      f"{'(any)' if need is None else f'{need:.4f}'}"
                      f"  ({wall:.0f} s)", flush=True)
    finally:
        mat.write_text(backup, encoding="utf-8")
    return out, (z0_v, z0_I)


def report(records, z0, p_m, grid=None):
    z0_v, z0_I = z0
    L = ["# The co-growth window, and whether the material sits in it", "",
         "## The criterion", "",
         "With `A` the vacancy arrival rate `Dbar_v c_v` and `B` the "
         "interstitial arrival rate `sum_m Dbar_m c_m |m|`, and one anisotropy "
         "`p_I` shared by the interstitial species:", "",
         "```",
         "<c> grows   <=>   A/B  >  (Z0_I/Z0_v) * p_I / p_v",
         "<a> grows   <=>   A/B  <  (Z0_I/Z0_v) * f(p_I) / f(p_v)",
         "",
         "f(p) = (p + p^-2)/2        g(p) = p/f(p) = 2/(1 + p^-3)",
         "```", "",
         "Both hold for some `A/B` only if the window is non-empty, i.e. only "
         "if `g(p_I) < g(p_v)`. `g` is strictly increasing, so", "",
         "> **simultaneous growth is possible if and only if `p_I < p_v`** --",
         "> the interstitial clusters must be biased into the basal plane "
         "*relative to* the vacancies.", "",
         "This is a property of the capture efficiencies alone: it does not "
         "depend on the dose, the microstructure or the mobile field. What "
         "those decide is whether the realized `A/B` lands inside the window.",
         "",
         f"Here `Z0_v = {z0_v:.6f}`, `Z0_I = {z0_I:.6f}`, so "
         f"`Z0_I/Z0_v = {z0_I / z0_v:.4f}`.", ""]

    if grid is not None:
        L += ["## Window width over the grid", "",
              "`W = g(p_v)/g(p_I)` -- the factor in `A/B` that both families "
              "tolerate. `W <= 1` means no `A/B` whatever admits both.", "",
              "| `p_v` \\ `p_I` | " + " | ".join(f"{p:.4g}" for p in grid[1])
              + " |",
              "|---:|" + "---:|" * len(grid[1])]
        for p_v in grid[0]:
            cells = []
            for p_I in grid[1]:
                w = float(g(p_v) / g(p_I))
                cells.append(f"**{w:.3f}**" if w > 1.0 else f"{w:.3f}")
            L.append(f"| {p_v:.4g} | " + " | ".join(cells) + " |")
        L += ["", "Bold = a window exists.", ""]

    L += ["## The realized arrival ratio", "",
          f"Measured by one fast solve per dose at "
          f"`p_m = ({', '.join(f'{p:g}' for p in p_m)})`, median over interior "
          "nodes (innermost quartile by distance to a face).", "",
          "| dose | median `A/B` | IQR | `Cv` | `Ci` | `p_I` needed for "
          "`<a>` growth |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        need = r["p_I_needed"]
        L.append(f"| {r['dose']:.4g} | {r['ratio_med']:.4g} | "
                 f"{r['ratio_q1']:.3g} - {r['ratio_q3']:.3g} | "
                 f"{r['Cv']:.3e} | {r['Ci']:.3e} | "
                 + ("any `p_I <= 1`" if need is None else f"**{need:.4f}**")
                 + " |")

    # ── who carries the interstitial flux ───────────────────────────────────
    L += ["", "## Which species carries the interstitial arrival", "",
          "In the mixed criterion the window opens by "
          "`f(p) - p = (1 - p^3)/(2 p^2)`, which is **zero for an isotropic "
          "species**. A species left at `p = 1` enters `p_bar` and `f_bar` "
          "identically and contributes nothing. So the anisotropic species "
          "help only in proportion to the share of the interstitial arrival "
          "they deliver:", "",
          "| dose | `w_i` | `w_2i` | `w_3i` | cluster share |",
          "|---:|---:|---:|---:|---:|"]
    for r in records:
        cl = r["w_2i"] + r["w_3i"]
        L.append(f"| {r['dose']:.4g} | {r['w_i']:.4f} | {r['w_2i']:.4f} | "
                 f"{r['w_3i']:.4f} | **{cl:.4f}** |")
    L += ["",
          "This is the number that decides whether Li et al.'s structure -- "
          "DAD on the di- and tri-interstitial, single interstitial isotropic "
          "-- can work in this model. If the cluster share is small, the "
          "cluster-only DAD is nearly inert however extreme `p_2i` is made, "
          "and the anisotropy has to be carried by the species that actually "
          "delivers the flux.", ""]

    L += ["", "The `p_I` column inverts the `<a>` condition at the measured "
          "`A/B`, holding `p_v = 1`: it is the root of `f(p_I) = "
          "(A/B)(Z0_v/Z0_I) f(p_v)`. For `A/B >> 1` it behaves as "
          "`p_I ~ 1/sqrt(2 A/B)`, so the required anisotropy tightens only as "
          "the square root of the vacancy excess.", "",
          "**Caveat.** `A/B` is itself a function of the anisotropy, because "
          "the tensor reshapes the mobile field. The column is therefore a "
          "first estimate obtained at the stated `p_m`, not a fixed point; a "
          "march at the required `p_I` moves `A/B` and the number should be "
          "re-measured there.", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--doses", nargs="*", type=float, default=[0.01, 0.1, 1, 10])
    ap.add_argument("--p-m", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0])
    ap.add_argument("--temperature", type=float, default=573.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    recs, z0 = run(args.sim, args.state, args.doses, tuple(args.p_m),
                   args.temperature)
    grid = ([0.7, 0.85, 1.0, 1.178808], [0.3, 0.5, 0.7, 0.91372, 1.0, 1.2])
    text = report(recs, z0, args.p_m, grid)
    print("\n" + text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
