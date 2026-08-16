"""
dad_sweep.py -- which DAD parameters let <a> AND <c> loops grow at the same time.

THE QUESTION
------------
Irradiated Zr is observed to hold both prismatic interstitial <a> loops and basal
vacancy <c> loops. A model reproduces that only if, over some range of dose, the
growth rate of BOTH families is positive at the same time and in the same place.
This module maps the region of DAD parameter space where that happens.

WHY THIS CANNOT BE DONE IN 0-D
------------------------------
In the 0-D model the diffusional anisotropy enters only through the capture
efficiencies: it re-weights how strongly each family absorbs, and the mobile
concentrations themselves are set by a well-mixed balance that knows nothing
about direction. In 3-D the anisotropy does that AND reshapes the mobile field,
because a species that diffuses faster along c reaches the basal-pole boundary
sooner and is depleted near it. Li et al. make exactly this point, and the
measurement that motivated this module is an instance of it: giving vacancies
p_v = 1.179 lowered Cv and raised Ci near the basal-pole face enough to dissolve
the <c> loops there, which no 0-D evaluation would have shown.

So each parameter point needs one FAST SOLVE. That is the cost driver and the
reason the sweep runs on the 200 nm case (27 720 nodes, ~50 s per point) rather
than the 500 nm one (90 617 nodes, ~200 s).

WHAT IS HELD FIXED
------------------
- The IMMOBILE state. Growth rates are evaluated from one frozen microstructure,
  so the sweep compares parameter sets and not the different histories they would
  have produced. Sign changes here are a statement about the instantaneous
  competition, not about the trajectory.
- D_eff per species. `staging/anisotropy.py` splits the migration energies at
  fixed (D_a^2 D_c)^(1/3), so a point in this sweep changes the DIRECTIONALITY of
  diffusion and not its overall rate. Without that the sweep would confound
  anisotropy with mobility.

WHAT IS COMPUTED
----------------
The same growth flux `solveImmobileClusters()` forms, Eqs. (41)-(45):

    gdot_k = phi_like_k - gate * phi_opp_k
    phi_k  = sum_m Dbar_m Z(row(k), m) S_k c_m |m_m|

with `Dbar_m = (det D_m)^(1/3)`, `S_k` the family's sink strength, and Z the
Eq. (15) capture efficiencies from `loopDADbias()`:

    Z_c(m) = Z0_m * p_m                      <c>, vacancy-type
    Z_a(m) = Z0_m * (p_m + p_m^-2) / 2       <a>, interstitial-type

A vacancy <c> loop grows on Cv and shrinks on the interstitials; an interstitial
<a> loop does the reverse. Co-growth therefore needs the vacancy supply and the
interstitial supply to be simultaneously favourable to their own family, which is
only possible because Z_c and Z_a respond differently to p_m: Z_c = Z0 p is
monotone, while Z_a = Z0 (p + p^-2)/2 is a rising plus a falling term with an
interior minimum at p = 2^(1/3) = 1.2599 (where it is 0.9449 Z0), so it is still
DECREASING at p = 1 and dips below Z0 over 1 < p < (1+sqrt5)/2 = 1.618.

`studies/dad_window.py` turns that difference into a closed-form criterion:
co-growth is possible iff p_I < p_v. This module measures where the realized
arrival ratio falls relative to the window that criterion defines.

USAGE
-----
    python -m dislocluster_code.studies.dad_sweep --sim <staged qssa dir> \\
        --state <march_state.npz> --dose 1.0 [--out report.md]

Sweeps `--p-v` against `--p-cluster` (2i and 3i together, i optionally tied).
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import field as mf, qssa as mq
from dislocluster_code.post.discrete_loops import FAMILIES
from dislocluster_code.post.fields import gb_distance
from dislocluster_code.staging import anisotropy as ani
from dislocluster_code.staging.anisotropy import KB_EV, split_migration

# Mobile species: (name, signed size m). Negative = vacancy-type.
MOBILE = [("Cv", -1), ("Ci", 1), ("C2i", 2), ("C3i", 3)]


def capture_efficiencies(p_m, z0):
    """``(Z_c, Z_a)`` per mobile species -- `loopDADbias()` in Python.

    Row 0 is the <c> (vacancy-type) family, row 1 the <a> (interstitial-type)
    families, exactly as ``ClusterDynamicsParameters::loopDADbias`` returns them.
    """
    p = np.asarray(p_m, float)
    z0 = np.asarray(z0, float)
    return z0 * p, z0 * 0.5 * (p + p ** -2.0)


def growth_rates(cm, p_m, z0, dbar, sink_c, sink_a):
    """``(gdot_c, gdot_a)`` per node, in the sign convention of `gdot` in C++.

    ``cm`` is (n_nodes, 4). Positive means the family gains defects. The
    minimum-stable-size gate is NOT applied: it only ever reduces the shrinking
    term, so omitting it is conservative for a growth criterion -- a family this
    says is shrinking would shrink at least as fast with the gate.
    """
    Zc, Za = capture_efficiencies(p_m, z0)
    like_c = np.zeros(len(cm))
    opp_c = np.zeros(len(cm))
    like_a = np.zeros(len(cm))
    opp_a = np.zeros(len(cm))
    for m, (_name, size) in enumerate(MOBILE):
        w = dbar[m] * cm[:, m] * abs(size)
        if size < 0:                      # vacancy species
            like_c += Zc[m] * w
            opp_a += Za[m] * w
        else:                             # interstitial species
            opp_c += Zc[m] * w
            like_a += Za[m] * w
    return sink_c * (like_c - opp_c), sink_a * (like_a - opp_a)


def _material_of(sim_dir):
    """The staged COPY of the material file that DDomp actually reads."""
    cands = list((Path(sim_dir) / "inputFiles").glob("Zr*.txt"))
    if not cands:
        raise FileNotFoundError(f"no material file in {sim_dir}/inputFiles")
    return cands[0]


def evaluate(sim_dir, Y, p_m, z0, T=573.0, verbose=True):
    """One parameter point: set the tensor, fast-solve, return growth rates.

    Writes the STAGED copy of the material file, never the one in
    ``Library/Materials``. Editing the Library copy would change
    ``cfg.domain_key`` -- which now hashes the material's content -- and force a
    re-stage and a fresh bootstrap for every point in the sweep.
    """
    mat = _material_of(sim_dir)
    ani.apply(p_m, material_file=mat, T=T)

    t0 = time.perf_counter()
    fast = mq.MobileQSSASolver(sim_dir, mat, verbose=False)
    cm = fast.solve(Y)
    wall = time.perf_counter() - t0

    # Dbar is (det D)^(1/3) = D_eff, which the fixed-D_eff split leaves
    # UNCHANGED by p_m. Read it back from the file actually used rather than
    # assuming, so a mistake in the split shows up here rather than silently.
    E = ani.read_migration(mat)
    kB = ani.KB_EV
    d0 = _read_d0(mat)
    dbar = np.array([(d0[k, 0] * np.exp(-E[k, 0] / (kB * T))) ** (2 / 3)
                     * (d0[k, 5] * np.exp(-E[k, 5] / (kB * T))) ** (1 / 3)
                     for k in range(E.shape[0])])
    return cm, dbar, wall


def _read_d0(material_file):
    import re
    txt = Path(material_file).read_text(encoding="utf-8")
    m = re.search(r"^mobileSpeciesD0_SI\s*=(.*?);", txt, re.S | re.M)
    v = [float(x) for x in m.group(1).split()]
    return np.array(v, float).reshape(-1, 6)


def sweep(sim_dir, state_npz, dose, p_v_list, p_cluster_list, tie_i=True,
          z0=None, T=573.0, verbose=True):
    """Run the grid. Returns a list of records, one per parameter point."""
    sim_dir = Path(sim_dir)
    mat = _material_of(sim_dir)
    backup = mat.read_text(encoding="utf-8")

    z = np.load(state_npz)
    i_dose = int(np.argmin(np.abs(np.asarray(z["doses"], float) - dose)))
    Y = np.array(z["Y"][i_dose], dtype=float)
    nodes = mf.FieldBridge(sim_dir, mat).nodes
    if Y.shape[0] != nodes.shape[0]:
        raise ValueError(f"state has {Y.shape[0]} nodes, the staged case has "
                         f"{nodes.shape[0]} — they must be the same case")
    d = gb_distance(nodes)
    interior = d > 0.5 * float(d.max())

    omega = mf.cluster_atomic_volume(mat)
    cd = mf.immobile_0d_to_modelib(Y, omega)
    nF = len(FAMILIES)
    # Sink strength S = 2 pi r N, per family, from the frozen microstructure.
    sink = {}
    for j, fam in enumerate(FAMILIES):
        n, c = cd[:, j], cd[:, nF + j]
        ok = (n > 0) & (c > 0)
        r = np.zeros_like(n)
        r[ok] = np.sqrt(c[ok] / (n[ok] * np.pi * fam["b_cd"]))
        sink[fam["key"]] = 2.0 * np.pi * r * n
    sink_c = sink["c"]
    sink_a = sum(sink[k] for k in ("a1", "a2", "a3")) / 3.0

    if z0 is None:
        z0 = _read_dadz0(mat)

    out = []
    try:
        for p_v, p_cl in itertools.product(p_v_list, p_cluster_list):
            p_i = p_cl if tie_i else 1.0
            p_m = (p_v, p_i, p_cl, p_cl)
            cm, dbar, wall = evaluate(sim_dir, Y, p_m, z0, T, verbose)
            gc, ga = growth_rates(cm, p_m, z0, dbar, sink_c, sink_a)
            rec = dict(
                p_v=p_v, p_i=p_i, p_cluster=p_cl, wall_s=wall,
                gdot_c=float(np.median(gc[interior])),
                gdot_a=float(np.median(ga[interior])),
                frac_c_growing=float((gc[interior] > 0).mean()),
                frac_a_growing=float((ga[interior] > 0).mean()),
                frac_both=float(((gc[interior] > 0) & (ga[interior] > 0)).mean()),
                Cv_med=float(np.median(cm[interior, 0])),
                Ci_med=float(np.median(cm[interior, 1])))
            out.append(rec)
            if verbose:
                print(f"  p_v={p_v:.4f} p_cl={p_cl:.4f}  "
                      f"gdot_c={rec['gdot_c']:+.3e}  gdot_a={rec['gdot_a']:+.3e}"
                      f"  both={rec['frac_both']:.2f}  ({wall:.0f} s)",
                      flush=True)
    finally:
        mat.write_text(backup, encoding="utf-8")     # always restore
        if verbose:
            print(f"  restored {mat}")
    return out


def _read_dadz0(material_file):
    import re
    txt = Path(material_file).read_text(encoding="utf-8")
    m = re.search(r"^dadZ0\s*=(.*?);", txt, re.S | re.M)
    return np.array([float(x) for x in m.group(1).split()], float)


def report(records, dose, e_eff=None, T=573.0):
    # ASCII only: this text is printed to a console that is cp1252 on Windows,
    # and a UnicodeEncodeError on the PRINT would lose a sweep that has already
    # cost half an hour of fast solves. The file is written utf-8 either way.
    L = ["# DAD parameters for simultaneous <a> and <c> loop growth", "",
         f"Growth rates evaluated at **{dose:g} dpa** from one frozen immobile "
         "state, on interior nodes. Each row is one fast solve — the mobile "
         "field is re-solved for every parameter point, because the anisotropy "
         "reshapes it and not only the capture efficiencies.", "",
         "`D_eff` is held fixed per species, so a row changes the "
         "**directionality** of diffusion, not its overall rate.", "",
         "| `p_v` | `p_i` | `p_2i=p_3i` | median `gdot_c` | median `gdot_a` | "
         "frac `<c>` grow | frac `<a>` grow | **frac both** |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        L.append(f"| {r['p_v']:.4f} | {r['p_i']:.4f} | {r['p_cluster']:.4f} | "
                 f"{r['gdot_c']:+.3e} | {r['gdot_a']:+.3e} | "
                 f"{r['frac_c_growing']:.3f} | {r['frac_a_growing']:.3f} | "
                 f"**{r['frac_both']:.3f}** |")
    best = max(records, key=lambda r: r["frac_both"]) if records else None
    median_both = [r for r in records
                   if r["gdot_c"] > 0 and r["gdot_a"] > 0]
    L += [""]
    if best:
        L += [f"**Widest co-growth region**: `p_v = {best['p_v']:.4f}`, "
              f"`p_i = {best['p_i']:.4f}`, "
              f"`p_2i = p_3i = {best['p_cluster']:.4f}` -- both families growing "
              f"on {best['frac_both']:.1%} of interior nodes.", ""]
    if median_both:
        L += ["Points where the **median** interior node grows in both "
              "families at once (a stronger statement than the fraction "
              "column, which can be satisfied by disjoint parts of the "
              "domain):", ""]
        for r in median_both:
            L.append(f"- `p_v = {r['p_v']:.4f}`, `p_i = {r['p_i']:.4f}`, "
                     f"`p_2i = p_3i = {r['p_cluster']:.4f}`")
        L.append("")
    else:
        L += ["**No grid point has both medians positive.** Co-growth in this "
              "state is regional, not domain-wide: the fraction column above "
              "counts nodes, and where it is non-zero the two families are "
              "growing in different parts of the interior.", ""]

    # ── the energies, which is what actually goes into the material file ────
    L += ["## Migration energies for these anisotropy factors", "",
          "The DAD factor is not itself an input. What the material file "
          "carries is `mobileSpeciesEnergyMigration_eV`, split at fixed "
          "`D_eff = (D_a^2 D_c)^(1/3)` by `staging/anisotropy.py`:", "",
          "```",
          "E_m<11,22> = E_m_eff + 2 kT ln(p_m)",
          "E_m<33>    = E_m_eff - 4 kT ln(p_m)",
          "```", ""]
    if e_eff is not None:
        kT = KB_EV * float(T)
        names = ["v", "i", "2i", "3i"][:len(e_eff)]
        p_used = sorted({r[k] for r in records
                         for k in ("p_v", "p_i", "p_cluster")})
        L += [f"At T = {T:g} K (kT = {kT:.6f} eV), per species `E_m_eff` = "
              + ", ".join(f"`{n}` {e:.6f}" for n, e in zip(names, e_eff))
              + " eV:", "",
              "| `p_m` | " + " | ".join(f"`{n}` E11=E22 / E33" for n in names)
              + " |",
              "|---:|" + "---:|" * len(names)]
        for p in p_used:
            cells = []
            for e in e_eff:
                e11, e33 = split_migration(e, p, T)
                cells.append(f"{e11:.6f} / {e33:.6f}")
            L.append(f"| {p:.4f} | " + " | ".join(cells) + " |")
        L += ["",
              "Read the column for the species that carries that `p_m` in the "
              "chosen row: e.g. a `p_v` of 1.1788 takes the `v` entry of the "
              "1.1788 row, and `p_2i = 0.70` takes the `2i` entry of the 0.70 "
              "row.", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", required=True, help="staged _qssa directory")
    ap.add_argument("--state", required=True, help="march_state.npz")
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--p-v", nargs="*", type=float,
                    default=[1.0, 1.05, 1.1, 1.178808])
    ap.add_argument("--p-cluster", nargs="*", type=float,
                    default=[0.5, 0.6, 0.7, 0.8, 0.913720, 1.0])
    ap.add_argument("--free-i", action="store_true",
                    help="hold p_i = 1 instead of tying it to the clusters")
    ap.add_argument("--temperature", type=float, default=573.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    recs = sweep(args.sim, args.state, args.dose, args.p_v, args.p_cluster,
                 tie_i=not args.free_i, T=args.temperature)
    e_eff = ani.effective_energies(_material_of(args.sim), args.temperature)
    text = report(recs, args.dose, e_eff, args.temperature)
    print("\n" + text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(recs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
