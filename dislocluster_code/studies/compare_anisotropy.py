"""
compare_anisotropy.py -- isotropic vs anisotropic diffusion, on the same case.

WHAT IS BEING COMPARED
----------------------
Two marches of the same configuration, differing only in the diffusion tensor
the fast solve uses:

    reference   mobileSpeciesEnergyMigration_eV isotropic, while
                dadAnisotropy carried fitted values -- so the code believed
                diffusion was isotropic AND that the sinks were biased by its
                anisotropy, which cannot both be true
    anisotropic both keys generated from one p_m per species by
                staging/anisotropy.py, so the tensor the fast solve diffuses
                with and the capture efficiencies the immobile families absorb
                through finally describe the same physics

Unlike the fast-solve optimization comparison, this one is EXPECTED to change
the answer: it is a physics change, not a refactor. So bit-identity is not the
test. What this module reports instead is where the answer moved and whether it
moved in the direction the anisotropy predicts.

WHAT IT DOES NOT ESTABLISH
--------------------------
The 28-parameter set was fitted at isotropic D and at the old atomic volume.
Neither run here is calibrated against experiment, and this comparison cannot
say which is closer to it. It is a controlled measurement of what the tensor
does to the model, nothing more.

USAGE
-----
    python -m dislocluster_code.studies.compare_anisotropy \\
        <reference_run> <anisotropic_run> [--out report.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dislocluster_code.post.discrete_loops import FAMILIES
from dislocluster_code.post.fields import B_SI, gb_distance
from dislocluster_code.post import movies as movies_mod

MOBILE = [("Cv", 0), ("Ci", 1), ("C2i", 2), ("C3i", 3)]


def _hms(x):
    x = float(x)
    h, r = divmod(x, 3600.0)
    m, s = divmod(r, 60.0)
    if h >= 1:
        return f"{int(h)} h {int(m):02d} m {int(s):02d} s"
    if m >= 1:
        return f"{int(m)} m {int(s):02d} s"
    return f"{s:.1f} s"


def _load(run):
    run = Path(run)
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    doses, nodes, frames = movies_mod.cd_blocks(run)
    return run, s, np.asarray(doses, float), nodes, frames


def _interior(nodes):
    d = gb_distance(nodes)
    return d > 0.5 * float(d.max())


def _ratio_rows(fa, fb, mask, cols):
    """Median interior ratio B/A for each named column."""
    out = []
    for name, j in cols:
        a = fa[:, j][mask]
        b = fb[:, j][mask]
        ok = (a > 0) & (b > 0)
        out.append((name, float(np.median(b[ok] / a[ok])) if ok.any()
                    else float("nan")))
    return out


def compare(ref_dir, ani_dir):
    (rp, rs, rd, rn, rf) = _load(ref_dir)
    (ap, as_, ad, an, af) = _load(ani_dir)

    L = ["# Isotropic vs anisotropic diffusion", "",
         f"- reference (isotropic) : `{rp.name}`",
         f"- anisotropic           : `{ap.name}`",
         f"- CD nodes              : {rn.shape[0]:,}", ""]

    same_grid = np.array_equal(rd, ad)
    same_nodes = rn.shape == an.shape and np.allclose(rn, an)
    L += ["## What changed", "",
          "Both runs use the same driver, the same mesh and the same optimized",
          "DDomp binary. They differ only in the diffusion tensor, and in that",
          "`dadAnisotropy` is now generated from the same `p_m` that sets the",
          "tensor rather than fitted independently of it.", "",
          "| | reference | anisotropic |",
          "|---|---|---|",
          "| `mobileSpeciesEnergyMigration_eV` | isotropic | anisotropic, from `p_m` |",
          "| `dadAnisotropy` | fitted, independent of the tensor | derived from the same `p_m` |",
          "| tensor and capture efficiencies agree | **no** | **yes** |", "",
          f"- dose grid identical: **{same_grid}**",
          f"- node set identical: **{same_nodes}**", ""]

    if not (same_grid and same_nodes):
        L += ["", "**The two runs are not the same case**; the comparison below",
              "is not meaningful. Stop here.", ""]
        return "\n".join(L) + "\n"

    mask = _interior(rn)
    L += [f"- interior nodes (innermost quartile by distance to a face): "
          f"{int(mask.sum()):,} of {mask.size:,}", ""]

    # ── physics ─────────────────────────────────────────────────────────────
    L += ["## Where the answer moved", "",
          "Median over interior nodes of the ratio anisotropic / isotropic. The",
          "interior is used because the boundary shell carries a known artifact",
          "-- loops accumulate there at their nucleation size because the only",
          "removal channel is driven by a mobile flux that Dirichlet pins to",
          "zero -- and a whole-domain mean would report that shell.", ""]
    cols = ([(n, j) for n, j in MOBILE]
            + [(f"N_{f['key']}", f["ncol"]) for f in FAMILIES]
            + [(f"c_{f['key']}", f["ccol"]) for f in FAMILIES])
    L += ["| dose | " + " | ".join(n for n, _ in cols) + " |",
          "|---:|" + "---:|" * len(cols)]
    for i, dose in enumerate(rd):
        rows = _ratio_rows(rf[i][1], af[i][1], mask, cols)
        L.append(f"| {dose:.4g} | "
                 + " | ".join(f"{v:.3f}" if np.isfinite(v) else "--"
                              for _, v in rows) + " |")
    L.append("")

    # ── coarsening detector ─────────────────────────────────────────────────
    L += ["## Coarsening detector", ""]
    ac = (as_.get("diagnostics", {}) or {}).get("coarsening")
    rc = (rs.get("diagnostics", {}) or {}).get("coarsening")
    phi_star = float(ac["phi_star"]) if ac else 0.15
    note = []
    if rc is None:
        # The reference predates the in-march detector. Re-derive it from the
        # stored snapshots so the column is not simply blank -- flagged, because
        # snapshot sampling is far coarser than the substep sampling the
        # anisotropic run used and the two crossings are not equally resolved.
        try:
            from dislocluster_code.post import coarsening as co
            _, tr = co.trajectory(rp, frac_star=(ac["frac_star"] if ac
                                                 else co.FRAC_STAR))
            rc = {"d_coarsen": co.detect(tr, phi_star),
                  "phi_gate_max": {f["key"]: max(r["families"][f["key"]]
                                                 ["phi_gate"] for r in tr)
                                   for f in FAMILIES},
                  "n_substeps": len(tr), "derived": True}
            note.append("The reference column is **re-derived from its "
                        "snapshots** by `post/coarsening.py`, because that run "
                        "predates the in-march detector. Its crossing is "
                        "interpolated between snapshots; the anisotropic one is "
                        "resolved to a substep. Compare the plateaus, which are "
                        "sampling-independent, rather than the crossing doses.")
        except Exception as exc:
            note.append(f"Could not re-derive the reference detector: {exc}")

    def _d(v):
        return f"{v:.4g} dpa" if isinstance(v, (int, float)) and v else "--"

    def _g(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "--"

    if ac or rc:
        L += [f"- `phi*` = {phi_star}, `frac*` = "
              f"{ac['frac_star'] if ac else '--'}, "
              f"hold = {ac['hold'] if ac else '--'}"
              + (f", fed at {ac['n_substeps']} substeps" if ac else ""), "",
              "| family | ref `d_coarsen` | aniso `d_coarsen` | "
              "ref max `phi_gate` | aniso max `phi_gate` |",
              "|---|---:|---:|---:|---:|"]
        for f in FAMILIES:
            k = f["key"]
            L.append(
                f"| {f['label']} "
                f"| {_d((rc or {}).get('d_coarsen', {}).get(k))} "
                f"| {_d((ac or {}).get('d_coarsen', {}).get(k))} "
                f"| {_g((rc or {}).get('phi_gate_max', {}).get(k))} "
                f"| {_g((ac or {}).get('phi_gate_max', {}).get(k))} |")
        L += [""] + ([" ".join(note), ""] if note else [])
    else:
        L += ["No coarsening diagnostics in either run.", ""]

    # ── cost ────────────────────────────────────────────────────────────────
    L += ["## Wall clock", "",
          "The anisotropic tensor fills the same sparsity pattern as the scalar",
          "it replaces, so any difference here is conditioning of the mobile",
          "solve, not extra assembly.", "",
          "| quantity | reference | anisotropic | ratio |",
          "|---|---:|---:|---:|"]

    def row(label, a, b, fmt=_hms):
        r = (b / a) if a else float("nan")
        L.append(f"| {label} | {fmt(a)} | {fmt(b)} | {r:.3f}x |")

    rfs = (rs.get("diagnostics", {}).get("fast_solver") or {})
    afs = (as_.get("diagnostics", {}).get("fast_solver") or {})
    r_slow = sum(float(t.get("slow_s", 0)) for t in rs.get("timing", []))
    a_slow = sum(float(t.get("slow_s", 0)) for t in as_.get("timing", []))
    row("march total", rs.get("wall_s", 0), as_.get("wall_s", 0))
    if rfs.get("wall_s") and afs.get("wall_s"):
        row(f"fast solves ({rfs.get('calls')} x DDomp)",
            rfs["wall_s"], afs["wall_s"])
        row("one fast solve", rfs["wall_s"] / max(rfs.get("calls", 1), 1),
            afs["wall_s"] / max(afs.get("calls", 1), 1))
    row("slow substeps (control)", r_slow, a_slow)
    L.append("")
    if rfs.get("calls") and afs.get("calls"):
        L += [f"Fast-solve convergence: reference "
              f"{rfs.get('unconverged', 0)} unconverged of {rfs['calls']}, "
              f"anisotropic {afs.get('unconverged', 0)} of {afs['calls']}.", ""]

    L += ["## Caveat", "",
          "**Neither run is calibrated.** The 28-parameter set was fitted at",
          "isotropic `D` and at the pre-correction atomic volume, so neither",
          "column above should be compared with experiment. What this measures",
          "is what the tensor does to the model, under a controlled change.",
          "The refit is deferred pending experimental data.", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference_run")
    ap.add_argument("anisotropic_run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = compare(args.reference_run, args.anisotropic_run)
    print(text)
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
