"""
coarsening.py -- locate ``d_coarsen``, the dose at which the CONTINUUM treatment
of loop coarsening stops being trustworthy.

WHAT THIS MEASURES, AND WHY IT IS NOT "WHEN COARSENING STARTS"
-------------------------------------------------------------
The continuum model already carries both coarsening channels:
``solveImmobileClusters()`` forms an Avrami overlap fraction for like-loop and
loop-network encounters and integrates them implicitly. So coarsening is
modelled from the first dose step. The question this module answers is the
different one of when that MEAN-FIELD treatment stops being valid, which is what
governs the switch to the discrete-continuum coupled mode.

    phi_LL = 1 - exp(-kappaLL * Lam_LL),   Lam_LL = (4/3) pi r^3 n
    phi_LN = 1 - exp(-kappaLN * Lam_LN),   Lam_LN = pi r^2 rho_N

``1 - exp(-kappa*Lam)`` is a POISSON estimate: it assumes the loops are placed
independently at random. Its leading error is the two-body correlation, entering
relatively at O(Lam) -- and coarsening is precisely the process that correlates
positions. The form is therefore asymptotically exact as Lam -> 0 and
progressively self-invalidating as Lam -> 1, which is what makes phi itself the
natural detector rather than some external criterion.

Note the identity

    Lam_LL = (4/3) pi r^3 n = (pi/6) (2r/d)^3       with d = n^(-1/3)

so the Avrami argument and the geometric crowding ratio ``2r/d`` are the same
measure. This is not a new quantity invented for the switch: it is the model's
own estimate of the effect in question.

WHICH RADIUS
------------
``r`` must be the radius the CONTINUUM model itself uses, so that phi here is
the phi that ``solveImmobileClusters()` computes. That means the CD-internal
Burgers magnitude ``b_cd``, NOT the ``b_dd`` that ``discrete_loops`` uses.

The two differ, and for <c> they differ by a lot: the cluster-dynamics material
file gives the <c> family a FULL <0001> Burgers vector (1.632993 b) while
MoDELib3's aLoop generator builds a HALF <0001> loop (0.8165 b). Defect count is
the conserved quantity, not radius, so the same loop has

    r_dd / r_cd = sqrt(b_cd / b_dd) = sqrt(2)

Using the discrete radius here would inflate Lam_LL by 2^(3/2) = 2.83 and Lam_LN
by 2, and would fire the detector roughly a decade of dose too early.

INTERIOR ONLY
-------------
Evaluated on interior nodes, by the same innermost-quartile rule
``discrete_loops`` uses. The boundary shell carries 99.8% of the <a> loop density
at 21 dpa -- an artifact of cascade nucleation being uniform while the only
removal channel is coalescence driven by an absorbed flux that vanishes where
Dirichlet pins the mobile field. A whole-domain reduction would fire the detector
on that artifact immediately.

USAGE
-----
    python -m dislocluster_code.post.coarsening <run_dir> [--phi 0.2]
                                                [--frac 0.10] [--json out.json]

This is REPORTING ONLY. It switches nothing; it exists so that d_coarsen can be
seen on runs that already exist before it is allowed to drive anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dislocluster_code import paths                                # noqa: E402
from dislocluster_code.coupling.field import read_material_scalar  # noqa: E402
from dislocluster_code.post import movies as movies_mod            # noqa: E402
from dislocluster_code.post.discrete_loops import FAMILIES         # noqa: E402
from dislocluster_code.post.fields import (                        # noqa: E402
    B_SI, OMEGA_B3, gb_distance,
)

# Defaults. phi* = 0.2 is a ~20% error in the coalescence rate from the
# neglected pair correlation; it lands at 2r/d = 0.597 in the like-loop channel.
# frac* requires a real region of the crystal to have crossed rather than one
# outlying node, and HOLD requires it to stay crossed, so the switch cannot
# chatter on a single noisy substep.
PHI_STAR = 0.20
FRAC_STAR = 0.10
HOLD = 2


def material_params(material_file=None):
    """``(kappaLL, kappaLN, rho_N_in_b^-2)`` from the MoDELib material file."""
    mf = Path(material_file or paths.MODELIB_MATERIAL)
    b = read_material_scalar(mf, "b_SI")
    return (read_material_scalar(mf, "kappaLL"),
            read_material_scalar(mf, "kappaLN"),
            read_material_scalar(mf, "rhoNetwork_SI") * b * b)


def loop_radius_b(n, c, bmag):
    """Continuum mean loop radius in units of b, from number density and content.

    ``n`` is loops per b^3 and ``c`` a defect volume fraction, which is MoDELib's
    own convention and the one ``clusterRadius()`` assumes. Defects per loop is
    ``m = c/(n*Omega)`` and the radius that stores them follows from
    ``pi r^2 |b| = m Omega``.
    """
    n = np.asarray(n, float)
    c = np.asarray(c, float)
    ok = (n > 0.0) & (c > 0.0)
    m = np.zeros_like(n)
    m[ok] = c[ok] / (n[ok] * OMEGA_B3)
    return np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * bmag))


def avrami(n, c, bmag, kLL, kLN, rhoN):
    """``(phi_LL, phi_LN, r_b, lam_LL, lam_LN)`` per node, all arrays."""
    r = loop_radius_b(n, c, bmag)
    lam_LL = (4.0 / 3.0) * np.pi * r ** 3 * np.asarray(n, float)
    lam_LN = np.pi * r ** 2 * rhoN
    return (1.0 - np.exp(-kLL * lam_LL), 1.0 - np.exp(-kLN * lam_LN),
            r, lam_LL, lam_LN)


def interior_mask(nodes):
    """Innermost quartile by distance to the nearest domain face.

    The same rule ``discrete_loops.build`` applies, so the two agree about what
    "interior" means and their numbers can be compared directly.
    """
    d = gb_distance(nodes)
    return d > 0.5 * float(d.max())


def trajectory(run_dir, material_file=None, frac_star=FRAC_STAR):
    """``(doses, [per-dose dict])`` -- phi and its inputs at every snapshot.

    ``phi_gate`` is the ``1 - frac_star`` quantile of phi over interior nodes,
    and it is the quantity the detector actually thresholds. The two statements
    "a fraction ``frac_star`` of nodes exceed ``phi_star``" and "``phi_gate`` >=
    ``phi_star``" are the same statement, but the second is a smooth function of
    dose while the first is a step -- on this run the fraction goes 0.000 to
    1.000 between two snapshots, because the interior field is smooth enough
    that every node crosses at once. Interpolating a step gives a meaningless
    crossing dose; interpolating ``phi_gate`` gives the right one.
    """
    kLL, kLN, rhoN = material_params(material_file)
    doses, nodes, frames = movies_mod.cd_blocks(run_dir)
    mask = interior_mask(nodes)
    nm = B_SI * 1e9

    out = []
    for i, dose in enumerate(doses):
        _, F = frames[i]
        rec = {"dose": float(dose), "families": {}}
        for fam in FAMILIES:
            n = F[:, fam["ncol"]][mask]
            c = F[:, fam["ccol"]][mask]
            pLL, pLN, r, lLL, lLN = avrami(n, c, fam["b_cd"], kLL, kLN, rhoN)
            phi = np.maximum(pLL, pLN)
            with np.errstate(divide="ignore", invalid="ignore"):
                d_sp = np.where(n > 0, n ** (-1.0 / 3.0), np.nan)
            rec["families"][fam["key"]] = {
                "r_nm": float(np.nanmedian(np.where(n > 0, r, np.nan)) * nm),
                "d_nm": float(np.nanmedian(d_sp) * nm),
                "two_r_over_d": float(np.nanmedian(
                    np.where(n > 0, 2.0 * r / d_sp, np.nan))),
                "lam_LL": float(np.nanmedian(np.where(n > 0, lLL, np.nan))),
                "lam_LN": float(np.nanmedian(np.where(n > 0, lLN, np.nan))),
                "phi_LL": float(np.nanmedian(np.where(n > 0, pLL, np.nan))),
                "phi_LN": float(np.nanmedian(np.where(n > 0, pLN, np.nan))),
                "phi_median": float(np.nanmedian(np.where(n > 0, phi, np.nan))),
                "phi_gate": float(np.quantile(phi, 1.0 - frac_star)),
                "phi_max": float(np.max(phi)) if phi.size else 0.0,
                "frac_over": float(np.mean(phi >= PHI_STAR)),
            }
        out.append(rec)
    return [float(d) for d in doses], out


def detect(traj, phi_star=PHI_STAR, hold=HOLD):
    """``{family: d_coarsen or None}`` -- first sustained crossing of ``phi_gate``.

    ``hold`` requires the criterion to be satisfied on that many CONSECUTIVE
    snapshots. On a snapshot grid this is a weak test -- the real detector runs
    per immobile substep, where consecutive samples are close enough for the hold
    to mean something. It is applied here so the two agree in form.

    The crossing dose is interpolated in ``log(dose)``, because the fields move
    by decades across a decade of dose and a linear interpolant would sit at the
    upper endpoint for the whole interval -- the same reason the movie frames
    blend geometrically.
    """
    result = {}
    for fam in FAMILIES:
        k = fam["key"]
        gate = [rec["families"][k]["phi_gate"] for rec in traj]
        over = [g >= phi_star for g in gate]
        hit = None
        for i in range(len(over)):
            if len(over[i:i + hold]) == hold and all(over[i:i + hold]):
                hit = i
                break
        if hit is None:
            result[k] = None
            continue
        d1, g1 = traj[hit]["dose"], gate[hit]
        if hit == 0:
            result[k] = float(d1)
            continue
        d0, g0 = traj[hit - 1]["dose"], gate[hit - 1]
        if d0 > 0.0 and g1 > g0:
            t = (phi_star - g0) / (g1 - g0)
            t = min(max(t, 0.0), 1.0)
            result[k] = float(np.exp(np.log(d0) + t * (np.log(d1) - np.log(d0))))
        else:
            result[k] = float(d1)
    return result


def report(run_dir, phi_star=PHI_STAR, frac_star=FRAC_STAR, hold=HOLD,
           material_file=None):
    run_dir = Path(run_dir)
    doses, traj = trajectory(run_dir, material_file, frac_star)
    kLL, kLN, rhoN = material_params(material_file)
    d_c = detect(traj, phi_star, hold)

    L = [f"# Coarsening detector -- {run_dir.name}", "",
         f"- `kappaLL` = {kLL:g}, `kappaLN` = {kLN:g}, "
         f"`rhoNetwork` = {rhoN:.4g} b^-2 "
         f"(spacing {1.0 / np.sqrt(rhoN) * B_SI * 1e9:.1f} nm)",
         f"- threshold: phi >= {phi_star} on >= {frac_star:.0%} of interior "
         f"nodes, held {hold} snapshots",
         "- radius from the CD-internal `b_cd`, not the discrete `b_dd`", "",
         "## Trajectory (interior median, `frac` = share of interior nodes "
         "over threshold)", "",
         "| dose | fam | r (nm) | 2r/d | Lam_LL | phi_LL | Lam_LN | phi_LN | "
         "phi | phi_gate | frac |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rec in traj:
        for fam in FAMILIES:
            f = rec["families"][fam["key"]]
            if not np.isfinite(f["r_nm"]) or f["r_nm"] <= 0.0:
                continue
            L.append(
                f"| {rec['dose']:.4g} | {fam['label']} | {f['r_nm']:.2f} | "
                f"{f['two_r_over_d']:.3f} | {f['lam_LL']:.4f} | "
                f"{f['phi_LL']:.4f} | {f['lam_LN']:.4f} | {f['phi_LN']:.4f} | "
                f"{f['phi_median']:.4f} | {f['phi_gate']:.4f} | "
                f"{f['frac_over']:.3f} |")
    L += ["", "## `d_coarsen`", ""]
    for fam in FAMILIES:
        v = d_c[fam["key"]]
        L.append(f"- {fam['label']}: "
                 + (f"**{v:.4g} dpa**" if v is not None
                    else "not reached on this dose range"))
    fired = [k for k, v in d_c.items() if v is not None]
    L += ["", ("**Switch to the coupled mode at the earliest crossing.**"
               if fired else
               "**No family crosses; the continuum treatment holds throughout "
               "this run.**")]
    if fired:
        first = min(d_c[k] for k in fired)
        L.append(f"Earliest: **{first:.4g} dpa** ({', '.join(sorted(fired))}).")
        L.append("")
        L.append("Note the snapshot grid bounds the resolution: the crossing is "
                 "interpolated between two snapshots that may be a decade of "
                 "dose apart. The in-march detector runs per immobile substep.")

    # How close phi* sits to the plateau decides whether d_coarsen is a robust
    # number or a coin toss, so report the sweep rather than a single value.
    # phi saturates here -- growth and coarsening balance -- so a threshold set
    # near the plateau is either never reached or reached at a dose that moves
    # wildly for a small change in phi*.
    L += ["", "## Sensitivity to `phi*`", "",
          "| `phi*` | " + " | ".join(f["label"] for f in FAMILIES) + " |",
          "|---:|" + "---:|" * len(FAMILIES)]
    for ps in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = detect(traj, ps, hold)
        cells = [(f"{d[f['key']]:.3g}" if d[f["key"]] is not None else "--")
                 for f in FAMILIES]
        L.append(f"| {ps:.2f} | " + " | ".join(cells) + " |")
    plateau = {f["label"]: max(r["families"][f["key"]]["phi_gate"] for r in traj)
               for f in FAMILIES}
    L += ["", "Highest `phi_gate` reached: "
          + ", ".join(f"{k} {v:.3f}" for k, v in plateau.items()) + ".",
          "",
          "A `phi*` within ~0.05 of a family's plateau makes its `d_coarsen` "
          "unstable or unreachable. Check the default 0.20 against the numbers "
          "above before trusting a switch dose."]
    return "\n".join(L) + "\n", {"doses": doses, "trajectory": traj,
                                 "d_coarsen": d_c,
                                 "phi_star": phi_star, "frac_star": frac_star,
                                 "hold": hold}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--phi", type=float, default=PHI_STAR)
    ap.add_argument("--frac", type=float, default=FRAC_STAR)
    ap.add_argument("--hold", type=int, default=HOLD)
    ap.add_argument("--material", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args(argv)

    text, data = report(args.run_dir, args.phi, args.frac, args.hold,
                        args.material)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
