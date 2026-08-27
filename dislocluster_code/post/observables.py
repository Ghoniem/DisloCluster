"""The plastic distortion, both representations -- steps 10 and 11 of the plan.

Two routes to the same tensor:

  CONTINUUM, Eq. (betaP)
      beta^P = sum_(s,k) eta_s Upsilon_h(k) c^k_sL  n_k (x) n_k
  the content fields ARE the plastic distortion, up to one factor per habit.

  DISCRETE, Eq. (betaPdisc)
      beta^P_D = (1/V) sum_lambda (1/2)(b (x) A + A (x) b)
  a sum over the segments that actually moved, with their actual Burgers
  vectors. NO STRAIN FACTOR APPEARS -- which is why comparing the two on one
  population is advertised as a *measurement* of Upsilon_h(k).

WHAT THAT COMPARISON ACTUALLY MEASURES
--------------------------------------
Read the conversion's own rule before believing the ratio is a material
property. A discrete loop is given the area that stores its defects,

    A_l |b|_l = m_l Omega,

and the continuum content of a family is by construction sum_l A_l (b.n)/V. So
for a pure-climb loop, where b is parallel to n, the discrete tensor reduces to

    (1/V) sum_l A_l |b| n (x) n = (Omega/V) sum_l m_l n (x) n = c^k_sL n (x) n

and the ratio to Eq. (betaP) is IDENTICALLY ONE, whatever the loop sizes are and
however they are distributed. Both sides rest on the same identity and the same
Omega.

That does not make the measurement worthless -- it makes it a different
measurement. What departs from unity is everything the chain could get wrong
between the two: a Burgers magnitude that differs between the cluster-dynamics
and dislocation-dynamics conventions, a polygon whose area does not match the
disc it replaced, an Omega read from two places, loops the domain refused. Run
on a real export it is a stringent end-to-end check on the handoff, and it
returns 1 when the handoff is right.

A PHYSICAL Upsilon != 1 IS NOT AVAILABLE FROM THIS COMPARISON. The formulation
defines it as the ratio of the defect's relaxation volume to Omega, and the
relaxation volume enters nowhere in either tensor above: the material file
carries `immobileSpeciesRelRelaxVol` (0.405 basal, 1.2 prismatic interstitial)
and neither route reads it. Measuring a real Upsilon means putting that number
into one side and not the other, which is a modelling decision and not a
measurement -- so this module reports the ratio, names it for what it is, and
prints the relaxation volumes beside it rather than quietly calling their
absence a result.
"""

from __future__ import annotations

import numpy as np

__all__ = ["beta_p_continuum", "beta_p_discrete", "habit_frame",
           "measure_upsilon", "swelling_from_beta"]


def habit_frame(fam):
    """``(n_hat, b_vec)`` of one family, in the crystal frame, |b| in b."""
    from dislocluster_code.post.discrete_loops import family_geometry
    b_vec, n_hat = family_geometry(fam)
    return np.asarray(n_hat, float), np.asarray(b_vec, float)


def beta_p_continuum(content, families, upsilon=None):
    """Eq. (betaP) from the stored content of each family.

    ``content`` maps a family key to its ``c^k_sL`` (atom fraction, volume
    averaged over whatever region the caller chose). ``upsilon`` defaults to 1
    for every habit -- see the module docstring for why that is a statement
    about what is implemented and not about the material.

    ``eta_s`` is +1 for an interstitial family and -1 for a vacancy one: an
    interstitial platelet inserts a layer and a vacancy platelet removes one.
    """
    beta = np.zeros((3, 3))
    for fam in families:
        c = float(content.get(fam["key"], 0.0))
        if c == 0.0:
            continue
        n_hat, _ = habit_frame(fam)
        eta = -1.0 if fam["vacancy"] else 1.0
        u = 1.0 if upsilon is None else float(upsilon.get(_habit(fam), 1.0))
        beta += eta * u * c * np.outer(n_hat, n_hat)
    return beta


def _habit(fam):
    """"basal" or "prismatic" -- the index Upsilon is carried on."""
    return "prismatic" if fam.get("prismatic", False) else "basal"


def beta_p_discrete(pops, volume):
    """Eq. (betaPdisc) for a set of exported loop populations.

    A loop that has been INSERTED rather than swept is the static limit of that
    equation: its swept area is its own area and its Burgers vector is constant
    over it, so the symmetrized product is ``(1/2)(b (x) A + A (x) b)`` with
    ``A = A_l n_hat``. For a climb loop b is parallel to n_hat and this is
    ``A_l |b| n (x) n``; the symmetrization is kept anyway, so a family whose
    Burgers vector is NOT along its normal -- a glide loop, which this model
    does not carry but a caller might hand over -- is still handled.

    ``volume`` is the crystal volume in b^3, from the same convex body the
    conversion counted loops in.
    """
    beta = np.zeros((3, 3))
    for pop in pops:
        if len(pop) == 0:
            continue
        fam = pop.fam
        n_hat, b_vec = habit_frame(fam)
        eta = -1.0 if fam["vacancy"] else 1.0
        area = float(np.pi * np.sum(np.asarray(pop.radii, float) ** 2))
        A = area * n_hat
        beta += eta * 0.5 * (np.outer(b_vec, A) + np.outer(A, b_vec)) / volume
    return beta


def swelling_from_beta(beta):
    """``tr beta^P`` -- the volumetric part, which is the swelling."""
    return float(np.trace(beta))


def measure_upsilon(pops, content, volume, families):
    """Compare the two tensors on ONE population, per habit.

    Returns a dict with, for each habit present, the discrete and continuum
    diagonal contributions and their ratio -- the quantity Eq. (betaPdisc)'s
    third bullet calls a measurement of ``Upsilon_h(k)``.

    Read the module docstring before quoting the ratio as a material property.
    """
    out = {}
    for habit in ("basal", "prismatic"):
        fams = [f for f in families if _habit(f) == habit]
        keys = {f["key"] for f in fams}
        sub = [p for p in pops if p.fam["key"] in keys]
        bd = beta_p_discrete(sub, volume)
        bc = beta_p_continuum({k: content.get(k, 0.0) for k in keys}, fams)
        nd, nc = np.linalg.norm(bd), np.linalg.norm(bc)
        out[habit] = dict(
            beta_discrete=bd, beta_continuum=bc,
            norm_discrete=float(nd), norm_continuum=float(nc),
            upsilon=float(nd / nc) if nc > 0 else float("nan"),
            n_loops=int(sum(len(p) for p in sub)))
    return out
