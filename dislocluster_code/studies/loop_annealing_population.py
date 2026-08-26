"""loop_annealing_population.py -- annealing of a loop POPULATION.

WHAT THIS IS
------------
Sec. 4.3 of ``Docs/Formulation/self-consistent/self_consistent_SRCD.tex`` anneals
ONE loop, alone, in a crystal pinned at ``c^v = c^{v,eq}_inf``. Its own closing
paragraph lists three limits, and the first two are the same limit twice: the
loop has no neighbors, so nothing re-absorbs what it emits, and the ambient
field is imposed rather than solved. **A real anneal has neither.** When the
source is switched off, thermal vacancies are emitted by some components of the
microstructure and absorbed by others, and the concentration that every loop
actually sees is whatever that exchange settles at.

This module solves for it. The construction is Ghoniem's thesis Eq. (9.18)-(9.20)
(``Docs/Formulation/key_references/Pages from Ghoniem-Thesis.pdf``, pp. 269-272),
carried onto the nine families of Eq. (families) and onto the three prismatic
variants in particular:

    total thermal vacancy emission  =  total vacancy removal              (9.18)

    cbar_v = SUM_k S_k c^{v,eq}_k  /  SUM_k S_k                           (9.20)

with ``S_k`` the family's own vacancy sink strength. **The denominator of (9.20)
is k^2_v of Eq. (KL_matrix), term for term** -- the remote concentration is a
sink-strength-weighted average of the levels the components would each impose on
their own, and the model already assembles those weights.

Three things follow, and they are what Secs. 4.3 and 4.4 cannot say:

  1. every family has its OWN level, so the ladder of the thesis' Fig. (9.16)
     becomes a ladder of loop families;                     -> ``figure_ladder``
  2. with one size per family, which families grow and which shrink is decided
     by where cbar_v falls in that ladder;                  -> ``figure_ladder``
  3. with a size DISTRIBUTION, each family's level becomes a CURVE in R, and
     cbar_v cuts it at a CRITICAL RADIUS -- so a family coarsens internally
     rather than dissolving as a block.                     -> ``figure_ripening``

WHAT DECIDES THE SIGN
---------------------
Each family sits at

    c^{v,eq}_k(R) = c^v_inf exp{ [ varsigma_s(v) mu_k(R) + Sigma_k Omega ] / kT }

with ``mu_k = dE_k/dm`` of Sec. 4.1 and ``varsigma_s(v) = +1`` for a vacancy
family, ``-1`` for an interstitial one. So vacancy families sit ABOVE the
stress-free level, interstitial families BELOW it, and the network sits ON it.
The isolated-loop law of Eq. (isolatedR) then reads, with cbar_v in place of
c^v_inf,

    dR_k/dt = varsigma_s(v) Z_k Dbar_v [ cbar_v - c^{v,eq}_k(R_k) ] / (b.n)_k

so a VACANCY family grows when its own level is below cbar_v and shrinks when it
is above, and an INTERSTITIAL family does the opposite. Since cbar_v is a
weighted average of the levels it always lies between the extreme ones, so in a
mixed microstructure something is always growing while something else shrinks --
which is the whole content of Fig. (9.16).

THE FAULT TERM REMOVES THE COARSENING BRANCH
--------------------------------------------
``mu_k`` has a fault part that does NOT vanish with size and a capillary part
that does. For an UNFAULTED family c^{v,eq}_k(R) decays to c^v_inf from above;
for a FAULTED one it decays to a FLOOR, c^v_inf exp[gamma_k Omega/((b.n)_k kT)],
which is 2.53 c^v_inf for c_f at 873 K. Whenever cbar_v lies below a family's
asymptote **no loop of that family is ever large enough to grow** and the family
dissolves at every size at once; :func:`critical_radius` returns ``None`` in
exactly that case.

**A family ON ITS OWN always ripens**, faulted or not: cbar_v is then its own
sink-weighted mean level, so it necessarily lies inside the family's own spread
and R* falls inside the distribution. What the fault energy changes is the
COMPETITION -- it lifts the whole family's ladder rung, so that a network and an
interstitial population holding cbar_v near c^v_inf can put the entire faulted
family above the line at once. That is the difference between ripening and
wholesale dissolution, and it is a property of the mixture, not of the family.

TWO CAPTURE EFFICIENCIES, AND WHY THE ANSWER IS "THE SAME ONE TWICE"
--------------------------------------------------------------------
``S_k = Z_k rho_k`` needs a Z, and the model carries two that are not the same
number:

  ``weight="woo"``  Z_{sk,v} of Eq. (Zsk): Z0_v A_h(p_v), the mean-field
                    efficiency Eq. (KL_matrix) assembles k^2_v from, so that the
                    denominator of (9.20) is literally the model's own sink
                    strength.
  ``weight="iso"``  Z^iso_k = 2 pi/ln(8R/r_0) of Eq. (Ziso), the isolated-loop
                    efficiency Sec. 4.3's own kinetics use. **The DEFAULT**,
                    because it makes this calculation reduce to Sec. 4.3's
                    EXACTLY in the single-component limit -- which is the
                    comparison the whole subsection is about.

Either may be chosen. **What may NOT be done is to choose differently in the two
places.** Eq. (9.20) makes ``SUM_k S_k (c^{v,eq}_k - cbar_v) = 0``, and that sum
is the net vacancy exchange with the matrix only if the S_k that weight it are
the S_k that drive the growth law. Mixing them broke conservation of stored
vacancies by a factor of 85 in :func:`verify` before it was caught -- a ripening
run that quietly destroyed vacancies and otherwise looked entirely reasonable.
So a single ``weight`` reaches both sides everywhere in this module.

The two differ in their RATIO between habits, which is the only thing cbar_v
sees: on the 300 nm reference microstructure they put the <a>/<c> weight ratio a
factor of ~1.7 apart and cbar_v up to 27% apart. :func:`report_reference` prints
it both ways rather than picking one.

USAGE
-----
    python -m dislocluster_code.studies.loop_annealing_population verify
    python -m dislocluster_code.studies.loop_annealing_population ladder   [--T 873]
    python -m dislocluster_code.studies.loop_annealing_population variants [--sigma 200]
    python -m dislocluster_code.studies.loop_annealing_population reference [--run <dir>]
    python -m dislocluster_code.studies.loop_annealing_population figure-ladder    --out <png>
    python -m dislocluster_code.studies.loop_annealing_population figure-ripening  --out <png>
    python -m dislocluster_code.studies.loop_annealing_population figure-reference --out <png>
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling.field import read_material_vector
from dislocluster_code.studies import loop_annealing as la
from dislocluster_code.studies import loop_annealing_greens as gr
from dislocluster_code.studies.loop_annealing import (EV_J, KB_EV, Family,
                                                      Material, families,
                                                      load_material)

# The 300 nm reference case: the hardening cube of
# `Simulations/output/<stamp>_hardening_300nmHex`, whose loop population is drawn
# from the matched self-consistent march. Its microstructure is what Sec. 6 of
# this module anneals.
REFERENCE_RUN = "20260817_075527_563ab7d_500nmHex_Matched_SelfConsistent"
REFERENCE_CUBE_NM = 300.0


# ── the microstructure: one entry per component that emits or absorbs ────────

@dataclass
class Component:
    """One emitting/absorbing component of the microstructure.

    ``fam`` supplies the crystallography and the energetics (Sec. 4.1); ``n`` and
    ``R`` the population. ``Sigma`` is the resolved stress of
    ``loop_annealing.sigma_resolved`` -- it is what separates the three prismatic
    variants under load and is zero for all of them without it.
    """
    name: str
    fam: Family
    n: float                  # number density [m^-3]
    R: float                  # mean radius [m]
    Sigma: float = 0.0        # resolved stress on the habit plane [Pa]
    label: str = ""

    @property
    def rho(self):
        """Line density 2 pi R n [m^-2] -- the model's own rho^k_sL."""
        return 2.0 * math.pi * self.R * self.n

    def c_eq(self, T, mat: Material):
        """This component's own equilibrium vacancy concentration."""
        return gr.c_line_local(1.0 / self.R, self.fam, T, mat, self.Sigma)


@dataclass
class Network:
    """The straight-dislocation network: a sink at the stress-free level.

    A straight edge dislocation has no capillarity and no fault, so ``mu = 0``
    and its level is ``c^v_inf exp(Sigma Omega/kT)`` -- exactly ``c^v_inf`` at
    zero stress. It is the rung Fig. (9.16) draws ``c_v^e`` on, and it is what
    pins cbar_v when the loop population is thin.
    """
    rho: float                                   # [m^-2]
    Z: float = 1.0                               # Z_{N,v}; k^2_v = rho_N
    Sigma: float = 0.0
    name: str = "network"
    label: str = "network dislocations"

    def c_eq(self, T, mat: Material):
        omega = mat.omega
        return mat.c_v_inf(T) * math.exp(self.Sigma * omega / (KB_EV * T * EV_J))


# ── the two capture efficiencies ─────────────────────────────────────────────

def p_anisotropy(T, material_file=None, species=0):
    """``p_m(T) = (D_c/D_a)^{1/6}``, from the tensor rather than from the file's
    ``dadAnisotropy``, which is pinned at the fitted temperature."""
    D = gr.diffusion_tensor(T, material_file, species)
    return float((D[2, 2] / D[0, 0]) ** (1.0 / 6.0))


def Z_woo(fam: Family, T, material_file=None):
    """``Z_{sk,v}`` of Eq. (Zsk) at zero stress: ``Z0_v A_h(p_v)``.

    ``A_c(p) = p`` for a basal habit and ``A_a(p) = (p + p^-2)/2`` for a
    prismatic one, Eq. (Afactors). ``Z0_v`` is ``dadZ0[0]`` from the material
    file. This is the efficiency Eq. (KL_matrix) builds ``k^2_v`` from, so using
    it here makes the denominator of Eq. (9.20) the model's own sink strength.
    """
    mf = material_file or paths.MODELIB_MATERIAL
    Z0 = float(np.asarray(read_material_vector(mf, "dadZ0"), float)[0])
    p = p_anisotropy(T, mf)
    A = p if fam.name.startswith("c") else 0.5 * (p + p ** -2)
    return Z0 * A


def Z_of(comp, T, weight="iso", material_file=None, R=None):
    """The capture efficiency of one component (or of one size bin of one).

    ``weight="iso"`` is size-INDEPENDENT and ``weight="iso"`` is not, which is
    why ``R`` is taken explicitly rather than from the component: a
    :class:`DistributionFamily` has one Z per bin.
    """
    if isinstance(comp, Network):
        return comp.Z
    if weight == "woo":
        return Z_woo(comp.fam, T, material_file)
    if weight == "iso":
        return la.Z_isolated(comp.R if R is None else R, comp.fam)
    raise ValueError("weight must be 'woo' or 'iso'")


def sink_strengths(comps, T, weight="iso", material_file=None):
    """``S_k = Z_k rho_k`` for every component, in m^-2.

    For a family carrying a distribution this is ``sum_j Z_j 2 pi R_j n_j``, bin
    by bin, so the same Z reaches the balance and the kinetics for every bin.

    **WHICH Z, AND WHY THE ANSWER IS "THE SAME ONE TWICE".** The formulation
    carries two efficiencies -- the mean-field ``Z_{sk,v}`` of Eq. (Zsk), from
    which Eq. (KL_matrix) assembles ``k^2_v``, and the isolated-loop
    ``Z^iso_k = 2 pi/ln(8R/r_0)`` of Eq. (Ziso), which Sec. 4.3's own kinetics
    use -- and either may be chosen. What may NOT be done is to choose
    differently in the two places. Eq. (9.20) makes
    ``SUM_k S_k (c^{v,eq}_k - cbar_v) = 0``, and that sum is the net vacancy
    exchange with the matrix ONLY if the S_k that weight it are the same S_k that
    drive Eq. (popR). Mixing them broke conservation of stored vacancies by a
    factor of 85 in :func:`verify` before this was fixed -- a ripening run that
    quietly destroyed vacancies and otherwise looked entirely reasonable.

    ``"iso"`` is the default, because it makes the population calculation reduce
    to Sec. 4.3's EXACTLY in the single-component limit, which is the comparison
    the whole subsection is about. ``"woo"`` makes the denominator of Eq. (9.20)
    literally ``k^2_v``; the two put cbar_v up to 27% apart on the reference
    microstructure and both are reported.
    """
    out = []
    for c in comps:
        if isinstance(c, DistributionFamily):
            Z = Z_of(c, T, weight, material_file, R=c.R)
            out.append(float((np.asarray(Z) * 2.0 * math.pi * c.R * c.n).sum()))
        else:
            out.append(float(Z_of(c, T, weight, material_file)) * c.rho)
    return np.array(out, float)


# ── Eq. (9.18)-(9.20): the remote thermal vacancy concentration ──────────────

def remote_concentration(comps, T, mat: Material = None, weight="iso",
                         material_file=None, return_parts=False):
    """``cbar_v``, the matrix vacancy concentration a mixed microstructure holds.

    Thesis Eq. (9.18): with the source off and the interstitial thermal
    population negligible (``E^f_i = 3.0`` eV against ``E^f_v = 1.6`` eV, so
    ``c^i_eq/c^v_eq = e^{-1.4/kT} = 1.0e-8`` at 873 K), mutual recombination
    drops out and the vacancy balance is emission = absorption. Component ``k``
    emits ``S_k Dbar_v c^{v,eq}_k/Omega`` and absorbs ``S_k Dbar_v cbar_v/Omega``,
    so

        SUM_k S_k c^{v,eq}_k  =  cbar_v SUM_k S_k

    which is Eq. (9.20). ``Dbar_v`` and ``Omega`` cancel: **cbar_v is a property
    of the microstructure and the temperature alone, not of the mobility.**
    """
    mat = mat or load_material(material_file)
    S = sink_strengths(comps, T, weight, material_file)
    ceq = np.array([c.c_eq(T, mat, weight, material_file)
                    if isinstance(c, DistributionFamily) else c.c_eq(T, mat)
                    for c in comps], float)
    tot = float(S.sum())
    cbar = float((S * ceq).sum() / tot) if tot > 0 else mat.c_v_inf(T)
    if return_parts:
        return cbar, S, ceq
    return cbar


def critical_radius(fam: Family, cbar, T, mat: Material = None, Sigma=0.0,
                    R_lo=None, R_hi=1e-4):
    """``R*``: the size at which a family of this kind neither grows nor shrinks.

    Solves ``c^{v,eq}_k(R*) = cbar_v``. Returns ``None`` when no root exists in
    ``[R_lo, R_hi]``, which is not a numerical failure but the physical statement
    that **every loop of that family moves the same way at every size** -- the
    case a fault energy forces, because it puts a floor under the level that the
    capillary term cannot bring down to cbar_v.

    The search runs over the model's OWN validity range: from ``R_min`` (the
    material file's ``r_min``, below which Sec. 4.3 says the continuum loop
    energy is at the edge of its validity, and below which a family does not
    exist) up to 100 um. Going lower would be meaningless twice over -- the
    capillary logarithm in ``dEdm_local`` saturates at ``R ~ |b|/2.94``, three
    decades below ``R_min``, so the level is flat there by construction.
    """
    mat = mat or load_material()
    R_lo = mat.r_min if R_lo is None else R_lo

    def f(R):
        return math.log(gr.c_line_local(1.0 / R, fam, T, mat, Sigma)
                        / max(cbar, 1e-300))

    a, b = R_lo, R_hi
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        return None
    for _ in range(200):
        m = math.sqrt(a * b)                      # bisection in log R
        fm = f(m)
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return math.sqrt(a * b)


def dRdt(comp: Component, cbar, T, mat: Material = None, weight="iso",
         material_file=None):
    """Eq. (isolatedR) with the SOLVED remote concentration in place of c^v_inf.

    This is the one substitution that turns Sec. 4.3's calculation into a
    population one. Everything else -- the capillarity, the fault term, the
    stress term, the sign convention -- is unchanged. ``weight`` must be the one
    :func:`sink_strengths` used, or the two sides of the balance disagree; see
    that function.
    """
    mat = mat or load_material()
    Z = float(Z_of(comp, T, weight, material_file))
    return (comp.fam.zeta_v * Z * mat.Dbar_v(T)
            * (cbar - comp.c_eq(T, mat)) / comp.fam.bdotn)


# ── the coupled population march ─────────────────────────────────────────────

def anneal_population(comps, T, mat: Material = None, weight="iso",
                      material_file=None, R_min=None, t_end=None,
                      n_steps=200000, cfl=2e-3):
    """March every component together, recomputing cbar_v at every step.

    Returns ``dict(t, R, n, cbar)`` with ``R`` and ``n`` of shape
    ``(n_out, n_comp)``. A component that reaches ``R_min`` is removed from the
    population -- its ``n`` drops to zero and it stops contributing to
    Eq. (9.20), which is what makes the late stage of the anneal accelerate.
    """
    mat = mat or load_material(material_file)
    R_min = mat.r_min if R_min is None else R_min
    comps = [Component(c.name, c.fam, c.n, c.R, c.Sigma, c.label)
             if isinstance(c, Component) else c for c in comps]
    live = [isinstance(c, Component) for c in comps]

    ts, Rs, ns, cs = [], [], [], []

    def snapshot(t, cbar):
        ts.append(t)
        Rs.append([c.R if isinstance(c, Component) else np.nan for c in comps])
        ns.append([c.n if isinstance(c, Component) else np.nan for c in comps])
        cs.append(cbar)

    t = 0.0
    cbar = remote_concentration(comps, T, mat, weight, material_file)
    snapshot(t, cbar)
    for _ in range(int(n_steps)):
        active = [i for i, c in enumerate(comps)
                  if isinstance(c, Component) and c.n > 0.0]
        if not active:
            break
        cbar = remote_concentration(comps, T, mat, weight, material_file)
        rates, ok = {}, False
        for i in active:
            rates[i] = dRdt(comps[i], cbar, T, mat, weight, material_file)
            ok = ok or abs(rates[i]) > 0.0
        if not ok:
            break
        dt = cfl * min(comps[i].R / max(abs(rates[i]), 1e-300) for i in active)
        if t_end is not None:
            dt = min(dt, t_end - t)
        for i in active:
            comps[i].R += dt * rates[i]
            if comps[i].R <= R_min:
                comps[i].R = R_min
                comps[i].n = 0.0                  # the family has annealed out
        t += dt
        snapshot(t, cbar)
        if t_end is not None and t >= t_end:
            break
    return dict(t=np.array(ts), R=np.array(Rs), n=np.array(ns),
                cbar=np.array(cs), comps=comps)


# ── size distributions, and the ripening they produce ────────────────────────

def lognormal_bins(N_total, R_mean, sigma_ln=0.45, n_bins=41, span=3.2):
    """A log-normal size distribution with the given total density and MEAN R.

    The model itself carries a mean size per family (Sec. 2, and Sec. 2's
    ``q^k_sL`` only widens it), so the width here is an ASSUMPTION and is the one
    number in the ripening figure that is not measured. ``sigma_ln = 0.45`` is
    the middle of what TEM size histograms of irradiated Zr report. The mean of
    the returned distribution is ``R_mean`` to machine precision, so changing
    ``sigma_ln`` re-partitions a fixed population rather than changing it.
    """
    mu = math.log(R_mean) - 0.5 * sigma_ln ** 2          # so that <R> = R_mean
    R = np.exp(np.linspace(mu - span * sigma_ln, mu + span * sigma_ln, n_bins))
    w = np.exp(-0.5 * ((np.log(R) - mu) / sigma_ln) ** 2) / R
    w /= w.sum()
    R_bar = float((w * R).sum())
    R *= R_mean / R_bar                                  # exact mean, by scaling
    return R, w * N_total


@dataclass
class DistributionFamily:
    """One family carrying a discretized size distribution instead of one size."""
    name: str
    fam: Family
    R: np.ndarray             # bin radii [m]
    n: np.ndarray             # bin number densities [m^-3]
    Sigma: float = 0.0
    label: str = ""

    @property
    def rho(self):
        return float(2.0 * math.pi * (self.R * self.n).sum())

    @property
    def N(self):
        return float(self.n.sum())

    @property
    def R_mean(self):
        N = self.N
        return float((self.R * self.n).sum() / N) if N > 0 else 0.0

    def c_eq(self, T, mat: Material, weight="iso", material_file=None):
        """The distribution's sink-weighted mean level.

        **The weights are ``Z_j 2 pi R_j n_j``, the bins' own sink strengths, and
        that is not a refinement -- it is what makes the balance close.**
        Eq. (9.20) weights every EMITTER by its sink strength, so a family
        carrying a distribution contributes ``sum_j S_j c^{v,eq}_j / sum_j S_j``.
        Pairing that with the ``S_k`` :func:`sink_strengths` returns for the same
        family reproduces the bin-by-bin sum exactly, because
        ``S_fam * c_eq_fam = sum_j S_j c^{v,eq}_j`` identically.

        Weighting by ``R_j n_j`` instead -- dropping the per-bin ``Z_j`` -- is a
        few per cent in ``cbar_v`` and looks harmless. It is not: the stored
        content evolves as ``sum_j S_j (cbar_v - c^{v,eq}_j)``, a near-cancellation
        of terms spanning decades, so a few per cent in the weighting became a
        **factor of 85** in the conservation check of :func:`verify`. That is the
        whole reason this method takes a ``weight``.

        Note also that this is NOT the level of the mean size: the level is
        exponential in 1/R, so the two differ by 1.8x on a sigma_ln = 0.5
        distribution.
        """
        if self.N <= 0:
            return mat.c_v_inf(T)
        Z = np.broadcast_to(np.asarray(
            Z_of(self, T, weight, material_file, R=self.R), float), self.R.shape)
        w = Z * self.R * self.n
        tot = float(w.sum())
        if tot <= 0:
            return mat.c_v_inf(T)
        lev = gr.c_line_local(1.0 / self.R, self.fam, T, mat, self.Sigma)
        return float((w * lev).sum() / tot)


def anneal_distribution(dfams, T, mat: Material = None, weight="iso",
                        material_file=None, R_min=None, t_end=None,
                        n_steps=200000, cfl=2e-3, others=(), store_every=200,
                        stop_frac=0.02):
    """March size distributions, recomputing cbar_v from every bin at every step.

    Bins drift by Eq. (isolatedR) evaluated at their own radius, so a bin below
    the critical radius shrinks while one above it grows -- within a single
    family. Bins that reach ``R_min`` are removed, and their loops leave the
    population. ``others`` are fixed components (e.g. the network) that
    contribute to cbar_v but do not evolve.

    ``store_every`` thins the stored snapshots (the march itself is not thinned);
    ``stop_frac`` ends it once that fraction of the initial loop NUMBER is left,
    which is where a ripening run would otherwise crawl on a single surviving bin.
    """
    mat = mat or load_material(material_file)
    R_min = mat.r_min if R_min is None else R_min
    dfams = [DistributionFamily(d.name, d.fam, d.R.copy(), d.n.copy(), d.Sigma,
                                d.label) for d in dfams]
    parts = list(dfams) + list(others)
    ts, snaps, cs = [0.0], [[(d.R.copy(), d.n.copy()) for d in dfams]], []
    cbar = remote_concentration(parts, T, mat, weight, material_file)
    cs.append(cbar)
    t = 0.0
    N0 = sum(float(d.n.sum()) for d in dfams)
    for step in range(int(n_steps)):
        cbar = remote_concentration(parts, T, mat, weight, material_file)
        rates, scale = [], []
        for d in dfams:
            Z = np.broadcast_to(np.asarray(
                Z_of(d, T, weight, material_file, R=d.R), float), d.R.shape)
            lev = gr.c_line_local(1.0 / d.R, d.fam, T, mat, d.Sigma)
            v = (d.fam.zeta_v * Z * mat.Dbar_v(T) * (cbar - lev) / d.fam.bdotn)
            v = np.where(d.n > 0, v, 0.0)
            rates.append(v)
            live = d.n > 0
            if live.any():
                scale.append(float(np.min(d.R[live]
                                          / np.maximum(np.abs(v[live]), 1e-300))))
        if not scale:
            break
        dt = cfl * min(scale)
        if t_end is not None:
            dt = min(dt, t_end - t)
        for d, v in zip(dfams, rates):
            d.R = d.R + dt * v
            gone = (d.R <= R_min) & (d.n > 0)
            if gone.any():
                d.R[gone] = R_min
                d.n[gone] = 0.0
        t += dt
        N = sum(float(d.n.sum()) for d in dfams)
        done = (t_end is not None and t >= t_end) or N <= stop_frac * N0
        if step % store_every == 0 or done:
            ts.append(t)
            snaps.append([(d.R.copy(), d.n.copy()) for d in dfams])
            cs.append(cbar)
        if done:
            break
    return dict(t=np.array(ts), snaps=snaps, cbar=np.array(cs), dfams=dfams)


# ── the 300 nm reference microstructure ──────────────────────────────────────

def reference_state(run_dir=None, dose=0.1, frac=0.5):
    """The interior per-family state of the reference march, at one dose.

    The 300 nm case is the hardening cube
    ``Simulations/output/<stamp>_hardening_300nmHex``; its loop population is
    drawn from the matched self-consistent march by ``hardening.cube_population``,
    so the microstructure to anneal is that march's INTERIOR state -- the same
    reduction, and the same interior-only rule, that built the cube.
    """
    from dislocluster_code.studies import hardening
    if run_dir is None:
        for r in paths.find_runs():
            if r.name == REFERENCE_RUN:
                run_dir = r
                break
        else:
            raise FileNotFoundError(
                f"reference march {REFERENCE_RUN} not found under "
                f"{[str(p) for p in paths.OUTPUT_DIRS]}")
    return hardening.interior_state(run_dir, dose=dose, frac=frac)


def components_from_state(state, mat: Material = None, c_state="c_f",
                          rho_N=None, material_file=None, stress=None):
    """Turn an interior state into the component list Eq. (9.20) sums over.

    The march's <c> family has ``b.n = c/2``, which is ``c_f``'s geometry (and
    ``lambda_c = 0.170 nm``), so ``c_state="c_f"`` maps it onto the faulted basal
    state of Table (cstates) and ``c_state="c_p"`` onto the perfect one.
    **That choice decides whether the family has a coarsening branch at all**,
    because only ``c_f`` carries the size-independent fault term, so both are
    reported rather than one being chosen.

    ``stress`` is the 3x3 Cauchy tensor in the crystal frame; it resolves onto
    each variant's own habit plane through ``loop_annealing.sigma_resolved`` and
    is what splits ``a_1``, ``a_2``, ``a_3``.
    """
    mat = mat or load_material(material_file)
    fams = families(mat)
    out = []
    for f in state["families"]:
        key = f["key"]
        if f["N_m3"] <= 0.0 or f["r_nm"] <= 0.0:
            continue
        fam = fams[c_state] if key == "c" else fams["a_i"]
        Sigma = la.sigma_resolved(fam, stress, mat) if stress is not None else 0.0
        out.append(Component(name=key, fam=fam, n=f["N_m3"],
                             R=f["r_nm"] * 1e-9, Sigma=Sigma,
                             label=f.get("label", key)))
    if rho_N is None:
        mf = material_file or paths.MODELIB_MATERIAL
        rho_N = float(np.asarray(read_material_vector(mf, "otherSinks_SI"),
                                 float)[0])
    out.append(Network(rho=rho_N))
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def _level_table(comps, T, mat, weight, material_file, cbar=None):
    cbar_, S, ceq = remote_concentration(comps, T, mat, weight, material_file,
                                         return_parts=True)
    cbar = cbar_ if cbar is None else cbar
    cinf = mat.c_v_inf(T)
    rows = []
    for c, s, ce in zip(comps, S, ceq):
        if isinstance(c, Network):
            rows.append((c.name, np.nan, c.rho, s, s / S.sum(), ce / cinf,
                         np.nan, "-"))
            continue
        rate = dRdt(c, cbar, T, mat, weight, material_file)
        rows.append((c.name, c.R, c.rho, s, s / S.sum(), ce / cinf, rate,
                     "grow" if rate > 0 else "shrink"))
    return cbar, rows


def report_ladder(T=873.0, run_dir=None, dose=0.1, c_state="c_f",
                  weight="iso", material_file=None, stress=None):
    """The ladder of Fig. (9.16), as numbers, for the reference microstructure."""
    mat = load_material(material_file)
    state = reference_state(run_dir, dose)
    comps = components_from_state(state, mat, c_state, material_file=material_file,
                                  stress=stress)
    cbar, rows = _level_table(comps, T, mat, weight, material_file)
    cinf = mat.c_v_inf(T)
    print(f"# Remote thermal vacancy concentration, Eq. (9.20), at T = {T:g} K")
    print(f"#   microstructure: {state['run']}")
    print(f"#   dose = {state['dose']:g} dpa, interior, <c> mapped onto "
          f"'{c_state}', weights = '{weight}'")
    print(f"#   c^v_inf = {cinf:.6e}     cbar_v = {cbar:.6e}     "
          f"cbar_v/c^v_inf = {cbar/cinf:.6f}")
    print(f"# {'component':<10}{'R [nm]':>9}{'rho [m^-2]':>12}{'S [m^-2]':>12}"
          f"{'weight':>9}{'c_eq/c_inf':>12}{'dR/dt [m/s]':>14}{'':>8}")
    for name, R, rho, s, w, lev, rate, verdict in rows:
        Rs = f"{R*1e9:>9.2f}" if np.isfinite(R) else f"{'-':>9}"
        rs = f"{rate:>14.4e}" if np.isfinite(rate) else f"{'-':>14}"
        print(f"  {name:<10}{Rs}{rho:>12.4e}{s:>12.4e}{w:>9.4f}{lev:>12.6f}"
              f"{rs}{verdict:>8}")
    # the critical radius of every loop family present
    print(f"# critical radius R* where c_eq(R*) = cbar_v "
          f"(None = one sign at every size)")
    for c in comps:
        if isinstance(c, Network):
            continue
        Rc = critical_radius(c.fam, cbar, T, mat, c.Sigma)
        s = f"{Rc*1e9:.3f} nm" if Rc else "None"
        print(f"  {c.name:<10}{c.fam.name:<8}{s:>14}"
              f"   (present at R = {c.R*1e9:.2f} nm)")
    return cbar, comps, state


def report_variants(T=873.0, sigma_MPa=200.0, run_dir=None, dose=0.1,
                    c_state="c_f", weight="iso", material_file=None):
    """What an applied stress does to the THREE prismatic variants.

    At zero deviatoric stress the three are equivalent by symmetry (Sec. 2), so
    they share one level and one fate. A stress resolves differently onto each
    habit normal, ``Sigma_k = a_k . sigma . a_k``, which shifts each level by
    ``exp(Sigma_k Omega/kT)`` -- and because cbar_v is a single number, **the
    variants can then end up on opposite sides of it.**
    """
    mat = load_material(material_file)
    fams = families(mat)
    state = reference_state(run_dir, dose)
    # uniaxial tension along [10-10], i.e. x: a_1's habit normal is x, so a_1
    # feels the full sigma and a_2, a_3 feel sigma cos^2(60 deg) = sigma/4.
    s = np.zeros((3, 3))
    s[0, 0] = sigma_MPa * 1e6
    normals = {"a1": np.array([1.0, 0.0, 0.0]),
               "a2": np.array([-0.5, math.sqrt(3) / 2, 0.0]),
               "a3": np.array([-0.5, -math.sqrt(3) / 2, 0.0])}
    base = components_from_state(state, mat, c_state,
                                 material_file=material_file)
    comps = []
    for c in base:
        if isinstance(c, Network) or c.name == "c":
            comps.append(c)
            continue
        nvec = normals[c.name]
        comps.append(Component(c.name, c.fam, c.n, c.R,
                               float(nvec @ s @ nvec), c.label))
    cbar, rows = _level_table(comps, T, mat, weight, material_file)
    cinf = mat.c_v_inf(T)
    print(f"# The three prismatic variants under sigma_11 = {sigma_MPa:g} MPa "
          f"along [10-10], T = {T:g} K")
    print(f"#   dose = {state['dose']:g} dpa, <c> as '{c_state}'.  "
          f"FULL mixture: cbar_v/c^v_inf = {cbar/cinf:.6f}")
    print(f"# {'variant':<10}{'Sigma [MPa]':>13}{'c_eq/c_inf':>12}"
          f"{'dR/dt [m/s]':>14}{'':>9}")
    for c, (name, R, rho, S, w, lev, rate, verdict) in zip(comps, rows):
        Sg = c.Sigma / 1e6 if not isinstance(c, Network) else 0.0
        rs = f"{rate:>14.4e}" if np.isfinite(rate) else f"{'-':>14}"
        print(f"  {name:<10}{Sg:>13.2f}{lev:>12.6f}{rs}{verdict:>9}")

    # The <a> SUB-BALANCE. Restricting Eq. (9.20) to the prismatic family alone
    # puts cbar_v inside the variants' own spread by construction, which is the
    # only arrangement in which the split can change a SIGN rather than a rate.
    subs = [c for c in comps
            if isinstance(c, Component) and c.name.startswith("a")]
    if subs:
        cb_a, rows_a = _level_table(subs, T, mat, weight, material_file)
        print(f"# The <a> SUB-BALANCE alone (network and <c> removed): "
              f"cbar_v/c^v_inf = {cb_a/cinf:.6f}")
        for c, (name, R, rho, S, w, lev, rate, verdict) in zip(subs, rows_a):
            print(f"  {name:<10}{c.Sigma/1e6:>13.2f}{lev:>12.6f}"
                  f"{rate:>14.4e}{verdict:>9}")
    # What stress WOULD flip a variant's sign in the full mixture
    a0 = next((c for c in comps
               if isinstance(c, Component) and c.name == "a1"), None)
    if a0 is not None:
        lev0 = gr.c_line_local(1.0 / a0.R, a0.fam, T, mat, 0.0)
        sig_star = (KB_EV * T * EV_J * math.log(cbar / lev0)) / mat.omega
        print(f"# sigma that would lift an <a> variant's level to cbar_v in the "
              f"FULL mixture: {sig_star/1e6:.0f} MPa")
        print("#   -- far beyond yield, so on this microstructure the variant "
              "split moves RATES,")
        print("#      not signs; the sub-balance above is what a nearly "
              "<a>-only microstructure would do.")
    return cbar, comps


def report_reference(T=873.0, run_dir=None, doses=(0.01, 0.1, 1.0, 10.0),
                     c_state="c_f", material_file=None):
    """Sec. 4.3's calculation repeated on the 300 nm reference microstructure.

    Two columns per family: the ISOLATED lifetime of Sec. 4.3 (ambient pinned at
    ``c^v_inf``) and the POPULATION lifetime (ambient solved from Eq. 9.20). The
    ratio is what the neighbors are worth.
    """
    mat = load_material(material_file)
    cinf = mat.c_v_inf(T)
    print(f"# Sec. 4.3 repeated on the {REFERENCE_CUBE_NM:g} nm reference "
          f"microstructure, T = {T:g} K, <c> as '{c_state}'")
    print(f"#   c^v_inf = {cinf:.6e}")
    print(f"# {'dpa':>7}{'R_c [nm]':>10}{'R_a [nm]':>10}"
          f"{'cbar/cinf woo':>15}{'cbar/cinf iso':>15}"
          f"{'t_c iso':>12}{'t_c pop':>12}{'t_a iso':>12}{'t_a pop':>12}")
    rows = []
    for d in doses:
        state = reference_state(run_dir, d)
        comps = components_from_state(state, mat, c_state,
                                      material_file=material_file)
        cb = {w: remote_concentration(comps, T, mat, w, material_file)
              for w in ("woo", "iso")}
        loops = [c for c in comps if isinstance(c, Component)]
        res = anneal_population(comps, T, mat, "iso", material_file)
        t_iso, t_pop = {}, {}
        for c in loops:
            try:
                t_iso[c.name] = la.anneal_time(c.R, c.fam, T, mat=mat,
                                               Sigma=c.Sigma)
            except ValueError:
                t_iso[c.name] = float("nan")
            j = [k.name for k in res["comps"]].index(c.name)
            gone = np.flatnonzero(res["n"][:, j] == 0.0)
            t_pop[c.name] = (float(res["t"][gone[0]]) if gone.size
                             else float("inf"))
        Rc = next((c.R for c in loops if c.name == "c"), float("nan"))
        Ra = next((c.R for c in loops if c.name == "a1"), float("nan"))
        rows.append((d, Rc, Ra, cb, t_iso, t_pop))
        def fm(v):
            return la._fmt_time(v) if np.isfinite(v) and v != float("inf") \
                else ("never" if v == float("inf") else "n/a")
        print(f"  {d:>7g}{Rc*1e9:>10.2f}{Ra*1e9:>10.2f}"
              f"{cb['woo']/cinf:>15.6f}{cb['iso']/cinf:>15.6f}"
              f"{fm(t_iso.get('c', float('nan'))):>12}"
              f"{fm(t_pop.get('c', float('nan'))):>12}"
              f"{fm(t_iso.get('a1', float('nan'))):>12}"
              f"{fm(t_pop.get('a1', float('nan'))):>12}")
    return rows


def verify(material_file=None):
    """Checks that can be done by hand."""
    mat = load_material(material_file)
    fams = families(mat)
    T = 873.0
    cinf = mat.c_v_inf(T)
    ok = True

    def chk(what, got, want, tol):
        nonlocal ok
        rel = abs(got - want) / max(abs(want), 1e-300)
        good = rel <= tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {what:<56}"
              f"{got:>13.6g}  vs {want:<12.6g} ({rel:.1e})")

    def flag(what, good, value=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok ' if good else 'FAIL'}] {what:<56}{value:>13}")

    # (1) one component alone: Eq. (9.20) must return that component's own level,
    #     and the population anneal must then be static.
    c = Component("solo", fams["c_p"], 1e21, 20e-9)
    chk("cbar_v of a single component == its own level",
        remote_concentration([c], T, mat), c.c_eq(T, mat), 1e-14)
    flag("dR/dt of a single component is zero",
         abs(dRdt(c, c.c_eq(T, mat), T, mat)) < 1e-30)
    # (2) network alone: the level is c^v_inf exactly
    chk("cbar_v of the network alone == c^v_inf",
        remote_concentration([Network(rho=1e14)], T, mat), cinf, 1e-14)
    # (3) cbar_v is bounded by the extreme levels, always
    comps = [Component("c", fams["c_f"], 7.7e20, 25e-9),
             Component("a", fams["a_i"], 4e21, 3.6e-9),
             Network(rho=1.88e14)]
    cbar = remote_concentration(comps, T, mat)
    lv = [x.c_eq(T, mat) for x in comps]
    flag("cbar_v lies between the extreme levels",
         min(lv) <= cbar <= max(lv), f"{cbar/cinf:.5f}")
    # (4) a vacancy family is ABOVE c^v_inf and an interstitial one BELOW
    flag("vacancy family sits above c^v_inf",
         fams["c_p"].zeta_v > 0 and comps[0].c_eq(T, mat) > cinf)
    flag("interstitial family sits below c^v_inf",
         comps[1].c_eq(T, mat) < cinf)
    # (5) the critical radius really is the zero of the growth law
    Rs = critical_radius(fams["c_p"], cbar, T, mat)
    if Rs:
        probe = Component("probe", fams["c_p"], 1.0, Rs)
        flag("dR/dt = 0 at the critical radius",
             abs(dRdt(probe, cbar, T, mat)) < 1e-16,
             f"{dRdt(probe, cbar, T, mat):.2e}")
        lo = Component("lo", fams["c_p"], 1.0, 0.7 * Rs)
        hi = Component("hi", fams["c_p"], 1.0, 1.4 * Rs)
        flag("below R* a vacancy loop shrinks, above it grows",
             dRdt(lo, cbar, T, mat) < 0 < dRdt(hi, cbar, T, mat))
    # (6) A FAULTED family has a level FLOOR, so above some cbar_v there is no R*
    fam = fams["c_f"]
    floor = cinf * math.exp(fam.gamma * fam.omega / fam.bdotn
                            / (KB_EV * T * EV_J))
    chk("c_f level floor as R -> infinity",
        float(gr.c_line_local(1e-12, fam, T, mat)), floor, 1e-3)
    flag("no critical radius for c_f when cbar_v is below its floor",
         critical_radius(fam, 0.5 * floor, T, mat) is None)
    flag("a critical radius DOES exist for c_f above its floor",
         critical_radius(fam, 1.5 * floor, T, mat) is not None)
    # (7) the unfaulted counterpart always has one, when cbar_v > c^v_inf
    flag("c_p always has a critical radius above c^v_inf",
         critical_radius(fams["c_p"], 1.05 * cinf, T, mat) is not None)
    # (8) Z_woo reproduces Eq. (Afactors) at the fitted temperature
    mf = material_file or paths.MODELIB_MATERIAL
    Z0 = float(np.asarray(read_material_vector(mf, "dadZ0"), float)[0])
    p573 = p_anisotropy(573.0, mf)
    pfile = float(np.asarray(read_material_vector(mf, "dadAnisotropy"),
                             float)[0])
    chk("p_v(573 K) from the tensor == dadAnisotropy", p573, pfile, 1e-5)
    chk("Z_woo(c_p, 573 K) == Z0_v p_v", Z_woo(fams["c_p"], 573.0, mf),
        Z0 * pfile, 1e-5)
    chk("Z_woo(a_i, 573 K) == Z0_v (p+p^-2)/2", Z_woo(fams["a_i"], 573.0, mf),
        Z0 * 0.5 * (pfile + pfile ** -2), 1e-5)
    # (9) the lognormal helper preserves the mean exactly
    R, n = lognormal_bins(1e21, 5e-9, 0.5)
    chk("lognormal_bins preserves <R>", float((R * n).sum() / n.sum()), 5e-9,
        1e-12)
    chk("lognormal_bins preserves N", float(n.sum()), 1e21, 1e-12)
    # (10) a distribution's sink-weighted level differs from the level of its
    #      mean size -- the reason Sec. 5's figure is not the same as Sec. 4's
    d = DistributionFamily("c", fams["c_p"], R, n)
    lev_dist = d.c_eq(T, mat)
    lev_mean = float(gr.c_line_local(1.0 / d.R_mean, fams["c_p"], T, mat))
    flag("distribution level != level of the mean size",
         abs(lev_dist / lev_mean - 1.0) > 1e-3,
         f"{lev_dist/lev_mean:.5f}")
    # (11) RIPENING CONSERVES STORED VACANCIES, up to what leaves through R_min.
    #      A family alone exchanges no net vacancies with anything -- Eq. (9.20)
    #      makes SUM_k S_k (c_eq_k - cbar_v) vanish identically -- so the total
    #      defect content may only fall by the content of the bins that dissolve,
    #      which is m(R_min) each. That bound is the check; without it a ripening
    #      run that quietly created or destroyed vacancies would look identical.
    fam = fams["c_f"]
    Rb, nb = lognormal_bins(7.7e20, 25e-9, 0.45, n_bins=41)
    d0 = DistributionFamily("c", fam, Rb, nb)

    def content(Rr, nn):
        return float((np.pi * Rr ** 2 * fam.bdotn / fam.omega * nn).sum())

    res = anneal_distribution([d0], T, mat, stop_frac=0.25)
    R0, n0 = res["snaps"][0][0]
    R1, n1 = res["snaps"][-1][0]
    lost_n = float((n0 - n1).sum())
    m_min = math.pi * mat.r_min ** 2 * fam.bdotn / fam.omega
    drop = content(R0, n0) - content(R1, n1)
    flag("ripening conserves content to the floor current",
         0.0 <= drop <= 1.05 * lost_n * m_min,
         f"{drop/(lost_n*m_min):.4f}")
    flag("ripening: N falls and <R> rises",
         n1.sum() < n0.sum()
         and (R1 * n1).sum() / n1.sum() > (R0 * n0).sum() / n0.sum(),
         f"{(R1*n1).sum()/n1.sum()/((R0*n0).sum()/n0.sum()):.3f}x")
    print("VERIFY:", "all checks passed" if ok else "FAILURES ABOVE")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "ladder", "variants", "reference", "figure-ladder",
                 "figure-ripening", "figure-reference"):
        q = sub.add_parser(name)
        q.add_argument("--material", default=None)
        if name != "verify":
            q.add_argument("--T", type=float, default=873.0)
            q.add_argument("--run", default=None)
            q.add_argument("--c-state", default="c_f",
                           choices=("c_f", "c_p"))
        if name in ("ladder", "variants", "figure-ladder", "figure-ripening"):
            q.add_argument("--dose", type=float, default=0.1)
        if name == "variants":
            q.add_argument("--sigma", type=float, default=200.0, help="MPa")
        if name.startswith("figure"):
            q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "verify":
        return 0 if verify(a.material) else 1
    if a.cmd == "ladder":
        report_ladder(a.T, a.run, a.dose, a.c_state, material_file=a.material)
    elif a.cmd == "variants":
        report_variants(a.T, a.sigma, a.run, a.dose, a.c_state,
                        material_file=a.material)
    elif a.cmd == "reference":
        report_reference(a.T, a.run, c_state=a.c_state, material_file=a.material)
    elif a.cmd.startswith("figure"):
        from dislocluster_code.studies import loop_annealing_population_figs as F
        getattr(F, a.cmd.replace("-", "_"))(
            a.out, T=a.T, run_dir=a.run, c_state=a.c_state,
            dose=getattr(a, "dose", 0.1), material_file=a.material)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
