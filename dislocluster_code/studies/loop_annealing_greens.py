"""loop_annealing_greens.py -- the isolated-loop anneal solved THREE ways.

WHAT THIS IS
------------
Sec. 4.3 of ``Docs/Formulation/self-consistent/self_consistent_SRCD.tex``
("An isolated basal loop: the faulted and the perfect state") integrates one
growth law, Eq. (isolatedR), in which the whole of the transport is compressed
into a single closed-form capture efficiency

    Z^iso_k = 2 pi / ln(8 R / r_0)                             Eq. (Ziso)

and the loop is a circle by assumption. This module solves the SAME anneal by
the other two constructions the manuscript carries -- the boundary-integral
Green's function solve of Sec. 3 (Eq. cDD / Gkernel / I01 / climbsystem) and the
two-parameter ellipse ROM (Eq. romA / romB) -- so that all three can be laid on
one axis. It answers four questions, in order:

  1. the diffusion problem the Green's function method actually solves, and the
     R(t) it gives, superimposed on Fig. 3(a)                  -> ``figure_rt``
  2. the Z^iso_k that makes the two coincide                   -> ``fit_Ziso``
  3. the ROM run on the same anneal, with the loop SHAPE       -> ``figure_shape``
  4. that shape against the Green's function's own outline     -> ``figure_shape``

THE DIFFUSION EQUATION THE GREEN'S FUNCTION METHOD SOLVES
---------------------------------------------------------
One loop, alone, in an infinite crystal held at c^v -> c^{v,eq}_inf, no
irradiation and no other sink. The mobile vacancy field is quasi-steady on the
climb time scale, so outside the loop core it satisfies the ANISOTROPIC steady
diffusion equation with the line held at its own local equilibrium:

    div( D_v grad c^v(x) ) = 0        x outside the core tube of radius r_0
    c^v(x) -> c^{v,eq}_inf            |x| -> infinity                       (I)
    c^v(x)  = c^{v,eq}_sk(kappa(x))   x on the loop line

    v_n(x) = varsigma_s(v) * Omega/(b.n)_k * Phi^v(x)                      (II)

with Phi^v the vacancy flux collected per unit length of line. (I) is a
DIRICHLET problem on a closed line in an infinite anisotropic medium, and (II)
is the kinematic statement Eq. (vn-common). It is solved by superposition,
c^v = c^v_C + c^v_D, with c^v_C = c^{v,eq}_inf the (here uniform) continuum part
and c^v_D the field the climbing line itself emits. A line element climbing at
v_n inserts material at the volumetric rate (b x xi).eta per unit length, so
c^v_D is the line integral of the anisotropic point-source Green's function

    div( D_m grad G_m ) = -delta(x)
    G_m(x) = 1 / [ 4 pi sqrt(det D_m) sqrt( x . D_m^{-1} x ) ]

against that source. Carrying the network's own linear shape functions along
each chord turns the line integral into the closed form Eq. (Gkernel) with the
elementary integrals Eq. (I01) -- reproduced here to the letter of
``DislocationSegment::concentrationMatrices``. Imposing (I) on every segment in a
Galerkin sense closes the system, Eq. (climbsystem):

    sum_{nu'} K_{nu nu'} v_{nu'} = F_nu
    K = int N_nu (b x L).eta_nu G ,   F = int N_nu (b x L).eta_nu [c_line - c_C]

**Nothing is assumed about the shape and no capture efficiency appears.** Z^iso
is an OUTPUT here, which is what makes question 2 well posed: solve the boundary
integral on a circle of radius R with a unit drive, read the velocity, and invert
Eq. (vn-common) for Z. That is :func:`Z_effective`.

WHAT IS HELD FIXED BETWEEN THE THREE METHODS
--------------------------------------------
The thermodynamics, and only the thermodynamics. All three are driven by the
SAME c^{v,eq}_sk built from the same dE_k/dm of Sec. 4.1 -- fault term plus
anisotropic capillarity -- generalized from a circle to a local curvature by
:func:`dEdm_local`, which reduces to ``loop_annealing.dEdm`` EXACTLY at
kappa = 1/R (checked in :func:`verify`). Every difference reported below is
therefore a difference of TRANSPORT and of nothing else. That is deliberate: the
ROM's own Eq. (romselfstress) uses mu/(1-nu) where Sec. 4.1 uses <K>_k and has no
place for the fault term at all, and mixing the two would put a thermodynamic
difference into a transport comparison.

THE ONE PLACE THE THREE GENUINELY DIFFER
-----------------------------------------
    mean field   Phi = Z^iso Dbar c / Omega     Dbar = (det D)^(1/3), circle
    Green fn     Phi = solved from (I)          FULL tensor D, free shape
    ROM          Phi = Z_m D^(axis) c / L_cap   per-axis D, ellipse, free L_cap

USAGE
-----
    python -m dislocluster_code.studies.loop_annealing_greens verify
    python -m dislocluster_code.studies.loop_annealing_greens ziso
    python -m dislocluster_code.studies.loop_annealing_greens lifetimes
    python -m dislocluster_code.studies.loop_annealing_greens lcap
    python -m dislocluster_code.studies.loop_annealing_greens closure
    python -m dislocluster_code.studies.loop_annealing_greens mechanism
    python -m dislocluster_code.studies.loop_annealing_greens shape
    python -m dislocluster_code.studies.loop_annealing_greens figure-rt    --out <png>
    python -m dislocluster_code.studies.loop_annealing_greens figure-shape --out <png>
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import ellipse_rom
from dislocluster_code.coupling.field import read_material_vector
from dislocluster_code.studies import loop_annealing as la
from dislocluster_code.studies.loop_annealing import (ALPHA_CORE, EV_J, KB_EV,
                                                      Family, Material,
                                                      families, load_material)

# ── the diffusion tensor, and the two habit planes ───────────────────────────
#
# Crystal frame throughout, the same one every figure in this repository uses:
#   x || [10-10]      y || [-2-1-10]      z || [0001]
# so D is diagonal with D_11 = D_22 = D_a (basal) and D_33 = D_c.
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])


def diffusion_tensor(T, material_file=None, species=0):
    """``D_m(T)``, the FULL 3x3 tensor, from the material file's own rows.

    ``mobileSpeciesD0_SI`` and ``mobileSpeciesEnergyMigration_eV`` carry
    [11 12 13 22 23 33] per species. The mean-field law of Eq. (isolatedR) uses
    only the invariant ``Dbar = (det D)^(1/3)``; the Green's function uses the
    tensor itself, which is one of the two reasons the two do not agree.
    """
    mf = material_file or paths.MODELIB_MATERIAL
    D0 = np.asarray(read_material_vector(mf, "mobileSpeciesD0_SI"), float)
    Em = np.asarray(read_material_vector(mf, "mobileSpeciesEnergyMigration_eV"),
                    float)
    off = 6 * species
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    D = np.zeros((3, 3))
    for (i, j), a, e in zip(idx, D0[off:off + 6], Em[off:off + 6]):
        val = a * math.exp(-e / (KB_EV * T)) if a != 0.0 else 0.0
        D[i, j] = D[j, i] = val
    return D


def habit_frame(fam: Family):
    """``(n_hat, e1, e2)`` for the family's habit plane, with ``e1 x e2 = n_hat``.

    BASAL families: n_hat = [0001], so both in-plane axes are basal, the tensor
    restricted to the habit plane is isotropic, and the loop is circular by
    symmetry and STAYS circular. PRISMATIC families: n_hat = b_hat lies in the
    basal plane, so the habit plane CONTAINS the c-axis and ``e1`` IS that
    c-axis -- which is why an <a> loop is the one whose shape moves.
    """
    if fam.name.startswith("c"):
        return EZ, EX, EY                 # e1 x e2 = z   (basal habit plane)
    return EY, EZ, EX                     # e1 x e2 = y   (prismatic, e1=[0001])


# ── the thermodynamics, generalized from a circle to a local curvature ───────

def dEdm_local(kappa, fam: Family, split=False):
    """``dE_k/dm`` at a point of LOCAL curvature ``kappa``, in J.

    Reduces to ``loop_annealing.dEdm(R)`` exactly at ``kappa = 1/R``:

        cap(R) = <K> b^2 lam^2 /(4R) * (1 + ln(alpha R/|b|))
               = <K> b^2 lam^2 /4 * kappa * ln( alpha e / (kappa |b|) )

    since ``1 + ln x = ln(e x)``. The fault term is curvature-INDEPENDENT, which
    is the whole reason the two basal states anneal by different laws.
    """
    kappa = np.asarray(kappa, float)
    fault = fam.gamma * fam.omega / fam.bdotn
    arg = np.maximum(ALPHA_CORE * math.e / np.maximum(kappa * fam.bmag, 1e-300),
                     1.0 + 1e-12)
    cap = fam.Kbar * fam.bmag ** 2 * fam.lam ** 2 / 4.0 * kappa * np.log(arg)
    return (fault, cap) if split else fault + cap


def c_line_local(kappa, fam: Family, T, mat: Material, Sigma=0.0):
    """``c^{v,eq}_sk`` at a point of local curvature ``kappa``.

    The same expression as ``loop_annealing.c_eq_vacancy``, with dE/dm taken
    locally:  ``c_eq = exp(-[E^f_v - varsigma_s(v) dE/dm - Sigma Omega]/kT)``.
    """
    Eb = (mat.Ef_v - fam.zeta_v * dEdm_local(kappa, fam) / EV_J
          - Sigma * fam.omega / EV_J)
    return np.exp(-Eb / (KB_EV * T))


def polygon_curvature(nodes):
    """Discrete curvature at every node of a closed polygon, 1/circumradius.

    Exact for a circle at any number of sides, so the boundary-integral solve
    reproduces the mean-field DRIVE with no discretization error at all and the
    comparison isolates the transport.
    """
    p = np.asarray(nodes, float)
    prv = np.roll(p, 1, axis=0)
    nxt = np.roll(p, -1, axis=0)
    a = np.linalg.norm(p - prv, axis=1)
    b = np.linalg.norm(nxt - p, axis=1)
    c = np.linalg.norm(nxt - prv, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(p - prv, nxt - p), axis=1)
    return 4.0 * area / np.maximum(a * b * c, 1e-300)


# ── the boundary-integral (Green's function) solve ───────────────────────────
#
# Eq. (cDD), Eq. (Gkernel), Eq. (I01) and Eq. (climbsystem), reproduced to the
# letter of DislocationSegment::concentrationMatrices and GalerkinClimbSolver.

def _gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w                     # mapped to [0, 1]


def loop_polygon(A, B, n_side, e1, e2, center=None):
    """A closed ``n_side``-gon inscribed in the ellipse of semi-axes (A, B).

    Traversed CLOCKWISE about ``n_hat = e1 x e2``, so that ``b x xi`` with
    ``b = (b.n) n_hat`` points OUTWARD. With that orientation a positive
    climb-velocity scalar means the loop is EMITTING, which is MoDELib's own
    convention and the one every sign below is written in.
    """
    th = -2.0 * np.pi * np.arange(n_side) / n_side
    p = (A * np.cos(th))[:, None] * e1 + (B * np.sin(th))[:, None] * e2
    return p if center is None else p + np.asarray(center, float)


def _geometry(nodes, bvec):
    """Chords, unit tangents, and segment / node climb directions.

    ``node_climb`` is ``DislocationNode::climbDirection`` verbatim: the
    chord-length-weighted average of the adjacent segments' ``b x chord``,
    renormalized.
    """
    p = np.asarray(nodes, float)
    L = np.roll(p, -1, axis=0) - p
    Lmag = np.linalg.norm(L, axis=1)
    xi = L / Lmag[:, None]
    seg = np.cross(bvec, xi)
    seg /= np.linalg.norm(seg, axis=1)[:, None]
    acc = seg * Lmag[:, None] + np.roll(seg * Lmag[:, None], 1, axis=0)
    node = acc / np.linalg.norm(acc, axis=1)[:, None]
    return L, Lmag, xi, seg, node


def _influence(x, src_p0, src_L, src_Lmag, Dinv, detD, reg):
    """``(pref*I0, pref*I1)`` of Eq. (Gkernel)/(I01), for a stack of field points.

    ``x`` is (n_field, 3); the return is (n_field, n_src). ``reg`` is
    ``r_0^2/(det D)^(1/3)``, the core term of Eq. (ABC) scaled by the tensor's
    own determinant so the regularization is isotropic in the transformed
    coordinates.
    """
    Rv = x[:, None, :] - src_p0                          # (n_field, n_src, 3)
    DiL = src_L @ Dinv.T                                 # (n_src, 3)
    A = np.einsum("ij,ij->i", src_L, DiL)                # (n_src,)
    B = -2.0 * np.einsum("fij,ij->fi", Rv, DiL)
    C = np.einsum("fij,jk,fik->fi", Rv, Dinv, Rv) + reg
    u = B / A
    w = C / A
    sq1 = np.sqrt(np.maximum(1.0 + u + w, 1e-300))
    sq0 = np.sqrt(np.maximum(w, 1e-300))
    Q = np.log(np.maximum((2.0 * sq1 + 2.0 + u) / (2.0 * sq0 + u), 1e-300))
    I0 = (1.0 + 0.5 * u) * Q - sq1 + sq0
    I1 = -0.5 * u * Q + sq1 - sq0
    pref = src_Lmag / (4.0 * np.pi * np.sqrt(A * detD))
    return pref * I0, pref * I1


def climb_system(nodes, bvec, D, r_core, drive, ZD=1.0, n_quad=16,
                 return_system=False):
    """Assemble and solve Eq. (climbsystem) for one isolated loop.

    ``drive`` is ``c_line - c_C`` at each NODE, interpolated linearly to the
    quadrature points exactly as the network's own shape functions do. Returns
    the climb-velocity scalar at every node; positive means EMITTING.
    """
    p = np.asarray(nodes, float)
    n = len(p)
    L, Lmag, xi, _, eta = _geometry(p, bvec)
    Dinv = np.linalg.inv(D)
    detD = float(np.linalg.det(D))
    reg = r_core ** 2 / detD ** (1.0 / 3.0)

    nxt = np.roll(np.arange(n), -1)
    # source strengths: (b x xi_hat) . eta at each end of each chord, Eq. (Gkernel)
    bxt = np.cross(bvec, xi)
    q0 = np.einsum("ij,ij->i", bxt, eta) / ZD
    q1 = np.einsum("ij,ij->i", bxt, eta[nxt]) / ZD
    # Galerkin test weights: (b x L) . eta at each end of each field chord
    bxL = np.cross(bvec, L)
    w0 = np.einsum("ij,ij->i", bxL, eta)
    w1 = np.einsum("ij,ij->i", bxL, eta[nxt])

    ug, wg = _gauss(n_quad)
    K = np.zeros((n, n))
    F = np.zeros(n)
    drive = np.asarray(drive, float)
    for u, wq in zip(ug, wg):
        x = p + u * L                                     # field quadrature pts
        G0, G1 = _influence(x, p, L, Lmag, Dinv, detD, reg)
        # column j of `col` is the influence of NODE j; source segment s feeds
        # its own node s through I0 and node s+1 through I1.
        col = G0 * q0 + np.roll(G1 * q1, 1, axis=1)       # (n_field, n_node)
        dq = (1.0 - u) * drive + u * drive[nxt]
        r0w = wq * (1.0 - u) * w0
        r1w = wq * u * w1
        K += r0w[:, None] * col
        K += np.roll(r1w[:, None] * col, 1, axis=0)
        F += r0w * dq
        F += np.roll(r1w * dq, 1)
    if return_system:
        return K, F
    return np.linalg.solve(K, F)


# ── question 2: the capture efficiency the Green's function implies ──────────

def Z_effective(R, fam: Family, D, r_core=None, n_side=48, n_quad=16, ZD=1.0):
    """``Z^iso_k`` READ OFF the boundary-integral solve of a circle of radius R.

    Invert Eq. (vn-common) as the mean-field law writes it,

        v = Z Dbar Delta_c / (b.n)_k    ==>    Z = v (b.n)_k / (Dbar Delta_c),

    with a unit drive. ``Dbar = (det D)^(1/3)`` is the same invariant
    Eq. (isolatedR) carries, so the Z returned absorbs BOTH the geometry and the
    tensor's departure from that invariant -- which is exactly the number
    Eq. (Ziso) is standing in for.
    """
    r_core = fam.bmag if r_core is None else r_core
    nhat, e1, e2 = habit_frame(fam)
    bvec = fam.bdotn * nhat
    Dbar = float(np.linalg.det(D)) ** (1.0 / 3.0)
    Rs = np.atleast_1d(np.asarray(R, float))
    out = np.empty_like(Rs)
    for i, r in enumerate(Rs):
        nodes = loop_polygon(r, r, n_side, e1, e2)
        v = climb_system(nodes, bvec, D, r_core, np.ones(n_side), ZD, n_quad)
        out[i] = float(np.mean(v)) * fam.bdotn / Dbar
    return out if np.ndim(R) else float(out[0])


def fit_Ziso(fam: Family, T, mat: Material = None, R=None, D=None, **kw):
    """The ``Z^iso_k`` that makes the two solutions coincide.

    The mean-field FORM is kept -- ``Z = 2 pi / ln(alpha R/r_0)`` -- and the one
    number in it, the log's prefactor (``alpha = 8`` in Eq. Ziso), is fitted to
    the Green's function's own ``Z_effective`` over the annealing size range.
    Reported alongside is the plain multiplicative factor ``f`` of the best
    ``Z = f * 2 pi/ln(8R/r_0)``, which is what a code unwilling to change the
    functional form would use instead.
    """
    mat = mat or load_material()
    D = diffusion_tensor(T) if D is None else D
    R = np.geomspace(2e-9, 60e-9, 12) if R is None else np.asarray(R, float)
    Zg = np.asarray(Z_effective(R, fam, D, **kw), float)
    lnR = np.log(R / fam.bmag)
    # 2pi/ln(alpha R/r0) = Zg  <=>  ln alpha = 2pi/Zg - ln(R/r0)
    ln_alpha = float(np.mean(2.0 * np.pi / Zg - lnR))
    Za = 2.0 * np.pi / (ln_alpha + lnR)
    Z0 = np.asarray([la.Z_isolated(r, fam) for r in R], float)
    f = float(np.exp(np.mean(np.log(Zg / Z0))))
    return dict(alpha=math.exp(ln_alpha), factor=f, R=R, Z_gf=Zg, Z_mf=Z0,
                Z_fit=Za,
                max_rel_alpha=float(np.max(np.abs(Za / Zg - 1.0))),
                max_rel_factor=float(np.max(np.abs(f * Z0 / Zg - 1.0))),
                max_rel_raw=float(np.max(np.abs(Z0 / Zg - 1.0))))


# ── question 1: R(t) by the Green's function method ──────────────────────────
#
# A BASAL loop stays a circle -- its habit plane is the basal plane, in which the
# tensor is isotropic, so the boundary-integral solve returns a UNIFORM velocity
# at every R (asserted in `verify`). Its trajectory is therefore obtained the way
# `loop_annealing.anneal` obtains it, integrating dt/dR because R is the monotone
# variable and the rate diverges as the loop disappears, with Z^iso replaced by
# the solved Z_effective(R). No shape freedom is discarded by doing so; the
# genuinely nodal march is `anneal_gf_shape`, and `verify` checks they agree.

def anneal_gf(R0, fam: Family, T, mat: Material = None, D=None, c_far=None,
              Sigma=0.0, R_end=None, n=241, n_side=48, n_quad=16, r_core=None,
              Z=None):
    """``(t, R)`` for an isolated loop, with the transport from Eq. (climbsystem)."""
    mat = mat or load_material()
    D = diffusion_tensor(T) if D is None else D
    c_far = mat.c_v_inf(T) if c_far is None else c_far
    R_end = mat.r_min if R_end is None else R_end
    R = np.geomspace(R0, R_end, n)
    if Z is None:
        Z = np.asarray(Z_effective(R, fam, D, r_core=r_core, n_side=n_side,
                                   n_quad=n_quad), float)
    Dbar = float(np.linalg.det(D)) ** (1.0 / 3.0)
    ceq = c_line_local(1.0 / R, fam, T, mat, Sigma)
    rate = fam.zeta_v * Z * Dbar * (c_far - ceq) / fam.bdotn
    if np.any(rate >= 0):
        raise ValueError(f"{fam.name} does not shrink over the requested range")
    inv = 1.0 / np.abs(rate)
    dR = np.abs(np.diff(R))
    t = np.concatenate([[0.0], np.cumsum(0.5 * (inv[:-1] + inv[1:]) * dR)])
    return t, R


# ── question 3: the same anneal by the ellipse ROM ───────────────────────────

def rom_capture_length(fam: Family, R, mat: Material, ZD=1.0):
    """The ``L_cap`` of Eq. (romA) that reproduces the isolated-loop limit.

    Eq. (romA) writes the arrival as ``Z_m D/L_cap``; the mean field and the
    Green's function both write it as ``Z^iso Dbar/Omega`` -- a concentration is
    an atom fraction, so ``c/Omega`` is the number density and no length appears.
    Matching the two at the reference radius,

        L_cap = Omega Z_m / ( b^2 Z^iso(R) )   ~   Omega/b^2 = 0.69 |b|,

    a fraction of a Burgers magnitude and NOT the ~59 nm screening length
    ``ellipse_rom`` defaults to. Since L_cap enters both axis equations
    identically it sets the TIME SCALE only and cancels from the aspect ratio, so
    this calibration moves the ROM's lifetime and leaves its shape untouched.
    """
    return mat.omega * ZD / (mat.b ** 2 * la.Z_isolated(R, fam))


def axis_curvatures(A, B, convention="paper"):
    """``(kappa_A, kappa_B)`` -- the curvature at the END of each semi-axis.

    **THE TWO CONVENTIONS ARE NOT THE SAME, AND ONE OF THEM IS WRONG.**
    Parametrize the habit-plane ellipse as ``(A cos t) e1 + (B sin t) e2``. Its
    curvature is ``kappa(t) = AB/(A^2 sin^2 t + B^2 cos^2 t)^{3/2}``, so

        at the A vertex (t=0)     kappa = A/B^2       <- BLUNT when A is minor
        at the B vertex (t=pi/2)  kappa = B/A^2       <- SHARP when B is major

    ``convention="geometric"`` returns that. ``convention="paper"`` returns the
    assignment Eq. (romselfstress) and ``ellipse_rom.curvature`` actually carry,
    ``kappa_A = B/A^2`` and ``kappa_B = A/B^2``, which is the pair SWAPPED --
    verified numerically against ``polygon_curvature`` on a 1:3 ellipse.

    The swap is not cosmetic: it reverses the sign of the capillary shape
    feedback. Eq. (romselfstress)'s own prose ("as the loop elongates, kappa
    rises at the MAJOR-axis ends, the back stress there raises c_eq and growth
    throttles") describes the geometric assignment; the equation beneath it puts
    the large curvature on the minor axis instead, which turns a restoring term
    into a runaway one. :func:`report_shape` reports both, and
    ``ellipse_rom.curvature`` is deliberately left alone -- it has no other
    caller in the repository, so nothing published depends on either answer yet.
    """
    A = max(float(A), 1e-300)
    B = max(float(B), 1e-300)
    if convention == "geometric":
        return A / B ** 2, B / A ** 2
    if convention == "paper":
        return B / A ** 2, A / B ** 2
    raise ValueError("convention must be 'paper' or 'geometric'")


def rom_rates(A, B, fam: Family, T, mat: Material, D, L_cap, c_far,
              closure="geometric", Sigma=0.0, ZD=1.0, self_stress_on=True,
              convention="paper"):
    """``(A_dot, B_dot)`` -- Eq. (romA)/(romB) for the pure vacancy anneal.

    ``A`` is the semi-axis along ``e1`` of :func:`habit_frame` and ``B`` along
    ``e2``; for a prismatic family ``e1 = [0001]``, so ``A`` is ``A_ell`` of
    Eq. (romstate) and ``B/A`` is ``varrho_ell``. The capture diffusivities come
    from ``ellipse_rom.capture_diffusivities`` unchanged; the driving
    supersaturation comes from the SAME ``dEdm_local`` the other two methods use,
    so what is compared is transport -- plus, through ``convention``, the
    curvature assignment of :func:`axis_curvatures`.
    """
    D_a, D_c = float(D[0, 0]), float(D[2, 2])
    D_basal, D_mixed = ellipse_rom.capture_diffusivities(D_a, D_c, closure)
    if fam.name.startswith("c"):
        # Basal habit plane: the tangent at BOTH axis endpoints lies in the
        # basal plane, so both capture planes span c and b. The loop cannot
        # become an ellipse whatever the anisotropy -- Eq. (romattractor) is
        # identically 1 here.
        DA = DB = D_mixed
    else:
        # Prismatic: e1 = [0001], so the tangent at the end of A is basal (mixed
        # capture plane) and the tangent at the end of B is along [0001] (basal
        # capture plane). Sec. 3's table, verbatim.
        DA, DB = D_mixed, D_basal
    kA, kB = axis_curvatures(A, B, convention)
    out = []
    for Dax, kap in ((DA, kA), (DB, kB)):
        ceq = c_line_local(kap if self_stress_on else 0.0, fam, T, mat, Sigma)
        out.append(float(fam.zeta_v * ZD * Dax / L_cap * mat.omega
                         / (mat.b ** 2 * fam.bdotn) * (c_far - ceq)))
    return out[0], out[1]


def anneal_rom(A0, B0, fam: Family, T, mat: Material = None, D=None, L_cap=None,
               c_far=None, closure="geometric", Sigma=0.0, R_end=None,
               n_steps=200000, ZD=1.0, cfl=0.002, convention="paper"):
    """``(t, A, B)`` -- the ROM anneal, marched explicitly to extinction."""
    mat = mat or load_material()
    D = diffusion_tensor(T) if D is None else D
    c_far = mat.c_v_inf(T) if c_far is None else c_far
    R_end = mat.r_min if R_end is None else R_end
    if L_cap is None:
        L_cap = rom_capture_length(fam, math.sqrt(A0 * B0), mat, ZD)
    ts, As, Bs = [0.0], [float(A0)], [float(B0)]
    A, B, t = float(A0), float(B0), 0.0
    for _ in range(int(n_steps)):
        dA, dB = rom_rates(A, B, fam, T, mat, D, L_cap, c_far, closure, Sigma,
                           ZD, convention=convention)
        rate = max(abs(dA), abs(dB), 1e-300)
        dt = cfl * min(A, B) / rate
        A, B, t = A + dt * dA, B + dt * dB, t + dt
        if not (np.isfinite(A) and np.isfinite(B)) or A <= 0.0 or B <= 0.0:
            break
        ts.append(t)
        As.append(A)
        Bs.append(B)
        if math.sqrt(A * B) <= R_end:
            break
    return np.array(ts), np.array(As), np.array(Bs)


# ── questions 3 & 4: the shape, from the NODAL Green's function solve ────────

def _signed_area(q, nhat):
    return 0.5 * float(np.dot(np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0),
                              nhat))


def equiv_radius(q, nhat):
    """The equal-area radius of a closed polygon -- the quantity comparable
    with the circular R(t) of Eq. (isolatedR)."""
    return math.sqrt(abs(_signed_area(q, nhat)) / math.pi)


def anneal_gf_shape(R0, fam: Family, T, mat: Material = None, D=None,
                    c_far=None, Sigma=0.0, R_end=None, n_side=32, n_quad=12,
                    r_core=None, n_steps=20000, cfl=0.004, n_snap=6,
                    capillarity="local"):
    """Full NODAL Green's function anneal: no assumed shape anywhere.

    Every node carries its own climb-velocity scalar, its own local curvature and
    therefore its own ``c_line``; the outline is whatever those velocities make
    it. Returns ``(t_snap, nodes_snap, t, R_eq)``.

    ``capillarity="uniform"`` is a DIAGNOSTIC, not a model: it evaluates the
    driving supersaturation at the equal-area circle's curvature everywhere on
    the line instead of at each node's own. The loop still shrinks, but the
    capillary shape feedback is switched off, so what is left is transport plus
    the geometric concentration of flux that the boundary integral supplies and
    Eq. (romA)/(romB) do not. Differencing the two isolates that term --- see
    :func:`report_shape_mechanism`.
    """
    mat = mat or load_material()
    D = diffusion_tensor(T) if D is None else D
    c_far = mat.c_v_inf(T) if c_far is None else c_far
    R_end = mat.r_min if R_end is None else R_end
    r_core = fam.bmag if r_core is None else r_core
    nhat, e1, e2 = habit_frame(fam)
    bvec = fam.bdotn * nhat

    p = loop_polygon(R0, R0, n_side, e1, e2)
    t = 0.0
    ts, Rs = [0.0], [R0]
    snap_R = np.linspace(R0, R_end, n_snap)
    snaps, snap_t, k_snap = [p.copy()], [0.0], 1

    for _ in range(int(n_steps)):
        kap = (polygon_curvature(p) if capillarity == "local"
               else np.full(len(p), 1.0 / equiv_radius(p, nhat)))
        drive = c_line_local(kap, fam, T, mat, Sigma) - c_far
        v = climb_system(p, bvec, D, r_core, drive, 1.0, n_quad)
        _, _, _, _, eta = _geometry(p, bvec)
        vout = -fam.zeta_v * v                      # outward node speed
        step = float(np.max(np.abs(vout)))
        if not np.isfinite(step) or step <= 0.0:
            break
        dt = cfl * equiv_radius(p, nhat) / step
        p = p + dt * vout[:, None] * eta
        t += dt
        Req = equiv_radius(p, nhat)
        ts.append(t)
        Rs.append(Req)
        while k_snap < n_snap and Req <= snap_R[k_snap]:
            snaps.append(p.copy())
            snap_t.append(t)
            k_snap += 1
        if Req <= R_end:
            break
    return np.array(snap_t), snaps, np.array(ts), np.array(Rs)


def shape_metrics(q, fam: Family):
    """``(R_eq, A, B, varrho)`` for one polygon: half-extents along the habit
    frame's two axes, and the aspect ratio Eq. (romstate) calls ``varrho_ell``."""
    nhat, e1, e2 = habit_frame(fam)
    A = float(np.max(q @ e1) - np.min(q @ e1)) / 2.0
    B = float(np.max(q @ e2) - np.min(q @ e2)) / 2.0
    return equiv_radius(q, nhat), A, B, B / A


# ── reporting ────────────────────────────────────────────────────────────────

def report_ziso(T=873.0, names=("c_f", "c_p", "a_v", "a_i"), material_file=None):
    """Question 2 as a table: Z from Eq. (Ziso), Z from the boundary integral,
    and the one-parameter refit of the former to the latter."""
    mat = load_material(material_file)
    fams = families(mat)
    D = diffusion_tensor(T, material_file)
    Dbar = float(np.linalg.det(D)) ** (1.0 / 3.0)
    print(f"# Z^iso_k of Eq. (Ziso) against the Green's function solve, "
          f"T = {T:g} K")
    print(f"#   D_a = {D[0,0]:.4e}   D_c = {D[2,2]:.4e}   Dbar = {Dbar:.4e} "
          f"m^2/s   D_c/D_a = {D[2,2]/D[0,0]:.4f}   p_v = "
          f"{(D[2,2]/D[0,0])**(1/6):.5f}")
    print(f"# {'family':<7}{'alpha fit':>11}{'(Eq. 8)':>9}{'factor f':>11}"
          f"{'err(alpha)':>12}{'err(f)':>10}{'err(raw)':>11}")
    out = {}
    for n in names:
        r = fit_Ziso(fams[n], T, mat, D=D)
        out[n] = r
        print(f"  {n:<7}{r['alpha']:>11.4f}{8:>9}{r['factor']:>11.5f}"
              f"{r['max_rel_alpha']:>12.2%}{r['max_rel_factor']:>10.2%}"
              f"{r['max_rel_raw']:>11.2%}")
    n0 = names[0]
    print(f"# Z(R) for {n0}:   R [nm]   Z_meanfield   Z_greens   Z_fit(alpha)")
    r = out[n0]
    for R, a, b, c in zip(r["R"], r["Z_mf"], r["Z_gf"], r["Z_fit"]):
        print(f"  {R*1e9:>10.2f}{a:>14.5f}{b:>11.5f}{c:>13.5f}")
    return out


def report_lifetimes(R0=25e-9, temperatures=(773, 823, 873, 923, 973, 1023),
                     names=("c_f", "c_p"), material_file=None):
    """Question 1 as a table: Tab. (anneal-lifetime) with the Green's function
    column beside it."""
    mat = load_material(material_file)
    fams = families(mat)
    print(f"# Anneal-out time from R0 = {R0*1e9:g} nm to r_min = "
          f"{mat.r_min*1e9:g} nm.")
    print("# MF = Eq. (isolatedR) with Eq. (Ziso);  GF = the same law with the "
          "boundary-integral Z.")
    head = "".join(f"{n+' MF':>13}{n+' GF':>13}{'GF/MF':>8}" for n in names)
    print(f"# {'T [K]':<7}{head}")
    rows = {}
    for T in temperatures:
        D = diffusion_tensor(T, material_file)
        line = ""
        for n in names:
            tm = la.anneal_time(R0, fams[n], T, mat=mat)
            tg = float(anneal_gf(R0, fams[n], T, mat=mat, D=D)[0][-1])
            rows[(T, n)] = (tm, tg)
            line += (f"{la._fmt_time(tm):>13}{la._fmt_time(tg):>13}"
                     f"{tg/tm:>8.3f}")
        print(f"  {T:<7g}{line}")
    return rows


def report_lcap(R0=25e-9, T=873.0, names=("c_f", "c_p", "a_v", "a_i"),
                material_file=None):
    """The ROM's one free parameter, calibrated against the isolated-loop limit."""
    mat = load_material(material_file)
    fams = families(mat)
    print(f"# L_cap of Eq. (romA) that reproduces the isolated-loop limit, "
          f"T = {T:g} K")
    print(f"#   Omega/b^2 = {mat.omega/mat.b**3:.4f} |b| = "
          f"{mat.omega/mat.b**2*1e9:.4f} nm")
    print(f"# {'family':<7}{'R0 [nm]':>9}{'Z^iso':>9}{'L_cap [nm]':>13}"
          f"{'L_cap [b]':>12}{'L_s/L_cap':>12}")
    for n in names:
        R = R0 if n.startswith("c") else 3.6e-9
        Lc = rom_capture_length(fams[n], R, mat)
        print(f"  {n:<7}{R*1e9:>9.2f}{la.Z_isolated(R, fams[n]):>9.4f}"
              f"{Lc*1e9:>13.5f}{Lc/mat.b:>12.4f}{59e-9/Lc:>12.0f}")
    print("# L_s is the ~59 nm screening length ellipse_rom defaults L_cap to on")
    print("# the 500 nm reference case; the last column is how much too slow the")
    print("# ROM's isolated-loop climb would be at that default.")


def report_closure(temperatures=(773.0, 873.0, 1023.0),
                   radii=(2e-9, 3.6e-9, 10e-9, 25e-9), name="a_v",
                   n_side=96, n_quad=16, material_file=None):
    """WHICH CAPTURE-DIFFUSIVITY CLOSURE Eq. (romattractor) SHOULD USE.

    Sec. 3 leaves this open -- "*this is not settled* ... only a nodal solve of
    the same case can collapse it" -- because the diffusivity feeding the end of
    the [0001] axis is not fixed by geometry: the geometric mean gives an
    axis-velocity ratio ``D^(A)/D^(B) = p_m^3``, the normal projection ``p_m^6``.

    THE ISOLATED LOOP IS A NODAL SOLVE OF EXACTLY THAT CASE. Put a CIRCULAR
    prismatic loop in the boundary-integral solver with a UNIFORM drive, so the
    only thing that can make the two axis endpoints move at different speeds is
    the diffusion tensor, and read the ratio off. Reported as the effective
    exponent ``q = ln(v_A/v_B)/ln(p_m)``, to be compared with 3 and with 6.
    """
    mat = load_material(material_file)
    fam = families(mat)[name]
    nhat, e1, e2 = habit_frame(fam)
    print(f"# Which closure? Axis-velocity ratio of a CIRCULAR {name} loop under "
          f"a UNIFORM drive.")
    print("#   geometric-mean closure predicts q = 3;  normal-projection "
          "closure predicts q = 6.")
    print(f"# {'T [K]':>7}{'R [nm]':>9}{'p_v':>10}{'v_A/v_B':>10}{'q_eff':>9}"
          f"{'p_v^3':>9}{'p_v^6':>9}")
    out = []
    for T in temperatures:
        D = diffusion_tensor(T, material_file)
        p = (D[2, 2] / D[0, 0]) ** (1.0 / 6.0)
        for R in radii:
            q = loop_polygon(R, R, n_side, e1, e2)
            v = climb_system(q, fam.bdotn * nhat, D, fam.bmag,
                             np.ones(n_side), 1.0, n_quad)
            iA = int(np.argmax(np.abs(q @ e1)))
            iB = int(np.argmax(np.abs(q @ e2)))
            r = float(v[iA] / v[iB])
            qe = math.log(r) / math.log(p)
            out.append((T, R, p, r, qe))
            print(f"  {T:>7.0f}{R*1e9:>9.1f}{p:>10.5f}{r:>10.4f}{qe:>9.3f}"
                  f"{p**3:>9.4f}{p**6:>9.4f}")
    print("# q_eff is independent of T (hence of p_v) to 3 decimal places and")
    print("# rises with R toward 3: the GEOMETRIC MEAN is the large-loop limit,")
    print("# and the normal projection is not a candidate. The shortfall at "
          "small R")
    print("# is the core regularization blunting the anisotropy.")
    return out


def axis_velocities(fam: Family, D, R_eq, rho, n_side=128, n_quad=20,
                    r_core=None):
    """``(v_A, v_B)`` on a FIXED ellipse of equal-area radius ``R_eq``.

    The drive is uniform, so capillarity plays no part and what is measured is
    transport plus geometry alone.
    """
    r_core = fam.bmag if r_core is None else r_core
    nhat, e1, e2 = habit_frame(fam)
    p = loop_polygon(R_eq / math.sqrt(rho), R_eq * math.sqrt(rho), n_side,
                     e1, e2)
    v = climb_system(p, fam.bdotn * nhat, D, r_core, np.ones(n_side), 1.0,
                     n_quad)
    iA = int(np.argmax(np.abs(p @ e1)))
    iB = int(np.argmax(np.abs(p @ e2)))
    return float(v[iA]), float(v[iB])


def report_shape_mechanism(T=873.0, name="a_i", R_eq=3.6e-9,
                           rhos=(1.0, 1.15, 1.35, 1.6, 2.0, 2.5, 3.0),
                           material_file=None, anneal=True):
    """WHY the ROM and the boundary integral give different shapes.

    The kinematics first, because ``v_A/v_B`` on its own is not the comparison
    that matters. With ``A = R/sqrt(rho)``, ``B = R sqrt(rho)`` at fixed area and
    both axes retreating,

        d(ln rho)/dt = |v_A|/A - |v_B|/B
                     = (|v_B|/R) rho^{-1/2} [ rho (v_A/v_B) - 1 ],

    so the ELONGATION DRIVE is ``E(rho) = rho (v_A/v_B) - 1`` and a steady shape
    needs ``E = 0``. In the ROM ``v_A/v_B = D^(A)/D^(B) = p^3`` is a CONSTANT, so
    ``E`` grows without bound in rho. In the boundary integral it is solved, and
    it falls -- because Eq. (isoBVP) is a Dirichlet problem on a line, hence a
    capacitance problem, whose source density concentrates at high curvature.

    Three measurements, in the order the argument needs them:

      1. an ISOTROPIC tensor at varying rho, where the ROM says ``v_A/v_B = 1``
         at every shape. Whatever comes out instead is the missing term ALONE.
         It is exactly 1 at rho = 1 -- a solver check, and the reason this term
         does not contaminate the closure exponent of `report_closure`.
      2. the true tensor at varying rho, against the ROM's constant.
      3. the nodal anneal with the capillary SHAPE feedback switched off
         (`capillarity="uniform"`), which bounds how much of the gap capillary
         resolution could possibly explain. On a_i it is 8.5%, against a 37% gap:
         **capillarity is not the explanation, the geometric term is.**
    """
    mat = load_material(material_file)
    fam = families(mat)[name]
    D = diffusion_tensor(T, material_file)
    Dbar = float(np.linalg.det(D)) ** (1.0 / 3.0)
    Diso = np.eye(3) * Dbar
    p_v = (D[2, 2] / D[0, 0]) ** (1.0 / 6.0)
    q3 = p_v ** 3
    print(f"# Axis-velocity response to SHAPE. {name}, R_eq = {R_eq*1e9:g} nm, "
          f"T = {T:g} K, uniform drive.")
    print(f"#   p_v = {p_v:.5f};  ROM: v_A/v_B = p_v^3 = {q3:.4f} at EVERY rho")
    print(f"# {'rho':>6}{'iso v_A/v_B':>13}{'true v_A/v_B':>14}{'E solved':>10}"
          f"{'E ROM':>9}{'ratio':>8}")
    iso = []
    for rho in rhos:
        vA_i, vB_i = axis_velocities(fam, Diso, R_eq, rho)
        vA_a, vB_a = axis_velocities(fam, D, R_eq, rho)
        ri, ra = vA_i / vB_i, vA_a / vB_a
        iso.append(ri)
        E, E0 = rho * ra - 1.0, rho * q3 - 1.0
        print(f"  {rho:>6.2f}{ri:>13.4f}{ra:>14.4f}{E:>10.4f}{E0:>9.4f}"
              f"{E/E0:>8.3f}")
    if len(rhos) > 2:
        g = -float(np.polyfit(np.log(rhos[1:]), np.log(iso[1:]), 1)[0])
        print(f"# isotropic column ~ rho^-{g:.3f}: the term the ROM has no place "
              f"for, and it is")
        print("# exactly 1.0000 at rho = 1, so it acts only off the circle.")
    if not anneal:
        return
    print("# Capillary shape feedback ON vs OFF, same nodal anneal:")
    R0, R_end = R_eq, 0.30 * R_eq
    out = {}
    for mode in ("local", "uniform"):
        _, snaps, _, _ = anneal_gf_shape(R0, fam, T, mat=mat, D=D, R_end=R_end,
                                         n_snap=5, n_side=32, n_steps=4000,
                                         capillarity=mode)
        out[mode] = [shape_metrics(q, fam) for q in snaps]
    print(f"# {'R/R0':>7}{'rho nodal':>12}{'rho frozen':>12}{'ratio':>9}")
    for a, b in zip(out["local"], out["uniform"]):
        print(f"  {a[0]/R0:>7.2f}{a[3]:>12.4f}{b[3]:>12.4f}{b[3]/a[3]:>9.4f}")
    print("# The last ratio bounds what capillary RESOLUTION could explain; "
          "compare it")
    print("# with the ROM's own excess in `shape`.")


ROM_VARIANTS = (("paper", "geometric"), ("paper", "normal"),
                ("geometric", "geometric"), ("geometric", "normal"))


def report_shape(T=873.0, names=("a_v", "a_i", "c_p"), material_file=None,
                 n_side=32):
    """Questions 3 and 4 as a table: the aspect ratio each construction gives.

    Four ROM columns, because there are TWO independent ambiguities and they are
    usually conflated: the capture-diffusivity closure of Eq. (romattractor)
    (``geometric`` vs ``normal``), and the curvature assignment of
    :func:`axis_curvatures` (``paper`` = Eq. romselfstress as written, vs
    ``geometric`` = the ellipse's actual curvature).
    """
    mat = load_material(material_file)
    fams = families(mat)
    D = diffusion_tensor(T, material_file)
    p_v = (D[2, 2] / D[0, 0]) ** (1.0 / 6.0)
    print(f"# Loop shape as it anneals, T = {T:g} K, p_v = {p_v:.5f}. "
          f"rho = B/A, A || [0001].")
    print(f"#   ROM axis-velocity ratio D^(A)/D^(B): geometric p_v^3 = "
          f"{p_v**3:.4f}, normal p_v^6 = {p_v**6:.4f}")
    print("#   curv='paper' is Eq. (romselfstress) as written; "
          "curv='geom' is the ellipse's true curvature")
    head = "".join(f"{('%s/%s' % (c[:4], k[:3])):>12}" for c, k in ROM_VARIANTS)
    print(f"# {'family':<7}{'R/R0':>7}{'rho GF':>10}{head}")
    for n in names:
        fam = fams[n]
        R0 = 25e-9 if n.startswith("c") else 3.6e-9
        R_end = 0.15 * R0
        _, snaps, _, _ = anneal_gf_shape(R0, fam, T, mat=mat, D=D, n_side=n_side,
                                         R_end=R_end, n_snap=6)
        Lc = rom_capture_length(fam, R0, mat)
        roms = {}
        for conv, cl in ROM_VARIANTS:
            _, A, B = anneal_rom(R0, R0, fam, T, mat=mat, D=D, L_cap=Lc,
                                 closure=cl, R_end=R_end, convention=conv)
            roms[(conv, cl)] = (np.sqrt(A * B), B / A)
        for q in snaps:
            Req, _, _, rho = shape_metrics(q, fam)
            row = f"  {n:<7}{Req/R0:>7.2f}{rho:>10.4f}"
            for key in ROM_VARIANTS:
                Rr, rr = roms[key]
                row += f"{rr[int(np.argmin(np.abs(Rr - Req)))]:>12.4f}"
            print(row)


def verify(material_file=None):
    """Checks that can be done by hand, so every number below is testable."""
    mat = load_material(material_file)
    fams = families(mat)
    T = 873.0
    D = diffusion_tensor(T, material_file)
    ok = True

    def chk(what, got, want, tol):
        nonlocal ok
        rel = abs(got - want) / max(abs(want), 1e-300)
        good = rel <= tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {what:<58}"
              f"{got:>13.6g}  vs {want:<12.6g} ({rel:.1e})")

    def flag(what, good, value=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok ' if good else 'FAIL'}] {what:<58}{value:>13}")

    # (1) the local capillarity reduces EXACTLY to Sec. 4.1's circular form
    for n in fams:
        R = la.REFERENCE_RADII[n]
        chk(f"dEdm_local(1/R) == dEdm(R), {n}",
            float(dEdm_local(1.0 / R, fams[n])), float(la.dEdm(R, fams[n])),
            1e-12)
    # (2) the tensor's invariant IS the Dbar the mean-field law uses
    chk("(det D)^(1/3) == Dbar_v(T)", float(np.linalg.det(D)) ** (1 / 3),
        mat.Dbar_v(T), 1e-10)
    # (3) THE KERNEL TEST. In an ISOTROPIC medium the boundary-integral solve of
    #     a circular loop must return Eq. (Ziso) itself, because the self-field
    #     of a ring of uniform strength is Q ln(8R/r0)/(2 pi R) -- the toroidal
    #     capacitance. This is what says the Green's function implementation is
    #     the same physics as Z^iso and not a different model.
    Diso = np.eye(3) * float(np.linalg.det(D)) ** (1.0 / 3.0)
    for n in ("c_p", "a_i"):
        for R in (5e-9, 25e-9):
            chk(f"Z_greens == 2pi/ln(8R/|b|), isotropic, {n}, R={R*1e9:g} nm",
                float(Z_effective(R, fams[n], Diso, n_side=128, n_quad=24)),
                float(la.Z_isolated(R, fams[n])), 2e-2)
    # (4) a BASAL loop's solved velocity is uniform to round-off, whatever the
    #     anisotropy -- so it stays a circle and `anneal_gf` discards nothing.
    fam = fams["c_p"]
    nhat, e1, e2 = habit_frame(fam)
    v = climb_system(loop_polygon(25e-9, 25e-9, 32, e1, e2), fam.bdotn * nhat,
                     D, fam.bmag, np.ones(32), 1.0, 12)
    spread = float(np.max(np.abs(v / np.mean(v) - 1.0)))
    flag("basal loop: solved velocity is uniform", spread < 1e-9,
         f"{spread:.2e}")
    # (5) a PRISMATIC loop's is not -- that is the shape signal of Q3/Q4
    fam = fams["a_v"]
    nhat, e1, e2 = habit_frame(fam)
    v = climb_system(loop_polygon(3.6e-9, 3.6e-9, 32, e1, e2),
                     fam.bdotn * nhat, D, fam.bmag, np.ones(32), 1.0, 12)
    ratio = float(np.max(v) / np.min(v))
    flag("prismatic loop: velocity varies around the line", ratio > 1.02,
         f"{ratio:.4f}")
    # (6) polygon curvature is exact for a circle
    chk("polygon_curvature(circle) == 1/R",
        float(np.mean(polygon_curvature(loop_polygon(7e-9, 7e-9, 24, EX, EY)))),
        1.0 / 7e-9, 1e-12)
    # (7) the nodal march and the dt/dR quadrature agree on a BASAL loop, which
    #     is the check that `anneal_gf` is the same solve as `anneal_gf_shape`
    fam = fams["c_p"]
    R0, Rend = 25e-9, 5e-9
    tq = float(anneal_gf(R0, fam, T, mat=mat, D=D, R_end=Rend)[0][-1])
    _, _, tn, _ = anneal_gf_shape(R0, fam, T, mat=mat, D=D, R_end=Rend,
                                  n_side=32, n_quad=12)
    chk("nodal march == dt/dR quadrature, c_p 25->5 nm", float(tn[-1]), tq, 5e-2)
    # (8) the calibrated ROM reproduces the mean-field lifetime on a BASAL loop,
    #     where its shape closure is inert -- the residual is the D^(axis)/Dbar
    #     ratio alone.
    fam = fams["c_p"]
    tm = la.anneal_time(R0, fam, T, mat=mat, R_end=Rend)
    Lc = rom_capture_length(fam, R0, mat)
    tr, A, B = anneal_rom(R0, R0, fam, T, mat=mat, D=D, L_cap=Lc, R_end=Rend)
    chk("calibrated ROM == mean field, c_p 25->5 nm", float(tr[-1]), tm, 0.5)
    flag("basal ROM stays circular", abs(B[-1] / A[-1] - 1.0) < 1e-12,
         f"{B[-1]/A[-1]:.10f}")
    print("VERIFY:", "all checks passed" if ok else "FAILURES ABOVE")
    return ok


# ── the figures ──────────────────────────────────────────────────────────────

CF, CP = la.CF, la.CP
CROM = "#b3243c"
CGF = "#0f766e"


def figure_rt(out, R0=25e-9, T_show=873.0, material_file=None):
    """Questions 1 and 2 on one sheet.

    (a) Fig. 3(a) repeated, with the Green's function solution superimposed.
    (b) Z(R): the closed form Eq. (Ziso) against the boundary-integral solve,
        and the one-parameter refit that makes the two coincide.
    (c) what is left over -- the elapsed-time ratio along the anneal, before and
        after the refit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_material(material_file)
    fams = families(mat)
    D = diffusion_tensor(T_show, material_file)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.1))

    fits, traj = {}, {}
    for n, col, lab in (("c_f", CF, r"$c_f$ faulted"),
                        ("c_p", CP, r"$c_p$ perfect")):
        fam = fams[n]
        t, R = la.anneal(R0, fam, T_show, mat=mat, n=241)
        tg, Rg = anneal_gf(R0, fam, T_show, mat=mat, D=D, n=241)
        traj[n] = (t, R, tg, Rg)
        ax[0].plot(t / 3600.0, R * 1e9, "-", color=col, lw=2.2,
                   label=lab + r",  Eq. (Z$^{\rm iso}$)")
        ax[0].plot([t[-1] / 3600.0], [R[-1] * 1e9], "o", ms=5, color=col)
        ax[0].plot(tg / 3600.0, Rg * 1e9, "--", color=col, lw=1.9, alpha=0.95,
                   label=lab + ",  Green's function")
        ax[0].plot([tg[-1] / 3600.0], [Rg[-1] * 1e9], "s", ms=5, color=col,
                   mfc="none")
        ax[0].annotate(f"{la._fmt_time(t[-1])} $\\to$ {la._fmt_time(tg[-1])}",
                       (max(t[-1], tg[-1]) / 3600.0, R[-1] * 1e9),
                       textcoords="offset points", xytext=(-5, 9), ha="right",
                       fontsize=8.0, color=col)
        fits[n] = fit_Ziso(fam, T_show, mat, D=D)
    ax[0].set_xlabel("annealing time  [h]")
    ax[0].set_ylabel(r"loop radius  $R$  [nm]")
    ax[0].set_ylim(0, R0 * 1e9 * 1.06)
    ax[0].set_xlim(0, None)
    ax[0].set_title(r"(a)  $R(t)$ at $T=%g$ K,  $R_0=%g$ nm"
                    % (T_show, R0 * 1e9), fontsize=10, pad=16)
    ax[0].legend(fontsize=7.4, frameon=False, loc="upper right")

    for n, col, lab in (("c_f", CF, r"$c_f$"), ("c_p", CP, r"$c_p$")):
        r = fits[n]
        ax[1].semilogx(r["R"] * 1e9, r["Z_mf"], "-", color=col, lw=2.0,
                       label=lab + r"  $2\pi/\ln(8R/|b|)$")
        ax[1].semilogx(r["R"] * 1e9, r["Z_gf"], "o", ms=4.4, color=col,
                       mfc="none", label=lab + "  Green's function")
        ax[1].semilogx(r["R"] * 1e9, r["Z_fit"], ":", color=col, lw=1.7,
                       label=lab + r"  fit  $\alpha=%.2f$" % r["alpha"])
    ax[1].set_xlabel(r"loop radius  $R$  [nm]")
    ax[1].set_ylabel(r"capture efficiency  $Z^{\rm iso}_k$")
    ax[1].set_title(r"(b)  $Z^{\rm iso}_k$ assumed, and $Z^{\rm iso}_k$ solved",
                    fontsize=10, pad=16)
    ax[1].legend(fontsize=6.9, frameon=False, loc="upper right", ncol=2)

    for n, col, lab in (("c_f", CF, r"$c_f$"), ("c_p", CP, r"$c_p$")):
        t, R, tg, _ = traj[n]
        ratio = tg[1:] / t[1:]
        ax[2].semilogx(R[1:] * 1e9, ratio, "-", color=col, lw=2.0,
                       label=lab + "  as published")
        ax[2].semilogx(R[1:] * 1e9, ratio * fits[n]["factor"], "--", color=col,
                       lw=1.7, label=lab + r"  with $f=%.3f$" % fits[n]["factor"])
    ax[2].axhline(1.0, color="0.4", lw=0.9, zorder=0)
    ax[2].set_xlabel(r"radius reached  $R$  [nm]")
    ax[2].set_ylabel(r"$t_{\rm Green}\,/\,t_{\rm Eq.(isolatedR)}$")
    ax[2].invert_xaxis()
    ax[2].set_title("(c)  elapsed time, ratio along the anneal", fontsize=10,
                    pad=16)
    ax[2].legend(fontsize=7.4, frameon=False, loc="best")

    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return out


def figure_shape(out, T_show=873.0, R0_a=3.6e-9, R0_c=25e-9, n_side=32,
                 material_file=None):
    """Questions 3 and 4: the loop SHAPE as it shrinks, ROM against Green's.

    Three families, chosen because they answer different halves of the question.
    The two PRISMATIC states have a habit plane containing [0001], so their two
    in-plane directions sample different diffusivities and the shape genuinely
    moves -- that is where the ROM's closure ambiguity, Eq. (romattractor), meets
    a solve that makes no closure at all. The BASAL state is the control: its
    habit plane is basal, the tensor restricted to it is isotropic, and both
    constructions must keep it a circle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_material(material_file)
    fams = families(mat)
    D = diffusion_tensor(T_show, material_file)
    p_v = (D[2, 2] / D[0, 0]) ** (1.0 / 6.0)

    cases = (("a_v", R0_a, r"$a_v$ prismatic, vacancy"),
             ("a_i", R0_a, r"$a_i$ prismatic, interstitial"),
             ("c_p", R0_c, r"$c_p$ basal, perfect"))
    fig, ax = plt.subplots(2, 3, figsize=(13.4, 8.2))

    for j, (n, R0, lab) in enumerate(cases):
        fam = fams[n]
        nhat, e1, e2 = habit_frame(fam)
        R_end = 0.15 * R0
        _, snaps, _, _ = anneal_gf_shape(R0, fam, T_show, mat=mat, D=D,
                                         n_side=n_side, R_end=R_end, n_snap=6)
        L_cap = rom_capture_length(fam, R0, mat)
        rom = {}
        for conv, cl in ROM_VARIANTS:
            _, a_, b_ = anneal_rom(R0, R0, fam, T_show, mat=mat, D=D,
                                   L_cap=L_cap, closure=cl, R_end=R_end,
                                   convention=conv)
            rom[(conv, cl)] = (np.sqrt(a_ * b_), a_, b_)

        top = ax[0, j]
        th = np.linspace(0.0, 2.0 * np.pi, 241)
        for k, q in enumerate(snaps):
            alpha = 0.28 + 0.72 * (1.0 - k / max(len(snaps) - 1, 1))
            x = np.append(q @ e2, (q @ e2)[0]) * 1e9
            y = np.append(q @ e1, (q @ e1)[0]) * 1e9
            top.plot(x, y, "-", color=CGF, lw=1.9, alpha=alpha)
            Req = equiv_radius(q, nhat)
            for (conv, cl), st, c in ((("paper", "geometric"), "--", CROM),
                                      (("geometric", "geometric"), "-.",
                                       "#7a5195")):
                Rr, a_, b_ = rom[(conv, cl)]
                i = int(np.argmin(np.abs(Rr - Req)))
                top.plot(b_[i] * np.cos(th) * 1e9, a_[i] * np.sin(th) * 1e9,
                         st, color=c, lw=1.25, alpha=alpha)
        top.set_aspect("equal")
        top.set_xlabel((r"$\hat n_k\times[0001]$  [nm]" if j < 2
                        else r"$[10\bar 1 0]$  [nm]"))
        top.set_ylabel((r"$[0001]$  [nm]" if j < 2
                        else r"$[\bar 2\bar 1\bar 1 0]$  [nm]"))
        top.set_title(f"({'abc'[j]})  {lab}", fontsize=10, pad=13)
        top.grid(alpha=0.22, lw=0.5)
        if j == 0:
            top.plot([], [], "-", color=CGF, lw=1.9, label="Green's function")
            top.plot([], [], "--", color=CROM, lw=1.25,
                     label=r"ROM, Eq. (romselfstress) $\kappa$")
            top.plot([], [], "-.", color="#7a5195", lw=1.25,
                     label=r"ROM, true ellipse $\kappa$")
            top.legend(fontsize=7.0, frameon=False, loc="upper right")

        # The aspect ratio is plotted on a LOG axis because the ROM's shrinking
        # branch has no attractor at all: with A and B advancing at the fixed
        # ratio D^(A)/D^(B) = p_v^3 (or p_v^6), whichever axis is faster reaches
        # zero first and varrho diverges. Eq. (romattractor) is the attractor of
        # the GROWING branch and is not a bound on this one -- drawing it as a
        # reference line here would be reading the wrong branch.
        bot = ax[1, j]
        m = np.array([shape_metrics(q, fam) for q in snaps])
        bot.semilogy(m[:, 0] * 1e9, m[:, 3], "o-", color=CGF, lw=2.1, ms=5,
                     label="Green's function")
        for (conv, cl), st, c in (
                (("paper", "geometric"), "--", CROM),
                (("paper", "normal"), ":", "#ef8a3c"),
                (("geometric", "geometric"), "-.", "#7a5195"),
                (("geometric", "normal"), (0, (1, 1)), "#2e7d32")):
            Rr, a_, b_ = rom[(conv, cl)]
            bot.semilogy(Rr * 1e9, b_ / a_, ls=st, color=c, lw=1.7,
                         label=r"ROM $\kappa$=%s, $D$=%s ($p_v^{%d}=%.3f$)"
                               % ("Eq." if conv == "paper" else "true", cl[:4],
                                  ellipse_rom.CLOSURES[cl],
                                  p_v ** ellipse_rom.CLOSURES[cl]))
        bot.axhline(1.0, color="0.55", lw=0.9, zorder=0)
        bot.invert_xaxis()
        bot.set_xlabel(r"equal-area radius  $\sqrt{A_\ell B_\ell}$  [nm]")
        bot.set_ylabel(r"aspect ratio  $\varrho_\ell=B_\ell/A_\ell$")
        bot.set_title(f"({'def'[j]})  {lab}", fontsize=10, pad=13)
        bot.grid(alpha=0.22, lw=0.5, which="both")
        bot.legend(fontsize=6.7, frameon=False, loc="best")

    fig.suptitle(r"Shape of an isolated loop as it anneals at $T=%g$ K:  "
                 r"$D_c/D_a=%.3f$,  $p_v=%.4f$.   "
                 r"$A_\ell\parallel[0001]$, so $\varrho_\ell>1$ is elongation "
                 r"IN the basal plane" % (T_show, D[2, 2] / D[0, 0], p_v),
                 fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "ziso", "lifetimes", "lcap", "closure",
                 "mechanism", "shape", "figure-rt", "figure-shape"):
        q = sub.add_parser(name)
        q.add_argument("--material", default=None)
        if name not in ("verify", "lifetimes", "closure"):
            q.add_argument("--T", type=float, default=873.0)
        if name in ("lifetimes", "lcap", "figure-rt"):
            q.add_argument("--R0", type=float, default=25.0, help="nm")
        if name.startswith("figure"):
            q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "verify":
        return 0 if verify(a.material) else 1
    if a.cmd == "ziso":
        report_ziso(a.T, material_file=a.material)
    elif a.cmd == "lifetimes":
        report_lifetimes(a.R0 * 1e-9, material_file=a.material)
    elif a.cmd == "lcap":
        report_lcap(a.R0 * 1e-9, a.T, material_file=a.material)
    elif a.cmd == "closure":
        report_closure(material_file=a.material)
    elif a.cmd == "mechanism":
        report_shape_mechanism(a.T, material_file=a.material)
    elif a.cmd == "shape":
        report_shape(a.T, material_file=a.material)
    elif a.cmd == "figure-rt":
        figure_rt(a.out, a.R0 * 1e-9, a.T, a.material)
    elif a.cmd == "figure-shape":
        figure_shape(a.out, a.T, material_file=a.material)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
