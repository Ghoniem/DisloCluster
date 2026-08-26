
"""
woo_three_way_bias_comparison.py

Three-way comparison of point-defect bias at prismatic a-loops in alpha-Zr:

    1. Woo (1981): isotropic point-defect model, finite circular loop
    2. Woo (1982): anisotropic saddle-point elastic-dipole model
    3. March-Rico & Wirth (2023): atomistic thermal-drift capture radii

The common bias convention used in ALL plots is Woo's convention

        B = 1 - Z_v / Z_i

so that:
    B > 0  : SIA-biased sink
    B < 0  : vacancy-biased sink

Scientific caveat
-----------------
Woo (1982) did not provide Zr saddle-point elastic-dipole tensors.
Therefore, to numerically evaluate Woo's anisotropic theory for Zr, this
script uses later atomistic saddle-point dipole tensors reported for
hcp Zr by Chulkin, Chernov & Sivak (AIP Conf. Proc. 999, 146-156, 2008).

This is an EXTERNAL INPUT to Woo's 1982 theory, not a value from Woo's
paper.  Both in-basal-plane and out-of-basal-plane saddle configurations
are retained so the sensitivity can be inspected.

The most important consequence is that Woo (1982), populated with those
later Zr saddle tensors, does NOT automatically reproduce the sign pattern
reported by March-Rico & Wirth.  This is scientifically useful: it shows
that "anisotropic elastic saddle-point bias" and "atomistic thermal-drift
capture bias" are closely related ideas but are not numerically identical.

References
----------
C.H. Woo, J. Nucl. Mater. 98 (1981) 279-294.
C.H. Woo, J. Nucl. Mater. 107 (1982) 20-30.
D.A. Chulkin, V.M. Chernov, A.B. Sivak,
AIP Conf. Proc. 999 (2008) 146-156.
J.F. March-Rico, B.D. Wirth,
J. Nucl. Mater. 587 (2023) 154752.
"""

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import ellipk, ellipe


# =============================================================================
# 0. Common constants and Zr parameters
# =============================================================================

KB = 1.380649e-23                 # J/K
ANGSTROM = 1.0e-10

# Representative isotropic elastic properties for alpha-Zr, consistent with
# the Woo-1981 reproduction used previously.
MU_ZR = 3.3e10                    # Pa
NU_ZR = 0.33
B_A = 0.323115e-9                 # a-type Burgers-vector magnitude [m]
OMEGA_ZR = 23.27e-30              # atomic volume [m^3]

# Equivalent spherical point-defect radius used in Woo's continuum model.
R0_ZR = (3.0 * OMEGA_ZR / (4.0 * np.pi)) ** (1.0 / 3.0)


# =============================================================================
# 1. WOO 1981: ISOTROPIC FINITE-LOOP MODEL
# =============================================================================
#
# Woo 1981 treats the point defect as an isotropic center of dilatation /
# contraction.  Vacancy and interstitial loops therefore do not acquire an
# intrinsic loop-character-dependent bias: the same B(r_L) applies to either
# loop character.  Loop-size dependence comes from the finite-loop elastic
# field, effective capture radius, and sink-sink interaction.
#
# The implementation below follows the validated version constructed from the
# uploaded 1981 paper:
#   - Woo Eq. (30): circular-loop interaction energy
#   - spherical symmetrization
#   - Woo Eq. (39): effective-medium sink strength
#
# The half-integer toroidal Q-function normalization in modern mpmath differs
# from the historical convention.  TOROIDAL_Q_SCALE is fixed by one published
# Woo Fig. 10 reference point; the rest of the 1981 curve is predictive.
# =============================================================================

# Woo's Zr relaxation-strain values used in the 1981 discussion.
# IMPORTANT: the vacancy and interstitial strains have opposite signs.
EPS_V_1981 = +0.167
EPS_I_1981 = -0.334

# High-resolution re-digitization of Woo 1981 Fig. 12, T=500 K.
FIG12_X = np.array([
    1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5,
    15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0
])

FIG12_RP_I = np.array([
    0.781, 1.399, 1.939, 2.298, 2.583, 3.167, 3.596, 3.974,
    4.311, 4.614, 4.860, 5.070, 5.219, 5.351, 5.430
])

FIG12_RP_V = np.array([
    0.544, 0.947, 1.289, 1.548, 1.754, 2.140, 2.395, 2.614,
    2.789, 2.930, 3.044, 3.140, 3.224, 3.294, 3.350
])

_rp_i_interp = PchipInterpolator(FIG12_X, FIG12_RP_I)
_rp_v_interp = PchipInterpolator(FIG12_X, FIG12_RP_V)


def woo1981_rp(rL, species, T):
    """
    Effective toroidal capture/pipe radius r_p.

    Woo Fig. 12 supplies the 500-K values to r_L=30b.  Above 30b we hold the
    terminal value constant rather than extrapolating a spline.

    The 500/T factor is a simple extension of Woo's thermal E~kT criterion.
    In the March-Rico comparison range (0.65-5 nm), this is a modest correction.
    """
    x = np.clip(rL / B_A, FIG12_X[0], FIG12_X[-1])

    if species == "i":
        rp_over_b = float(_rp_i_interp(x))
    elif species == "v":
        rp_over_b = float(_rp_v_interp(x))
    else:
        raise ValueError("species must be 'i' or 'v'")

    return rp_over_b * B_A * (500.0 / T)


def woo1981_interaction_energy(r, z, rL, epsilon):
    """
    Woo 1981 Eq. (30): isotropic point-defect interaction with a circular
    pure-edge loop.  E is returned in joules.
    """
    outer = (r + rL)**2 + z**2
    inner = (rL - r)**2 + z**2

    m = 4.0 * r * rL / outer
    m = np.clip(m, 0.0, 1.0 - 1.0e-13)

    K = ellipk(m)
    E = ellipe(m)

    prefactor = (
        -(4.0 / 3.0)
        * (1.0 + NU_ZR) / (1.0 - NU_ZR)
        * MU_ZR * R0_ZR**3 * B_A * epsilon
        / np.sqrt(outer)
    )

    return prefactor * (
        ((rL**2 - r**2 - z**2) / inner) * E + K
    )


# Gauss-Legendre nodes for Woo's spherical symmetrization.
_GL_U, _GL_W = np.polynomial.legendre.leggauss(64)


def woo1981_g(spherical_r, rL, epsilon, T):
    """
    Spherically averaged Boltzmann factor
        g(r) = (1/4pi) integral exp[-E(r,Omega)/kT] dOmega.
    """
    u = _GL_U
    r = spherical_r * np.sqrt(1.0 - u*u)
    z = spherical_r * u

    energies = np.array([
        woo1981_interaction_energy(rr, zz, rL, epsilon)
        for rr, zz in zip(r, z)
    ])

    vals = np.exp(np.clip(-energies / (KB*T), -200.0, 200.0))
    return 0.5 * np.dot(_GL_W, vals)


def _woo1981_toroidal_series(x, nmax=100):
    """
    Half-integer Legendre series entering Woo's transfer velocity.
    """
    mp.mp.dps = 40

    total = (
        mp.legenq(-0.5, 0, x, type=3)
        / mp.legenp(-0.5, 0, x)
    )

    for n in range(1, nmax + 1):
        term = (
            2.0
            * mp.legenq(n - 0.5, 0, x, type=3)
            / mp.legenp(n - 0.5, 0, x)
        )
        total += term

        if n > 8 and abs(term) < 1.0e-12 * abs(total):
            break

    return float(mp.re(total))


# Historical toroidal-Q normalization, previously validated against Woo Fig. 10.
TOROIDAL_Q_SCALE = 10.212116595178147


@lru_cache(maxsize=4096)
def _woo1981_kbar_over_D_cached(rL_over_b, rp_over_b):
    """
    Woo transfer velocity divided by D, returned in units of 1/b.
    """
    r_sigma = rL_over_b + rp_over_b
    x = rL_over_b / rp_over_b

    series = TOROIDAL_Q_SCALE * _woo1981_toroidal_series(x)

    return (
        2.0 / (np.pi * r_sigma**2)
        * np.sqrt(rL_over_b**2 - rp_over_b**2)
        * series
    )


def woo1981_R(NL):
    """Woo 1981 Eq. (33): spherical sink-free-region radius."""
    return (4.0 * np.pi * NL / 3.0) ** (-1.0 / 3.0)


def woo1981_k2(rL, NL, T, species):
    """
    Woo 1981 Eq. (39), effective-medium sink strength k^2.

    The comparison here is restricted to r_L < R/2, so there is NO branch
    switch and therefore no artificial discontinuity.
    """
    R = woo1981_R(NL)

    if rL >= R / 2.0:
        raise ValueError("Woo 1981 Eq. (39) is used only for r_L < R/2.")

    epsilon = EPS_I_1981 if species == "i" else EPS_V_1981

    rp = woo1981_rp(rL, species, T)
    r_sigma = rL + rp

    kbar_over_D = (
        _woo1981_kbar_over_D_cached(
            round(rL / B_A, 8),
            round(rp / B_A, 8),
        )
        / B_A
    )

    gamma = 1.0 / (r_sigma * kbar_over_D)

    cache = {}

    def gfun(r):
        key = round(r / B_A, 8)
        if key not in cache:
            cache[key] = woo1981_g(r, rL, epsilon, T)
        return cache[key]

    # Only finite differences of F and X are required.
    Fdiff = -quad(
        lambda t: t / gfun(t),
        r_sigma, R,
        epsabs=0.0, epsrel=3.0e-5, limit=150,
    )[0]

    Xdiff = quad(
        lambda t: 1.0 / (t*t*gfun(t)),
        r_sigma, R,
        epsabs=0.0, epsrel=3.0e-5, limit=150,
    )[0]

    h_sigma = gfun(r_sigma)

    denominator = (
        (4.0 * np.pi * NL / 3.0) * Fdiff
        + Xdiff
        + (
            4.0 * np.pi * NL * gamma
            / (3.0 * r_sigma * h_sigma)
            * (R**3 - r_sigma**3)
        )
    )

    return 4.0 * np.pi * NL / denominator


def woo1981_bias(rL, NL=1.0e22, T=573.0):
    """
    Woo's bias convention:
        B = 1 - Z_v/Z_i.
    """
    Zi = woo1981_k2(rL, NL, T, "i")
    Zv = woo1981_k2(rL, NL, T, "v")
    return 1.0 - Zv / Zi


# =============================================================================
# 2. WOO 1982: ANISOTROPIC SADDLE-POINT MODEL
# =============================================================================
#
# Woo 1982 introduces the normalized eigenvalues p_alpha of the elastic dipole
# tensor P_ij^s at the MIGRATION SADDLE POINT.
#
# Let
#       P = (1/3) Tr(P_ij^s)
# and
#       p_alpha = P_alpha / P,
# so p1+p2+p3 = 3.
#
# Woo assumes the first principal direction e^(1) is the jump direction.
#
# Eqs. (29a,b):
#       A = (4/5)(1+nu)(1-p1) sign(P Q)
#
#       B = 1.6(1-2nu)^2
#           +1.5 C1 (p_alpha p_alpha - 3)
#           +1.5 C2 (p1^2 - 1)
#           +1.5 C3 (p1 - 1)
#           -0.32(1+nu)^2(1-p1)^2
#
# where p_alpha p_alpha = p1^2+p2^2+p3^2.
#
# In the low-density infinitesimal-loop limit:
#
#       S(A,B) = integral_1^infinity t^-2 exp(A t^-3 - B t^-6) dt
#
# Woo Eqs. (38,39):
#
#   B_vL = 1 - |P_v/P_i|^(1/3) * S(A_i^vL,B_i)/S(A_v^vL,B_v)
#   B_iL = 1 - |P_v/P_i|^(1/3) * S(A_i^iL,B_i)/S(A_v^iL,B_v)
#
# and A_j^iL = - A_j^vL.
#
# IMPORTANT:
# This infinitesimal-loop result has no explicit loop-radius dependence.
# =============================================================================


def woo1982_C_coefficients(nu):
    """Woo 1982 coefficients appearing in Eq. (25)/(29b)."""
    C1 = (11000.0*nu**2 - 7568.0*nu + 7760.0) / 17325.0
    C2 = (2112.0*nu**2 + 4224.0*nu + 42720.0) / 17325.0
    C3 = (25344.0*nu**2 - 82368.0*nu - 16272.0) / 17325.0
    return C1, C2, C3


def woo1982_AB_from_eigenvalues(eigenvalues_eV, nu=NU_ZR):
    """
    Convert three saddle-point dipole eigenvalues to Woo's A and B parameters.

    Parameters
    ----------
    eigenvalues_eV : iterable of length 3
        Principal values of P_ij^s in eV.
        The FIRST eigenvalue must correspond to the jump direction, consistent
        with Woo's assumption h = e^(1).

    Returns
    -------
    dictionary containing:
        trace, Pmean, normalized p-vector,
        A_vloop, A_iloop, Bshape
    """
    eig = np.asarray(eigenvalues_eV, dtype=float)

    if eig.shape != (3,):
        raise ValueError("Need exactly three principal dipole eigenvalues.")

    trace = eig.sum()
    Pmean = trace / 3.0
    p = eig / Pmean

    p1, p2, p3 = p
    C1, C2, C3 = woo1982_C_coefficients(nu)

    p2norm = p1*p1 + p2*p2 + p3*p3

    Bshape = (
        1.6 * (1.0 - 2.0*nu)**2
        + 1.5*C1*(p2norm - 3.0)
        + 1.5*C2*(p1*p1 - 1.0)
        + 1.5*C3*(p1 - 1.0)
        - 0.32*(1.0 + nu)**2*(1.0 - p1)**2
    )

    # Woo Eq. (42) for a vacancy loop.
    A_vloop = (
        (4.0/5.0)
        * (1.0 + nu)
        * (1.0 - p1)
        * np.sign(Pmean)
    )

    # Woo Eq. (40): loop-character reversal.
    A_iloop = -A_vloop

    return {
        "trace": trace,
        "Pmean": Pmean,
        "p": p,
        "A_vloop": A_vloop,
        "A_iloop": A_iloop,
        "Bshape": Bshape,
    }


def woo1982_S(A, Bshape):
    """
    Woo 1982 Eq. (36):
        S(A,B) = integral_1^inf t^-2 exp(A t^-3 - B t^-6) dt.
    """
    value, _ = quad(
        lambda t: t**(-2.0) * np.exp(A*t**(-3.0) - Bshape*t**(-6.0)),
        1.0, np.inf,
        epsabs=1.0e-12, epsrel=1.0e-10, limit=500,
    )
    return value


def woo1982_bias(vacancy_saddle_eigs, interstitial_saddle_eigs, nu=NU_ZR):
    """
    Woo 1982 Eqs. (38)-(39).

    Returns
    -------
    B_vloop, B_iloop, details

    Both values use Woo's convention B = 1 - Z_v/Z_i.
    """
    v = woo1982_AB_from_eigenvalues(vacancy_saddle_eigs, nu)
    i = woo1982_AB_from_eigenvalues(interstitial_saddle_eigs, nu)

    size_factor = abs(v["Pmean"] / i["Pmean"]) ** (1.0/3.0)

    # Vacancy loop, Woo Eq. (38).
    B_vloop = 1.0 - size_factor * (
        woo1982_S(i["A_vloop"], i["Bshape"])
        / woo1982_S(v["A_vloop"], v["Bshape"])
    )

    # Interstitial loop, Woo Eq. (39).
    B_iloop = 1.0 - size_factor * (
        woo1982_S(i["A_iloop"], i["Bshape"])
        / woo1982_S(v["A_iloop"], v["Bshape"])
    )

    details = {
        "vacancy": v,
        "interstitial": i,
        "size_factor": size_factor,
    }

    return B_vloop, B_iloop, details


# -----------------------------------------------------------------------------
# 2a. Validate the Woo-1982 equations against Woo's own Table 3.
# -----------------------------------------------------------------------------
#
# Woo gives normalized p_alpha values and Tr(P) for Cu and Fe.
# Using nu=0.30 and the rounded table entries reproduces the published biases
# closely.  Small residual differences are expected because the printed
# parameters are rounded and Woo also discusses approximations in t_sigma.
#
WOO1982_VALIDATION = {
    "Cu M0": {
        "Pv_trace": 5.61,
        "pv": np.array([-0.73, 0.21, 3.52]),
        "Pi_trace": 68.9,
        "pi": np.array([1.10, 0.87, 1.03]),
        "published": (0.527, 0.327),   # (B_vL, B_iL)
    },
    "Cu M1": {
        "Pv_trace": 3.4,
        "pv": np.array([-1.13, 0.00, 4.13]),
        "Pi_trace": 45.6,
        "pi": np.array([1.14, 0.83, 1.03]),
        "published": (0.498, 0.294),
    },
    "Fe Johnson": {
        "Pv_trace": -2.82,
        "pv": np.array([2.73, 0.135, 0.135]),
        "Pi_trace": 63.1,
        "pi": np.array([1.50, 0.89, 0.61]),
        "published": (0.604, 0.445),
    },
}


def woo1982_bias_from_normalized_table(Pv_trace, pv, Pi_trace, pi, nu=0.30):
    """
    Same Woo-1982 equations, but accepts Woo's published normalized p_alpha
    entries directly.  Used only to validate against Woo Table 3.
    """
    def AB(P_trace, p):
        p = np.asarray(p, dtype=float)
        C1, C2, C3 = woo1982_C_coefficients(nu)
        p1 = p[0]

        Bshape = (
            1.6*(1.0 - 2.0*nu)**2
            + 1.5*C1*(np.dot(p, p) - 3.0)
            + 1.5*C2*(p1*p1 - 1.0)
            + 1.5*C3*(p1 - 1.0)
            - 0.32*(1.0 + nu)**2*(1.0 - p1)**2
        )

        A_v = (
            (4.0/5.0)
            * (1.0 + nu)
            * (1.0 - p1)
            * np.sign(P_trace)
        )

        return A_v, -A_v, Bshape

    Av_v, Ai_v, Bv = AB(Pv_trace, pv)
    Av_i, Ai_i, Bi = AB(Pi_trace, pi)

    factor = abs(Pv_trace / Pi_trace) ** (1.0/3.0)

    B_vL = 1.0 - factor * woo1982_S(Av_i, Bi) / woo1982_S(Av_v, Bv)
    B_iL = 1.0 - factor * woo1982_S(Ai_i, Bi) / woo1982_S(Ai_v, Bv)

    return B_vL, B_iL


# -----------------------------------------------------------------------------
# 2b. Zr saddle-point elastic dipoles used as external input to Woo 1982.
# -----------------------------------------------------------------------------
#
# Chulkin, Chernov & Sivak (2008), Table 1, eV.
#
# Coordinate system in that paper:
#     [2 -1 -1 0], [0 1 -1 0], [0001]
#
# "SP_in" corresponds to migration within the basal plane.
# "SP_out" corresponds to migration between basal planes.
#
# These are NOT from Woo 1982.  They are supplied to Woo's equations as a
# later Zr-specific atomistic estimate of the saddle-point dipole tensor.
#
ZR_SADDLE_DIPOLES = {
    "in_plane": {
        "vacancy": np.array([-9.08, -0.66, -2.81]),   # SP_in V
        "interstitial": np.array([19.03, 21.80, 18.21]),  # SP_in BC
    },
    "out_of_plane": {
        "vacancy": np.array([-9.23, 1.32, -6.66]),    # SP_out V
        "interstitial": np.array([18.74, 17.63, 25.95]), # SP_out BC
    },
}


# =============================================================================
# 3. MARCH-RICO & WIRTH 2023
# =============================================================================
#
# Thermal-drift radial capture radii at 573 K.
# Values are in Angstrom and correspond to the atomistic data underlying
# March-Rico & Wirth (2023).
#
# The comparison uses the toroidal sink-strength approximation reported in the
# paper:
#
#   Z^T = 4 pi^2 R_L rho_L /
#         ln[1 + 8 R_L/(r1 + r_d)]
#
# rho_L cancels from the bias ratio.
#
# March-Rico's paper commonly writes bias as Z_i/Z_v - 1.
# We convert it here to Woo's convention 1 - Z_v/Z_i.
# =============================================================================

MR_R_A = np.array([6.5, 12.0, 20.0, 30.0, 50.0])

MR_CAPTURE_573K = {
    "vacancy_a_loop": {
        "ri": np.array([6.3, 5.7, 5.2, 6.4, 8.0]),
        "rv": np.array([4.8, 5.4, 10.1, 9.5, 14.1]),
    },
    "interstitial_a_loop": {
        "ri": np.array([9.5, 10.1, 15.6, 18.9, 21.8]),
        "rv": np.array([6.6, 6.9, 7.7, 6.7, 7.7]),
    },
}

MR_MONOMER_RADIUS_A = 1.6


def march_rico_toroidal_Z(R_A, capture_A, r1_A=MR_MONOMER_RADIUS_A):
    """March-Rico toroidal sink strength up to a common density factor."""
    return (
        4.0 * np.pi**2 * R_A
        / np.log(1.0 + 8.0*R_A/(r1_A + capture_A))
    )


def march_rico_bias_woo_convention(loop_type):
    """
    Compute March-Rico bias in Woo convention:
        B = 1 - Z_v/Z_i.
    """
    d = MR_CAPTURE_573K[loop_type]

    Zi = march_rico_toroidal_Z(MR_R_A, d["ri"])
    Zv = march_rico_toroidal_Z(MR_R_A, d["rv"])

    return 1.0 - Zv/Zi


# =============================================================================
# 4. Main calculation, validation, and plots
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # A. Validate Woo 1982 implementation against Woo's Table 3.
    # -------------------------------------------------------------------------
    print("="*82)
    print("WOO 1982 ANISOTROPIC FORMULATION: VALIDATION AGAINST TABLE 3")
    print("="*82)
    print("Using nu=0.30 and the rounded dipole parameters printed by Woo.")
    print()
    print("case          calculated B_vL  published B_vL   calculated B_iL  published B_iL")

    for name, d in WOO1982_VALIDATION.items():
        bv, bi = woo1982_bias_from_normalized_table(
            d["Pv_trace"], d["pv"],
            d["Pi_trace"], d["pi"],
            nu=0.30,
        )
        pbv, pbi = d["published"]

        print(
            f"{name:12s}   {bv:10.4f}        {pbv:10.4f}"
            f"        {bi:10.4f}        {pbi:10.4f}"
        )

    # -------------------------------------------------------------------------
    # B. Evaluate Woo 1982 for Zr using later atomistic saddle dipoles.
    # -------------------------------------------------------------------------
    print()
    print("="*82)
    print("WOO 1982 APPLIED TO ZR USING CHULKIN et al. (2008) SADDLE DIPOLES")
    print("="*82)

    woo82_zr = {}

    for path, tensors in ZR_SADDLE_DIPOLES.items():
        bv, bi, details = woo1982_bias(
            tensors["vacancy"],
            tensors["interstitial"],
            nu=NU_ZR,
        )

        woo82_zr[path] = (bv, bi, details)

        print(f"\n{path.replace('_', ' ').title()}")
        print(f"  B_v-loop = {bv:+.5f}")
        print(f"  B_i-loop = {bi:+.5f}")
        print(f"  |P_v/P_i|^(1/3) = {details['size_factor']:.5f}")
        print(f"  vacancy normalized p = {details['vacancy']['p']}")
        print(f"  SIA normalized p     = {details['interstitial']['p']}")

    # We will use the in-plane saddle as the primary Zr comparison because
    # both monovacancies and SIAs in alpha-Zr have important basal-plane
    # migration channels.  The out-of-plane result is retained as sensitivity.
    B82_v_primary = woo82_zr["in_plane"][0]
    B82_i_primary = woo82_zr["in_plane"][1]

    B82_v_out = woo82_zr["out_of_plane"][0]
    B82_i_out = woo82_zr["out_of_plane"][1]

    # -------------------------------------------------------------------------
    # C. Woo 1981 isotropic finite-loop curve over the March-Rico radius range.
    # -------------------------------------------------------------------------
    T = 573.0
    NL = 1.0e22

    radius_nm = np.linspace(0.65, 5.0, 45)

    B81 = np.array([
        woo1981_bias(Rnm*1.0e-9, NL=NL, T=T)
        for Rnm in radius_nm
    ])

    # -------------------------------------------------------------------------
    # D. March-Rico thermal-drift bias.
    # -------------------------------------------------------------------------
    MR_B_i = march_rico_bias_woo_convention("interstitial_a_loop")
    MR_B_v = march_rico_bias_woo_convention("vacancy_a_loop")

    # -------------------------------------------------------------------------
    # E. Print direct comparison at March-Rico radii.
    # -------------------------------------------------------------------------
    print()
    print("="*82)
    print("THREE-WAY ZR COMPARISON, COMMON CONVENTION B = 1 - Z_v/Z_i")
    print("="*82)
    print("Woo-1982 values below use the in-plane Chulkin saddle tensors.")
    print()
    print(
        "R(nm)   Woo81 isotropic   Woo82 i-loop   Woo82 v-loop"
        "   March-Rico i-loop   March-Rico v-loop"
    )

    for RA, bmi, bmv in zip(MR_R_A, MR_B_i, MR_B_v):
        Rnm = 0.1*RA
        b81 = woo1981_bias(Rnm*1e-9, NL=NL, T=T)

        print(
            f"{Rnm:5.2f}       {b81:+9.4f}"
            f"        {B82_i_primary:+9.4f}"
            f"       {B82_v_primary:+9.4f}"
            f"          {bmi:+9.4f}"
            f"          {bmv:+9.4f}"
        )

    # -------------------------------------------------------------------------
    # F. Main comparison plot.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    # Woo 1981: same curve for vacancy and interstitial loop character.
    ax.plot(
        radius_nm,
        B81,
        linewidth=2.0,
        label=r"Woo 1981 isotropic: common $B_L$",
    )

    # Woo 1982: radius-independent infinitesimal-loop values.
    ax.plot(
        radius_nm,
        np.full_like(radius_nm, B82_i_primary),
        linestyle="--",
        linewidth=2.0,
        label=r"Woo 1982 anisotropic: $i$-loop",
    )

    ax.plot(
        radius_nm,
        np.full_like(radius_nm, B82_v_primary),
        linestyle="--",
        linewidth=2.0,
        label=r"Woo 1982 anisotropic: $v$-loop",
    )

    # March-Rico atomistic thermal drift.
    ax.scatter(
        0.1*MR_R_A,
        MR_B_i,
        marker="o",
        s=60,
        label=r"March-Rico 2023: $i$-loop",
    )

    ax.scatter(
        0.1*MR_R_A,
        MR_B_v,
        marker="s",
        s=60,
        label=r"March-Rico 2023: $v$-loop",
    )

    # Show sensitivity of the Woo-1982 Zr extension to choosing out-of-plane
    # rather than in-plane saddle configurations.
    ax.axhspan(
        min(B82_i_primary, B82_i_out),
        max(B82_i_primary, B82_i_out),
        alpha=0.10,
        label="Woo 1982 i-loop: in/out saddle sensitivity",
    )

    ax.axhspan(
        min(B82_v_primary, B82_v_out),
        max(B82_v_primary, B82_v_out),
        alpha=0.10,
        label="Woo 1982 v-loop: in/out saddle sensitivity",
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xlabel("Loop radius (nm)")
    ax.set_ylabel(r"Bias factor $B=1-Z_v/Z_i$")
    ax.set_title(
        "Three-way point-defect bias comparison for a-loops in alpha-Zr\n"
        "Woo 1981 isotropic / Woo 1982 saddle anisotropy / March-Rico 2023"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    fig.savefig(
        "woo_isotropic_anisotropic_march_rico_comparison.png",
        dpi=300,
    )

    # -------------------------------------------------------------------------
    # G. Separate plot emphasizing ONLY loop-character splitting.
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.8))

    # Isotropic Woo has zero intrinsic splitting between loop characters.
    ax2.plot(
        radius_nm,
        np.zeros_like(radius_nm),
        linewidth=2.0,
        label=r"Woo 1981: $B_{iL}-B_{vL}=0$",
    )

    # Woo 1982 splitting is constant in the infinitesimal-loop approximation.
    delta82 = B82_i_primary - B82_v_primary
    ax2.plot(
        radius_nm,
        np.full_like(radius_nm, delta82),
        linestyle="--",
        linewidth=2.0,
        label=r"Woo 1982: $B_{iL}-B_{vL}$",
    )

    deltaMR = MR_B_i - MR_B_v
    ax2.scatter(
        0.1*MR_R_A,
        deltaMR,
        s=60,
        label=r"March-Rico: $B_{iL}-B_{vL}$",
    )

    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_xlabel("Loop radius (nm)")
    ax2.set_ylabel(r"Intrinsic loop-character splitting $B_{iL}-B_{vL}$")
    ax2.set_title("Intrinsic vacancy/interstitial loop bias differential")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()

    fig2.savefig(
        "woo_march_rico_bias_differential.png",
        dpi=300,
    )

    plt.show()


if __name__ == "__main__":
    main()
