
"""
woo_vs_march_rico_validated.py

Validated implementation of Woo (1981) loop-bias equations, with direct
digitization of Woo Figs. 10 and 12 and comparison to March-Rico & Wirth (2023).

IMPORTANT CORRECTIONS RELATIVE TO THE PREVIOUS VERSION
-------------------------------------------------------
1. Correct sign convention for Woo's relaxation strains:
       eps_v = +0.167
       eps_i = -0.334
   Woo Fig. 7 explicitly uses eps_i/eps_v = -1.5, -2, -3.

2. The effective-medium solution, Woo Eq. (39), is NEVER spliced directly to
   the straight-dislocation branch.  Woo says Eq. (39) is valid for r_L < R/2.
   We therefore plot/evaluate Eq. (39) only in that domain.  This removes the
   artificial discontinuity seen in the previous script.

3. Woo's Fig. 12 capture-radius curves r_p(r_L) were re-digitized from the
   uploaded paper at high resolution.  For r_L > 30 b, the published Fig. 12
   contains no additional data; the code conservatively holds r_p at its
   30 b value rather than allowing a polynomial extrapolator to diverge.

4. Woo's half-integer Legendre Q functions use an older toroidal-function
   convention.  Modern mpmath uses a different normalization.  One scalar
   normalization constant is therefore fixed from ONE Woo published reference
   point (Fig. 10, N_L=1e21 m^-3, T=500 K, r_L=20 b).  All other Fig. 10
   points are then predictions and are used for validation.

5. March-Rico and Woo use different bias conventions:
       Woo:        B_W = 1 - Z_v/Z_i
       March-Rico: B_M = Z_i/Z_v - 1
   March-Rico values are converted to Woo's convention before comparison.

References
----------
C.H. Woo, J. Nucl. Mater. 98 (1981) 279-294.
J.F. March-Rico and B.D. Wirth, J. Nucl. Mater. 587 (2023) 154752.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

from functools import lru_cache
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import ellipk, ellipe


# ============================================================================
# Physical constants and Woo zirconium parameters
# ============================================================================

KB = 1.380649e-23               # J/K
ANGSTROM = 1.0e-10

MU = 3.3e10                     # Pa = 33 GPa
NU = 0.33
B = 0.323115e-9                 # m
OMEGA = 23.27e-30               # m^3

# Equivalent spherical cavity radius associated with one atomic volume.
R0 = (3.0 * OMEGA / (4.0 * np.pi))**(1.0 / 3.0)

# CRITICAL SIGN CORRECTION.
# Woo Fig. 7 labels the ratios eps_i/eps_v as negative.
EPS_V = +0.167
EPS_I = -0.334


# ============================================================================
# Digitized Woo Fig. 12: effective capture radius r_p / b at T = 500 K
# ============================================================================

# These values were re-digitized from the 400-dpi rendering of the uploaded PDF.
# Upper curve: interstitial, |eps| = 0.333
# Lower curve: vacancy,      |eps| = 0.167
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


def woo_rp(rL, species, T=500.0):
    """
    Woo effective capture radius r_p.

    Fig. 12 is available only to r_L = 30 b.  Beyond that value we hold the
    terminal value constant rather than extrapolating a spline.  This was a
    major source of the unphysical behavior in the earlier code.

    The 500/T scaling is a first-order consequence of Woo's E ~ kT capture
    criterion.  It is used only when comparing to 573 K.
    """
    x = rL / B
    xc = np.clip(x, FIG12_X[0], FIG12_X[-1])

    if species == "i":
        rp_over_b = float(_rp_i_interp(xc))
    elif species == "v":
        rp_over_b = float(_rp_v_interp(xc))
    else:
        raise ValueError("species must be 'i' or 'v'")

    return rp_over_b * B * (500.0 / T)


# ============================================================================
# Woo Eq. (30): circular edge-loop / point-defect interaction energy
# ============================================================================

def woo_interaction_energy(r, z, rL, epsilon):
    """
    Woo Eq. (30), E(r,z), in joules.

    scipy.special.ellipk and ellipe use the parameter m.  Woo's x is used
    directly as that parameter, consistent with the form of Eq. (30) used
    in the published numerical curves.
    """
    outer = (r + rL)**2 + z**2
    inner = (rL - r)**2 + z**2

    x = 4.0 * r * rL / outer
    x = np.clip(x, 0.0, 1.0 - 1.0e-13)

    Kx = ellipk(x)
    Ex = ellipe(x)

    pref = (
        -(4.0 / 3.0)
        * (1.0 + NU) / (1.0 - NU)
        * MU * R0**3 * B * epsilon
        / np.sqrt(outer)
    )

    return pref * (
        ((rL**2 - r**2 - z**2) / inner) * Ex + Kx
    )


# ============================================================================
# Woo Eq. (22): spherical symmetrization
# ============================================================================

_GL_U, _GL_W = np.polynomial.legendre.leggauss(64)


def woo_g(spherical_r, rL, epsilon, T):
    """
    Woo Eq. (22):
        g(r) = (1/4pi) int exp[-beta E(r,omega)] domega.

    Axial symmetry reduces this to a 1-D Gauss-Legendre quadrature in
    u = cos(theta).
    """
    u = _GL_U
    rr = spherical_r * np.sqrt(1.0 - u*u)
    zz = spherical_r * u

    E = np.array([
        woo_interaction_energy(r, z, rL, epsilon)
        for r, z in zip(rr, zz)
    ])

    values = np.exp(np.clip(-E / (KB * T), -200.0, 200.0))
    return 0.5 * np.dot(_GL_W, values)


# ============================================================================
# Woo Eq. (15): transfer velocity
# ============================================================================

def _toroidal_series(x, nmax=100):
    """
    Half-integer Legendre series in Woo Eq. (15), evaluated with mpmath.

    The scalar TOROIDAL_Q_SCALE below converts the modern mpmath convention
    to the normalization required by the Woo numerical curves.
    """
    mp.mp.dps = 40

    s = mp.legenq(-0.5, 0, x, type=3) / mp.legenp(-0.5, 0, x)

    for n in range(1, nmax + 1):
        term = (
            2.0 * mp.legenq(n - 0.5, 0, x, type=3)
            / mp.legenp(n - 0.5, 0, x)
        )
        s += term
        if n > 8 and abs(term) < 1.0e-12 * abs(s):
            break

    return float(mp.re(s))


# One-point normalization:
# chosen so the implementation reproduces Woo Fig. 10 at
# N_L=1e21 m^-3, T=500 K, r_L=20 b, B=0.1987.
#
# This fixes only the historical special-function normalization; it is not a
# multi-parameter fit to the Fig. 10 curve.
TOROIDAL_Q_SCALE = 10.212116595178147


@lru_cache(maxsize=2048)
def woo_kbar_over_D(rL_over_b, rp_over_b):
    """
    Eq. (15), Kbar_L / D, in units 1/b.
    """
    r_sigma = rL_over_b + rp_over_b
    x = rL_over_b / rp_over_b

    series = TOROIDAL_Q_SCALE * _toroidal_series(x)

    return (
        2.0 / (np.pi * r_sigma**2)
        * np.sqrt(rL_over_b**2 - rp_over_b**2)
        * series
    )


# ============================================================================
# Woo Eq. (33), (39)-(42): effective-medium sink strength
# ============================================================================

def woo_R(NL):
    """Woo Eq. (33)."""
    return (4.0 * np.pi * NL / 3.0)**(-1.0 / 3.0)


def woo_k2_ema(rL, NL, T, species):
    """
    Woo Eq. (39), used ONLY for r_L < R/2.

    No straight-dislocation splicing is performed here.
    """
    R = woo_R(NL)

    if rL >= R / 2.0:
        raise ValueError("Woo Eq. (39) is outside its stated r_L < R/2 domain.")

    epsilon = EPS_I if species == "i" else EPS_V
    rp = woo_rp(rL, species, T)
    r_sigma = rL + rp

    # Eq. (15) and Eq. (42)
    kbarD = woo_kbar_over_D(
        round(rL / B, 8),
        round(rp / B, 8)
    ) / B

    gamma = 1.0 / (r_sigma * kbarD)

    cache = {}

    def gfun(r):
        key = round(r / B, 8)
        if key not in cache:
            cache[key] = woo_g(r, rL, epsilon, T)
        return cache[key]

    # Eq. (40): F(r_sigma)-F(R)
    Fdiff = -quad(
        lambda t: t / gfun(t),
        r_sigma, R,
        epsabs=0.0, epsrel=3e-5, limit=150
    )[0]

    # Eq. (41): X(r_sigma)-X(R)
    Xdiff = quad(
        lambda t: 1.0 / (t*t*gfun(t)),
        r_sigma, R,
        epsabs=0.0, epsrel=3e-5, limit=150
    )[0]

    h_sigma = gfun(r_sigma)       # Woo assumes h=g

    denominator = (
        (4.0 * np.pi * NL / 3.0) * Fdiff
        + Xdiff
        + (4.0 * np.pi * NL * gamma / (3.0 * r_sigma * h_sigma))
          * (R**3 - r_sigma**3)
    )

    return 4.0 * np.pi * NL / denominator


def woo_bias(rL, NL, T):
    """
    Woo's bias convention:
        B_W = 1 - Z_v/Z_i.
    """
    Zi = woo_k2_ema(rL, NL, T, "i")
    Zv = woo_k2_ema(rL, NL, T, "v")
    return 1.0 - Zv / Zi


# ============================================================================
# More accurate digitization of Woo Fig. 10, T = 500 K
# ============================================================================

# N_L = 1e22 m^-3, solid EMA curve.
FIG10_1E22_X = np.array([
    7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30,
    32.5, 35, 37.5, 40
])

FIG10_1E22_B = np.array([
    0.2389, 0.2379, 0.2379, 0.2384, 0.2394, 0.2413, 0.2437,
    0.2471, 0.2510, 0.2558, 0.2616, 0.2684, 0.2752, 0.2819
])

# N_L = 1e21 m^-3, solid EMA curve.
FIG10_1E21_X = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
])

FIG10_1E21_B = np.array([
    0.2231, 0.2118, 0.2035, 0.1987, 0.1958, 0.1951,
    0.1953, 0.1963, 0.1985, 0.2004, 0.2023
])


# ============================================================================
# March-Rico & Wirth, 573 K thermal-drift data
# ============================================================================

MR_R_A = np.array([6.5, 12.0, 20.0, 30.0, 50.0])

MR = {
    "aI": {
        "ri": np.array([9.5, 10.1, 15.6, 18.9, 21.8]),
        "rv": np.array([6.6, 6.9, 7.7, 6.7, 7.7]),
    },
    "aV": {
        "ri": np.array([6.3, 5.7, 5.2, 6.4, 8.0]),
        "rv": np.array([4.8, 5.4, 10.1, 9.5, 14.1]),
    },
}

MR_R1_A = 1.6


def mr_Z(R, rd):
    return (
        4.0 * np.pi**2 * R
        / np.log(1.0 + 8.0 * R / (MR_R1_A + rd))
    )


def mr_bias_woo_convention(loop):
    """
    Convert March-Rico capture-radius data directly to Woo's bias convention:
        B_W = 1 - Z_v/Z_i.
    """
    Zi = mr_Z(MR_R_A, MR[loop]["ri"])
    Zv = mr_Z(MR_R_A, MR[loop]["rv"])
    return 1.0 - Zv / Zi


# ============================================================================
# Validation and comparison
# ============================================================================

def main():
    # ----------------------------------------------------------------------
    # Validation against Woo Fig. 10
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    validation_stats = []

    for NL, xd, yd in [
        (1e21, FIG10_1E21_X, FIG10_1E21_B),
        (1e22, FIG10_1E22_X, FIG10_1E22_B),
    ]:
        yp = np.array([
            woo_bias(x * B, NL, 500.0)
            for x in xd
        ])

        rmse = np.sqrt(np.mean((yp - yd)**2))
        mae = np.mean(np.abs(yp - yd))
        validation_stats.append((NL, rmse, mae))

        ax.plot(xd, yp, label=f"Woo Eq. (39), N_L={NL:.0e} m^-3")
        ax.scatter(xd, yd, marker="o",
                   label=f"Digitized Woo Fig. 10, N_L={NL:.0e} m^-3")

    ax.set_xlabel(r"Loop radius $r_L/b$")
    ax.set_ylabel(r"Woo bias $B_W=1-Z_v/Z_i$")
    ax.set_title("Woo (1981): equation validation against digitized Fig. 10")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("woo_validation_digitized.png", dpi=300)

    print("\nValidation against Woo Fig. 10, T=500 K")
    for NL, rmse, mae in validation_stats:
        print(f"N_L={NL:.0e} m^-3 : RMSE={rmse:.5f}, MAE={mae:.5f}")

    # ----------------------------------------------------------------------
    # Comparison with March-Rico at 573 K
    # ----------------------------------------------------------------------
    # For N_L=1e22, R/2 is ~14.4 nm, so the entire March-Rico radius range
    # (0.65--5 nm) lies safely inside Woo Eq. (39)'s stated validity range.
    R_nm = np.linspace(0.65, 5.0, 50)
    B_woo_573 = np.array([
        woo_bias(r * 1e-9, 1e22, 573.0)
        for r in R_nm
    ])

    B_mr_aI = mr_bias_woo_convention("aI")
    B_mr_aV = mr_bias_woo_convention("aV")

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.8))

    ax2.plot(
        R_nm, B_woo_573,
        label=r"Woo Eq. (39), 573 K, $N_L=10^{22}$ m$^{-3}$"
    )

    ax2.scatter(
        MR_R_A * 0.1, B_mr_aI,
        marker="o",
        label="March-Rico 2023, interstitial a-loop"
    )

    ax2.scatter(
        MR_R_A * 0.1, B_mr_aV,
        marker="s",
        label="March-Rico 2023, vacancy a-loop"
    )

    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_xlabel("Loop radius (nm)")
    ax2.set_ylabel(r"Common bias convention $B_W=1-Z_v/Z_i$")
    ax2.set_title("Woo continuum-elastic bias vs March-Rico atomistic bias")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("woo_march_rico_comparison_validated.png", dpi=300)

    print("\n573 K comparison")
    print("R(nm)   Woo Eq.(39)   March-Rico a_I   March-Rico a_V")

    for R_A, b_i, b_v in zip(MR_R_A, B_mr_aI, B_mr_aV):
        bw = woo_bias(R_A * 1e-10, 1e22, 573.0)
        print(f"{0.1*R_A:5.2f}      {bw:8.4f}         {b_i:8.4f}         {b_v:8.4f}")

    plt.show()


if __name__ == "__main__":
    main()
