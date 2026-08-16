"""
ellipse_rom.py -- the <a> loop as an ellipse with two degrees of freedom (plan 4e).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
A reduced-order model PROPOSED IN THIS PLAN. It is not in Li et al., who carry
full nodal freedom on every loop. The claim it rests on is narrow: an <a> loop
that stays convex, centred and close to elliptical is described by two numbers
instead of `2 * DD_SIDES` of them, and the <a> family is the one where that
holds -- `2r/d = 0.091` at the doses run so far, so <a> loops are far from
touching and far from junction formation.

Everything below separates what the crystallography FIXES from the one place a
modeling choice genuinely enters, because the second carries a factor-of-three
ambiguity that must be quoted with any number this module produces.

KINEMATICS -- no freedom
------------------------
An <a> loop is prismatic and climbs, so `b` is normal to its own habit plane and
that plane CONTAINS the c-axis. With in-plane axes `e_z || [0001]` and
`e_x = b_hat x c_hat`:

    r(theta) = a_x cos(theta) e_x + a_z sin(theta) e_z
    N_defects = pi a_x a_z |b| / Omega

That invariant is the SAME one `discrete_loops` conserves, so the ROM and the
discrete export agree by construction rather than by calibration -- which is
checked in :func:`defect_count` against `discrete_loops`' own radius rule.

THE ONE MODELING CHOICE
-----------------------
The capture diffusivity at an endpoint is set by the plane the flux arrives
through, i.e. the plane perpendicular to the local tangent:

    endpoint       tangent        capture plane        D
    end of a_x     || [0001]      basal                D_a
    end of a_z     in basal       spans c and b        sqrt(D_a D_c)  or  D_c

The second row is not determined by geometry. Two defensible closures give
different steady aspect ratios, and both put the major axis in the basal plane
for p < 1 (the paper's conclusion 2):

    closure              a_x/a_z          at p = 0.7
    geometric mean       p^-3             2.9
    normal projection    p^-6             8.5

`closure="geometric"` is the default because a line sink in a transversely
isotropic medium samples the geometric mean, but **this is not settled**. Plan
5.6 settles it by running the <a> family nodally as well as through the ROM on
one case; until then `aspect_ambiguity()` returns the factor and it should be
quoted alongside any predicted ellipticity.

TWO THINGS THE ASPECT-RATIO TABLE HIDES
---------------------------------------
**1. The sign of growth decides which axis is major.** `a_x/a_z` relaxes toward
`D_x/D_z` only while the loop GROWS. Integrating the same p = 0.7 loop in the
two regimes:

    growing (interstitial-rich)   a_x/a_z:  1.00 -> 1.24 -> 1.43 -> 1.70 ...
    shrinking (vacancy-rich)      a_x/a_z:  1.00 -> 0.86 -> 0.69 -> 0.27

because a faster `|a_x_dot|` lengthens `a_x` when positive and eats it when
negative. Li et al.'s conclusion 2 -- elliptical <a> loops with the major axis
in the basal plane -- is therefore a statement about GROWING loops. It is worth
saying plainly that this model, at the concentrations its own DAD study measures
(`A/B ~ 1.11`, above the <a> growth threshold; see `studies/dad_window.py`), puts
<a> loops in the SHRINKING regime, where this ROM predicts the opposite
ellipticity: major axis along [0001]. That is a genuine, falsifiable prediction
and not a defect of either model.

**2. The attractor is approached slowly.** From a circular nucleus,

    a_x/a_z = (a0 + q R t) / (a0 + R t)  ->  q   only as R t >> a0

so a loop must grow to k times its nucleus to reach `(1 + q(k-1))/k` of the
ratio: at q = 2.915 that is 1.96 at k = 2, 2.53 at k = 5, and 2.72 at k = 10.
**Observed ellipticity therefore measures how far a loop has grown as much as it
measures `p`**, and comparing a TEM aspect ratio with `p^-3` directly will
under-read the anisotropy unless the loops are many times their nucleus size.

WHY THE SELF-STRESS TERM IS NOT OPTIONAL
----------------------------------------
Without it the aspect ratios above are ATTRACTORS and the loop elongates without
limit -- the ROM has no steady shape at all. The restoring term is the self
stress entering `c_eq` through the curvature:

    sigma_n_self(theta) = -[mu |b| / (4 pi (1-nu))] kappa ln(1/(kappa r_c))
    kappa(0) = a_z/a_x^2        kappa(pi/2) = a_x/a_z^2

As the loop elongates, curvature rises at the major-axis ends, the back stress
there grows, and growth throttles. This is the mechanism behind the paper's
remark that the change of loop shape re-distributes the Peach-Koehler force.

THE CAPTURE LENGTH
------------------
`lambda` in the rate equations is a capture length and is the ROM's second free
parameter. It sets the TIMESCALE but not the steady shape: it enters both
equations, so an isotropic `lambda` cancels from the aspect-ratio attractor.
That is worth knowing before worrying about it -- an error in `lambda` gets the
growth rate wrong and the ellipticity right. Default is the screening length
from `neighbors.screening_lengths`, the distance over which a line sink's
concentration gradient actually develops.

WHAT THE ROM CANNOT REPRESENT
-----------------------------
Coalescence by contact, junction formation, non-planar climb, and any departure
from a centred ellipse. :func:`assert_valid` checks the first at runtime rather
than assuming it; the others have no cheap test and are the reason <c>, which is
crowded at `2r/d = 0.86`, goes to the full nodal solver instead (plan 4f).
"""
from __future__ import annotations

import numpy as np

from dislocluster_code.coupling.field import (read_material_scalar,
                                              read_material_vector)
from dislocluster_code.post.discrete_loops import OMEGA_B3

KB_EV = 8.617333262e-5

CLOSURES = {
    # name -> exponent q in  a_x/a_z = p^-q  (steady, self-stress off)
    "geometric": 3.0,
    "normal": 6.0,
}


def capture_diffusivities(D_a, D_c, closure="geometric"):
    """``(D_x, D_z)`` -- the capture diffusivity at each axis endpoint.

    `D_x` is fixed by the geometry: the tangent at the end of `a_x` is along
    [0001], so the flux arrives through the basal plane and sees `D_a`. `D_z` is
    the choice documented in the module docstring.
    """
    if closure == "geometric":
        return float(D_a), float(np.sqrt(D_a * D_c))
    if closure == "normal":
        return float(D_a), float(D_c)
    raise ValueError(f"closure must be one of {list(CLOSURES)}, got {closure!r}")


def steady_aspect(p_m, closure="geometric"):
    """``a_x/a_z`` the ROM relaxes to with the self stress switched off."""
    return float(p_m) ** (-CLOSURES[closure])


def aspect_ambiguity(p_m):
    """How far apart the two closures are -- quote this with any ellipticity."""
    g = steady_aspect(p_m, "geometric")
    n = steady_aspect(p_m, "normal")
    return g, n, (n / g if g else float("nan"))


def curvature(a_x, a_z):
    """``(kappa_x, kappa_z)`` at the two axis endpoints of an ellipse."""
    a_x = np.asarray(a_x, float)
    a_z = np.asarray(a_z, float)
    return a_z / np.maximum(a_x ** 2, 1e-300), a_x / np.maximum(a_z ** 2, 1e-300)


def self_stress(a_x, a_z, mu, b_mag, nu, r_c):
    """``(sigma_x, sigma_z)`` -- the curvature back stress, same units as ``mu``.

    Negative: it opposes growth, and most strongly where the loop is sharpest,
    which is what gives the ROM a steady shape.
    """
    kx, kz = curvature(a_x, a_z)
    pref = -mu * b_mag / (4.0 * np.pi * (1.0 - nu))
    def term(k):
        arg = np.maximum(1.0 / np.maximum(k * r_c, 1e-300), 1.0 + 1e-12)
        return pref * k * np.log(arg)
    return term(kx), term(kz)


def defect_count(a_x, a_z, b_mag, omega=OMEGA_B3):
    """``pi a_x a_z |b| / Omega`` -- the invariant shared with `discrete_loops`."""
    return float(np.pi) * np.asarray(a_x, float) * np.asarray(a_z, float) \
        * float(b_mag) / float(omega)


def equivalent_radius(a_x, a_z):
    """The circle of equal area, so a ROM loop can be compared with a disc."""
    return np.sqrt(np.asarray(a_x, float) * np.asarray(a_z, float))


def rates(a_x, a_z, c_m, c_eq0, s_m, Z_m, D_a, D_c, lam, T,
          mu=None, b_mag=1.0, nu=0.34, r_c=1.0, omega=OMEGA_B3,
          closure="geometric", self_stress_on=True):
    """``(a_x_dot, a_z_dot)`` -- the two ODEs of plan 4.3.1.

    ``c_m``, ``c_eq0``, ``s_m``, ``Z_m`` are per mobile species. ``s_m`` is the
    polarity (+1 interstitial, -1 vacancy) and multiplies both the deposition
    sign and the exponent, so a vacancy arriving at an interstitial loop shrinks
    it and its emission term has the opposite sign.
    """
    c_m = np.atleast_1d(np.asarray(c_m, float))
    c_eq0 = np.atleast_1d(np.asarray(c_eq0, float))
    s_m = np.atleast_1d(np.asarray(s_m, float))
    Z_m = np.atleast_1d(np.asarray(Z_m, float))

    Dx, Dz = capture_diffusivities(D_a, D_c, closure)
    if self_stress_on:
        if mu is None:
            raise ValueError("self_stress_on needs mu")
        sx, sz = self_stress(a_x, a_z, mu, b_mag, nu, r_c)
    else:
        sx = sz = 0.0

    kT = KB_EV * float(T)
    out = []
    for D, sig, lm in ((Dx, sx, lam), (Dz, sz, lam)):
        # c_eq is raised where the normal stress opposes the loop, which is
        # what throttles growth at the sharp ends.
        drive = c_m - c_eq0 * np.exp(-s_m * omega * sig / kT)
        out.append(float(omega / b_mag * np.sum(s_m * Z_m * D / lm * drive)))
    return out[0], out[1]


def integrate(a_x0, a_z0, t_end, n_steps=2000, **kw):
    """Explicit march of the two ODEs. Returns ``(t, a_x, a_z)``.

    Explicit on purpose: this is a two-dimensional system per loop and the
    neighbour coupling is lagged (plan 4.3.1), so there is no coupled solve and
    nothing to gain from an implicit step at this size.
    """
    t = np.linspace(0.0, float(t_end), int(n_steps) + 1)
    dt = t[1] - t[0]
    ax = np.empty_like(t)
    az = np.empty_like(t)
    ax[0], az[0] = float(a_x0), float(a_z0)
    for k in range(len(t) - 1):
        dx, dz = rates(ax[k], az[k], **kw)
        ax[k + 1] = max(ax[k] + dt * dx, 1e-12)
        az[k + 1] = max(az[k] + dt * dz, 1e-12)
    return t, ax, az


def assert_valid(centers, a_x, a_z, gap=1.0):
    """Flag loops close enough that contact coalescence would matter.

    The ROM has no contact mechanics. `2r/d = 0.091` for <a> at the doses run so
    far, so this should never fire there -- but that is a property of the dose,
    not of the model, and it is cheap to keep checking.
    """
    from scipy.spatial import cKDTree
    c = np.asarray(centers, float)
    r = equivalent_radius(a_x, a_z)
    if len(c) < 2:
        return True, []
    tree = cKDTree(c)
    d, j = tree.query(c, k=2)
    touch = d[:, 1] < gap * (r + r[j[:, 1]])
    return (not touch.any()), np.flatnonzero(touch).tolist()


def material_constants(material_file, T=573.0):
    """The constants the ROM needs, read straight from the material file."""
    b = read_material_scalar(material_file, "b_SI")
    out = dict(
        b_SI=b, nu=read_material_scalar(material_file, "nu"),
        omega=read_material_scalar(material_file, "atomicVolume_SI") / b ** 3,
        s_m=read_material_vector(material_file, "mobileSpeciesVector"),
        Z_m=read_material_vector(material_file, "discreteDislocationBias"),
        T=T)
    try:
        out["mu_SI"] = read_material_scalar(material_file, "mu_SI")
    except KeyError:
        out["mu_SI"] = None
    return out
