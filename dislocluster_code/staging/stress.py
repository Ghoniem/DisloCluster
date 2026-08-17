"""One applied load, resolved onto the loop habit planes.

WHY THIS MODULE EXISTS
----------------------
The applied stress had TWO independent sources of truth and they disagreed.

``BOUNDARY['applied_stress_MPa']`` goes into ``ElasticDeformation.txt`` and
drives the 3-D elastic solve. The 0-D slow step instead read ``sigma_n`` from
the workbook's ``Material_Environment`` sheet, where it has stood at
``1.0e8 Pa`` -- **100 MPa** -- and it used that for the aligned/non-aligned loop
split and for the stress-dependent vacancy emission. Nothing reconciled them, so
every run in the 500 nm series was simultaneously

    at 100 MPa   in the 0-D loop alignment (f_a = 0.4015) and emission, and
    at   0 MPa   in the 3-D elastic solve, which logged
                 "elastic : SKIPPED -- every applied load is zero".

The symptom that exposed it: at zero applied stress the aligned fraction must be
exactly 1/3, because ``f_a = (1 + 2f)/3`` with
``f = (exp(x) - 1)/(exp(x) + 2)`` and ``x = sigma_n * Omega / kT`` gives
``f = 0`` at ``sigma_n = 0``. It was 0.4015.

THE TWO IMPLEMENTATIONS WERE ALREADY THE SAME FORMULA
-----------------------------------------------------
``coupling/export._variant_weights`` gives Boltzmann weights over the three
prism variants, ``w_k = exp(sigma_k Omega/kT) / sum_j exp(sigma_j Omega/kT)``.
Algebraically,

    f_a = (1 + 2f)/3 = exp(x)/(exp(x) + 2)

which is exactly ``w_aligned`` when the other two variants see zero resolved
stress. So the 0-D two-family split is the three-variant Boltzmann weighting in
disguise, and unifying them is an identity rather than a modelling choice. That
is verified in ``self_test`` below to 1e-15.

CONVENTION
----------
Prismatic <a> loops are edge loops with Burgers vector along <11-20>, so their
habit normal is parallel to b; basal <c> loops have normal [0001]. The three
<a> normals sit at 120 deg in the basal plane. Same basis as
``coupling/export._DEFAULT_BASIS``, which this module now shares rather than
duplicating.

``sigma_n`` for the legacy two-family split is the resolved normal stress of the
MOST FAVORED variant measured against the mean of the other two. That reference
choice is what makes the legacy formula exact in the case it was written for --
one variant loaded, the other two degenerate -- and it is the only choice that
sends a hydrostatic or c-axis load to zero bias, which is correct: neither
distinguishes the three prism variants.
"""
from __future__ import annotations

import numpy as np

_KB_EV = 8.617e-5          # eV/K, as input_data uses
_EV_J = 1.602e-19

# a1/a2/a3 prism habit normals and the basal normal, c along z. Imported from
# the export module so there is ONE basis, not two that can drift.
from dislocluster_code.coupling.export import _DEFAULT_BASIS  # noqa: E402


def habit_normals(orientation=None):
    """``(A, c)`` with ``A`` the three unit <a> habit normals and ``c`` the basal one."""
    b = dict(_DEFAULT_BASIS) if orientation is None else dict(orientation)
    A = np.array([b["a1"], b["a2"], b["a3"]], dtype=float)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    c = np.asarray(b["c"], dtype=float)
    return A, c / np.linalg.norm(c)


def voigt_to_tensor(voigt_MPa):
    """Voigt ``(s11,s22,s33,s23,s13,s12)`` in MPa -> a 3x3 tensor in Pa.

    The Voigt order is the one `config._voigt` validates and
    `staging/inputs` writes to ElasticDeformation.txt, so the same six numbers
    mean the same thing on both sides of the coupling.
    """
    v = np.asarray(voigt_MPa, dtype=float).ravel()
    if v.size != 6:
        raise ValueError(f"expected 6 Voigt components, got {v.size}")
    s11, s22, s33, s23, s13, s12 = v * 1.0e6          # MPa -> Pa
    return np.array([[s11, s12, s13],
                     [s12, s22, s23],
                     [s13, s23, s33]], dtype=float)


def resolved(voigt_MPa, orientation=None):
    """Resolved normal stresses ``(sigma_a1, sigma_a2, sigma_a3, sigma_c)`` [Pa]."""
    sig = voigt_to_tensor(voigt_MPa)
    A, c = habit_normals(orientation)
    sa = np.einsum("ki,ij,kj->k", A, sig, A)
    sc = float(c @ sig @ c)
    return sa, sc


def variant_weights(voigt_MPa, Omega, T, orientation=None):
    """Boltzmann weights over the three prism variants; equal thirds at zero load."""
    sa, _ = resolved(voigt_MPa, orientation)
    kT = _KB_EV * float(T) * _EV_J
    x = sa * float(Omega) / kT
    x = x - x.max()                       # stabilize
    w = np.exp(x)
    return w / w.sum()


def sigma_n(voigt_MPa, orientation=None):
    """The scalar the legacy two-family split needs, in Pa.

    ``max_k sigma_k - mean(the other two)``. Zero for any load that does not
    distinguish the three prism variants -- hydrostatic, or uniaxial along
    [0001] -- which is the property that matters, because such a load must leave
    ``f_a`` at 1/3.
    """
    sa, _ = resolved(voigt_MPa, orientation)
    k = int(np.argmax(sa))
    others = np.delete(sa, k)
    return float(sa[k] - others.mean())


def sigma_h(voigt_MPa):
    """Hydrostatic stress ``tr(sigma)/3`` in Pa."""
    return float(np.trace(voigt_to_tensor(voigt_MPa)) / 3.0)


def overrides(voigt_MPa, orientation=None):
    """``{'sigma_n': .., 'sigma_h': ..}`` in Pa, for ``build_sim(extra=...)``."""
    return dict(sigma_n=sigma_n(voigt_MPa, orientation),
                sigma_h=sigma_h(voigt_MPa))


def f_a(voigt_MPa, Omega, T, orientation=None):
    """The legacy aligned fraction implied by this load — exactly 1/3 at zero."""
    x = sigma_n(voigt_MPa, orientation) * float(Omega) / (_KB_EV * float(T) * _EV_J)
    f = (np.exp(x) - 1.0) / (np.exp(x) + 2.0)
    return (1.0 + 2.0 * f) / 3.0


def self_test(Omega=2.326552782049238e-29, T=573.0):
    """Check the identity and the limits. Returns a list of (name, ok, detail)."""
    out = []

    zero = (0.0,) * 6
    out.append(("zero load -> f_a = 1/3",
                abs(f_a(zero, Omega, T) - 1 / 3) < 1e-15,
                f"{f_a(zero, Omega, T):.16f}"))
    out.append(("zero load -> equal thirds",
                np.allclose(variant_weights(zero, Omega, T), 1 / 3, atol=1e-15),
                str(variant_weights(zero, Omega, T))))

    # Hydrostatic and c-axis loads do not distinguish the prism variants.
    for name, v in (("hydrostatic", (100.0, 100.0, 100.0, 0, 0, 0)),
                    ("uniaxial [0001]", (0.0, 0.0, 100.0, 0, 0, 0))):
        out.append((f"{name} -> f_a = 1/3",
                    abs(f_a(v, Omega, T) - 1 / 3) < 1e-12,
                    f"{f_a(v, Omega, T):.16f}"))

    # THE IDENTITY: f_a is the Boltzmann weight of the aligned variant when the
    # other two are degenerate, which is what a uniaxial load along a1 gives
    # (sigma_2 = sigma_3 = sigma/4 by symmetry, so the reference cancels).
    v = (100.0, 0.0, 0.0, 0, 0, 0)
    w = variant_weights(v, Omega, T)
    out.append(("uniaxial [10-10] -> f_a == max Boltzmann weight",
                abs(f_a(v, Omega, T) - w.max()) < 1e-12,
                f"f_a={f_a(v, Omega, T):.12f} w_max={w.max():.12f}"))
    out.append(("the two non-favored variants stay degenerate",
                abs(np.sort(w)[0] - np.sort(w)[1]) < 1e-15,
                str(np.sort(w))))

    # THE WORKBOOK'S sigma_n WAS NOT A UNIAXIAL APPLIED STRESS, whatever its
    # label said. Fed 100 MPa, the old code put x = sigma_n*Omega/kT directly
    # and got f_a = 0.401549. A real 100 MPa uniaxial load along a1 resolves to
    # (100, 25, 25) MPa on (a1, a2, a3) -- the off-axis normals see
    # cos^2(120 deg) = 1/4 -- so the DIFFERENTIAL stress that distinguishes the
    # variants is 75 MPa and f_a = 0.384013. The workbook was therefore
    # consuming its value as a differential resolved stress while calling it a
    # uniaxial applied stress, so even a deliberate 100 MPa would have produced
    # a bias 33% larger than intended. Asserted here as the INEQUALITY it is,
    # so the distinction cannot quietly be lost again.
    direct = (1.0 + 2.0 * ((np.exp(0.2941302439022814) - 1)
                           / (np.exp(0.2941302439022814) + 2))) / 3.0
    out.append(("100 MPa uniaxial != sigma_n = 100 MPa (differential is 75)",
                abs(f_a(v, Omega, T) - direct) > 1e-3,
                f"resolved {f_a(v, Omega, T):.6f} vs direct {direct:.6f}"))
    out.append(("sigma_n(100 MPa uniaxial along a1) == 75 MPa",
                abs(sigma_n(v) - 75.0e6) < 1.0,
                f"{sigma_n(v) / 1e6:.6f} MPa"))
    return out


if __name__ == "__main__":
    print(f"{'check':58} {'ok':>5}  detail")
    bad = 0
    for name, ok, detail in self_test():
        print(f"{name:58} {'OK' if ok else 'FAIL':>5}  {detail}")
        bad += not ok
    raise SystemExit(bad)
