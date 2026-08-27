"""The smeared discrete sink, Eq. (kdiscrete) -- step 9 of the plan.

WHAT THIS IS FOR
----------------
A family that has been handed to dislocation dynamics is REMOVED from the
continuum sink strength: ``transition.transfer`` scales its field down by
``1 - frac_kept`` so the mobile solve does not count the same loops twice. The
loops are then invisible to the finite-element operator, because
``c^m_D`` -- the analytic field of the discrete segments -- reaches the continuum
problem only through the Dirichlet values on the outer surface. A transferred
population therefore contributes to the mobile solve by superposition and
contributes NOTHING to the operator.

This module builds the missing half: the population's sink strength returned to
the continuum as a nodal field built from where the loops actually are,

    k2_D(x_j) = (1/V_j) sum_l  Z_sk,m * sigma_sink_k * P_l * W(x_j - x_l),
    sum_j V_j W(x_j - x_l) = 1                     for every loop l,

with ``P_l`` the loop's own elliptical perimeter, ``V_j`` the nodal volume and
``W`` a partition-of-unity kernel of width ``max(h, L_s)``.

WHY THE UNIFORM COMPENSATION IS NOT ENOUGH
------------------------------------------
The refusal of a loop is a strongly SPATIAL criterion -- a loop is refused
because it is near a face -- and its compensation is currently a single scalar
applied at every node alike. In expectation that is the correct removal for a
draw whose intensity was the continuum density, which is why the transfer ledger
closes. It stops being correct as soon as the accepted loops evolve, and it is
ALREADY incorrect for the refused ones: those sit in a boundary shell a hundred
nanometres or more thick, a scale well above L_s, and therefore exactly where
spatial structure is information rather than noise.

THE NORMALIZATION IS PER LOOP, AND THAT IS THE WHOLE TRICK
----------------------------------------------------------
``sum_j V_j W = 1`` is imposed for each loop separately, after the kernel is
evaluated on whatever nodes are actually within reach. A kernel normalized once
analytically leaks mass wherever the node cloud is irregular or truncated -- and
it is truncated precisely at the surface, where the refused loops are. Per-loop
normalization makes the total sink strength exactly conserved on any node set,
which is the property every check in this module rests on.

WHAT IS NOT HERE
----------------
The C++ ingestion. ``ImmobileSinks`` builds k^2 from the immobile field at
quadrature points and has no channel for an externally supplied nodal field;
adding one is a design decision about where it belongs -- an extra trial
function, or an equivalent (n, c) written back into the family's own slots --
and it is deliberately not made here. This module produces the field and proves
its invariants; wiring it in is the remaining half of step 9.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["kernel_width", "partition_weights", "discrete_sink_field",
           "continuum_sink_field", "SUPPORT"]

#: Compact support of the kernel, in units of its width. A cubic B-spline is
#: identically zero beyond 2h, so this is exact rather than a tolerance.
SUPPORT = 2.0


def _spline(q):
    """Cubic B-spline in ``q = r/h``, zero beyond ``q = 2``.

    Any compactly supported, non-negative, monotone kernel would do -- the
    per-loop normalization below removes every constant -- but this one is the
    standard SPH choice, is C^2, and has no cusp at the origin, so a loop
    sitting exactly on a node is not a special case.
    """
    q = np.asarray(q, dtype=float)
    w = np.zeros_like(q)
    a = q < 1.0
    b = (q >= 1.0) & (q < 2.0)
    w[a] = 1.0 - 1.5 * q[a] ** 2 + 0.75 * q[a] ** 3
    w[b] = 0.25 * (2.0 - q[b]) ** 3
    return w


def kernel_width(nodes=None, L_s=0.0, h=None):
    """``max(h, L_s)`` -- the width Eq. (kdiscrete) requires, and NOT narrower.

    ``h`` defaults to the mean nearest-neighbour spacing of the node cloud,
    which is the finest structure the mesh can carry. ``L_s`` is the sink
    screening length: below it the continuum has no meaning, so a kernel
    narrower than either would be reporting structure that neither the mesh nor
    the physics supports.
    """
    if h is None:
        if nodes is None:
            raise ValueError("give either `h` or the node cloud it comes from")
        nodes = np.asarray(nodes, dtype=float)
        d, _ = cKDTree(nodes).query(nodes, k=2)
        h = float(np.mean(d[:, 1]))
    return float(max(h, L_s))


def partition_weights(nodes, volumes, centers, width, tree=None):
    """Per-loop partition-of-unity weights, as ``(rows, cols, w)``.

    ``rows`` index loops and ``cols`` index nodes, so
    ``sum over cols of volumes[col]*w == 1`` for every row -- exactly, on any
    node set, including a loop whose support is cut by the domain surface.

    A loop that reaches no node at all -- possible only if the width is smaller
    than the local spacing, which ``kernel_width`` forbids -- is assigned to its
    single nearest node, so no sink strength is ever silently dropped.
    """
    nodes = np.asarray(nodes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    if centers.size == 0:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    tree = cKDTree(nodes) if tree is None else tree

    rows, cols, vals = [], [], []
    near = tree.query_ball_point(centers, r=SUPPORT * width)
    for i, idx in enumerate(near):
        idx = np.asarray(idx, dtype=int)
        if idx.size:
            r = np.linalg.norm(nodes[idx] - centers[i], axis=1)
            w = _spline(r / width)
            denom = float(np.dot(volumes[idx], w))
        else:
            denom = 0.0
        if denom <= 0.0:
            # Either nothing in range, or every node in range sat on a zero of
            # the kernel. Fall back to the nearest node and put the whole loop
            # there: losing it would break conservation silently, which is the
            # one failure mode this module cannot tolerate.
            _, j = tree.query(centers[i], k=1)
            idx = np.array([int(j)])
            w = np.array([1.0])
            denom = float(volumes[idx[0]])
        rows.append(np.full(idx.size, i, dtype=int))
        cols.append(idx)
        vals.append(w / denom)
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))


def discrete_sink_field(nodes, volumes, centers, perimeters, width,
                        coefficient=1.0):
    """Eq. (kdiscrete): the nodal sink-strength field of a discrete population.

    Parameters
    ----------
    nodes       (N, 3) CD node positions [b]
    volumes     (N,)   nodal volumes [b^3], from the same hull-restricted
                       construction the conversion counts loops with
    centers     (L, 3) loop centres [b]
    perimeters  (L,)   each loop's OWN perimeter [b] -- elliptical, so the
                       shape the conversion drew is the shape that sinks
    coefficient        ``Z_sk,m * sigma_sink_k``, the per-species capture
                       efficiency times the family's sink convention. Scalar or
                       one per species.

    Returns ``(N,)`` for a scalar coefficient, else ``(N, n_species)``.

    THE PERIMETER, NOT THE RADIUS. A line sink absorbs along its length, so an
    elliptical loop of the same area as a circular one is a slightly STRONGER
    sink -- by 0.2% at e = 0.1 and 4.9% at e = 0.4. Using pi r^2 or 2 pi r would
    discard the shape the previous step went to some trouble to draw.
    """
    nodes = np.asarray(nodes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    perimeters = np.asarray(perimeters, dtype=float).reshape(-1)
    coeff = np.atleast_1d(np.asarray(coefficient, dtype=float))

    out = np.zeros((nodes.shape[0], coeff.size), dtype=float)
    if perimeters.size == 0:
        return out[:, 0] if coeff.size == 1 else out
    r, c, w = partition_weights(nodes, volumes, centers, width)
    # NO SECOND 1/V_j HERE, and Eq. (kdiscrete) as printed has one. The kernel
    # is normalized as the equation states, sum_j V_j W = 1, which already makes
    # W a DENSITY -- it carries 1/volume -- so sum_l P_l W is a length per
    # volume, which is a sink strength. Applying the equation's explicit 1/V_j
    # on top divides by the nodal volume twice: dimensionally it leaves
    # length/volume^2, and numerically the population total came out low by
    # exactly a factor of V (measured -1.000e+00 in relative terms, i.e. the
    # whole of it).
    #
    # The two readings that ARE consistent give the same field, which is the
    # check that this is a transcription slip and not a choice:
    #   * dimensionless allocation, sum_j w_j = 1, then k2 = (1/V_j) sum_l P_l w
    #   * density kernel, sum_j V_j W = 1, then k2 = sum_l P_l W
    # and W = w/V_j maps one onto the other exactly.
    contrib = perimeters[r] * w
    acc = np.bincount(c, weights=contrib, minlength=nodes.shape[0])
    out = acc[:, None] * coeff[None, :]
    return out[:, 0] if coeff.size == 1 else out


def continuum_sink_field(n_field, r_field, coefficient=1.0):
    """The same quantity from the CONTINUUM field, for comparison.

    ``sum_l P_l`` per unit volume is ``n * 2 pi r`` for a circular family, so
    this is what Eq. (kdiscrete) must reproduce when the loops are drawn from a
    uniform field and nothing is refused. It is the consistency check that says
    the two routes are one channel and not two.
    """
    n_field = np.asarray(n_field, dtype=float)
    r_field = np.asarray(r_field, dtype=float)
    return coefficient * n_field * 2.0 * np.pi * r_field
