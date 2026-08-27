"""seed.py — the configuration a march starts from.

Two kinds of seed, and the choice is just the seed dose:

``dose_seed > 0``   the calibrated 0-D state at that dose, integrated in ONE
                    adaptive call with tight tolerances so the seed carries no
                    accumulated step error, then written uniformly onto every
                    CD node.
``dose_seed <= 0``  pristine material: every species at the concentration
                    floor. Not literally zero -- both codes floor their
                    populations, and the loop mean size is carried as
                    ``r = l_a*sqrt(c/N)``, which is 0/0 at exactly zero. The
                    floor is what both models already mean by "no loops".

Seeding EARLIER is safer, not riskier. Mobile transients settle by 1e-4 dpa
while loop nucleation runs over 1e-3 to 1e-2 dpa -- a 100x separation -- so
decreasing the seed dose moves further into the regime where the operator split
is valid. The loop-free limit is not singular.

The mobile columns of a seed are immaterial: the march overwrites them with its
first fast solve before any immobile marching, so a seed's mobile field never
influences the result.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from dislocluster_code import paths
from dislocluster_code.coupling import field as mfield
from dislocluster_code.zerod.cpp_bridge import collect_solver_args

__all__ = ["C_FLOOR", "STATE_NAMES", "pristine_state", "zero_d_state",
           "build_seed", "lumped"]

C_FLOOR = 1.0e-20

STATE_NAMES = ["Cv", "Ci", "C2i", "C3i", "CiL", "CaiL", "CvL", "CavL",
               "CiL_i", "CaiL_i", "CvL_v", "CavL_v"]


def pristine_state():
    """Unirradiated material: every species at the concentration floor."""
    y = np.zeros(19)
    y[0:12] = C_FLOOR
    return y


def zero_d_state(sim, dose, n_points=4000, rtol=1e-10, atol=1e-24,
                 loop_model=0, material_file=None):
    """The calibrated 0-D state vector at `dose`, from one adaptive run.

    Integrated to the requested dose in a single call rather than marched, so
    the seed carries no accumulated step error.

    `loop_model` must match the march's, or the seed is a state vector in the
    other formulation's layout: the same twelve numbers meaning
    aligned/non-aligned rather than <c>/<a1..a3>.
    """
    G = float(sim.input_data.material_params["G"])
    cli = collect_solver_args(sim, dict(
        t_begin=1e-1, t_end=dose / G, n_points=n_points, log_time=True,
        rtol=rtol, atol=atol, analytic_jac=True,
        loop_model=int(loop_model),
        material_file=material_file or paths.MODELIB_MATERIAL))
    p = subprocess.run([str(paths.zrmicro_solver_exe())] + cli,
                       capture_output=True, text=True)
    rows = np.array([[float(x) for x in L.split()]
                     for L in p.stdout.splitlines()
                     if L.strip() and L[0] not in "#[="])
    if rows.size == 0:
        raise RuntimeError(f"0-D solver produced no output:\n{p.stdout[-2000:]}"
                           f"\n{p.stderr[-2000:]}")
    return rows[-1, 1:].copy(), float(rows[-1, 0] * G)


def build_seed(sim, dose, dest, scaffold, variant_weights=(1 / 3, 1 / 3, 1 / 3),
               material_file=None, loop_model=0):
    """Write a uniform seed onto the CD nodes of `scaffold`.

    `scaffold` fixes the node count, so it must come from the SAME staged case
    the march will run in -- the CD node set is mesh-specific and is only known
    after the bootstrap (see :mod:`dislocluster_code.staging.case`).

    Returns ``(y, dose_reached)``.
    """
    if dose <= 0.0:
        y, hit = pristine_state(), 0.0
    else:
        y, hit = zero_d_state(sim, dose, loop_model=loop_model,
                              material_file=material_file)

    ev = mfield.EvlFile(Path(scaffold))
    N = ev.cd.shape[0]
    omega = mfield.cluster_atomic_volume(material_file or paths.MODELIB_MATERIAL)
    y19 = y[:19] if y.size >= 19 else np.r_[y, np.zeros(19 - y.size)]
    ev.cd[:, :mfield.M_SIZE] = y[0:4]
    ev.cd[:, mfield.M_SIZE:] = mfield.immobile_0d_to_modelib(
        np.tile(y19, (N, 1)), omega, variant_weights=variant_weights,
        loop_model=loop_model)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    ev.write(dest)
    return y, hit


def lumped(Y, loop_model=0):
    """The four reported aggregates: <a> and <c> loop density and content.

    The same four numbers in either formulation, but summed over different
    slots: legacy pairs each family with its aligned partner, the
    self-consistent model puts <c> alone in slot 0 and the three prism variants
    in slots 1..3.
    """
    Y = np.atleast_2d(Y)
    if loop_model:
        # Families 4..8 live at 19..28, numbers then contents. <a> is every
        # PRISMATIC family, so the three vacancy variants belong in N_a as much
        # as the interstitial ones do, and <c> is every BASAL one, so c_p
        # belongs in N_c -- both are habits, not polarities. The pyramid (slot
        # 8) is neither: it is not a loop and it is deliberately in neither sum.
        wide = Y.shape[1] >= 29
        n_a = Y[:, 5:8].sum(axis=1)
        c_a = Y[:, 9:12].sum(axis=1)
        n_c, c_c = Y[:, 4], Y[:, 8]
        if wide:
            n_a = n_a + Y[:, 19:22].sum(axis=1)     # a1v..a3v
            c_a = c_a + Y[:, 24:27].sum(axis=1)
            n_c = n_c + Y[:, 22]                    # c_p
            c_c = c_c + Y[:, 27]
        return dict(N_c=n_c, N_a=n_a, c_c=c_c, c_a=c_a)
    return dict(N_a=Y[:, 4] + Y[:, 5], N_c=Y[:, 6] + Y[:, 7],
                c_a=Y[:, 8] + Y[:, 9], c_c=Y[:, 10] + Y[:, 11])
