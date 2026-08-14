"""
integration_options.py — the two time-integration options, named once.

Both options solve the SAME problem by the SAME operator split. At every
substep the steady mobile field ``C_M*(x)`` is solved for the immobile state
currently held (MoDELib3's ``ClusterDynamicsFEM::solveMobileClusters``, which
has no time derivative at all), then frozen while the immobile population is
advanced. They differ only in how that second, slow step is integrated.

    Option A — "IMEX"
        Implicit-explicit Euler with Patankar stabilization, MoDELib3's own
        nodal scheme. Each species is advanced as

            n <- (n + dt * production) / (1 + dt * loss)

        which treats production explicitly and destruction implicitly. That is
        the Patankar trick: the update is unconditionally positive for any dt
        and any non-negative state, because a positive numerator is divided by
        a numerator greater than one. It is first order and carries no error
        estimate, so accuracy is controlled only by the step size (nSub = 20
        substeps per 1 dpa output step). Runs entirely inside DDomp.

    Option B — "Adaptive"
        SUNDIALS CVODE, adaptive-order adaptive-step BDF, applied at every
        quadrature node with the mobile species frozen. Order 1-5 and the step
        size are both chosen from a local error estimate against rtol/atol, so
        the accuracy is requested rather than inferred. Driven from Python
        (``modelib_coupling.run_immobile_step``) with acc_mode=2, the exact AD
        Jacobian, per-thread workspace reuse and deduplication.

HISTORICAL NAMES
----------------
These were called "standalone" (Option A) and "coupled" (Option B) up to and
including the seeding study. Those names described how each was *driven* rather
than what it *is*, and "coupled" was actively confusing, because both options
are coupled 0-D <-> 3-D in the sense that matters: both re-solve the mobile
field against the evolving immobile state. The word "coupled" is therefore
still correct for the ARCHITECTURE -- the 0-D <-> 3-D field handoff -- and is
retained there. It is no longer a route name.

    standalone -> IMEX       (Option A)
    coupled    -> Adaptive   (Option B)

Use ``OPTIONS`` for anything written to disk: directory names, figure
subdirectories, table columns, plot legends.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationOption:
    key: str                 # short name used in paths and columns
    option: str              # "A" or "B"
    integrator: str          # one line, for a legend or a table caption
    detail: str              # one paragraph, for a report or provenance file
    legacy: str              # the name used before this module existed

    def __str__(self):
        return self.key


IMEX = IntegrationOption(
    key="IMEX",
    option="A",
    integrator="implicit-explicit Euler, Patankar-stabilized (MoDELib3 nodal)",
    detail=(
        "Option A. MoDELib3's own nodal update, n <- (n + dt*production) / "
        "(1 + dt*loss): production explicit, destruction implicit. The "
        "Patankar form makes it unconditionally positive for any step size, "
        "but it is first order and has no error estimate, so accuracy follows "
        "the step size alone (20 substeps per 1 dpa output step). Runs inside "
        "DDomp."),
    legacy="standalone",
)

ADAPTIVE = IntegrationOption(
    key="Adaptive",
    option="B",
    integrator="SUNDIALS CVODE, adaptive-order adaptive-step BDF",
    detail=(
        "Option B. CVODE BDF at every node with the mobile species frozen. "
        "Order (1-5) and step size are both selected from a local error "
        "estimate against rtol/atol, so accuracy is requested rather than "
        "inferred. acc_mode=2, exact AD Jacobian, per-thread workspace reuse, "
        "base+delta protocol, deduplication at 1e-8."),
    legacy="coupled",
)

OPTIONS = {o.key: o for o in (IMEX, ADAPTIVE)}
BY_LEGACY = {o.legacy: o for o in (IMEX, ADAPTIVE)}
BY_OPTION = {o.option: o for o in (IMEX, ADAPTIVE)}


def resolve(name):
    """Accept a current name, a legacy name, or an option letter."""
    if isinstance(name, IntegrationOption):
        return name
    s = str(name).strip()
    for table in (OPTIONS, BY_LEGACY, BY_OPTION):
        if s in table:
            return table[s]
    for table in (OPTIONS, BY_LEGACY):
        for k, v in table.items():
            if k.lower() == s.lower():
                return v
    raise KeyError(f"unknown integration option {name!r}; "
                   f"expected one of {sorted(OPTIONS)} "
                   f"(or legacy {sorted(BY_LEGACY)})")


def describe_all():
    lines = ["Two options for time integration by operator splitting.", ""]
    for o in (IMEX, ADAPTIVE):
        lines += [f"Option {o.option} — {o.key}: {o.integrator}",
                  f"    {o.detail}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_all())
