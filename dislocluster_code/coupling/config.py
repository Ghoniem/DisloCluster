"""config.py — the march's settings, as a value rather than module state.

``coupling.march`` used to be configured by assigning to its module globals,
and its two callers did exactly that::

    rcs.SUBSTEPS_PER_INTERVAL = int(args.substeps)
    rcs.DOSE_SEED = float(dose_seed)
    ...
    rcs.run_coupled(sim, qdir, seed, snaps, out, fem_every=args.fem_every)

which made ``fem_every`` an argument and ``substeps`` an action at a distance,
and meant two marches could not be configured differently in one process.

WHY ``default_config()`` READS THE GLOBALS AT CALL TIME
-------------------------------------------------------
The globals are still there and still honoured, so the monkey-patch above keeps
working while callers migrate. That only holds because the defaults are read
inside a function. Writing them as dataclass field defaults::

    substeps: int = SUBSTEPS_PER_INTERVAL       # WRONG

binds the value at import, so a later ``rcs.SUBSTEPS_PER_INTERVAL = 5`` becomes
a silent no-op -- a march running 20 substeps where the caller asked for 5,
with no error anywhere. Do not "simplify" it that way.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

__all__ = ["MarchConfig", "default_config"]


@dataclass(frozen=True)
class MarchConfig:
    """Everything the operator-split march needs to know about itself."""

    dose_seed: float = 0.1
    dose_max: float = 20.1
    dose_interval: float = 5.0
    substeps: int = 20
    fem_every: int = 1
    mobile_mode: str = "qssa"
    dedup_rtol: float = 1e-8

    # An explicit snapshot grid. When set it wins over the
    # dose_seed/dose_max/dose_interval arithmetic, which is what a run over a
    # logarithmic grid through the nucleation transient needs.
    snaps: tuple | None = None

    variant_weights: tuple = (1 / 3, 1 / 3, 1 / 3)

    # How many nodes may take an identity step in one substep before the march
    # aborts. None tolerates any number (the behaviour every caller predating
    # this field expected); a whole-batch failure aborts either way.
    max_failed_nodes: int | None = None

    # Coarsening detector thresholds. See post/coarsening.py.
    phi_star: float = 0.20        # keep equal to post/coarsening.PHI_STAR
    frac_star: float = 0.10
    coarsen_hold: int = 2

    # Whether the march ACTS on the detector or only reports it. With
    # discrete_transition off -- the default -- the detector is pure
    # instrumentation and the march is unchanged. On, a family that has crossed
    # phi* is handed to DD at the next fast-solve boundary; see
    # coupling/transition.py.
    #
    # THESE MUST BE FORWARDED BY driver.march_config_from. The march reads its
    # own MarchConfig, not SimulationConfig.coupling, so a field added to one
    # and not the other is silently inert -- which is exactly how the first
    # attempt at this hook ran a whole march with the transition switched off
    # while the launcher printed that it was on.
    # Which immobile formulation the slow step integrates; see SOLVER in
    # config.py. Lives on the MarchConfig because run_coupled builds the slow
    # step's CLI and nothing else can reach it there.
    loop_model: int = 0
    # The implementation plan's model switches, steps 1-5. Every one defaults to
    # the value that reproduces the formulation before its step, so a march that
    # sets none of them is the march that ran before any of them existed.
    #
    #   n_fam          4 legacy | 8 the prismatic vacancy variants (step 1)
    #                  | 9 the stacking-fault pyramid as well (step 4)
    #   chi            character splitting X_iI X_vV/(X_iV X_vI); 1 = none
    #   emission_model 1 replaces the annealing lifetimes by peripheral
    #                  emission against c^{v,eq}_k (step 3)
    #   basal_chain    1 runs c_0 -> c_f -> c_p (step 4)
    #   moments        1 carries the second content moment q and the log-normal
    #                  closure (step 5)
    #
    # model_params carries the numeric inputs those switches need -- eps_sfp,
    # tau_sfp, nu_col, nu_uf, m_min, m_col, m_uf and the rest -- straight
    # through to the solver command line. It is a dict rather than a field per
    # parameter because the C++ already defaults every one of them, and an
    # unrecognized key must fail there rather than be dropped here.
    n_fam: int = 4
    chi: float = 1.0
    emission_model: int = 0
    basal_chain: int = 0
    moments: int = 0
    model_params: dict = field(default_factory=dict)
    discrete_transition: bool = False
    transition_units: tuple = ("c",)
    climb_cutoff_nL: float = 4.0

    # "warn" keeps a capped, unconverged mobile field and carries on -- it has
    # been measured at 54% error in Ci on a single node while the mean still
    # matched to four figures. "raise" refuses it.
    on_unconverged: str = "warn"

    # The frozen-mobile immobile solve.
    rtol: float = 1e-6
    atol: float = 1e-20
    analytic_jac: bool = False

    def __post_init__(self):
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1")
        if self.fem_every < 1:
            raise ValueError("fem_every must be >= 1")
        if self.mobile_mode not in ("qssa", "replay", "seed"):
            raise ValueError(f"mobile_mode {self.mobile_mode!r} is not "
                             f"'qssa', 'replay' or 'seed'")
        if self.on_unconverged not in ("warn", "raise"):
            raise ValueError("on_unconverged must be 'warn' or 'raise'")
        if self.n_fam not in (4, 8, 9):
            raise ValueError(
                f"n_fam must be 4 (legacy), 8 (with the prismatic vacancy "
                f"variants) or 9 (with the pyramid), got {self.n_fam}")
        if self.n_fam != 4 and not self.loop_model:
            raise ValueError(
                "n_fam > 4 needs loop_model = 1: the extra families exist only "
                "in the self-consistent formulation, and the legacy slots name "
                "different quantities")
        if self.basal_chain and self.n_fam != 9:
            raise ValueError("basal_chain = 1 needs n_fam = 9 (the pyramid)")
        if self.moments and not self.loop_model:
            raise ValueError("moments = 1 needs loop_model = 1")
        if self.dedup_rtol < 0:
            raise ValueError("dedup_rtol must be >= 0")
        s = self.dose_grid
        if len(s) < 2:
            raise ValueError(
                f"the march needs at least one interval, i.e. two grid points; "
                f"got {list(s)}. A one-point grid runs no substeps and returns "
                f"an empty timing table, which divides by zero downstream.")
        if any(b <= a for a, b in zip(s, s[1:])):
            raise ValueError(f"the dose grid must increase strictly, got "
                             f"{list(s)}; a decreasing pair runs CVODE "
                             f"backwards")

    @property
    def dose_grid(self):
        """The snapshot doses, explicit list or arithmetic sequence."""
        if self.snaps is not None:
            return tuple(float(x) for x in self.snaps)
        n = int(round((self.dose_max - self.dose_seed) / self.dose_interval))
        return tuple(self.dose_seed + self.dose_interval * i
                     for i in range(n + 1))

    @property
    def n_substeps(self):
        return (len(self.dose_grid) - 1) * self.substeps

    def replace(self, **over):
        return dataclasses.replace(self, **over)

    def to_dict(self):
        d = dataclasses.asdict(self)
        d["dose_grid"] = list(self.dose_grid)
        d["n_substeps"] = self.n_substeps
        return d

    def fingerprint(self):
        """Stable hash of the settings a checkpoint may not be resumed across."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def default_config(**over):
    """A MarchConfig from ``coupling.march``'s globals, read right now.

    See the module docstring: reading them here rather than at import is what
    keeps `rcs.SUBSTEPS_PER_INTERVAL = ...` working.
    """
    from dislocluster_code.coupling import march as _m
    base = dict(dose_seed=_m.DOSE_SEED, dose_max=_m.DOSE_MAX,
                dose_interval=_m.DOSE_INTERVAL,
                substeps=_m.SUBSTEPS_PER_INTERVAL, fem_every=_m.FEM_EVERY,
                mobile_mode=_m.MOBILE_MODE, dedup_rtol=_m.DEDUP_RTOL)
    base.update(over)
    return MarchConfig(**base)
