"""
transition.py -- the runtime continuum -> discrete transition, and its ledger.

WHAT THIS IS (plan 4b and 4c)
-----------------------------
`post/coarsening.py` decides WHEN the continuum description of a loop family
stops being adequate: it reports `d_coarsen`, the dose at which the Avrami
overlap fractions `phi_LL`/`phi_LN` cross `phi*`. This module performs the
switch itself -- it converts a family's continuum field into a discrete loop
population, removes that family from the continuum, and audits that nothing was
created or destroyed in the process.

It is deliberately PYTHON and deliberately BETWEEN march intervals. Nothing
here needs a MoDELib rebuild, and every transfer is inspectable as a table
before any of it is wired into a solve.

WHY THE LEDGER IS THE POINT
---------------------------
Once a loop is discrete it is a sink through the Green's function. If it is
ALSO still in the continuum immobile field it is a sink through `ImmobileSinks`
as well, and it absorbs twice. The same bug has a subtler second face:
`DislocationQuadraturePoint::cCD` samples the continuum mobile field AT the
loop line, so a loop whose own sink is still in the continuum sees its own
depletion twice over. Zeroing the transferred family fixes both at once, and
they are the same bug.

So the transfer is all-or-nothing per family. The continuum carries ONE mean
size per family per node, so a size-selective transfer is not representable --
there is no spectrum to split. Removing a fraction of the loops would have to
remove content in the same proportion to leave `r_bar` alone, which is the same
as transferring everything and is more bookkeeping.

WHAT IS TRANSFERABLE, AND WHY <a> IS ONE UNIT
---------------------------------------------
The 0-D state lumps the three prismatic variants: `CiL + CaiL` is ONE <a>
population, and `immobile_0d_to_modelib` splits it across a1/a2/a3 by
`variant_weights` only on the way out. There is therefore no state to zero for
`a1` alone. The transferable units are:

    "c"   vacancy basal loops        CvL, CavL, CvL_v, CavL_v
    "a"   interstitial prismatic     CiL, CaiL, CiL_i, CaiL_i  (all 3 variants)

THE THREE INVARIANTS
--------------------
Across a transfer, for each family, totals over the whole crystal regardless of
which carrier holds them:

    (I)   loop number     N = sum_j n_j V_j            + count of discrete loops
    (II)  stored defects  C = sum_j (c_j/Omega) V_j    + sum_i pi r_i^2 |b| / Omega
    (III) sink strength   S = sum_j 2 pi r_j n_j V_j   + sum_i 2 pi r_i

(I) and (II) are conservation statements and must hold to sampling error --
`sample_family` rounds the expected count to an integer, so the tolerance is
one loop, not zero. (III) is a MODELING statement, and it is the one that
catches mistakes: see `SINK_DISCONTINUITY` below.

USAGE
-----
    from dislocluster_code.coupling import transition as tr
    Y2, pops, ledger = tr.transfer(Y, nodes, omega, keys=("c",))
    print(tr.format_ledger(ledger))
"""
from __future__ import annotations

import numpy as np

from dislocluster_code.coupling.field import IDX, immobile_0d_to_modelib
from dislocluster_code.post.discrete_loops import (
    FAMILIES, OMEGA_B3, LoopPopulation, populate, domain_weights,
    write_microstructure, write_table)

# The 0-D columns each transferable unit owns: (number columns, content columns).
# Zeroing a unit means zeroing all four, which is what keeps `Delta c / Delta n`
# equal to `c/n` and hence leaves `r_bar` -- and invariant (III) -- alone.
UNITS = {
    "c": dict(n=("CvL", "CavL"), c=("CvL_v", "CavL_v"),
              families=("c",), label="<c> vacancy basal"),
    "a": dict(n=("CiL", "CaiL"), c=("CiL_i", "CaiL_i"),
              families=("a1", "a2", "a3"), label="<a> interstitial prismatic"),
}

FAM_BY_KEY = {f["key"]: f for f in FAMILIES}


def cd_block(Y, omega, variant_weights=(1 / 3, 1 / 3, 1 / 3)):
    """The ``(N,12)`` CD block `populate` indexes, from a live march state.

    `movies.cd_blocks` builds this from a finished run; the march has `Y` in
    hand and needs the same layout without going through a run directory.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    return np.column_stack([Y[:, :4],
                            immobile_0d_to_modelib(Y, omega, variant_weights)])


# ── the three invariants ─────────────────────────────────────────────────────
def continuum_totals(F, vol, fam):
    """``(N, C, S)`` held by the CONTINUUM for one family.

    ``vol`` is the per-node volume in b^3 (``weights * box_volume``). The radius
    is the continuum's own: ``r = sqrt(m Omega / (pi b_cd))`` with
    ``m = c/(n Omega)`` the defects per loop -- `b_cd`, not `b_dd`, because that
    is the Burgers magnitude `ClusterDynamicsFEM` forms its sink strength with.
    """
    n = np.asarray(F[:, fam["ncol"]], float)
    c = np.asarray(F[:, fam["ccol"]], float)
    ok = np.isfinite(n) & np.isfinite(c) & (n > 0) & (c > 0)
    N = float(np.sum(np.where(ok, n * vol, 0.0)))
    C = float(np.sum(np.where(ok, c / OMEGA_B3 * vol, 0.0)))
    m = np.zeros_like(n)
    m[ok] = c[ok] / (n[ok] * OMEGA_B3)
    r = np.sqrt(np.maximum(m, 0.0) * OMEGA_B3 / (np.pi * fam["b_cd"]))
    S = float(np.sum(np.where(ok, 2.0 * np.pi * r * n * vol, 0.0)))
    return N, C, S


def discrete_totals(pop):
    """``(N, C, S)`` held by a DISCRETE population.

    Note `stored_defects` uses ``b_dd``: `sample_family` sizes a loop by the DD
    Burgers magnitude. That used to be half the CD one for <c> and is now equal
    to it -- both sides carry ``|b| = c/2``. Keep the two names distinct anyway:
    they are read from different places (`FAMILIES` and the material file's
    lattice basis), and the ledger's job is to notice if they ever diverge
    again -- see `sink_discontinuity`.
    """
    if len(pop) == 0:
        return 0.0, 0.0, 0.0
    return (float(len(pop)), float(pop.stored_defects),
            float(2.0 * np.pi * np.sum(pop.radii)))


def sink_discontinuity(fam, material=None):
    """How much a loop's absorption jumps at the moment it becomes discrete.

    TWO independent factors stack, and both must be a deliberate choice rather
    than a discovered surprise:

    1. **The Burgers magnitude.** The continuum sizes a loop with ``b_cd`` and
       the discrete side with ``b_dd``. Since ``r ~ 1/sqrt(b)`` at fixed stored
       defects and sink strength is linear in `r`, (III) jumps by
       ``sqrt(b_cd/b_dd)``.

       **THIS IS NOW 1 FOR EVERY FAMILY.** It used to be ``sqrt(2) = 1.414``
       for <c>, because only the discrete side treated a basal loop as the
       ``1/2[0001]`` loop it physically is -- the continuum sized it with the
       full ``[0001]``. Both now carry ``|b| = c/2 = 2.575 A``, so the
       discontinuity this factor measured has been removed rather than
       accounted for. <a> was always 1.

    2. **The bias parameter.** `loopSinkScale` (continuum, 0.291528 for <c>)
       and `discreteDislocationBias` (discrete, 1.0) are unrelated numbers
       serving the same role. Left as they are, absorption jumps by
       ``1/0.291528 = 3.43``.

    Returned as ``(factor_b, factor_bias, total)``. With factor 1 removed the
    <c> product is 3.43, down from about 4.85 -- still not a rounding error,
    and still exactly what invariant (III) exists to surface.
    """
    fb = float(np.sqrt(fam["b_cd"] / fam["b_dd"]))
    scale = None
    if material is not None:
        # PER FAMILY, and it genuinely differs: 0.291528 for <c> against
        # 0.792317 for the three <a> variants. Reading it with
        # read_material_scalar would apply the <c> value to everything.
        from dislocluster_code.coupling.field import read_material_vector
        try:
            vals = read_material_vector(material, "loopSinkScale",
                                        len(FAMILIES))
            scale = float(vals[[f["key"] for f in FAMILIES].index(fam["key"])])
        except Exception:
            scale = None
    fbias = (1.0 / scale) if scale else float("nan")
    return fb, fbias, fb * fbias


# ── the transfer ─────────────────────────────────────────────────────────────
def scale_unit(Y, key, factor=0.0):
    """A copy of ``Y`` with one transferable unit's continuum field scaled.

    ``factor=0`` removes the family entirely, which is the whole-family transfer
    of plan 4.5. A non-zero factor is what a PARTIAL transfer needs: when the
    microstructure generator refuses some loops (see :func:`split_by_fit`),
    their defects must stay in the continuum, so the family is scaled to the
    refused fraction instead of zeroed.

    BOTH the number and the content columns take the same factor, which is what
    keeps `Delta c / Delta n = c/n` and therefore leaves the mean radius -- and
    invariant (III) -- unchanged. Scaling only one of them would silently
    resize every remaining loop.
    """
    if key not in UNITS:
        raise KeyError(f"{key!r} is not transferable; choose from {list(UNITS)}")
    Y = np.array(Y, dtype=float, copy=True)
    u = UNITS[key]
    for name in u["n"] + u["c"]:
        Y[:, IDX[name]] *= float(factor)
    return Y


def zero_unit(Y, key):
    """A copy of ``Y`` with one transferable unit's continuum field removed."""
    return scale_unit(Y, key, 0.0)


def transfer(Y, nodes, omega, keys=("c",), weights=None, faces=None,
             region="domain", variant_weights=(1 / 3, 1 / 3, 1 / 3),
             coalesce_pass=True, seed=0, material=None, mc_samples=2_000_000,
             fit_filter=True, conserve_defects=True):
    """Convert the named units to discrete loops and remove them from `Y`.

    Returns ``(Y_after, {family_key: LoopPopulation}, ledger)``.

    ``region="domain"`` is the default here, unlike in `discrete_loops.build`
    where "interior" is wanted for a bulk DD cell: a transfer must carry the
    WHOLE population, boundary shell included, or the invariants cannot close.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    nodes = np.asarray(nodes, float)
    if weights is None:
        weights, faces = domain_weights(nodes, mc_samples)

    F = cd_block(Y, omega, variant_weights)
    pops, stats, box, box_volume, _L, raws = populate(
        nodes, F, weights, faces, region=region,
        coalesce_pass=coalesce_pass, seed=seed, return_raw=True)
    by_key = {f["key"]: p for f, p in zip(FAMILIES, pops)}
    raw_by_key = {f["key"]: p for f, p in zip(FAMILIES, raws)}
    vol = np.asarray(weights, float) * box_volume

    Y_after = Y
    out, rows = {}, []
    for key in keys:
        u = UNITS[key]
        # Loops the generator will refuse must keep their share of the
        # continuum, so the fit is predicted BEFORE anything is zeroed. The
        # fraction is by stored defects and is taken over the whole unit --
        # <a> is three families sharing one 0-D population, so one factor.
        kept_defects = 0.0
        all_defects = 0.0
        if fit_filter:
            for fk in u["families"]:
                kept, _refused, _f = split_by_fit(by_key[fk], faces)
                kept_defects += kept.stored_defects
                all_defects += by_key[fk].stored_defects
                by_key[fk] = kept
            frac_kept = (kept_defects / all_defects) if all_defects > 0 else 1.0
        else:
            frac_kept = 1.0

        # Make the transfer conserve defects EXACTLY. The drawn population
        # misses the continuum total by the integer rounding in sample_family
        # -- 15.4% at six loops -- so each kept population is rescaled to the
        # share of the continuum it is supposed to carry. Skipped when
        # conserve_defects=False, which reproduces the raw draw.
        if conserve_defects:
            for fk in u["families"]:
                fam0 = FAM_BY_KEY[fk]
                _N, C_cont, _S = continuum_totals(F, vol, fam0)
                by_key[fk], _s = rescale_to_defects(by_key[fk],
                                                    frac_kept * C_cont)

        for fk in u["families"]:
            fam = FAM_BY_KEY[fk]
            pop, raw = by_key[fk], raw_by_key[fk]
            N0, C0, S0 = continuum_totals(F, vol, fam)
            # (I) and (II) are checked at the TRANSFER, i.e. against the
            # pre-coalescence draw. Coalescence is a separate physical step
            # that conserves area and deliberately reduces count, so comparing
            # the coalesced count with the continuum measures coarsening, not
            # conservation.
            Nr, Cr, Sr = discrete_totals(raw)
            N1, C1, S1 = discrete_totals(pop)
            fb, fbias, ftot = sink_discontinuity(fam, material)
            rows.append(dict(
                family=fk, unit=key,
                N_continuum=N0, N_raw=Nr, N_discrete=N1,
                C_continuum=C0, C_raw=Cr, C_discrete=C1,
                S_continuum=S0, S_raw=Sr, S_discrete=S1,
                dN=Nr - N0, dC_rel=(Cr - C0) / C0 if C0 else float("nan"),
                S_ratio_transfer=(Sr / S0) if S0 else float("nan"),
                S_ratio_coalesced=(S1 / S0) if S0 else float("nan"),
                coalescence_N=(N1 / Nr) if Nr else float("nan"),
                coalescence_area=(C1 / Cr) if Cr else float("nan"),
                f_burgers=fb, f_bias=fbias, f_total=ftot,
                frac_kept=frac_kept,
                C_discrete_kept=float(by_key[fk].stored_defects)))
            out[fk] = pop
        # Scale, not zero, by whatever the discrete side could not take.
        Y_after = scale_unit(Y_after, key, 1.0 - frac_kept)

    # After zeroing, the continuum must hold nothing of the transferred unit.
    F_after = cd_block(Y_after, omega, variant_weights)
    residual = {}
    for key in keys:
        for fk in UNITS[key]["families"]:
            residual[fk] = continuum_totals(F_after, vol, FAM_BY_KEY[fk])

    return Y_after, out, dict(rows=rows, residual=residual,
                              box_volume=box_volume, stats=stats)


def check(ledger, tol_loops=1.0, tol_defects=2e-2):
    """``(ok, [message])`` -- invariants (I) and (II), which must hold.

    (I) is exact only to the integer rounding in `sample_family`, so the
    tolerance is one loop. (II) inherits that same rounding through the loop
    count, so it is checked as a relative tolerance.

    (III) is NOT checked here: it is a modeling statement, it is expected to
    move by `sink_discontinuity`, and asserting on it would only encode
    whichever choice happened to be in force.
    """
    msgs, ok = [], True
    for r in ledger["rows"]:
        if abs(r["dN"]) > tol_loops:
            ok = False
            msgs.append(f"(I) {r['family']}: loop number moved by {r['dN']:+.2f} "
                        f"({r['N_continuum']:.2f} -> {r['N_discrete']:.0f})")
        # (II) on the population actually handed over, against the share it
        # was supposed to take. The raw draw's own rounding error is reported
        # separately as dC_rel and is not a failure -- rescale_to_defects
        # removes it.
        want = r.get("frac_kept", 1.0) * r["C_continuum"]
        got = r["C_discrete_kept"]
        if want > 0 and abs(got - want) > tol_defects * want:
            ok = False
            msgs.append(f"(II) {r['family']}: handed over {got:.6e} defects, "
                        f"intended {want:.6e} ({100*(got-want)/want:+.2f}%)")
    # The residual is EXPECTED to be non-zero under a partial transfer: it is
    # exactly the share the generator refused. What must hold is that it equals
    # that share, not that it vanishes.
    for r in ledger["rows"]:
        fk = r["family"]
        N, C, S = ledger["residual"][fk]
        want = (1.0 - r.get("frac_kept", 1.0)) * r["C_continuum"]
        if abs(C - want) > max(1e-6 * max(want, 1.0), 1e-9 * r["C_continuum"]):
            ok = False
            msgs.append(f"residual: {fk} continuum holds C={C:.6e} after "
                        f"transfer, expected {want:.6e} (the refused share)")
    return ok, msgs


def format_ledger(ledger):
    L = ["transfer ledger", "",
         "(I) and (II) across the TRANSFER -- continuum vs the pre-coalescence "
         "draw:", "",
         f"{'fam':<4} {'N_cont':>10} {'N_raw':>8} {'dN':>7} "
         f"{'C_cont':>12} {'C_raw':>12} {'dC':>8}"]
    for r in ledger["rows"]:
        L.append(f"{r['family']:<4} {r['N_continuum']:>10.2f} "
                 f"{r['N_raw']:>8.0f} {r['dN']:>+7.2f} "
                 f"{r['C_continuum']:>12.4e} {r['C_raw']:>12.4e} "
                 f"{100 * r['dC_rel']:>7.2f}%")
    L += ["", "Then COALESCENCE, a separate physical step that conserves area "
          "and reduces count:", "",
          f"{'fam':<4} {'N_raw':>8} {'N_final':>8} {'N ratio':>9} "
          f"{'area kept':>10}"]
    for r in ledger["rows"]:
        L.append(f"{r['family']:<4} {r['N_raw']:>8.0f} {r['N_discrete']:>8.0f} "
                 f"{r['coalescence_N']:>9.3f} {r['coalescence_area']:>10.4f}")
    L += ["", "(III) sink strength. A modeling statement, not a conservation "
          "law -- it is EXPECTED to move, and by how much is the decision:", "",
          f"{'fam':<4} {'S_cont':>11} {'S_raw':>11} {'S_final':>11} "
          f"{'raw/cont':>9} {'final/cont':>11}"]
    for r in ledger["rows"]:
        L.append(f"{r['family']:<4} {r['S_continuum']:>11.4e} "
                 f"{r['S_raw']:>11.4e} {r['S_discrete']:>11.4e} "
                 f"{r['S_ratio_transfer']:>9.3f} "
                 f"{r['S_ratio_coalesced']:>11.3f}")
    L += ["", "The geometric part of that jump decomposes as:", "",
          f"{'fam':<4} {'sqrt(b_cd/b_dd)':>16} {'1/loopSinkScale':>17} "
          f"{'product':>9}"]
    for r in ledger["rows"]:
        L.append(f"{r['family']:<4} {r['f_burgers']:>16.4f} "
                 f"{r['f_bias']:>17.4f} {r['f_total']:>9.4f}")
    ok, msgs = check(ledger)
    L += ["", "(I) and (II): " + ("OK" if ok else "FAILED")]
    L += ["  " + m for m in msgs]
    return "\n".join(L)


def _material_of(sim_dir):
    """The STAGED material file a case actually reads.

    Defined here rather than imported from `studies.dad_sweep`: the coupling
    path must not depend on a study driver.
    """
    from pathlib import Path
    cands = sorted((Path(sim_dir) / "inputFiles").glob("Zr*.txt"))
    if not cands:
        raise FileNotFoundError(f"no material file in {sim_dir}/inputFiles")
    return cands[0]


def enable_discrete_climb(sim_dir, cutoff_b, solver="Galerkin",
                          superposed_output=True):
    """Switch a staged case into discrete-climb mode, and set the pair cutoff.

    Four keys, in two files, and all four are needed:

      DD.txt   useDislocations=1   the network exists at all
      DD.txt   climbSolverType     GalerkinClimbSolver runs
      material climbNeighborCutoff_b   truncates its O(N_seg^2) assembly
      material outputSuperposedMobile  publishes c_FEM + c_DD at the CD nodes

    The last one is what makes the discrete field VISIBLE. MoDELib solves the
    mobile species by superposition, and the CD block of `evl_*.txt` carries
    only the corrective part c_FEM -- `initializeConfiguration` reads it back,
    so it cannot be anything else. The analytic loop field c_DD enters the solve
    through the Dirichlet values alone and is never stored, so a figure drawn
    from the CD block is smooth however many discrete loops exist. Without this
    key the transition is physically active and invisible.

    The cutoff is written to the STAGED material file, never to the one in
    `Library/Materials`: the staged copy is what DDomp reads, and editing the
    library copy would change `cfg.domain_key` and force a re-stage.

    Idempotent -- rewriting an existing key rather than appending a second one,
    because `TextFileParser` takes the FIRST match and a duplicate would be
    silently ignored in whichever order they happened to land.
    """
    import re
    from pathlib import Path
    from dislocluster_code.staging.inputs import set_dd_scalar

    sim_dir = Path(sim_dir)
    dd = sim_dir / "inputFiles" / "DD.txt"
    set_dd_scalar(dd, "useDislocations", "1")
    txt = dd.read_text(encoding="utf-8")
    if re.search(r"^\s*climbSolverType\s*=", txt, re.M):
        txt = re.sub(r"^\s*climbSolverType\s*=[^;]*;",
                     f"climbSolverType={solver};", txt, count=1, flags=re.M)
    else:
        txt += f"\nclimbSolverType={solver};\n"
    dd.write_text(txt, encoding="utf-8")

    mat = _material_of(sim_dir)
    mtxt = mat.read_text(encoding="utf-8")

    def _set(text, key, value, note):
        line = f"{key}={value};  # GENERATED by {note}"
        if re.search(rf"^\s*{key}\s*=", text, re.M):
            return re.sub(rf"^\s*{key}\s*=[^;]*;[^\n]*", line, text,
                          count=1, flags=re.M)
        return text.rstrip("\n") + "\n" + line + "\n"

    note = "coupling/transition.enable_discrete_climb"
    mtxt = _set(mtxt, "climbNeighborCutoff_b", f"{float(cutoff_b):.10g}", note)
    if superposed_output:
        mtxt = _set(mtxt, "outputSuperposedMobile", "1", note)
    mat.write_text(mtxt, encoding="utf-8")
    return dd, mat


def fits_in_crystal(pop, faces, sides_key="dd_sides"):
    """Boolean mask of loops every vertex of which lies inside the crystal.

    This PREDICTS what `microstructureGenerator` will accept. Its criterion is
    that a loop's nodes fall inside the grain -- it prints "nodes outside grain
    N" and silently drops the loop otherwise -- and the loops are built as
    polygons on exactly the `dd_sides` vertices `write_microstructure` exports,
    so the same test applied to the same vertices reproduces the same decision.

    Predicting it rather than discovering it afterwards is what makes the
    transfer conservative: a refused loop's defects must stay in the continuum,
    and that can only be arranged BEFORE the continuum is zeroed.
    """
    from dislocluster_code.post.discrete_loops import loop_polygon
    if faces is None or len(pop) == 0:
        return np.ones(len(pop), dtype=bool)
    N, bb = faces
    fam = dict(pop.fam)
    fam["sides"] = int(pop.fam.get(sides_key, pop.fam["sides"]))
    keep = np.ones(len(pop), dtype=bool)
    for k in range(len(pop)):
        P = loop_polygon(fam, pop.centers[k], pop.radii[k])
        keep[k] = bool(np.all(P @ N.T <= bb[None, :] + 1e-9))
    return keep


def rescale_to_defects(pop, target):
    """Rescale a population's radii so it stores exactly ``target`` defects.

    WHY THIS IS NEEDED, and why it is not a fudge. `sample_family` draws
    ``round(sum_j n_j V_j)`` loops -- an INTEGER count from a real expectation --
    so the drawn population misses the continuum total by the rounding. That is
    negligible when the population is large (1.6% at 98 loops) and it is not
    when it is small: 15.4% at 6 loops, measured. A transfer that loses 15% of
    the stored defects is not a transfer.

    Since stored defects go as ``pi r^2 |b| / Omega``, scaling every radius by
    ``sqrt(target/current)`` fixes the total EXACTLY while leaving every ratio
    between loops, and hence the shape of the size distribution, untouched. The
    alternative -- drawing a non-integer number of loops -- does not exist.

    The correction is applied only on the TRANSFER path, never in
    `discrete_loops.build`, so no published figure moves.
    """
    cur = pop.stored_defects
    if len(pop) == 0 or cur <= 0 or target <= 0:
        return pop, 1.0
    s = float(np.sqrt(target / cur))
    return LoopPopulation(pop.fam, pop.centers, pop.radii * s,
                          pop.n_merged), s


def split_by_fit(pop, faces):
    """``(kept, refused)`` populations, and the defect fraction kept.

    The fraction is by STORED DEFECTS, not by loop count, and the two are not
    interchangeable: on the first case this ran, one loop of six was refused,
    which is 83.3% by count but 98.4% by defects, because the refused loop
    happened to be the SMALLEST. What decides refusal is proximity to the
    surface, not size, so the count fraction carries no information about how
    much material is involved and only the defect fraction may be used to
    reconcile the continuum.
    """
    keep = fits_in_crystal(pop, faces)
    kept = LoopPopulation(pop.fam, pop.centers[keep], pop.radii[keep],
                          pop.n_merged[keep])
    refused = LoopPopulation(pop.fam, pop.centers[~keep], pop.radii[~keep],
                             pop.n_merged[~keep])
    tot = pop.stored_defects
    frac = (kept.stored_defects / tot) if tot > 0 else 1.0
    return kept, refused, float(frac)


def inject_discrete_loops(sim_dir, pops, ddomp_generator=None, tag="transfer",
                          verbose=True):
    """Put a discrete population into a march's `evl_0` WITHOUT losing its CD field.

    This is the piece that makes a runtime transition possible at all, and the
    obstacle it clears is worth stating. `write_microstructure` emits INPUT for
    MoDELib's `microstructureGenerator`, which builds the node/loop/link topology
    a dislocation network needs -- intricate enough that reproducing it in Python
    would be its own defect surface. But the generator writes a FRESH `evl_0`,
    and the march's `evl_0` already carries the CD field the whole coupling
    exists to advance. Running the generator naively destroys it.

    The two are separable because of how `EvlFile` is built: it parses ONLY the
    CD block numerically and keeps every other record as verbatim text. So the
    merge is exactly

        network  <- the generator's evl   (nodes, loops, links, displacement)
        CD block <- the march's evl       (the field being marched)

    and `EvlFile.write` rewrites the header's CD row count to match. Both files
    describe the same FE mesh, so `cdNodes.txt` and hence the CD row order are
    identical between them -- which is the assumption that makes this legal, and
    it is checked rather than trusted.
    """
    import subprocess
    from pathlib import Path
    from dislocluster_code import paths
    from dislocluster_code.coupling.field import EvlFile

    sim_dir = Path(sim_dir)
    evl0 = sim_dir / "evl" / "evl_0.txt"
    if not evl0.is_file():
        raise FileNotFoundError(f"no evl_0 in {sim_dir}")

    micro, table, n_loops = write(pops, sim_dir / "inputFiles", tag=tag)
    (sim_dir / "inputFiles" / "initialMicrostructure.txt").write_text(
        f"microstructureFile={micro.name};\n", encoding="utf-8")

    keep = EvlFile(evl0)                       # the CD field to preserve
    backup = evl0.with_suffix(".txt.premerge")
    backup.write_bytes(evl0.read_bytes())

    exe = ddomp_generator or paths.modelib_generator()
    cmd, cwd = paths.generator_cmd(sim_dir, exe=exe)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        backup.replace(evl0)                   # never leave a half-written evl
        raise RuntimeError(
            f"microstructureGenerator failed ({proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    made = EvlFile(evl0)                       # the network just generated
    if made.n_cd and made.n_cd != keep.n_cd:
        backup.replace(evl0)
        raise ValueError(
            f"the generated evl has {made.n_cd} CD rows and the marched one "
            f"{keep.n_cd}; they must describe the same mesh")
    made.cd = keep.cd                          # network from one, field from the other
    made.write(evl0)

    # THE GENERATOR CAN SILENTLY DROP LOOPS. It refuses any whose nodes fall
    # outside the grain ("nodes outside grain N"), which for a population sampled
    # right up to the crystal surface is not rare -- 6 requested, 5 created, in
    # the first case this was run on. That is a conservation LEAK and precisely
    # what the ledger exists to catch: the continuum was zeroed for every loop,
    # so any the generator refuses take their defects with them.
    realized = int(made.n_loops)
    if verbose:
        note = "" if realized == n_loops else \
            f"  <-- {n_loops - realized} REFUSED by the generator"
        print(f"      injected {realized}/{n_loops} discrete loops, CD block "
              f"preserved ({keep.cd.shape[0]} rows){note}")
    return dict(requested=int(n_loops), realized=realized,
                lost=int(n_loops) - realized,
                lost_fraction=(n_loops - realized) / n_loops if n_loops else 0.0,
                microstructure=micro, table=table, evl=evl0)


def write(pops, out_dir, tag="transfer"):
    """Emit the transferred population as a MoDELib microstructure and a table.

    Returns ``(microstructure_path, table_path, n_loops)``. Note the two
    writers themselves return COUNTS, not paths -- easy to misread, and the
    reason this returns both.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plist = list(pops.values()) if isinstance(pops, dict) else list(pops)
    micro = out_dir / f"aLoops_{tag}.txt"
    table = out_dir / f"loops_{tag}.csv"
    n = write_microstructure(plist, micro)
    write_table(plist, table)
    return micro, table, int(n)
