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

    Note `stored_defects` uses ``b_dd``: `sample_family` sizes a loop by the
    DD Burgers magnitude, which for <c> is half the CD one. That is not a
    bookkeeping detail -- see `sink_discontinuity`.
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
       the discrete side with ``b_dd``. For <c> those differ by 2 (full versus
       half [0001]), and since ``r ~ 1/sqrt(b)`` at fixed stored defects the
       DISCRETE radius is ``sqrt(2)`` times the continuum one. Sink strength is
       linear in `r`, so (III) jumps by ``sqrt(b_cd/b_dd)``.

    2. **The bias parameter.** `loopSinkScale` (continuum, 0.291528 for <c>)
       and `discreteDislocationBias` (discrete, 1.0) are unrelated numbers
       serving the same role. Left as they are, absorption jumps by
       ``1/0.291528 = 3.43``.

    Returned as ``(factor_b, factor_bias, total)``. For <c> the product is
    about 4.85, which is not a rounding error and is exactly what invariant
    (III) exists to surface.
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
def zero_unit(Y, key):
    """A copy of ``Y`` with one transferable unit's continuum field removed."""
    if key not in UNITS:
        raise KeyError(f"{key!r} is not transferable; choose from {list(UNITS)}")
    Y = np.array(Y, dtype=float, copy=True)
    u = UNITS[key]
    for name in u["n"] + u["c"]:
        Y[:, IDX[name]] = 0.0
    return Y


def transfer(Y, nodes, omega, keys=("c",), weights=None, faces=None,
             region="domain", variant_weights=(1 / 3, 1 / 3, 1 / 3),
             coalesce_pass=True, seed=0, material=None, mc_samples=2_000_000):
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
                f_burgers=fb, f_bias=fbias, f_total=ftot))
            out[fk] = pop
        Y_after = zero_unit(Y_after, key)

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
        if np.isfinite(r["dC_rel"]) and abs(r["dC_rel"]) > tol_defects:
            ok = False
            msgs.append(f"(II) {r['family']}: stored defects moved by "
                        f"{100 * r['dC_rel']:+.2f}%")
    for fk, (N, C, S) in ledger["residual"].items():
        if N != 0.0 or C != 0.0:
            ok = False
            msgs.append(f"residual: {fk} still carries N={N:.3e} C={C:.3e} "
                        "in the continuum after transfer")
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
