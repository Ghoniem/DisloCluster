"""
modelib_field.py — genuine per-node field exchange between the 0-D reduced
cluster dynamics (ZrMicro) and the spatially-resolved cluster dynamics
(MoDELib3).

WHY THIS REPLACES THE OLD HANDOFF
---------------------------------
The original 0-D -> 3-D carrier (``modelib_fem.write_sink_field``) wrote FOUR
SCALARS per dose step — one number density and one mean radius per loop family —
into ``inputFiles/aLoopsDensity.txt`` and ``inputFiles/frankLoopsDensity.txt``.
Those files drive MoDELib's *microstructure generator*, which sprinkles discrete
loops at a uniform target density. The coupling driver therefore had to collapse
the whole marched state to its volume average (``q_per_variant_from_state``)
before handing it over, and any spatial structure the slow march had produced
was discarded on the way out.

MoDELib3 does not actually need that. Its cluster-dynamics module already
carries the immobile population as a finite-element FIELD: ``ClusterDynamicsFEM``
holds ``immobileClusters``, a trial function with ``iSize = 27`` components per
node, and ``ImmobileSinks`` builds the sink strength from the LOCAL value at
every quadrature point. ``ClusterDynamics::output`` writes that field, together
with the ``mSize = 4`` mobile components, into the ``evl/evl_<N>.txt`` CD block,
and — the part that makes the exchange two-way —
``ClusterDynamicsFEM::initializeConfiguration`` READS IT BACK:

    if(size_t(configIO.cdMatrix().size())==mobileClusters.gSize()+immobileClusters.gSize())
    {
        mobileClusters   = configIO.cdMatrix().block(0,0,nNodes,mSize)...
        immobileClusters = configIO.cdMatrix().block(0,mSize,nNodes,iSize)...
    }

So the CD block of an ``evl`` file is a read/write field channel on exactly the
node set both codes already share. This module uses it, and nothing is averaged
in either direction.

FILE FORMAT (MoDELib3 ``DDconfigIO::writeTxtStream`` / ``readTxt``)
------------------------------------------------------------------
Ten integer header lines::

    nNodes nLoops nLoopLinks nLoopNodes nSphericalInclusions
    nPolyhedronInclusions nPolyhedronInclusionNodes nPolyhedronInclusionEdges
    nDisplacement nCD

followed by the corresponding records, then ``nDisplacement`` rows of ``2*dim``
columns, then ``nCD`` rows of ``mSize + iSize = 31`` columns.

CD COLUMN LAYOUT (``ClusterDynamicsParameters``: mSize=4, iSize=27)
------------------------------------------------------------------
=======  ======================================================
0 .. 3   mobile    Cv, Ci, C2i, C3i            (atom fraction)
4 .. 12  immobile  n_k, nine families          (number per b^3)
13 .. 21 immobile  c_k, nine families          (atom fraction)
22 .. 30 immobile  q_k, nine families          (second content moment)
=======  ======================================================

The nine families, in ``immobileSpeciesVector`` order: 0 the faulted basal <c>
vacancy loop c_f; 1-3 the prismatic <a> INTERSTITIAL variants; 4-6 the prismatic
<a> VACANCY variants (step 1 of the implementation plan); 7 the perfect basal
state c_p and 8 the stacking-fault pyramid c_0 (step 4).

**The block is numbers-then-contents, so a family added in the middle MOVES the
contents.** A narrower block written by an earlier revision is widened on read
rather than refused, and the relocation is what that widening has to get right.

UNITS
-----
MoDELib works in Burgers-vector units, so its nodal number density ``n`` is per
b^3 while ZrMicro's loop number density is an atom fraction. ``ImmobileSinks``
floors them as ``nFloor = concentrationFloor/omega`` and ``cFloor =
concentrationFloor``, which fixes the conversion:

    n_MoDELib = C_0D / omega          omega = atomicVolume_SI / b_SI^3
    c_MoDELib = c_0D                  (both atom fractions, 1:1)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from dislocluster_code import paths as _paths

# MoDELib3 ClusterDynamicsParameters
M_SIZE = 4          # Cv, Ci, C2i, C3i
# Step 1 of the implementation plan raised iSize from 8 to 16: eight immobile
# families, each with a (number, content) pair. The CD block is ordered numbers
# first then contents -- iVal(k) and iVal(nFamilies+k) in ImmobileSinks -- so
# the new columns land among the numbers and among the contents, NOT
# appended at the end. Step 4 added the ninth family, the pyramid c_0.
#
# Step 5 added a THIRD field per family, the second content moment q_k, so the
# block is moment-major: nine numbers, nine contents, nine q. The first eighteen
# columns are exactly where they were, which is what lets an evl written before
# this step widen into it. N_FAMILIES is its OWN constant now and not half of
# I_SIZE -- that spelling silently became 13 the moment the third moment landed.
N_MOMENTS = 3
N_FAMILIES = 9
I_SIZE = N_FAMILIES * N_MOMENTS     # 27
N_CD_COLS = M_SIZE + I_SIZE

#: The pre-step-1 width, kept as a named reference point. Any narrower block is
#: accepted on read and widened, not just this one.
N_CD_COLS_LEGACY = M_SIZE + 8
N_HEADER = 10       # integer header lines in an evl text file
DISP_COLS = 6       # 2*dim

# Family order in the CD block; family 0 is the <c> vacancy family.
FAMILIES = ("c", "a1", "a2", "a3")

# ZrMicro native state indices (see modelib_coupling.py)
IDX = dict(Cv=0, Ci=1, C2i=2, C3i=3,
           CiL=4, CaiL=5, CvL=6, CavL=7,
           CiL_i=8, CaiL_i=9, CvL_v=10, CavL_v=11,
           rho_N=18)


# ── material constants ───────────────────────────────────────────────────────

def read_material_scalar(material_file, key):
    """Read a ``key=value;`` scalar from a MoDELib material file."""
    txt = Path(material_file).read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^;#\n]+)", txt, re.M)
    if not m:
        raise KeyError(f"{key} not found in {material_file}")
    return float(m.group(1).split()[0])


def read_material_vector(material_file, key, n=None):
    """Read a ``key=v1 v2 ...;`` VECTOR from a MoDELib material file.

    :func:`read_material_scalar` returns ``split()[0]`` and silently discards
    the rest, which is correct for the scalars it was written for and a trap
    for everything else. Several keys that look scalar are per-species or
    per-family: `loopSinkScale` is ``0.291528`` for <c> but ``0.792317`` for
    the three <a> variants, and `otherSinks_SI`, `dadAnisotropy` and `dadZ0`
    all carry one entry per mobile species. Reading any of them as a scalar
    applies the <c> or vacancy value to everything.

    ``n`` broadcasts a single entry to that length, so a genuinely scalar
    spelling of a per-species key still works.
    """
    txt = Path(material_file).read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^;#\n]+)", txt, re.M)
    if not m:
        raise KeyError(f"{key} not found in {material_file}")
    v = np.array([float(x) for x in m.group(1).split()], dtype=float)
    if n is not None:
        if v.size == 1:
            v = np.repeat(v, n)
        elif v.size != n:
            raise ValueError(f"{key} has {v.size} entries, expected {n}")
    return v


def cluster_atomic_volume(material_file):
    """omega = atomicVolume_SI / b_SI^3 — MoDELib's atomic volume in b^3.

    This is the factor that converts a ZrMicro loop number density (an atom
    fraction) into MoDELib's nodal number density (per b^3), and it is the same
    quantity ``ClusterDynamicsParameters::getClusterAtomicVolume`` computes.
    """
    b = read_material_scalar(material_file, "b_SI")
    try:
        vol = read_material_scalar(material_file, "atomicVolume_SI")
    except KeyError:
        # MoDELib falls back to the lattice-derived volume when the optional
        # key is absent; require it explicitly rather than guess a structure.
        raise KeyError(
            f"{material_file} has no atomicVolume_SI key; the 0-D <-> 3-D "
            "number-density conversion is undefined without it.")
    return vol / b ** 3


# ── evl CD block I/O ─────────────────────────────────────────────────────────

class EvlFile:
    """An ``evl/evl_<N>.txt`` configuration, with its CD block exposed.

    Only the CD block is parsed numerically; every other record is retained
    verbatim as text, so rewriting a file cannot perturb the dislocation
    configuration it also carries.
    """

    def __init__(self, path):
        self.path = Path(path)
        lines = self.path.read_text(encoding="utf-8",
                                    errors="replace").splitlines()
        if len(lines) < N_HEADER:
            raise ValueError(f"{path}: too short to be an evl file")
        self.header = [int(lines[i].strip()) for i in range(N_HEADER)]
        (self.n_nodes, self.n_loops, self.n_loop_links, self.n_loop_nodes,
         self.n_sph_inc, self.n_poly_inc, self.n_poly_inc_nodes,
         self.n_poly_inc_edges, self.n_disp, self.n_cd) = self.header

        n_records = (self.n_nodes + self.n_loops + self.n_loop_links +
                     self.n_loop_nodes + self.n_sph_inc + self.n_poly_inc +
                     self.n_poly_inc_nodes + self.n_poly_inc_edges)
        r0 = N_HEADER
        r1 = r0 + n_records
        d1 = r1 + self.n_disp
        c1 = d1 + self.n_cd
        if len(lines) < c1:
            raise ValueError(
                f"{path}: header promises {c1} lines, file has {len(lines)}")

        self._prefix = lines[:d1]              # header + records + displacement
        self._suffix = lines[c1:]              # anything trailing (normally none)
        if self.n_cd:
            self.cd = np.array(
                [[float(x) for x in lines[i].split()] for i in range(d1, c1)],
                dtype=float)
            got = self.cd.shape[1]
            if got != N_CD_COLS:
                # An evl written when the model carried FEWER families. Widen it
                # rather than refuse it: those runs must stay post-processable,
                # since the figures, the movies and the transfer ledger all read
                # evl_<N>.txt directly.
                #
                # The CD block is moment-major, so widening is not a pad on the
                # end -- the old contents have to MOVE, from M_SIZE+nf_old to
                # M_SIZE+N_FAMILIES. Getting that wrong would silently read a
                # stale file's contents as the new families' numbers.
                #
                # Every block narrower than the current one carries TWO moments:
                # the third arrived with step 5, which is the same change that
                # widened the block to its present size, so a narrower file
                # predates it by construction. q comes back zero, which reads as
                # Delta = 1 -- the monodisperse limit those runs were computed
                # in, and the correction-free value of every closure factor.
                nf_old, rem = divmod(got - M_SIZE, 2)
                if rem or nf_old <= 0 or nf_old > N_FAMILIES:
                    raise ValueError(
                        f"{path}: CD block has {got} columns, which is not "
                        f"M_SIZE + 2*nf for any nf <= {N_FAMILIES} "
                        f"(expected {N_CD_COLS})")
                old = self.cd
                self.cd = np.zeros((old.shape[0], N_CD_COLS))
                self.cd[:, 0:M_SIZE] = old[:, 0:M_SIZE]
                self.cd[:, M_SIZE:M_SIZE + nf_old] =                     old[:, M_SIZE:M_SIZE + nf_old]
                self.cd[:, M_SIZE + N_FAMILIES:M_SIZE + N_FAMILIES + nf_old] =                     old[:, M_SIZE + nf_old:M_SIZE + 2 * nf_old]
        else:
            self.cd = np.zeros((0, N_CD_COLS))

    # -- views ---------------------------------------------------------------
    @property
    def mobile(self):
        """(nCD, 4) mobile field [Cv, Ci, C2i, C3i], atom fraction."""
        return self.cd[:, :M_SIZE]

    @property
    def immobile(self):
        """(nCD, 8) immobile field [n_c,n_a1,n_a2,n_a3, c_c,c_a1,c_a2,c_a3]."""
        return self.cd[:, M_SIZE:]

    # -- writing -------------------------------------------------------------
    def write(self, dest=None, fmt="%.15e"):
        """Write the configuration out, CD block replaced by ``self.cd``.

        The header's CD row count is rewritten to match, so the file stays
        self-consistent if the node count ever changes.
        """
        dest = Path(dest) if dest is not None else self.path
        prefix = list(self._prefix)
        prefix[9] = str(self.cd.shape[0])
        body = [" ".join(fmt % v for v in row) for row in self.cd]
        dest.write_text("\n".join(prefix + body + self._suffix) + "\n",
                        encoding="utf-8")
        return dest


def read_superposed_mobile(evl_dir, n_nodes=None):
    """``(N,4)`` physical mobile field ``c_FEM + c_DD``, or None if absent.

    Written by MoDELib as ``evl/cdTotalMobile_<runID>.txt`` when the material
    key ``outputSuperposedMobile`` is set — which ``transition.
    enable_discrete_climb`` does, because otherwise the discrete loop field is
    active in the solve and invisible in every figure.

    Why it has to come from MoDELib rather than be rebuilt here: the analytic
    field of a segment is ``concentrationMatrices(x) @ [v_source, v_sink]``
    (``DislocationSegment::clusterConcentration``), i.e. it is LINEAR IN THE
    NODAL CLIMB VELOCITY — and those velocities are the output of the climb
    solve itself. There is no way to evaluate the Green's function offline
    without first reproducing the solve that sets its amplitude.

    The highest runID present is returned, matching ``MobileQSSASolver``'s
    read-back rule.
    """
    import re as _re
    from pathlib import Path as _P
    evl_dir = _P(evl_dir)
    cands = sorted(
        (int(m.group(1)), p) for p in evl_dir.glob("cdTotalMobile_*.txt")
        if (m := _re.fullmatch(r"cdTotalMobile_(\d+)\.txt", p.name)))
    if not cands:
        return None
    arr = np.loadtxt(cands[-1][1], dtype=float).reshape(-1, M_SIZE)
    if n_nodes is not None and arr.shape[0] != n_nodes:
        raise ValueError(
            f"{cands[-1][1]} has {arr.shape[0]} rows, expected {n_nodes}")
    return arr


def read_cd_nodes(evl_dir):
    """(N,3) coordinates of the CD nodes, in Burgers-vector units.

    ``evl/cdNodes.txt`` is written by ``ClusterDynamicsFEM::writeNodePositions``
    and is in the SAME order as the CD block rows, so no search or interpolation
    is needed to associate a field value with a position.
    """
    p = Path(evl_dir) / "cdNodes.txt"
    return np.loadtxt(p, dtype=float).reshape(-1, 3)


# ── field mapping ────────────────────────────────────────────────────────────

def immobile_0d_to_modelib(Y, omega, variant_weights=(1 / 3, 1 / 3, 1 / 3),
                           loop_model=0):
    """Map the marched 0-D immobile state at every point to MoDELib's CD layout.

    Parameters
    ----------
    Y : (N, 19) array
        The native ZrMicro state at each node — exactly what
        ``modelib_coupling.run_immobile_step`` returns, one row per node.
    omega : float
        MoDELib atomic volume in b^3 (``cluster_atomic_volume``).
    variant_weights : 3-sequence
        Split of the lumped <a> interstitial population across a1/a2/a3. Equal
        thirds at zero resolved stress; ``modelib_export._variant_weights``
        supplies the stress-tilted values.

    Returns
    -------
    (N, 8) array  [n_c, n_a1, n_a2, n_a3, c_c, c_a1, c_a2, c_a3]

    ZrMicro's aligned/non-aligned pair is collapsed first (the spatial code does
    not carry that split), then the <a> total is distributed over the three
    prism variants — the same Option-A reduction ``modelib_export`` performs,
    but applied POINTWISE instead of to a volume average.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    # 19 is the pre-step-1 state; 27 carries the four families step 1 appended
    # at indices 19..26. Both are accepted, and a 19-wide state simply leaves
    # the new families empty -- which is what they are in every run made before
    # the step. Anything else is a layout error and must not be guessed at.
    # 19 is the pre-step-1 state; 29 carries the five families appended by
    # steps 1 and 4 (numbers 19..23, contents 24..28). A 19-wide state simply
    # leaves them empty, which is what they are in every run made before those
    # steps. Anything else is a layout error and must not be guessed at.
    # 38 adds step 5's second content moment, q, at 29..37.
    # 40 (and 21, 31) add the two-entry grain-boundary ledger after everything
    # else. They are diagnostics -- no family, no moment -- so every index below
    # is unchanged and the extra columns are simply not read here.
    if Y.shape[1] not in (19, 21, 29, 31, 38, 40):
        raise ValueError(
            f"expected (N,19/21), (N,29/31) or (N,38/40) 0-D states, "
            f"got {Y.shape} -- the second width of each pair carries the "
            f"grain-boundary ledger")
    w = np.asarray(variant_weights, dtype=float)
    if w.shape != (3,):
        raise ValueError("variant_weights must have three entries")

    if loop_model:
        # SELF-CONSISTENT MODE: y[4..11] already IS this block, in this order --
        # [n_c, n_a1, n_a2, n_a3, c_c, c_a1, c_a2, c_a3]. The slow step carries
        # the same four families as the 3-D code, so there is nothing to lump
        # and nothing to split, and `variant_weights` has already been applied
        # at the source (the nucleation split) rather than on the way out.
        #
        # The number densities still need the 1/omega conversion: the 0-D
        # carries them per ATOM and MoDELib per b^3.
        # The 0-D appends its step-1 families at 19..26 (numbers 19..22,
        # contents 23..26) so that every pre-existing state index keeps its
        # meaning; the CD block interleaves them instead, numbers 0..7 then
        # contents 8..15. The two orderings are different and the mapping below
        # is where that is reconciled -- writing Y[:, 19:27] straight into
        # out[:, 8:16] would put the new NUMBERS into the old CONTENTS.
        out = np.zeros((Y.shape[0], I_SIZE), dtype=float)
        out[:, 0:4] = Y[:, 4:8] / omega                       # n_0..n_3
        out[:, N_FAMILIES:N_FAMILIES + 4] = Y[:, 8:12]        # c_0..c_3
        nx = 5 if Y.shape[1] >= 29 else 0                     # appended families
        if nx:
            out[:, 4:4 + nx] = Y[:, 19:19 + nx] / omega
            out[:, N_FAMILIES + 4:N_FAMILIES + 4 + nx] = Y[:, 19 + nx:19 + 2 * nx]
        # Step 5's second moment, q_k, at 0-D indices 29..37 in family order --
        # already the CD block's own order, so this one IS a straight copy.
        #
        # q crosses UNSCALED, exactly as the content does: it is a moment of the
        # content and carries the content's units. Only the number density is
        # converted (per atom -> per b^3), which is why the dispersion is
        # q n / c^2 on the 0-D side and q n omega / c^2 in ImmobileSinks -- the
        # same single omega that makes MoDELib's mean size CI/N/omega.
        if Y.shape[1] >= 38 and N_MOMENTS > 2:
            out[:, 2 * N_FAMILIES:2 * N_FAMILIES + N_FAMILIES] = Y[:, 29:38]
        return out

    N_a = Y[:, IDX["CiL"]] + Y[:, IDX["CaiL"]]        # <a> loop number
    c_a = Y[:, IDX["CiL_i"]] + Y[:, IDX["CaiL_i"]]    # <a> loop content
    N_c = Y[:, IDX["CvL"]] + Y[:, IDX["CavL"]]        # <c> loop number
    c_c = Y[:, IDX["CvL_v"]] + Y[:, IDX["CavL_v"]]    # <c> loop content

    # N_FAMILIES, not the literal 4: the CD block is moment-major, so the
    # contents start where the numbers end, and that moved when steps 1 and 4
    # widened it. Writing c_c at column 4 put it in the a1v NUMBER slot.
    out = np.zeros((Y.shape[0], I_SIZE), dtype=float)
    out[:, 0] = N_c / omega                            # n_c
    for k in range(3):
        out[:, 1 + k] = w[k] * N_a / omega             # n_a1..n_a3
    out[:, N_FAMILIES] = c_c                           # c_c
    for k in range(3):
        out[:, N_FAMILIES + 1 + k] = w[k] * c_a        # c_a1..c_a3
    return out


def to_legacy_layout(Y, loop_model=0):
    """Re-express a self-consistent state in the LEGACY slot layout.

    The 0-D figure suite and ``zerod.post_process`` address the immobile state
    by the legacy names, and rewriting all of that for a second formulation
    would double the surface that has to stay correct. This instead moves the
    four families into the slots those names expect:

        n_c -> CvL,  c_c -> CvL_v            <c> is a vacancy loop in both
        sum(n_a1..3) -> CiL, c_a -> CiL_i    the three prism variants, lumped

    with the ALIGNED partners set to zero, which is the truthful statement: the
    self-consistent model carries no aligned/non-aligned split, so every figure
    that plots the pair sum is exactly right and every figure that plots the
    split shows all of one and none of the other rather than an invented ratio.

    ``loop_model=0`` returns the input unchanged, so callers can apply this
    unconditionally.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if not loop_model:
        return Y
    out = Y.copy()

    # Families 4..8 live at 19..28, numbers then contents, so a wider state has
    # more to lump than the original four. THE PARTNER SLOTS CARRY THEM. Every
    # legacy figure that plots a pair SUM is then exactly right -- CiL+CaiL is
    # the total prismatic population and CvL+CavL the total basal one -- and the
    # few that plot the split show interstitial against vacancy rather than
    # aligned against non-aligned, which is a relabelling and not an invention.
    # Dropping them instead, as this used to, understated the totals by however
    # much the vacancy variants and c_p carried.
    def fam(idx, moment):
        """Number (moment 0) or content (moment 1) of family `idx`."""
        if idx < 4:
            return Y[:, (4 if moment == 0 else 8) + idx]
        if Y.shape[1] < 29:
            return np.zeros(Y.shape[0])
        return Y[:, (19 if moment == 0 else 24) + (idx - 4)]

    for m, (i_slot, ai_slot, v_slot, av_slot) in enumerate(
            (("CiL", "CaiL", "CvL", "CavL"),
             ("CiL_i", "CaiL_i", "CvL_v", "CavL_v"))):
        prism_i = sum(fam(k, m) for k in (1, 2, 3))     # prismatic <a>, i-type
        prism_v = sum(fam(k, m) for k in (4, 5, 6))     # prismatic <a>, v-type
        out[:, IDX[i_slot]] = prism_i
        out[:, IDX[ai_slot]] = prism_v
        out[:, IDX[v_slot]] = fam(0, m)                 # c_f, faulted basal
        out[:, IDX[av_slot]] = fam(7, m)                # c_p, perfect basal
    # The PYRAMID (family 8) has no legacy slot and gets none. It is not a loop
    # -- no perimeter, no lambda sqrt(m) radius -- so a figure that reduces a
    # slot to a loop diameter would report a number that means nothing for it.
    #
    # Truncated to the legacy 19. This function's whole promise is "a state in
    # the layout the 0-D names address", and everything those names reach lives
    # below 19 -- the four slots, the six accumulators and rho_N. Returning the
    # appended families and the second moments as a tail would hand the 0-D
    # post-processing a state it has no reading for, in the one place that
    # exists to keep it from having to.
    #
    # THE GRAIN-BOUNDARY LEDGER IS THE ONE EXCEPTION, and it has to be. It is
    # an accumulator, not a family, so the 0-D balance does have a reading for
    # it -- and truncating it away is what made a `gb_absorption` run's
    # conservation figures omit the very channel they exist to show: the
    # absorbed atoms silently reappeared in `grain_boundary`, which is closure
    # by difference, so nothing looked wrong. It is always the last two
    # components (`immobile.N_GBACC`), and the result is width 21, which is
    # exactly `immobile.state_width`'s legacy-plus-ledger width.
    if out.shape[1] <= 19:
        return out
    if out.shape[1] in (21, 31, 40):
        return np.concatenate([out[:, :19], out[:, -2:]], axis=1)
    return out[:, :19]


def run_loop_model(run_dir):
    """Which immobile formulation a finished run's ``march_state.npz`` is in.

    Runs made before the self-consistent model existed carry no such key, and
    they are all legacy — so a missing key is 0, not an error. ``summary.json``
    is consulted as a fallback for a run whose npz predates the key but whose
    diagnostics do not.
    """
    from pathlib import Path as _P
    run_dir = _P(run_dir)
    p = run_dir / "march_state.npz"
    if p.is_file():
        with np.load(p) as z:
            if "loop_model" in z.files:
                return int(z["loop_model"])
    s = run_dir / "summary.json"
    if s.is_file():
        import json as _json
        try:
            d = _json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            return 0
        v = (d.get("diagnostics") or {}).get("loop_model")
        if v is None:
            v = ((d.get("config") or {}).get("solver") or {}).get("loop_model")
        return int(v or 0)
    return 0


def modelib_to_0d_mobile(cd):
    """(N,4) frozen mobile field for the slow march, straight from the CD block.

    MoDELib's mSize=4 mobile species ARE ZrMicro's [Cv, Ci, C2i, C3i], in the
    same order and the same units, so this is a slice — no interpolation, and
    no dependence on the unverified ``F/quadrature_*.txt`` column map that
    ``modelib_fem._read_quadrature_mobile_field`` had to guess at.
    """
    cd = np.atleast_2d(np.asarray(cd, dtype=float))
    return cd[:, :M_SIZE].copy()


def modelib_immobile_to_0d(immob, omega, Y_prev=None, loop_model=0):
    """Inverse map: MoDELib immobile field -> 0-D lumped loop state.

    Returns an (N, 8) block in ZrMicro's immobile order
    ``[CiL, CaiL, CvL, CavL, CiL_i, CaiL_i, CvL_v, CavL_v]``.

    The <a> variants are re-lumped by summation, and the aligned/non-aligned
    split — which the spatial code does not carry — is restored from the
    PREVIOUS 0-D state's ratio when one is supplied, so the information is
    preserved across a round trip instead of being reset to the f_a/f_na
    nominal. With ``Y_prev=None`` an even split is used.

    ``loop_model=1`` inverts the identity path instead: the slow step then
    carries MoDELib's own four families, so the block passes straight back with
    only the ``omega`` factor undone. Nothing is lumped and nothing is split, so
    unlike the legacy path this round trip is EXACT — there is no
    aligned/non-aligned information to lose in the first place.
    """
    immob = np.atleast_2d(np.asarray(immob, dtype=float))
    if loop_model:
        # Inverse of immobile_0d_to_modelib, and it returns the 0-D's OWN
        # layout: the original four families at 0..7 (numbers then contents) and
        # step 1's four at 8..15, which the caller places at state indices
        # 19..26. Width follows the CD block, so a pre-step-1 evl -- widened on
        # read -- comes back with its new families empty.
        nf = N_FAMILIES
        out = np.zeros((immob.shape[0], 2 * nf), dtype=float)
        out[:, 0:4] = immob[:, 0:4] * omega            # n_0..n_3
        out[:, 4:8] = immob[:, nf:nf + 4]              # c_0..c_3
        if nf > 4:
            out[:, 8:8 + (nf - 4)] = immob[:, 4:nf] * omega
            out[:, 8 + (nf - 4):] = immob[:, nf + 4:2 * nf]
        return out
    # LEGACY PATH: N_FAMILIES and not the literal 4 the block used to have.
    # The <c> content moved from column 4 to column N_FAMILIES when step 1
    # widened the block, and reading it at 4 would have returned the a1 NUMBER
    # as the <c> content -- a well-formed answer about the wrong quantity.
    n_c = immob[:, 0] * omega
    n_a = immob[:, 1:4].sum(axis=1) * omega
    c_c = immob[:, N_FAMILIES]
    c_a = immob[:, N_FAMILIES + 1:N_FAMILIES + 4].sum(axis=1)

    def split(tot, lo, hi):
        if Y_prev is None:
            return 0.5 * tot, 0.5 * tot
        a = np.asarray(Y_prev, dtype=float)[:, lo]
        b = np.asarray(Y_prev, dtype=float)[:, hi]
        s = a + b
        den = np.where(s > 0.0, s, 1.0)
        # Both fractions are formed directly rather than one as (1 - f): when
        # the pair is lopsided (CiL >> CaiL is normal at low dose) f rounds to
        # 1 - eps and 1-f keeps only the few bits that survive the
        # cancellation, costing the minor partner ~8 significant digits.
        fa = np.where(s > 0.0, a / den, 0.5)
        fb = np.where(s > 0.0, b / den, 0.5)
        return fa * tot, fb * tot

    CiL, CaiL = split(n_a, IDX["CiL"], IDX["CaiL"])
    CvL, CavL = split(n_c, IDX["CvL"], IDX["CavL"])
    CiL_i, CaiL_i = split(c_a, IDX["CiL_i"], IDX["CaiL_i"])
    CvL_v, CavL_v = split(c_c, IDX["CvL_v"], IDX["CavL_v"])
    return np.column_stack([CiL, CaiL, CvL, CavL,
                            CiL_i, CaiL_i, CvL_v, CavL_v])


def immobile_into_state(Y, immob, omega, loop_model=0, Y_prev=None):
    """Scatter a MoDELib immobile block into the 0-D state's own slots.

    The 0-D state and the CD block are BOTH grouped by moment, but at different
    offsets: the 0-D keeps its original four families at 4..11 and appends the
    five later ones at 19..28, so families 4..8 are not contiguous with 0..3 on
    that side. Doing this by hand at the call site is what put an 18-wide block
    into an 8-wide slice; there is now one place that knows the mapping.

    ``Y`` is modified in place and returned.
    """
    blk = modelib_immobile_to_0d(immob, omega, Y_prev=Y_prev,
                                 loop_model=loop_model)
    if not loop_model:
        Y[:, 4:12] = blk
        return Y
    nf = N_FAMILIES
    Y[:, 4:8] = blk[:, 0:4]                    # n_c, n_a1..a3
    Y[:, 8:12] = blk[:, 4:8]                   # c_c, c_a1..a3
    if Y.shape[1] >= 29 and nf > 4:
        Y[:, 19:19 + (nf - 4)] = blk[:, 8:8 + (nf - 4)]
        Y[:, 24:24 + (nf - 4)] = blk[:, 8 + (nf - 4):8 + 2 * (nf - 4)]
    # Step 5's second moment rides along unscaled, the way the content does.
    if Y.shape[1] >= 29 + nf and N_MOMENTS > 2:
        immob = np.atleast_2d(np.asarray(immob, dtype=float))
        if immob.shape[1] >= 3 * nf:
            Y[:, 29:29 + nf] = immob[:, 2 * nf:3 * nf]
    return Y


# ── the bridge ───────────────────────────────────────────────────────────────

class FieldBridge:
    """Two-way per-node field exchange over a MoDELib simulation directory.

    Usage inside the operator-split march::

        br = FieldBridge(sim_dir, material_file)
        C_M = br.read_mobile_field()             # (N,4) from the 3-D fast solve
        Y[:, 0:4] = C_M                          # freeze the mobile species
        Y = run_immobile_step(base_cli, Y, t0, t1)
        br.write_immobile_field(Y)               # (N,8) back, pointwise

    ``write_immobile_field`` rewrites only the CD block of the seed
    configuration; the dislocation records in the same file are copied through
    unchanged.
    """

    def __init__(self, sim_dir, material_file, evl_name="evl_0.txt"):
        self.sim_dir = Path(sim_dir)
        self.evl_dir = self.sim_dir / "evl"
        self.material_file = Path(material_file)
        self.evl_name = evl_name
        self.omega = cluster_atomic_volume(self.material_file)
        self._nodes = None

    # -- geometry ------------------------------------------------------------
    @property
    def nodes(self):
        """(N,3) CD node coordinates, cached."""
        if self._nodes is None:
            self._nodes = read_cd_nodes(self.evl_dir)
        return self._nodes

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    # -- reading -------------------------------------------------------------
    def latest_evl(self):
        """Path of the highest-numbered ``evl_<N>.txt`` present."""
        cands = []
        for p in self.evl_dir.glob("evl_*.txt"):
            m = re.fullmatch(r"evl_(\d+)\.txt", p.name)
            if m:
                cands.append((int(m.group(1)), p))
        if not cands:
            raise FileNotFoundError(f"no evl_<N>.txt in {self.evl_dir}")
        return max(cands)[1]

    def read(self, evl=None):
        return EvlFile(evl if evl is not None else self.latest_evl())

    def read_mobile_field(self, evl=None):
        """(N,4) steady mobile field C_M*(x) at the CD nodes."""
        return modelib_to_0d_mobile(self.read(evl).cd)

    def read_immobile_field(self, evl=None, Y_prev=None):
        """(N,8) 0-D-ordered immobile state at the CD nodes."""
        return modelib_immobile_to_0d(self.read(evl).immobile, self.omega,
                                      Y_prev)

    # -- writing -------------------------------------------------------------
    def write_immobile_field(self, Y, evl_src=None, dest=None,
                             variant_weights=(1 / 3, 1 / 3, 1 / 3),
                             backup=False, mobile=True, loop_model=0):
        """Write the marched per-node 0-D state into an evl CD block.

        Both halves of the CD block are written by default. Writing only the
        immobile half — which this method did originally — leaves the mobile
        columns at whatever ``evl_src`` carried, and ``evl_src`` is the seed:
        spatially uniform, with no boundary layer. The snapshot then claims a
        flat mobile field that the march never used, since ``run_coupled``
        replaces ``Y[:, 0:4]`` with the fast solve's ``C_M*(x)`` at every
        refresh. Any figure or grain-boundary profile of a mobile species drawn
        from such a snapshot shows the seed, not the solution.

        This does not affect the immobile results: those come from the block
        written here, and from ``march_state.npz``, both of which were always
        correct.

        Parameters
        ----------
        Y : (N,19) array   — one native ZrMicro state per CD node
        evl_src : path     — configuration to seed from (default: latest)
        dest : path        — output (default: overwrite ``evl_src``)
        mobile : bool      — also write Y[:, :4] into the mobile columns. Pass
                             False only to reproduce the historical behaviour.
        """
        src = Path(evl_src) if evl_src is not None else self.latest_evl()
        ev = EvlFile(src)
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] != ev.cd.shape[0]:
            raise ValueError(
                f"state has {Y.shape[0]} rows but the CD block of {src.name} "
                f"has {ev.cd.shape[0]} nodes — the march must run on the CD "
                "node set (FieldBridge.n_nodes), not an independent grid.")
        if backup and dest is None:
            shutil.copy2(src, src.with_suffix(".txt.bak"))
        # loop_model MUST be passed through. Without it this took the legacy
        # path unconditionally, which LUMPS a self-consistent state back into
        # four families and re-splits it -- so the archived snapshots of a
        # nine-family march held the four-family answer, with the five appended
        # families empty and the second moments zero, while the marched state
        # and the fast solve (which passes it) carried all of them. The
        # snapshot is the only thing an offline reader has.
        ev.cd[:, M_SIZE:] = immobile_0d_to_modelib(
            Y, self.omega, variant_weights, loop_model=loop_model)
        if mobile:
            ev.cd[:, :M_SIZE] = Y[:, :M_SIZE]
        return ev.write(dest if dest is not None else src)

    def write_mobile_field(self, C_M, evl_src=None, dest=None):
        """Write a (N,4) mobile field into an evl CD block (warm start)."""
        src = Path(evl_src) if evl_src is not None else self.latest_evl()
        ev = EvlFile(src)
        C_M = np.atleast_2d(np.asarray(C_M, dtype=float))
        if C_M.shape != (ev.cd.shape[0], M_SIZE):
            raise ValueError(
                f"expected ({ev.cd.shape[0]},{M_SIZE}) mobile field, got {C_M.shape}")
        ev.cd[:, :M_SIZE] = C_M
        return ev.write(dest if dest is not None else src)

    # -- diagnostics ---------------------------------------------------------
    def describe(self):
        ev = self.read()
        n = self.nodes
        d = np.linalg.norm(n - n.mean(axis=0), axis=1)
        lines = [
            f"sim_dir      : {self.sim_dir}",
            f"evl          : {ev.path.name}  ({ev.cd.shape[0]} CD nodes x "
            f"{ev.cd.shape[1]} cols)",
            f"cdNodes      : {n.shape[0]} nodes, extent "
            f"{n.min(axis=0)} .. {n.max(axis=0)} [b]",
            f"omega        : {self.omega:.6g} b^3  "
            f"(from {self.material_file.name})",
            f"radial span  : {d.min():.3g} .. {d.max():.3g} b",
        ]
        for j, nm in enumerate(("Cv", "Ci", "C2i", "C3i")):
            c = ev.cd[:, j]
            lines.append(f"  {nm:<4} {c.min():.4e} .. {c.max():.4e}")
        for k, fam in enumerate(FAMILIES):
            nk, ck = ev.cd[:, M_SIZE + k], ev.cd[:, M_SIZE + N_FAMILIES + k]
            lines.append(f"  n_{fam:<3} {nk.min():.4e} .. {nk.max():.4e}   "
                         f"c_{fam:<3} {ck.min():.4e} .. {ck.max():.4e}")
        return "\n".join(lines)


def rewrite_snapshots(run_dir, verbose=True):
    """Rebuild ``evl_coupled/evl_*.txt`` from ``march_state.npz``.

    The archived snapshots are a projection of the marched state, so they can
    always be regenerated from it without re-solving. This exists because they
    were being written through the LEGACY lumping on a self-consistent march --
    ``write_immobile_field`` did not take ``loop_model`` -- so a nine-family run
    archived the four-family answer with the appended families empty and the
    second moments zero. Everything the figures draw comes from
    ``march_state.npz`` through ``movies.cd_blocks``, which does pass it, and so
    does the fast solve; the snapshots were the one consumer that did not.

    Returns the list of files rewritten.
    """
    from pathlib import Path as _P
    run_dir = _P(run_dir)
    with np.load(run_dir / "march_state.npz") as z:
        doses, Y = z["doses"], z["Y"]
        lm = int(z["loop_model"]) if "loop_model" in z.files else 0
    evl_dir = run_dir / "evl_coupled"
    # ONE SNAPSHOT PER INTERVAL END, so the files correspond to doses[1:] and
    # not to doses: dose 0 is the seed and the march writes nothing for it.
    # Their names follow the march's own rule -- the dose-lattice index where
    # the dose has one, an ordinal where it does not -- and reproducing that is
    # the only safe pairing. Sorting the directory does NOT work: "evl_0.txt"
    # is 1 dpa, "evl_9.txt" is 10 dpa, and "evl_s01.txt" is 1e-4.
    used, names = set(), []
    for i, d1 in enumerate(doses[1:]):
        step = int(round(float(d1))) - 1
        if step < 0 or step in used:
            names.append(f"s{i + 1:02d}")
        else:
            used.add(step)
            names.append(str(step))
    files = [evl_dir / f"evl_{nm}.txt" for nm in names]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise ValueError(
            f"{run_dir.name}: no snapshot for {missing} -- the dose grid in "
            f"march_state.npz does not describe what evl_coupled/ holds")
    omega = cluster_atomic_volume(_paths.MODELIB_MATERIAL)
    out = []
    for f, y in zip(files, Y[1:]):
        ev = EvlFile(f)
        ev.cd[:, M_SIZE:] = immobile_0d_to_modelib(y, omega, loop_model=lm)
        ev.cd[:, :M_SIZE] = y[:, :M_SIZE]
        ev.write(f)
        out.append(f)
        if verbose:
            print(f"  rewrote {f.name}")
    return out
