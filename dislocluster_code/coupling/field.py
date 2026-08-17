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
holds ``immobileClusters``, a trial function with ``iSize = 8`` components per
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
columns, then ``nCD`` rows of ``mSize + iSize = 12`` columns.

CD COLUMN LAYOUT (``ClusterDynamicsParameters``: mSize=4, iSize=8)
-----------------------------------------------------------------
=======  ======================================================
0 .. 3   mobile    Cv, Ci, C2i, C3i            (atom fraction)
4 .. 7   immobile  n_c, n_a1, n_a2, n_a3       (number per b^3)
8 .. 11  immobile  c_c, c_a1, c_a2, c_a3       (atom fraction)
=======  ======================================================

``immobileSpeciesVector = -1 1 1 1`` in ``Zr3d_ghoniem.txt``: family 0 is the
basal <c> VACANCY loop family, families 1-3 the three prism <a> INTERSTITIAL
variants. That is the same partition ``modelib_export`` already uses.

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

# MoDELib3 ClusterDynamicsParameters
M_SIZE = 4          # Cv, Ci, C2i, C3i
I_SIZE = 8          # 4 families x (number, content)
N_FAMILIES = I_SIZE // 2
N_CD_COLS = M_SIZE + I_SIZE
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
            if self.cd.shape[1] != N_CD_COLS:
                raise ValueError(
                    f"{path}: CD block has {self.cd.shape[1]} columns, "
                    f"expected {N_CD_COLS}")
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
    if Y.shape[1] != 19:
        raise ValueError(f"expected (N,19) 0-D states, got {Y.shape}")
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
        out = np.zeros((Y.shape[0], I_SIZE), dtype=float)
        out[:, 0:4] = Y[:, 4:8] / omega
        out[:, 4:8] = Y[:, 8:12]
        return out

    N_a = Y[:, IDX["CiL"]] + Y[:, IDX["CaiL"]]        # <a> loop number
    c_a = Y[:, IDX["CiL_i"]] + Y[:, IDX["CaiL_i"]]    # <a> loop content
    N_c = Y[:, IDX["CvL"]] + Y[:, IDX["CavL"]]        # <c> loop number
    c_c = Y[:, IDX["CvL_v"]] + Y[:, IDX["CavL_v"]]    # <c> loop content

    out = np.zeros((Y.shape[0], I_SIZE), dtype=float)
    out[:, 0] = N_c / omega                            # n_c
    for k in range(3):
        out[:, 1 + k] = w[k] * N_a / omega             # n_a1..n_a3
    out[:, 4] = c_c                                    # c_c
    for k in range(3):
        out[:, 5 + k] = w[k] * c_a                     # c_a1..c_a3
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
    n_c, n_a = Y[:, 4], Y[:, 5:8].sum(axis=1)
    c_c, c_a = Y[:, 8], Y[:, 9:12].sum(axis=1)
    out[:, IDX["CiL"]], out[:, IDX["CaiL"]] = n_a, 0.0
    out[:, IDX["CvL"]], out[:, IDX["CavL"]] = n_c, 0.0
    out[:, IDX["CiL_i"]], out[:, IDX["CaiL_i"]] = c_a, 0.0
    out[:, IDX["CvL_v"]], out[:, IDX["CavL_v"]] = c_c, 0.0
    return out


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
        out = np.zeros((immob.shape[0], 8), dtype=float)
        out[:, 0:4] = immob[:, 0:4] * omega
        out[:, 4:8] = immob[:, 4:8]
        return out
    n_c = immob[:, 0] * omega
    n_a = immob[:, 1:4].sum(axis=1) * omega
    c_c = immob[:, 4]
    c_a = immob[:, 5:8].sum(axis=1)

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
                             backup=False, mobile=True):
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
        ev.cd[:, M_SIZE:] = immobile_0d_to_modelib(Y, self.omega,
                                                   variant_weights)
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
