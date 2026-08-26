"""loop_annealing_population_figs.py -- the three figures of the population anneal.

Companion to :mod:`dislocluster_code.studies.loop_annealing_population`, kept
separate so that module stays readable. Nothing here computes physics; every
number comes from that module.

    figure_ladder     Fig. (9.16) of the thesis, redrawn for the nine families:
                      the concentration ladder, and which families grow.
    figure_ripening   the same question once each family carries a SIZE
                      DISTRIBUTION -- the level becomes a curve in R and cbar_v
                      cuts it at a critical radius.
    figure_reference  Sec. 4.3's calculation repeated on the 300 nm reference
                      microstructure, isolated against population.
"""

from __future__ import annotations

import math

import numpy as np

from dislocluster_code.studies import loop_annealing as la
from dislocluster_code.studies import loop_annealing_greens as gr
from dislocluster_code.studies import loop_annealing_population as P
from dislocluster_code.studies.loop_annealing import families, load_material

C_C = "#1f5fa8"          # <c> basal vacancy
C_A = "#d1690c"          # <a> prismatic interstitial
C_N = "#4f5b66"          # network
C_BAR = "#b3243c"        # cbar_v
GROW = "#2e7d32"
SHRINK = "#b3243c"

FAM_COLOR = {"c": C_C, "a1": C_A, "a2": C_A, "a3": C_A, "network": C_N}

# The capture efficiency used on BOTH sides of Eq. (9.20) -- see
# `loop_annealing_population.sink_strengths` for why it must be one choice.
WEIGHT = "iso"


# ── (4) the ladder: which families grow and which shrink ─────────────────────

def figure_ladder(out, T=873.0, run_dir=None, dose=0.1, c_state="c_f",
                  material_file=None, doses=(0.01, 0.1, 1.0, 10.0)):
    """The concentration ladder, with one size per family.

    Panel (a) is the direct descendant of Fig. (9.16): a vertical axis of vacancy
    concentration, one rung per component at its own ``c^{v,eq}_k``, and the
    solved ``cbar_v`` of Eq. (9.20) drawn across them. Voids are gone -- Zr has
    none in this model -- and the rungs are the nine families instead. **A
    component above the cbar_v line is a net EMITTER, one below it a net
    ABSORBER**, and what that does to its size then depends on its character,
    which is why the arrows are labeled rather than left to be inferred.

    The rung widths are the sink-strength weights of Eq. (9.20), so the figure
    also shows WHY cbar_v sits where it does: it is dragged toward whichever
    component has the widest rung.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_material(material_file)
    cinf = mat.c_v_inf(T)
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.4),
                           gridspec_kw=dict(width_ratios=(1.15, 1.0)))

    # ---- panel (a): the ladder at one dose -----------------------------------
    state = P.reference_state(run_dir, dose)
    comps = P.components_from_state(state, mat, c_state,
                                    material_file=material_file)
    cbar, S, ceq = P.remote_concentration(comps, T, mat, WEIGHT, material_file,
                                          return_parts=True)
    w = S / S.sum()
    a = ax[0]
    drawn = []
    for c, lev, wt in zip(comps, ceq, w):
        if c.name in ("a2", "a3"):
            continue
        wt = wt * 3 if c.name == "a1" else wt          # the three variants
        drawn.append((c, lev, wt))
    for c, lev, wt in drawn:
        y = lev / cinf
        col = FAM_COLOR[c.name]
        half = 0.30 + 0.55 * wt
        a.plot([0.5 - half, 0.5 + half], [y, y], "-", color=col, lw=3.0,
               solid_capstyle="butt", zorder=3)
        name = {"c": r"$\langle c\rangle$ basal vacancy",
                "a1": r"$\langle a\rangle_{1,2,3}$ prismatic interstitial",
                "network": "network dislocations"}[c.name]
        a.text(0.5 + half + 0.04, y, f"{name}\n$S/\\Sigma S={wt:.2f}$",
               va="center", ha="left", fontsize=8.4, color=col)
    a.axhline(cbar / cinf, color=C_BAR, lw=2.4, ls="--", zorder=4)
    a.text(0.02, cbar / cinf, r"$\bar c_v$  (Eq. 9.20)", color=C_BAR,
           fontsize=9.5, va="bottom", ha="left", weight="bold")
    a.axhline(1.0, color="0.6", lw=1.0, ls=":", zorder=1)
    a.text(0.02, 1.0, r"$c^{v,\rm eq}_\infty$", color="0.35", fontsize=9,
           va="top", ha="left")
    # arrows: every loop family, from its own rung to cbar_v, labeled by fate
    for c, lev, wt in drawn:
        if isinstance(c, P.Network):
            continue
        y0, y1 = lev / cinf, cbar / cinf
        rate = P.dRdt(c, cbar, T, mat)
        # Above the line the component is a net EMITTER, below it a net
        # ABSORBER; what that does to its SIZE then depends on its character,
        # which is why both are written out rather than left to be inferred.
        role = "net EMITTER" if y0 > y1 else "net ABSORBER"
        fate = "GROWTH" if rate > 0 else "SHRINKAGE"
        col = GROW if rate > 0 else SHRINK
        x = 0.30 if c.name == "c" else 0.70
        a.annotate("", xy=(x, y1), xytext=(x, y0),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                   connectionstyle="arc3,rad=0.0"))
        a.text(x + 0.02, math.sqrt(max(y0 * y1, 1e-30)),
               f"{role}\n$\\Rightarrow$ {fate}", color=col, fontsize=8.4,
               rotation=90, va="center", ha="left", weight="bold",
               linespacing=1.5)
    a.set_yscale("log")
    a.set_xlim(0, 1.75)
    a.set_xticks([])
    a.set_ylabel(r"vacancy concentration  $/\,c^{v,\rm eq}_\infty$")
    a.set_title(r"(a)  the ladder at $%g$ dpa, $T=%g$ K   "
                r"($\bar c_v/c^{v,\rm eq}_\infty=%.3f$)"
                % (state["dose"], T, cbar / cinf), fontsize=10, pad=12)
    a.grid(alpha=0.22, lw=0.5, axis="y", which="both")

    # ---- panel (b): the same across dose --------------------------------------
    b = ax[1]
    rows = []
    for d in doses:
        st = P.reference_state(run_dir, d)
        cs = P.components_from_state(st, mat, c_state,
                                     material_file=material_file)
        cb, Sk, lv = P.remote_concentration(cs, T, mat, WEIGHT, material_file,
                                            return_parts=True)
        rows.append((st["dose"], cb, cs, lv))
    x = np.arange(len(rows))
    b.plot(x, [r[1] / cinf for r in rows], "o--", color=C_BAR, lw=2.2, ms=7,
           label=r"$\bar c_v$, Eq. (9.20)", zorder=5)
    for key, col, lab in (("c", C_C, r"$\langle c\rangle$ own level"),
                          ("a1", C_A, r"$\langle a\rangle$ own level")):
        y = []
        for d, cb, cs, lv in rows:
            i = [c.name for c in cs].index(key)
            y.append(lv[i] / cinf)
        b.plot(x, y, "s-", color=col, lw=2.0, ms=6, label=lab)
    b.axhline(1.0, color="0.6", lw=1.0, ls=":",
              label=r"$c^{v,\rm eq}_\infty$ (network)")
    b.set_yscale("log")
    b.set_xticks(x)
    b.set_xticklabels([f"{r[0]:g}" for r in rows])
    b.set_xlabel("dose  [dpa]")
    b.set_ylabel(r"vacancy concentration  $/\,c^{v,\rm eq}_\infty$")
    b.set_title(r"(b)  $\bar c_v$ stays between the two families at every dose",
                fontsize=10, pad=12)
    b.legend(fontsize=8.2, frameon=False, loc="center right")
    b.grid(alpha=0.22, lw=0.5, which="both")

    fig.suptitle(r"Which loops grow and which shrink: one size per family, "
                 r"$\langle c\rangle$ as $%s$" % c_state.replace("_", "_"),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return out


# ── (5) the same question with size distributions ────────────────────────────

def figure_ripening(out, T=873.0, run_dir=None, dose=0.1, c_state="c_f",
                    material_file=None, sigma_ln=0.45):
    """What a SIZE DISTRIBUTION changes.

    With one size per family, Eq. (9.20) decides the fate of a whole family at
    once. With a distribution, each family's level is a CURVE ``c^{v,eq}_k(R)``
    and ``cbar_v`` is still a single number, so the two can CROSS -- at a
    critical radius ``R*_k`` that splits the family in two.

    Panel (a) draws those curves and marks every crossing. Panel (b) puts the
    reference microstructure's own distributions underneath, so the reader can
    see which side they are on. Panel (c) is the contrast case: the SAME <c>
    population with the network and the <a> loops removed, which is the only way
    to put cbar_v inside the distribution -- and then the family ripens instead
    of dissolving.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_material(material_file)
    fams = families(mat)
    cinf = mat.c_v_inf(T)
    state = P.reference_state(run_dir, dose)
    comps = P.components_from_state(state, mat, c_state,
                                    material_file=material_file)
    cbar = P.remote_concentration(comps, T, mat, WEIGHT, material_file)

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))
    R = np.geomspace(mat.r_min, 1e-6, 400)

    # ---- (a) the level curves and the crossings ------------------------------
    #
    # Each basal state is drawn against the cbar_v THAT STATE'S OWN mixture
    # produces, because the mapping changes both the curve and the line: with the
    # faulted state the <c> family emits harder, cbar_v is 1.86 c_inf, and the
    # curve's own floor is ABOVE it, so there is no crossing at all.
    a = ax[0]
    cbars = {}
    for st in ("c_f", "c_p"):
        cs = P.components_from_state(state, mat, st, material_file=material_file)
        cbars[st] = P.remote_concentration(cs, T, mat, WEIGHT, material_file)
    cbars["a_i"] = cbars[c_state]
    curves = (("c_f", C_C, "-", r"$c_f$ basal, FAULTED"),
              ("c_p", "#5b9bd5", "--", r"$c_p$ basal, perfect"),
              ("a_i", C_A, "-", r"$a_i$ prismatic, interstitial"))
    for key, col, ls, lab in curves:
        fam = fams[key]
        cb = cbars[key]
        lev = gr.c_line_local(1.0 / R, fam, T, mat) / cinf
        a.loglog(R * 1e9, lev, ls, color=col, lw=2.1, label=lab)
        if key != "a_i":
            a.axhline(cb / cinf, color=col, lw=1.6, ls="--", alpha=0.75)
            a.text(1.15 if key == "c_f" else 12.0, cb / cinf,
                   r"$\bar c_v(%s)$" % key, color=col, fontsize=8.2,
                   va="bottom" if key == "c_f" else "top")
        floor = (cinf * math.exp(fam.gamma * fam.omega / fam.bdotn
                                 / (P.KB_EV * T * P.EV_J)) / cinf
                 if fam.gamma else 1.0)
        a.plot([R[-1] * 1e9 * 0.55, R[-1] * 1e9], [floor, floor], ":",
               color=col, lw=1.2, alpha=0.9)
        Rc = P.critical_radius(fam, cb, T, mat)
        if Rc is not None and R[0] <= Rc <= R[-1]:
            a.plot([Rc * 1e9], [cb / cinf], "o", ms=9, mfc="white", mec=col,
                   mew=2.2, zorder=6)
            a.annotate(r"$R^\ast=%.0f$ nm" % (Rc * 1e9),
                       (Rc * 1e9, cb / cinf), textcoords="offset points",
                       xytext=(-8, -16), ha="right", fontsize=8.6, color=col,
                       weight="bold")
    a.annotate(r"$c_f$ floor $2.53\,c^{v,\rm eq}_\infty$ is ABOVE its"
               "\n" r"own $\bar c_v$: NO $R^\ast$, all sizes shrink",
               (R[-1] * 1e9 * 0.9, 2.53), textcoords="offset points",
               xytext=(-4, 26), ha="right", fontsize=8.0, color=C_C)
    a.axhline(1.0, color="0.6", lw=1.0, ls=":")
    a.set_xlabel(r"loop radius  $R$  [nm]")
    a.set_ylabel(r"$c^{v,\rm eq}_k(R)\,/\,c^{v,\rm eq}_\infty$")
    a.set_title(r"(a)  each family's level is a CURVE in $R$", fontsize=10,
                pad=12)
    a.legend(fontsize=7.6, frameon=False, loc="upper right")
    a.grid(alpha=0.22, lw=0.5, which="both")

    # ---- (b) the reference distributions, on their side of cbar_v ------------
    b = ax[1]
    for key, col, lab in (("c", C_C, r"$\langle c\rangle$"),
                          ("a1", C_A, r"$\langle a\rangle$ (per variant)")):
        c = next(x for x in comps if getattr(x, "name", None) == key)
        Rb, nb = P.lognormal_bins(c.n, c.R, sigma_ln)
        b.semilogx(Rb * 1e9, nb / nb.max(), "-", color=col, lw=2.1,
                   label=lab + r"  $\bar R=%.1f$ nm" % (c.R * 1e9))
        b.fill_between(Rb * 1e9, 0, nb / nb.max(), color=col, alpha=0.16)
        Rc = P.critical_radius(c.fam, cbar, T, mat)
        if Rc is not None:
            b.axvline(Rc * 1e9, color=col, ls="--", lw=1.5)
            b.annotate(r"$R^\ast_{%s}$" % key, (Rc * 1e9, 0.9), color=col,
                       fontsize=9, ha="left", xytext=(4, 0),
                       textcoords="offset points")
    b.set_xlabel(r"loop radius  $R$  [nm]")
    b.set_ylabel("population density  (normalized)")
    b.set_ylim(0, 1.12)
    b.set_title(r"(b)  the $%g$ dpa distributions lie ENTIRELY on the "
                r"shrinking side" % state["dose"], fontsize=9.6, pad=12)
    b.legend(fontsize=8.0, frameon=False, loc="upper left")
    b.grid(alpha=0.22, lw=0.5, which="both")

    # ---- (c) the contrast: the same <c> population, alone -> it ripens --------
    c_ax = ax[2]
    cc = next(x for x in comps if getattr(x, "name", None) == "c")
    # More bins than panel (b) uses: cbar_v is a sum over DELTA functions, so
    # every bin that dissolves steps it, and R*(t) inherits that sawtooth.
    # 81 bins halves the step size; it does not remove it, and the caption
    # says so rather than the curve being smoothed.
    Rb, nb = P.lognormal_bins(cc.n, cc.R, sigma_ln, n_bins=81)
    d0 = P.DistributionFamily("c", fams[c_state], Rb, nb)
    cb_alone = P.remote_concentration([d0], T, mat, WEIGHT, material_file)
    Rstar = P.critical_radius(d0.fam, cb_alone, T, mat)
    res = P.anneal_distribution([d0], T, mat, material_file=material_file,
                                stop_frac=0.05)
    ts = np.maximum(res["t"], 1.0) / 3600.0
    # Every bin's TRAJECTORY, colored by which side of R* it started on, with
    # the self-consistent R*(t) drawn through them. A snapshot of the density
    # cannot show this: the bins are Lagrangian, so what moves is the axis.
    Rtraj = np.array([s[0][0] for s in res["snaps"]])          # (n_t, n_bin)
    ntraj = np.array([s[0][1] for s in res["snaps"]])
    Rstar_t = []
    for i in range(len(ts)):
        d = P.DistributionFamily("c", fams[c_state], Rtraj[i], ntraj[i])
        cb = P.remote_concentration([d], T, mat, WEIGHT, material_file)
        Rstar_t.append(P.critical_radius(d.fam, cb, T, mat) or np.nan)
    for j in range(Rtraj.shape[1]):
        live = ntraj[:, j] > 0
        if live.sum() < 2:
            continue
        grows = Rtraj[0, j] > (Rstar or 0.0)
        c_ax.plot(ts[live], Rtraj[live, j] * 1e9, "-",
                  color=(GROW if grows else SHRINK), lw=1.0, alpha=0.55)
    c_ax.plot(ts, np.array(Rstar_t) * 1e9, "--", color=C_BAR, lw=2.4,
              label=r"$R^\ast(t)$, self-consistent")
    c_ax.plot([], [], "-", color=GROW, lw=1.4,
              label=r"bins starting above $R^\ast$  (grow)")
    c_ax.plot([], [], "-", color=SHRINK, lw=1.4,
              label=r"bins starting below $R^\ast$  (dissolve)")
    Rm0 = d0.R_mean * 1e9
    Rm1 = float((Rtraj[-1] * ntraj[-1]).sum() / max(ntraj[-1].sum(), 1e-300)) * 1e9
    N0, N1 = ntraj[0].sum(), ntraj[-1].sum()
    c_ax.set_xscale("log")
    c_ax.set_yscale("log")
    c_ax.set_xlabel("annealing time  [h]")
    c_ax.set_ylabel(r"loop radius  $R$  [nm]")
    c_ax.set_title(r"(c)  the same $\langle c\rangle$ loops ALONE: they RIPEN"
                   "\n" r"$\bar R:\,%.0f\to%.0f$ nm,  $N:\,%.1e\to%.1e$ m$^{-3}$"
                   % (Rm0, Rm1, N0, N1), fontsize=9.2, pad=10)
    c_ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    c_ax.grid(alpha=0.22, lw=0.5, which="both")

    fig.suptitle(r"Size distributions: $\bar c_v$ cuts each family at a "
                 r"critical radius $R^\ast$  ($T=%g$ K, $\sigma_{\ln}=%.2f$, "
                 r"$\langle c\rangle$ as $%s$)"
                 % (T, sigma_ln, c_state), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return out


# ── (6) the 300 nm reference microstructure ──────────────────────────────────

def figure_reference(out, T=873.0, run_dir=None, dose=0.1, c_state="c_f",
                     material_file=None, doses=(0.01, 0.1, 1.0, 10.0)):
    """Sec. 4.3's anneal, repeated on the 300 nm reference microstructure.

    Panel (a) is Fig. 3(a) redrawn: ``R(t)`` for the two families, first with the
    ambient PINNED at ``c^{v,eq}_inf`` (Sec. 4.3, dashed) and then with it SOLVED
    from Eq. (9.20) at every step (solid). Panel (b) is the number that makes the
    difference -- ``cbar_v`` during the anneal, which is not constant, because the
    microstructure that sets it is the one being dissolved. Panel (c) is the
    lifetime ratio across dose.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_material(material_file)
    cinf = mat.c_v_inf(T)
    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.4))

    state = P.reference_state(run_dir, dose)
    comps = P.components_from_state(state, mat, c_state,
                                    material_file=material_file)
    res = P.anneal_population(comps, T, mat, material_file=material_file)
    names = [c.name for c in res["comps"]]

    # R/R0 against LOG time: the two families differ by a factor of 12 in size
    # and a factor of 10 in lifetime, so a linear axis in either would hide one
    # of them entirely.
    a = ax[0]
    for key, col, lab in (("c", C_C, r"$\langle c\rangle$ basal vacancy"),
                          ("a1", C_A, r"$\langle a\rangle$ prismatic "
                                      r"interstitial")):
        j = names.index(key)
        live = res["n"][:, j] > 0
        c0 = next(x for x in comps if getattr(x, "name", None) == key)
        a.semilogx(np.maximum(res["t"][live], 1.0) / 3600.0,
                   res["R"][live, j] / c0.R, "-", color=col, lw=2.4,
                   label=lab + r",  population, $R_0=%.1f$ nm" % (c0.R * 1e9))
        t, R = la.anneal(c0.R, c0.fam, T, mat=mat)
        a.semilogx(np.maximum(t, 1.0) / 3600.0, R / c0.R, "--", color=col,
                   lw=1.9, label=lab + ",  isolated (Sec. 4.3)")
        tp = float(res["t"][live][-1])
        a.annotate(f"{la._fmt_time(t[-1])} $\\to$ {la._fmt_time(tp)}",
                   (max(t[-1], tp) / 3600.0, 0.30 if key == "a1" else 0.14),
                   textcoords="offset points", xytext=(-6, 0), ha="right",
                   fontsize=8.4, color=col, weight="bold")
    a.set_xlabel("annealing time  [h]")
    a.set_ylabel(r"$R/R_0$")
    a.set_ylim(0, 1.05)
    a.set_xlim(1e-3, None)
    a.set_title(r"(a)  $%g$ dpa microstructure, $T=%g$ K"
                % (state["dose"], T), fontsize=10, pad=12)
    a.legend(fontsize=7.2, frameon=False, loc="lower left")
    a.grid(alpha=0.22, lw=0.5, which="both")

    b = ax[1]
    b.semilogy(res["t"] / 3600.0, res["cbar"] / cinf, "-", color=C_BAR, lw=2.3)
    for key, col in (("a1", C_A), ("c", C_C)):
        j = names.index(key)
        gone = np.flatnonzero(res["n"][:, j] == 0.0)
        if gone.size:
            b.axvline(res["t"][gone[0]] / 3600.0, color=col, ls=":", lw=1.6)
            b.text(res["t"][gone[0]] / 3600.0, b.get_ylim()[1],
                   {"c": r" $\langle c\rangle$ gone",
                    "a1": r" $\langle a\rangle$ gone"}[key],
                   color=col, fontsize=8.4, va="top", rotation=90)
    b.axhline(1.0, color="0.6", lw=1.0, ls=":")
    b.set_xlabel("annealing time  [h]")
    b.set_ylabel(r"$\bar c_v\,/\,c^{v,\rm eq}_\infty$")
    b.set_title(r"(b)  the ambient is NOT constant: it is set by what is left",
                fontsize=9.6, pad=12)
    b.grid(alpha=0.22, lw=0.5, which="both")

    c_ax = ax[2]
    xs, iso, pop = [], {"c": [], "a1": []}, {"c": [], "a1": []}
    for d in doses:
        st = P.reference_state(run_dir, d)
        cs = P.components_from_state(st, mat, c_state,
                                     material_file=material_file)
        r = P.anneal_population(cs, T, mat, material_file=material_file)
        nm = [x.name for x in r["comps"]]
        xs.append(st["dose"])
        for key in ("c", "a1"):
            c0 = next(x for x in cs if getattr(x, "name", None) == key)
            iso[key].append(la.anneal_time(c0.R, c0.fam, T, mat=mat))
            j = nm.index(key)
            g = np.flatnonzero(r["n"][:, j] == 0.0)
            pop[key].append(float(r["t"][g[0]]) if g.size else float("nan"))
    x = np.arange(len(xs))
    for key, col, lab in (("c", C_C, r"$\langle c\rangle$"),
                          ("a1", C_A, r"$\langle a\rangle$")):
        c_ax.plot(x, np.array(pop[key]) / np.array(iso[key]), "o-", color=col,
                  lw=2.2, ms=7, label=lab)
    c_ax.axhline(1.0, color="0.4", lw=1.0)
    c_ax.set_xticks(x)
    c_ax.set_xticklabels([f"{v:g}" for v in xs])
    c_ax.set_xlabel("dose  [dpa]")
    c_ax.set_ylabel(r"$t_{\rm population}\,/\,t_{\rm isolated}$")
    c_ax.set_title("(c)  what the neighbors are worth", fontsize=10, pad=12)
    c_ax.legend(fontsize=8.6, frameon=False, loc="best")
    c_ax.grid(alpha=0.22, lw=0.5)

    fig.suptitle(r"The %g nm reference microstructure annealed: "
                 r"Sec. 4.3's calculation with $\bar c_v$ solved instead of "
                 r"pinned  ($\langle c\rangle$ as $%s$)"
                 % (P.REFERENCE_CUBE_NM, c_state), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return out
