"""Package a fitted parameter set: figures, tables and provenance.

A refit's CSV is not a result anyone can check. This turns one into a run
directory of the same shape every other driver writes -- figures, the numbers
behind them, and a provenance file that records the formulation, the switches
and the git hash the set was produced at -- so the fit can be read, argued
with, and reproduced.

    python -m dislocluster_code.fitting.refit_report <refit-dir> [--tag NAME]

`<refit-dir>` is a directory `refit_production` wrote (it holds
optimal_parameters.csv and refit.json), or the CSV itself.

WHAT THE FIGURES SHOW, and one thing they deliberately do not.
Each panel is one temperature: model trajectory against dose, with the
experimental points on it. THE MODEL CARRIES ONE MEAN SIZE PER FAMILY, so the
"diameter" curve is a mean and there are no error bars on it that would mean
anything -- the spread a real micrograph shows is a distribution the 0-D does
not resolve. Do not read agreement in the mean as agreement in the
distribution.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import dislocluster_code.fitting.fit_cloops as F
from dislocluster_code import paths

FAMILY_LABEL = {'A': r'$\langle a \rangle$ prismatic (interstitial)',
                'C': r'$\langle c \rangle$ basal (vacancy)'}


def _resolve(where):
    """A refit directory, given either an absolute path or a bare name."""
    p = Path(where)
    return p if p.is_absolute() or p.exists() else paths.OUTPUT_DIR / where


def load(where):
    """(params, meta) from a refit directory or its CSV."""
    p = _resolve(where)
    meta = {}
    if p.is_dir():
        j = p / 'refit.json'
        if j.exists():
            meta = json.loads(j.read_text())
        p = p / 'optimal_parameters.csv'
    with open(p, newline='') as fh:
        rows = list(csv.DictReader(fh))
    key = 'refitted' if 'refitted' in rows[0] else 'optimal'
    return {r['parameter']: float(r[key]) for r in rows}, meta


def _trajectories(pdict):
    """Model history at every evaluated temperature, keyed T -> (dose, series)."""
    tm = [(T, max(float(F.GROUPS[k][T]['dpa'].max())
                  for k in ('A', 'C') if T in F.GROUPS[k]))
          for T in F.EVAL_TEMPS]
    return F.model_history_batch(pdict, tm)


def figures(pdict, out, hist=None):
    """One figure per family: a grid of temperature panels, N and d vs dose."""
    hist = hist or _trajectories(pdict)
    written = []
    for fam in ('A', 'C'):
        temps = [T for T in F.EVAL_TEMPS if T in F.GROUPS[fam]]
        if not temps:
            continue
        for what, idx, ylab, logy in (
                ('density', 0, r'$N_L$  [m$^{-3}$]', True),
                ('diameter', 1, r'$d$  [nm]', False)):
            n = len(temps)
            ncol = min(4, n)
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow),
                                     squeeze=False)
            for ax in axes.ravel()[n:]:
                ax.axis('off')
            for j, T in enumerate(temps):
                ax = axes.ravel()[j]
                h = hist.get(T)
                g = F.GROUPS[fam][T]
                col = 'N_L' if idx == 0 else 'd'
                ok = g[col].notna().to_numpy()
                if h is not None:
                    dose, ser = h
                    ax.plot(dose, ser[fam][idx], '-', lw=1.6, color='C0',
                            label='model')
                if ok.any():
                    ax.plot(g['dpa'].to_numpy()[ok], g[col].to_numpy()[ok],
                            'o', ms=5, color='C3', label='experiment')
                ax.set_xscale('log')
                if logy:
                    ax.set_yscale('log')
                ax.set_title(f"{T:.0f} K", fontsize=10)
                ax.set_xlabel('dose [dpa]', fontsize=9)
                ax.set_ylabel(ylab, fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.3, which='both', lw=0.4)
                if j == 0:
                    ax.legend(fontsize=8, frameon=False)
            fig.suptitle(f"{FAMILY_LABEL[fam]} -- {what}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            f = out / f"fit_{fam}_{what}.png"
            fig.savefig(f, dpi=150)
            plt.close(fig)
            written.append(f)
    return written


def parity(pdict, out, hist=None):
    """Model against experiment, every point, both families on one figure.

    A parity plot is the one view in which a fit cannot hide: a systematic
    offset is a line off the diagonal, and the factor-of-N bands say how far
    off without anyone having to read a ratio out of a table.
    """
    hist = hist or _trajectories(pdict)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    stats = {}
    for ax, (what, idx, lab, logs) in zip(axes, (
            ('density', 0, r'$N_L$  [m$^{-3}$]', True),
            ('diameter', 1, r'$d$  [nm]', True))):
        allx, ally = [], []
        for fam, mk, c in (('A', 'o', 'C0'), ('C', 's', 'C3')):
            xs, ys = [], []
            for T in F.EVAL_TEMPS:
                if T not in F.GROUPS[fam] or hist.get(T) is None:
                    continue
                dose, ser = hist[T]
                g = F.GROUPS[fam][T]
                col = 'N_L' if idx == 0 else 'd'
                ok = g[col].notna().to_numpy()
                if not ok.any():
                    continue
                N, d = F._sample(dose, ser[fam][0], ser[fam][1],
                                 g['dpa'].to_numpy())
                sim = (N if idx == 0 else d)[ok]
                xs.extend(g[col].to_numpy()[ok]); ys.extend(sim)
            if xs:
                ax.plot(xs, ys, mk, ms=5, mfc='none', color=c,
                        label=f"$\\langle {fam.lower()} \\rangle$  "
                              f"({len(xs)} points)")
                allx.extend(xs); ally.extend(ys)
                r = np.log10(np.maximum(np.array(ys), 1e-30)
                             / np.array(xs))
                stats[f"{fam}_{what}"] = (float(np.mean(r)), float(np.std(r)),
                                          len(xs))
        if allx:
            lo = min(min(allx), min(ally)) * 0.5
            hi = max(max(allx), max(ally)) * 2.0
            ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
            for f_ in (2.0, 10.0):
                ax.plot([lo, hi], [lo * f_, hi * f_], 'k--', lw=0.6, alpha=0.5)
                ax.plot([lo, hi], [lo / f_, hi / f_], 'k--', lw=0.6, alpha=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if logs:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(f"experiment  {lab}")
        ax.set_ylabel(f"model  {lab}")
        ax.set_title(what)
        ax.grid(alpha=0.3, which='both', lw=0.4)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('model vs experiment  (dashed: factors of 2 and 10)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = out / 'fit_parity.png'
    fig.savefig(f, dpi=150)
    plt.close(fig)
    return f, stats


def point_table(pdict, hist=None):
    """Every experimental point, model against measurement."""
    hist = hist or _trajectories(pdict)
    rows = []
    for fam in ('A', 'C'):
        for T in F.EVAL_TEMPS:
            if T not in F.GROUPS[fam] or hist.get(T) is None:
                continue
            dose, ser = hist[T]
            g = F.GROUPS[fam][T]
            N, d = F._sample(dose, ser[fam][0], ser[fam][1],
                             g['dpa'].to_numpy())
            for j, row in enumerate(g.itertuples()):
                rows.append({
                    'family': fam, 'T_K': T, 'dpa': row.dpa,
                    'N_exp': row.N_L, 'N_model': N[j],
                    'N_ratio': (N[j] / row.N_L) if row.N_L == row.N_L else '',
                    'd_exp': row.d, 'd_model': d[j],
                    'd_ratio': (d[j] / row.d) if row.d == row.d else ''})
    return rows


def bias_check(p):
    """The Woo efficiencies the fitted bias implies, and the co-growth test."""
    pv, pi = p.get('dad_p_v'), p.get('dad_p_i')
    z0v, z0i = p.get('dad_Z0_v'), p.get('dad_Z0_i')
    if None in (pv, pi, z0v, z0i):
        return None
    basal = lambda z0, pm: z0 * pm
    prism = lambda z0, pm: z0 * (pm + pm ** -2) / 2.0
    return {
        'p_v': pv, 'p_i': pi, 'Z0_v': z0v, 'Z0_i': z0i,
        'Z_basal_v': basal(z0v, pv), 'Z_basal_i': basal(z0i, pi),
        'Z_prismatic_v': prism(z0v, pv), 'Z_prismatic_i': prism(z0i, pi),
        'co_growth_p_i_lt_p_v': bool(pi < pv),
        'D_ratio_v': pv ** 6, 'D_ratio_i': pi ** 6,
    }


def report(refit_dir, tag=None):
    params, meta = load(refit_dir)
    model = meta.get('model') or dict(loop_model=1, n_fam=9, moments=1,
                                      emission_model=1,
                                      variant_weights=(1 / 3, 1 / 3, 1 / 3))
    F.set_model(**model)
    F.DIAM_LOG = bool(meta.get('diam_log', False))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"{stamp}_{paths.git_hash()}_{tag or 'refit_report'}"
    out = paths.OUTPUT_DIR / name
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    hist = _trajectories(params)
    figs = figures(params, out / 'figures', hist)
    pf, stats = parity(params, out / 'figures', hist)
    figs.append(pf)

    J, bd = F.objective_full(params, want_breakdown=True)
    rows = point_table(params, hist)
    with open(out / 'model_vs_experiment.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(out / 'parameters.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['parameter', 'value'])
        for k in sorted(params):
            w.writerow([k, params[k]])

    bias = bias_check(params)
    rails = []
    for k, v in params.items():
        if k not in F.CAND:
            continue
        lo, hi, lg = F.CAND[k]
        if abs(v - lo) <= 1e-9 * max(1.0, abs(lo)) or \
           abs(v - hi) <= 1e-9 * max(1.0, abs(hi)):
            rails.append(k)

    lines = [f"# Refit report -- {name}", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')}",
             f"from `{refit_dir}`", "",
             "## Formulation", "", "```"]
    for k, v in model.items():
        lines.append(f"{k} = {v}")
    lines += [f"diameter term = "
              f"{'log ratio' if F.DIAM_LOG else 'relative residual'}", "```", "",
              "## Objective", "",
              f"| | J | J_A | J_C |", "|---|---:|---:|---:|",
              f"| refitted | {J:.4f} | {bd['A']['J']:.4f} | {bd['C']['J']:.4f} |"]
    if meta.get('baseline_J'):
        lines.append(f"| stale set | {meta['baseline_J']:.4f} | | |")
    lines += ["", f"overshoot term  {bd.get('J_os', float('nan')):.4f}", ""]

    lines += ["## Agreement, in log10 ratio (model / experiment)", "",
              "| quantity | mean | sd | points |", "|---|---:|---:|---:|"]
    for k, (m, s, n) in stats.items():
        lines.append(f"| {k} | {m:+.3f} | {s:.3f} | {n} |")
    lines += ["", "A mean of 0 is unbiased; +0.3 is a factor of 2 high.", ""]

    if bias:
        lines += ["## The bias the fit chose", "", "```"]
        for k, v in bias.items():
            lines.append(f"{k:<24}{v}")
        lines += ["```", "",
                  "`p_m = (D_parallel/D_basal)^(1/6)`, so `D_ratio` is what the "
                  "fitted anisotropy implies for the diffusion tensor. THAT IS "
                  "THE NUMBER TO CHECK AGAINST THE MIGRATION ENERGIES -- a fit "
                  "is free to choose an anisotropy no atomistics supports.", ""]

    if rails:
        lines += ["## Levers resting on a bound", "",
                  "**A lever on its bound is not a fitted value.** The search "
                  "wanted to go further and the box stopped it, so the number "
                  "below is a floor or a ceiling, not an optimum:", ""]
        for k in sorted(rails):
            lo, hi, _ = F.CAND[k]
            lines.append(f"- `{k}` = {params[k]:.6g}  (bounds {lo:g} .. {hi:g})")
        lines.append("")

    lines += ["## Parameters", "", "| parameter | value |", "|---|---:|"]
    for k in sorted(params):
        lines.append(f"| `{k}` | {params[k]:.6g} |")
    lines += ["", "## Provenance", "", "```",
              f"git             {paths.git_hash()}",
              f"generated       {datetime.now().isoformat(timespec='seconds')}",
              f"source refit    {refit_dir}",
              f"targets         Targets_A / Targets_C in "
              f"{F.XL.name if hasattr(F, 'XL') else 'the workbook'}",
              f"a-loop points   {sum(1 for r in rows if r['family'] == 'A')}",
              f"c-loop points   {sum(1 for r in rows if r['family'] == 'C')}",
              "```", "",
              "### What this set is NOT calibrated for", "",
              "- the basal chain: every rate defaults to zero and none is "
              "fitted here, so `basal_chain = 1` runs a chain that does "
              "nothing. Do not read these parameters as endorsing it.",
              "- the discrete handoff and the hardening module take their own "
              "inputs; nothing here was fitted against either.",
              "- `m_vanish` and `nu_vanish` are the small-end leak's "
              "thresholds. They are fitted HERE, against loop densities, and "
              "no independent measurement pins them.", ""]
    (out / 'provenance.md').write_text("\n".join(lines), encoding='utf-8')

    src = _resolve(refit_dir)
    if src.is_dir():
        for f in ('refit.md', 'refit.json', 'optimal_parameters.csv'):
            if (src / f).exists():
                shutil.copy2(src / f, out / f)

    print(f"J {J:.4f}   J_A {bd['A']['J']:.4f}   J_C {bd['C']['J']:.4f}")
    for f in figs:
        print(f"  figure  {f.relative_to(out)}")
    print(f"\nwrote {out}")
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    tag = None
    if '--tag' in args:
        tag = args[args.index('--tag') + 1]
        args = [a for i, a in enumerate(args)
                if i not in (args.index('--tag'), args.index('--tag') + 1)]
    if not args:
        raise SystemExit(__doc__)
    report(args[0], tag)
