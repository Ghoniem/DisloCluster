"""Take a fitted parameter set from the 0-D refit into a 3-D run.

A fitted set does not live in one place, and that is the whole difficulty. The
28 workbook parameters travel as `MATERIAL['overrides']`; `m_min`, `nu_vanish`
and `m_vanish` are solver switches and travel as `SOLVER['model_params']`; and
THE BIAS LIVES IN THE MATERIAL FILE, where it is read by both codes. Splitting
them by hand is how a "refitted" run ends up carrying half a fit.

    python -m dislocluster_code.fitting.apply_refit <refit-dir> --show
    python -m dislocluster_code.fitting.apply_refit <refit-dir> --apply

`--show` prints the split and what the material file would become. `--apply`
writes the material file and saves a timestamped backup beside it.

WHY THE BIAS CANNOT SIMPLY BE WRITTEN IN. `p_m` sets the diffusion tensor AND
the capture efficiencies, and `staging/anisotropy.py` generates both from one
number precisely so they cannot disagree. Writing `dadAnisotropy` alone would
leave the migration energies describing a different anisotropy than the
efficiencies use -- the exact defect that module exists to prevent. So `p_m`
goes through `anisotropy.apply`, which rewrites the energies with it. `dadZ0`
and `loopSinkScale` are independent calibrations and are written directly.

THE MATERIAL FILE IS SHARED. Every staged case hashes it into `domain_key`, so
applying a refit re-stages every case -- which is correct, they are a different
model now -- and any run in flight against the old values is reading a file
that changed underneath it. Apply between runs, not during one.
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dislocluster_code import paths
from dislocluster_code.fitting import fit_cloops as F
from dislocluster_code.staging import anisotropy

#: Fitted names that are BIAS, and where each one lands in the material file.
BIAS_KEYS = ('dad_p_v', 'dad_p_i', 'dad_Z0_v', 'dad_Z0_i',
             'loop_sink_scale_c')

#: Fitted names that are solver switches rather than workbook parameters.
SOLVER_KEYS = ('m_min', 'chi', 'nu_vanish', 'm_vanish')


def split(params):
    """(workbook overrides, solver model_params, bias) from a fitted set."""
    workbook, solver, bias = {}, {}, {}
    for k, v in params.items():
        if k in BIAS_KEYS:
            bias[k] = v
        elif k in SOLVER_KEYS:
            solver[k] = v
        elif k in F.MODEL_LEVERS:
            solver[k] = v
        else:
            workbook[k] = v
    # `chi = 1` and `m_min = 1` are the solver's own defaults; emitting them
    # changes nothing but does change the command line, and a command line that
    # differs is a run that cannot be diffed against the artifacts before it.
    for k, default in (('chi', 1.0), ('m_min', 1.0)):
        if k in solver and solver[k] == default:
            del solver[k]
    if solver.get('nu_vanish', 0.0) <= 0.0 or solver.get('m_vanish', 0.0) <= 0.0:
        # The leak needs BOTH to be positive; the C++ gates on that. Half of it
        # is not a weaker leak, it is no leak, so ship neither rather than one.
        solver.pop('nu_vanish', None)
        solver.pop('m_vanish', None)
    return workbook, solver, bias


def _replace_vector(txt, key, values, comment):
    body = " ".join(f"{v:.10g}" for v in values)
    new, n = re.subn(rf"^{key}\s*=.*?;[^\n]*", f"{key}={body};{comment}",
                     txt, count=1, flags=re.S | re.M)
    if n != 1:
        raise RuntimeError(f"{key} not found in the material file")
    return new


def material_edits(bias, material_file=None, dry_run=True, T=573.0):
    """Write the bias into the material file. Returns a description."""
    mf = Path(material_file or paths.MODELIB_MATERIAL)
    from dislocluster_code.coupling.field import read_material_vector
    pm = list(read_material_vector(mf, 'dadAnisotropy', 4))
    z0 = list(read_material_vector(mf, 'dadZ0', 4))
    ls = list(read_material_vector(mf, 'loopSinkScale'))
    before = dict(dadAnisotropy=list(pm), dadZ0=list(z0), loopSinkScale=list(ls))

    if 'dad_p_v' in bias:
        pm[0] = float(bias['dad_p_v'])
    if 'dad_p_i' in bias:
        # The clusters follow the monomer. THE FIT ONLY SEES p_i THROUGH THE
        # CLUSTERS' ARRIVAL, which is 3% of the interstitial flux, so it cannot
        # resolve them separately -- giving 2i and 3i their own fitted values
        # would be reporting three numbers where the data supports one.
        pm[1] = pm[2] = pm[3] = float(bias['dad_p_i'])
    if 'dad_Z0_v' in bias:
        z0[0] = float(bias['dad_Z0_v'])
    if 'dad_Z0_i' in bias:
        z0[1] = z0[2] = z0[3] = float(bias['dad_Z0_i'])
    if 'loop_sink_scale_c' in bias:
        ls[0] = float(bias['loop_sink_scale_c'])
        if len(ls) >= 8:
            # Slot 7 is c_p, the collapsed pyramid -- also a basal family, and
            # it carried the same 0.291528 as c_f before this. Keeping the two
            # equal is a modelling choice and it is stated here rather than
            # left implicit in a number.
            ls[7] = ls[0]

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not dry_run:
        shutil.copy2(mf, mf.with_suffix(f'.txt.bak_{stamp}'))
        # p_m FIRST and through anisotropy.apply, so the migration energies are
        # regenerated with it. Doing this after the direct edits below would be
        # equally correct; doing it with a regex INSTEAD of anisotropy.apply
        # would not, and that is the failure this ordering makes obvious.
        anisotropy.apply(pm, material_file=mf, T=T)
        txt = mf.read_text(encoding='utf-8')
        txt = _replace_vector(
            txt, 'dadZ0', z0,
            '  # Z0_m per mobile species. FITTED against loop densities; '
            'see the refit report.')
        txt = _replace_vector(
            txt, 'loopSinkScale', ls,
            '  # [-] per immobile family. The <c> entries are FITTED; '
            'see the refit report.')
        mf.write_text(txt, encoding='utf-8')

    return {'file': mf, 'before': before,
            'after': dict(dadAnisotropy=pm, dadZ0=z0, loopSinkScale=ls),
            'backup': (mf.with_suffix(f'.txt.bak_{stamp}') if not dry_run
                       else None)}


def _woo(z0, pm):
    return z0 * pm, z0 * (pm + pm ** -2) / 2.0


def show(params, edits):
    workbook, solver, bias = split(params)
    print(f"  {'workbook overrides':<28}{len(workbook)} parameters")
    print(f"  {'solver model_params':<28}{solver}")
    print(f"  {'bias -> material file':<28}{edits['file']}")
    b, a = edits['before'], edits['after']
    for key in ('dadAnisotropy', 'dadZ0', 'loopSinkScale'):
        print(f"    {key}")
        print(f"      before  {' '.join(f'{v:.6g}' for v in b[key])}")
        print(f"      after   {' '.join(f'{v:.6g}' for v in a[key])}")
    zb_v, zp_v = _woo(a['dadZ0'][0], a['dadAnisotropy'][0])
    zb_i, zp_i = _woo(a['dadZ0'][1], a['dadAnisotropy'][1])
    print(f"\n    the Woo efficiencies that implies")
    print(f"      Z_basal(v)      {zb_v:.6f}   <c> gain")
    print(f"      Z_basal(i)      {zb_i:.6f}   <c> loss")
    print(f"      Z_prismatic(i)  {zp_i:.6f}   <a> gain")
    print(f"      Z_prismatic(v)  {zp_v:.6f}   <a> loss")
    pv, pi = a['dadAnisotropy'][0], a['dadAnisotropy'][1]
    print(f"      co-growth p_i < p_v : {pi:.4f} < {pv:.4f}  "
          f"{'OK' if pi < pv else 'VIOLATED'}")
    print(f"      implied D_par/D_bas : v {pv**6:.2f}   i {pi**6:.3f}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    src = argv[0]
    do_apply = '--apply' in argv
    from dislocluster_code.fitting.refit_report import load
    params, meta = load(src)
    T = 573.0
    edits = material_edits(*[dict(split(params)[2])], dry_run=not do_apply, T=T)
    show(params, edits)
    if do_apply:
        print(f"\n  applied. backup at {edits['backup'].name}")
        print("  every staged case will re-stage: domain_key hashes this file.")
    else:
        print("\n  dry run -- nothing written. Re-run with --apply.")
    return params


if __name__ == '__main__':
    main()
