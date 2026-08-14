"""Audit point-defect transport: attempt frequencies, jump frequencies, D_0,
and the recombination-vs-sink balance. Then test more realistic nu_i/nu_v and
recombination on x, the recombination fraction, and c-loop flux growth."""
import io, contextlib
import numpy as np
from dislocluster_code import paths as _paths
from dislocluster_code.zerod.simulation import ZrMicroSimulation
from dislocluster_code.zerod.cpp_bridge import run_cpp_solver
from dislocluster_code.zerod.reaction_rates import ReactionRates
from dislocluster_code.zerod.rate_equations import RateEquations
_N = contextlib.redirect_stdout(io.StringIO())
with _N:
    # was cwd-relative, so this only ran from ZrMicro/
    SIM = ZrMicroSimulation(_paths.INPUT_DIR / 'Zr_input_parameters.xlsx')
inp = SIM.input_data
pp = inp.physical_props
kB = 8.617e-5
T = 668.0
a = pp['a']; zc = pp['z_c']

print("=" * 74)
print("CURRENT TRANSPORT PARAMETERS (from Zr_input_parameters.xlsx)")
print("=" * 74)
print(f"  a       = {a:.3e} m     z_c = {zc}")
print(f"  nu_i    = {pp['nu_i']:.3e} Hz     E_m_i = {pp['E_m_i']:.3f} eV")
print(f"  nu_v    = {pp['nu_v']:.3e} Hz     E_m_v = {pp['E_m_v']:.3f} eV")
print(f"  recom   = {pp['recom']}")
print(f"\n  D_0_i = a^2*nu_i = {a*a*pp['nu_i']:.3e} m^2/s   "
      f"(physical Zr SIA ~1e-7)")
print(f"  D_0_v = a^2*nu_v = {a*a*pp['nu_v']:.3e} m^2/s   "
      f"(physical Zr vac ~1e-6 .. 1e-5)")
oi = zc*pp['nu_i']*np.exp(-pp['E_m_i']/(kB*T))
ov = zc*pp['nu_v']*np.exp(-pp['E_m_v']/(kB*T))
print(f"\n  at T={T:.0f} K:")
print(f"  omega_i = {oi:.3e} /s    D_i = {a*a/zc*oi:.3e} m^2/s")
print(f"  omega_v = {ov:.3e} /s    D_v = {a*a/zc*ov:.3e} m^2/s")
print(f"  omega_i/omega_v = {oi/ov:.2e}   (D_i/D_v same)")
print("  Physical Zr at ~670 K: D_i ~ 1e-12..1e-10, D_v ~ 1e-16..1e-14 m^2/s")
print("  => check whether nu_v (and E_m) make vacancies far too slow.")

# recombination vs sink balance proxy at a representative free Cv, Ci
def run_state(over, dpa=0.2):
    inp.material_params['T'] = T
    for k in ('nu_i', 'nu_v', 'recom'):
        if k in over: pp[k] = over[k]
    inp.model_params.update({'E_a_vL': over.get('E_a_vL', 1.3),
                             'delta_DAD': over.get('delta_DAD', 0.4),
                             'Z_N': over.get('Z_N', 1.3),
                             'n_vL_nuc': 30.0, 'tau_vL0': 1e-4})
    with _N:
        inp.calculate_derived_parameters()
        SIM.reaction_rates = ReactionRates(inp); SIM.rate_equations = RateEquations(inp, SIM.reaction_rates)
    G = inp.material_params['G']
    cfg = {'t_begin': 1e-1, 't_end': dpa/G*3, 'n_points': 150, 'rtol': 1e-6,
           'atol': 1e-20, 'log_time': True,
           'solver_method': {'backend': 'cvode', 'lmm': 'bdf', 'linsol': 'dense'}}
    with _N:
        r = run_cpp_solver(SIM, cfg, base_dir=_paths.ZRMICRO_DIR)
    rr, c = SIM.reaction_rates, r['concentrations']
    names = SIM.rate_equations.concentration_names
    dose = G*r['time']; i = int(np.argmin(np.abs(dose-dpa)))
    y = np.array([c[n][i] for n in names]); rr.update_state(y, r['time'][i])
    recomb = rr.R_i_v(y)+rr.R_2i_v(y)+rr.R_3i_v(y)
    sink_v = rr.R_v_s(y); la_i, la_v, _ = rr.loop_absorption(y)
    frac_recomb = recomb/max(rr.G_v(), 1e-300)
    gvL = rr.loop_growth_rate_vL(y)+rr.loop_growth_rate_avL(y)
    GvL = rr.G_vL()+rr.G_avL()
    return {'x': rr.flux_v()/rr.flux_i(), 'frac_recomb_v': frac_recomb,
            'growth/G': gvL/GvL, 'r_vL': r['loop_sizes']['r_vL'][i]*1e9,
            'Cv': c['Cv'][i], 'Ci': c['Ci'][i]}

print("\n" + "=" * 74)
print("EFFECT OF nu_v / nu_i / recom  (at 668 K, 0.2 dpa, with E_a_vL=1.3, DAD=0.4)")
print("=" * 74)
print(f"{'scenario':>26} | {'x':>7} | {'%recomb_v':>9} | {'growth/G':>9} | {'r_vL':>6}")
print("-"*74)
cases = [
    ('baseline (nu_v=2.3e8)', {}),
    ('nu_v 2.3e8->1e13',      {'nu_v': 1e13}),
    ('nu_i 3.8e11->1e13',     {'nu_i': 1e13}),
    ('both nu=1e13',          {'nu_v': 1e13, 'nu_i': 1e13}),
    ('both nu=1e13 + recom=1', {'nu_v': 1e13, 'nu_i': 1e13, 'recom': 1.0}),
    ('both nu=5e12 + recom=1', {'nu_v': 5e12, 'nu_i': 5e12, 'recom': 1.0}),
]
for name, over in cases:
    try:
        s = run_state(over)
        print(f"{name:>26} | {s['x']:>7.3f} | {s['frac_recomb_v']*100:>8.2f}% | "
              f"{s['growth/G']:>9.2e} | {s['r_vL']:>6.2f}")
    except Exception as e:
        print(f"{name:>26} | error: {e}")
    pp['nu_v'] = 228380603.667245; pp['nu_i'] = 376725167498.969543; pp['recom'] = 10.0  # restore
