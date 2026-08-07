/**
 * rate_equations.cpp – ODE right-hand side implementation.
 *
 * Faithfully translates the Python ODE system in
 *   py_utils/rate_equations.py  (dCx_dt methods, ode_system)
 *   py_utils/reaction_rates.py  (all R_*, G_*, nucleation/growth/annealing)
 *
 * All parameters arrive pre-computed from the Python side via Parameters.
 * No map lookups, no std::exp inside the RHS — only arithmetic.
 */
#include "rate_equations.h"

#include <cmath>

static constexpr double PI = 3.14159265358979323846;

// Clamp c to the concentration floor (inline, branch-free alternative).
static inline double fl(double c, double floor) {
    return c > floor ? c : floor;
}

int rhs_zrmicro(sunrealtype /*t*/, N_Vector y, N_Vector ydot, void* user_data) {
    const Parameters& P = *static_cast<const Parameters*>(user_data);
    const double C_floor = P.C_floor;

    // ── Extract and floor concentrations ────────────────────────────────────
    double Cv     = fl(NV_Ith_S(y,  0), C_floor);
    double Ci     = fl(NV_Ith_S(y,  1), C_floor);
    double C2i    = fl(NV_Ith_S(y,  2), C_floor);
    double C3i    = fl(NV_Ith_S(y,  3), C_floor);
    double CiL    = fl(NV_Ith_S(y,  4), C_floor);
    double CaiL   = fl(NV_Ith_S(y,  5), C_floor);
    double CvL    = fl(NV_Ith_S(y,  6), C_floor);
    double CavL   = fl(NV_Ith_S(y,  7), C_floor);
    double CiL_i  = fl(NV_Ith_S(y,  8), C_floor);
    double CaiL_i = fl(NV_Ith_S(y,  9), C_floor);
    double CvL_v  = fl(NV_Ith_S(y, 10), C_floor);
    double CavL_v = fl(NV_Ith_S(y, 11), C_floor);

    // Evolving network dislocation density (state row IDX_RHO_N). Starts at the
    // grown-in seed P.rho_N and grows as loops are absorbed into the network;
    // floored to the seed so the sink strengths never see a non-positive value.
    double rho_N_state = NV_Ith_S(y, IDX_RHO_N);
    double rho_N = rho_N_state > 0.0 ? rho_N_state : P.rho_N;

    // ── Loop radii (RateEquations.calculate_r_*) ─────────────────────────────
    // r = l_a * sqrt(CiL_i / CiL)  if CiL > floor, else 5 nm fallback
    double r_iL   = (CiL  > C_floor) ? P.l_a * std::sqrt(CiL_i  / CiL)  : 5e-9;
    double r_aiL  = (CaiL > C_floor) ? P.l_a * std::sqrt(CaiL_i / CaiL) : 5e-9;
    double r_vL   = (CvL  > C_floor) ? P.l_c * std::sqrt(CvL_v  / CvL)  : 5e-9;
    double r_avL  = (CavL > C_floor) ? P.l_c * std::sqrt(CavL_v / CavL) : 5e-9;

    // ── Loop dislocation densities (ReactionRates.calculate_sink_concentrations) ──
    double rho_iL  = 2.0 * PI * r_iL  * CiL  / P.Omega;
    double rho_aiL = 2.0 * PI * r_aiL * CaiL / P.Omega;
    double rho_vL  = 2.0 * PI * r_vL  * CvL  / P.Omega;
    double rho_avL = 2.0 * PI * r_avL * CavL / P.Omega;
    double rho_tot = rho_N + rho_iL + rho_aiL + rho_vL + rho_avL;

    // ── Sink concentrations (Equations 28, 30) ───────────────────────────────
    // NETWORK-ONLY: loop absorption is handled explicitly via the loop-growth
    // coupling below (which debits the free pools), so loops must NOT also appear
    // in the point-defect sink strength or their absorption would be double
    // counted. rho_tot/rho_*L are retained only for the loop dislocation density
    // diagnostics above.
    double a2_zc = (P.a * P.a) / P.z_c;
    double C_v_s = a2_zc * rho_N;
    double C_i_s = a2_zc * P.Z_N * rho_N;
    (void) rho_tot;

    // ── Defect fluxes (ReactionRates.flux_v / flux_i) ────────────────────────
    double flux_v = P.omega_v * Cv;
    double flux_i = P.omega_i * (Ci + 2.0 * C2i + 3.0 * C3i);

    // ── Reaction rates ────────────────────────────────────────────────────────
    // Note: omega_3i == omega_2i (same migration energy assumed in Python)
    double omega_3i = P.omega_2i;

    double R_i_v   = P.recom * (P.omega_i + P.omega_v) * Ci  * Cv;
    double R_2i_v  = (P.omega_2i + P.omega_v) * C2i * Cv;
    double R_3i_v  = (omega_3i   + P.omega_v) * C3i * Cv;
    double R_i_i   = 2.0 * P.omega_i * Ci * Ci;
    double R_i_2i  = (P.omega_i + P.omega_2i) * Ci * C2i;
    double R_i_3i  = (P.omega_i + omega_3i)   * Ci * C3i;
    double R_2i_2i = 4.0 * P.omega_2i * C2i * C2i;   // 2*(omega_2i+omega_2i)*C2i^2
    double R_2i_3i = 2.0 * P.omega_2i * C2i * C3i;   // (omega_2i+omega_3i)*C2i*C3i
    double R_v_s   = P.omega_v * Cv  * C_v_s;
    double R_i_s   = P.omega_i * Ci  * C_i_s;
    double R_2i_s  = P.omega_i * C2i * C_i_s;         // uses omega_i (matches Python)
    double R_3i_s  = P.omega_i * C3i * C_i_s;         // uses omega_i (matches Python)

    // ── Thermal emission rates (ReactionRates.emission_2i / emission_3i) ────
    double emission_2i = P.omega_2i * P.e_2i * C2i;
    double emission_3i = P.omega_2i * P.e_3i * C3i;   // omega_3i = omega_2i

    // ── Nucleation rates ──────────────────────────────────────────────────────
    // Loop-number nucleation: each i+3i / 2i+2i reaction forms one loop, split
    // between non-aligned (fraction f_na = 1 - f_a) and aligned (f_a).  The
    // fractions sum to 1 (previously both were f_a — a double-negation bug).
    double nuc_iL_frac  = P.f_na;   // non-aligned fraction (= 1 - f_a)
    double nuc_aiL_frac = P.f_a;    // aligned fraction

    // Two a-loop nucleation sources: (1) homogeneous clustering from i+3i / 2i+2i
    // reactions, and (2) cascade source G_iL/G_aiL per n_iL_nuc interstitials
    // (content seed G_iL/G_aiL added to dCiL_i/dCaiL_i below). G_iL/G_aiL already
    // carry the (1-f_a)/f_a split, so they are NOT re-scaled by nuc_*_frac.
    double nuc_iL  = nuc_iL_frac  * (R_i_3i + R_2i_2i) + P.G_iL  / P.n_iL_nuc;
    double nuc_aiL = nuc_aiL_frac * (R_i_3i + R_2i_2i) + P.G_aiL / P.n_iL_nuc;

    // Interstitial atoms consumed by the nucleation/loop reactions, deposited
    // into loop content so loops nucleate at finite size and mass is conserved.
    // This equals exactly the interstitial atoms removed from Ci/C2i/C3i by the
    // R_i_3i (1+3), R_2i_2i (2) and R_2i_3i (3) reactions.
    double nuc_content = 4.0 * R_i_3i + 2.0 * R_2i_2i + 3.0 * R_2i_3i;

    // Vacancy loops: G_vL/G_avL are cascade vacancy ATOM rates; loop NUMBER
    // nucleation = G_vL/n_vL_nuc (content seed G_vL added to dCvL_v below).
    double nuc_vL  = P.G_vL  / P.n_vL_nuc;
    double nuc_avL = P.G_avL / P.n_vL_nuc;

    // ── Thermal annealing rates ───────────────────────────────────────────────
    double ann_vL  = CvL  / P.tau_vL;     // loop number loss
    double ann_avL = CavL / P.tau_avL;
    // Vacancy content released to the free pool by annealing loops. tau_vL
    // dissolves newly-nucleated EMBRYO loops (birth content ~n_vL_nuc each),
    // not grown loops, so release = n_vL_nuc * (number anneal rate CvL/tau_vL).
    // At number saturation this balances the seed G_vL, leaving net content
    // growth = flux, so mean c-loop size grows with dose (see reaction_rates.py
    // annealing_content_vL). Internal transfer loop content -> Cv.
    double ann_cont_vL  = P.n_vL_nuc * CvL  / P.tau_vL;
    double ann_cont_avL = P.n_vL_nuc * CavL / P.tau_avL;

    // ── Loop growth rates (ReactionRates.loop_growth_rate_*) ─────────────────
    // Decomposed into the interstitial- and vacancy-absorption components so the
    // free pools can be debited consistently (mass-conserving rate-theory
    // coupling).  Each absorption component is >= 0.
    //   i-loops grow by absorbing interstitials, shrink by absorbing vacancies
    //           (the vacancy annihilates a stored interstitial).
    //   v-loops grow by absorbing vacancies,    shrink by absorbing interstitials
    //           (the interstitial annihilates a stored vacancy).
    double lc_l = P.l_c / P.l;

    double pref_iL  = lc_l        * std::sqrt(CiL_i  * CiL);
    double pref_aiL = lc_l        * std::sqrt(CaiL_i * CaiL);
    double pref_vL  = lc_l * P.Q  * std::sqrt(CvL    * CvL_v);
    double pref_avL = lc_l * P.Q  * std::sqrt(CavL   * CavL_v);

    // DAD: a-loops capture i with Z_i_a, v with Z_v_a; c-loops capture v with
    // Z_v_c, i with Z_i_c. With delta>0 (Z_i_a>1>Z_v_a, Z_v_c>1>Z_i_c) both
    // a-interstitial and c-vacancy loops can grow simultaneously.
    double iL_i  = pref_iL  * P.Z_i_a * flux_i;   double iL_v  = pref_iL  * P.Z_v_a * flux_v;
    double aiL_i = pref_aiL * P.Z_i_a * flux_i;   double aiL_v = pref_aiL * P.Z_v_a * flux_v;
    double vL_v  = pref_vL  * P.Z_v_c * flux_v;   double vL_i  = pref_vL  * P.Z_i_c * flux_i;
    double avL_v = pref_avL * P.Z_v_c * flux_v;   double avL_i = pref_avL * P.Z_i_c * flux_i;

    // ── Minimum-stable-size gate on the a-loop vacancy-absorption channel ─────
    // a-loops shrink by absorbing free vacancies (each annihilates a stored
    // interstitial). At high T / high dose the vacancy flux exceeds the
    // interstitial flux, the net climb flips negative, the stored content drains
    // to the floor, and the reconstructed mean diameter r = l_a*sqrt(C_cont/C_num)
    // collapses to zero. The gate g(r) -> 0 as r -> r_min_a, so loops stop
    // absorbing vacancies (and stop shrinking) at the minimum stable radius.
    // Gating iL_v/aiL_v themselves (used in growth, loop_abs_v AND loop_recomb)
    // keeps the point-defect balance closed: un-absorbed vacancies stay free.
    // C1-continuous smoothstep avoids solver stiffness.
    auto size_gate = [&](double r) -> double {
        if (P.r_min_a <= 0.0) return 1.0;          // gate disabled
        double x = (r / P.r_min_a - 1.0) / P.w_rmin;
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        return x * x * (3.0 - 2.0 * x);
    };
    iL_v  *= size_gate(r_iL);
    aiL_v *= size_gate(r_aiL);

    double growth_iL  = iL_i  - iL_v;            // net dCiL_i/dt
    double growth_aiL = aiL_i - aiL_v;
    double growth_vL  = vL_v  - vL_i;            // net dCvL_v/dt
    double growth_avL = avL_v - avL_i;

    // Free-pool depletion by loop absorption (all >= 0):
    double loop_abs_i = iL_i + aiL_i + vL_i + avL_i;   // interstitials captured by all loops
    double loop_abs_v = vL_v + avL_v + iL_v + aiL_v;   // vacancies captured by all loops
    // Defect-loop recombination: one i and one v annihilated per event
    // (free v on an i-loop, free i on a v-loop).
    double loop_recomb = iL_v + aiL_v + vL_i + avL_i;

    // ── ODE right-hand side ───────────────────────────────────────────────────

    // dCv/dt  (Equations 34-35)
    // Loop absorption (loop_abs_v) now debits the free vacancy pool explicitly;
    // network absorption is R_v_s (network-only C_v_s). Annealing vacancy loops
    // release their content back to the free pool (ann_cont_vL + ann_cont_avL).
    NV_Ith_S(ydot, 0) = P.G_v + ann_cont_vL + ann_cont_avL
                       - (R_i_v + R_2i_v + R_3i_v + R_v_s + loop_abs_v);

    // dCi/dt  (Equations 36-37)
    // 0.5*R_i_i: i+i -> 2i forms one di-interstitial per two monomers consumed
    // (R_i_i = 2*omega_i*Ci^2 is the monomer-loss rate).  loop_abs_i debits the
    // free interstitial pool for absorption into loops.
    NV_Ith_S(ydot, 1) = P.G_i + R_2i_v
                       + 2.0 * emission_2i + 3.0 * emission_3i
                       - (R_i_v + R_i_i + R_i_2i + R_i_3i + R_i_s + loop_abs_i);

    // dC2i/dt  (Equations 38-39)
    NV_Ith_S(ydot, 2) = P.G_2i + 0.5 * R_i_i + R_3i_v
                       - (R_2i_v + R_i_2i + R_2i_2i + R_2i_s) -emission_2i;

    // dC3i/dt  (Equations 40-41)
    NV_Ith_S(ydot, 3) = P.G_3i + R_i_2i
                       - (R_3i_v + R_i_3i + R_3i_s + R_2i_3i)-emission_3i;

    // dCiL/dt  (Equation 42)
    NV_Ith_S(ydot, 4) = nuc_iL;

    // dCaiL/dt  (Equation 43)
    NV_Ith_S(ydot, 5) = nuc_aiL;

    // dCvL/dt  (Equation 44)
    NV_Ith_S(ydot, 6) = nuc_vL - ann_vL;

    // dCavL/dt  (Equation 45)
    NV_Ith_S(ydot, 7) = nuc_avL - ann_avL;

    // dCiL_i/dt  (Equation 46) — growth + clustering nucleation content (same
    // split as number) + cascade content seed (G_iL interstitial atoms, the
    // eps_iL channel — counted in prod_i below).
    NV_Ith_S(ydot, 8) = growth_iL  + nuc_iL_frac  * nuc_content + P.G_iL;

    // dCaiL_i/dt  (Equation 47)
    NV_Ith_S(ydot, 9) = growth_aiL + nuc_aiL_frac * nuc_content + P.G_aiL;

    // dCvL_v/dt  (Equation 48) — growth + cascade content seed (G_vL vacancy
    // atoms, the eps_vL channel) - content released by annealing. The seed is
    // essential: without finite content the sqrt(CvL*CvL_v) growth prefactor
    // pins CvL_v at the floor and vacancy loops can never grow.
    NV_Ith_S(ydot, 10) = growth_vL  + P.G_vL  - ann_cont_vL;

    // dCavL_v/dt  (Equation 49)
    NV_Ith_S(ydot, 11) = growth_avL + P.G_avL - ann_cont_avL;

    // ── Point-defect conservation accumulators (Eq. 12-17) ───────────────────
    // Monotonic time integrals of the production and physical loss channels for
    // the interstitial and vacancy atom balances.  Internal transfers (clustering,
    // emission) cancel in the atom-weighted sums and are intentionally excluded;
    // the residual between stored atoms and (production − losses) measured in
    // post-processing therefore quantifies the model's conservation error.
    //
    //   I_stored = Ci + 2*C2i + 3*C3i + CiL_i + CaiL_i
    //   V_stored = Cv + CvL_v + CavL_v
    // Interstitial atoms = free monomers/clusters (G_i + 2 G_2i + 3 G_3i) +
    // cascade loop-borne (G_iL + G_aiL), the eps_iL channel now seeded as a-loop
    // content. Sums to the full dpa rate G, mirroring prod_v.
    double prod_i  = P.G_i + 2.0 * P.G_2i + 3.0 * P.G_3i + P.G_iL + P.G_aiL;  // interstitial atoms produced
    // Vacancy atoms = free (G_v) + cascade loop-borne (G_vL+G_avL), the eps_vL
    // channel now seeded as loop content. Sums to the full dpa rate G.
    double prod_v  = P.G_v + P.G_vL + P.G_avL;               // vacancies produced
    // Recombination: bulk Frenkel (R_*_v) plus defect-loop recombination.
    double recomb  = R_i_v + R_2i_v + R_3i_v + loop_recomb;  // 1 i and 1 v per event
    double sink_i  = R_i_s + 2.0 * R_2i_s + 3.0 * R_3i_s;    // interstitial absorption at network
    double sink_v  = R_v_s;                                  // vacancy absorption at network

    NV_Ith_S(ydot, 12) = prod_i;    // cum_prod_i
    NV_Ith_S(ydot, 13) = recomb;    // cum_recomb_i
    NV_Ith_S(ydot, 14) = sink_i;    // cum_sink_i
    NV_Ith_S(ydot, 15) = prod_v;    // cum_prod_v
    NV_Ith_S(ydot, 16) = recomb;    // cum_recomb_v  (same Frenkel events)
    NV_Ith_S(ydot, 17) = sink_v;    // cum_sink_v

    // ── Geometric loop coalescence (absorbed-flux–climb driven) ──────────────
    // Mirrors ReactionRates.coalescence_rates / RateEquations.ode_system.
    //   (1) like-loop coarsening: number density drops, content conserved
    //       (fewer loops -> larger mean size).
    //   (2) loop-network coalescence: number AND content removed; removed
    //       content booked as network sink absorption (point-defect conserving)
    //       and the absorbed loop line length grows rho_N (see dydt[IDX_RHO_N]).
    // The rate scale is the loop climb velocity from the ABSORBED (gain-side)
    // flux — a-loops absorb interstitials (gcont = iL_i/aiL_i), c-loops absorb
    // vacancies (gcont = vL_v/avL_v). Passing the gain-side absorption (rather
    // than the NET growth growth_iL/growth_vL, which collapses to ~0 once growth
    // saturates) keeps drdt positive at steady state, so coalescence persists:
    // N keeps falling gradually and the mean size keeps rising gradually.
    // Smooth Avrami gates use the loop-loop (N^-1/3) and loop-network
    // (rho_N^-1/2) distances; rho_N is the evolving network density.
    double sqrt_rhoN = std::sqrt(rho_N);
    double d_N = 1.0 / sqrt_rhoN;   // loop-network spacing [m]; caps overlap radius
    auto coal = [&](double Cnum, double Ccont, double lscale, double gcont,
                    double c_LL, double c_LN,
                    double& num_loss, double& cont_loss,
                    double& num_LN, double& r_out) {
        double r      = lscale * std::sqrt(Ccont / Cnum);
        double N      = Cnum / P.Omega;
        // drdt computed from the gain-side absorption gcont gives the persistent
        // absorbed-flux climb speed v_abs = lscale*(l_c/l)*[Q]*Z_gain*flux/2.
        double drdt   = lscale * gcont / (2.0 * std::sqrt(Cnum * Ccont));
        double drdt_p = drdt > 0.0 ? drdt : 0.0;
        // Bounded overlap gates: cap the radius at the relevant spacing so the
        // Avrami argument SATURATES instead of diverging as the number density
        // collapses.  d_LL = N^(-1/3) (loop-loop), d_N = rho_N^(-1/2) (network).
        double d_LL = std::cbrt(P.Omega / Cnum);
        double r_LL = r < d_LL ? r : d_LL;
        double r_LN = r < d_N  ? r : d_N;
        double phi_LL = 1.0 - std::exp(-P.kappa_LL * (4.0 / 3.0) * PI * r_LL * r_LL * r_LL * N);
        double phi_LN = 1.0 - std::exp(-P.kappa_LN * PI * r_LN * r_LN * rho_N);
        double nu_LL  = c_LL * drdt_p * std::cbrt(N);
        double nu_LN  = c_LN * drdt_p * sqrt_rhoN;
        num_LN    = nu_LN * phi_LN * Cnum;
        num_loss  = nu_LL * phi_LL * Cnum + num_LN;
        cont_loss = nu_LN * phi_LN * Ccont;   // channel 1 conserves content
        r_out     = r;
    };
    double coal_num_iL,  coal_cont_iL,  numLN_iL,  r_iL_c;   coal(CiL,  CiL_i,  P.l_a, iL_i,  P.c_LL_a, P.c_LN_a, coal_num_iL,  coal_cont_iL,  numLN_iL,  r_iL_c);
    double coal_num_aiL, coal_cont_aiL, numLN_aiL, r_aiL_c;  coal(CaiL, CaiL_i, P.l_a, aiL_i, P.c_LL_a, P.c_LN_a, coal_num_aiL, coal_cont_aiL, numLN_aiL, r_aiL_c);
    double coal_num_vL,  coal_cont_vL,  numLN_vL,  r_vL_c;   coal(CvL,  CvL_v,  P.l_c, vL_v,  P.c_LL_c, P.c_LN_c, coal_num_vL,  coal_cont_vL,  numLN_vL,  r_vL_c);
    double coal_num_avL, coal_cont_avL, numLN_avL, r_avL_c;  coal(CavL, CavL_v, P.l_c, avL_v, P.c_LL_c, P.c_LN_c, coal_num_avL, coal_cont_avL, numLN_avL, r_avL_c);

    NV_Ith_S(ydot, 4)  -= coal_num_iL;
    NV_Ith_S(ydot, 5)  -= coal_num_aiL;
    NV_Ith_S(ydot, 6)  -= coal_num_vL;
    NV_Ith_S(ydot, 7)  -= coal_num_avL;
    NV_Ith_S(ydot, 8)  -= coal_cont_iL;
    NV_Ith_S(ydot, 9)  -= coal_cont_aiL;
    NV_Ith_S(ydot, 10) -= coal_cont_vL;
    NV_Ith_S(ydot, 11) -= coal_cont_avL;
    NV_Ith_S(ydot, 14) += coal_cont_iL + coal_cont_aiL;   // cum_sink_i
    NV_Ith_S(ydot, 17) += coal_cont_vL + coal_cont_avL;   // cum_sink_v

    // ── Evolving network dislocation density rho_N (index IDX_RHO_N) ──────────
    // Source: each loop absorbed into the network contributes 2*pi*r of line
    // length per unit volume (num_LN is the loop-network NUMBER loss in atom-
    // fraction/s; /Omega converts to number density). Recovery: first-order
    // relaxation of the irradiation-grown excess toward the grown-in seed
    // (network climb annihilation), so rho_N saturates. Mirrors
    // ReactionRates.rho_N_source.
    double rho_source = P.c_rhoN * (2.0 * PI / P.Omega) *
        (r_iL_c * numLN_iL + r_aiL_c * numLN_aiL +
         r_vL_c * numLN_vL + r_avL_c * numLN_avL);
    double rho_recovery = P.k_rhoN_rec * (rho_N - P.rho_N);
    NV_Ith_S(ydot, IDX_RHO_N) = rho_source - rho_recovery;

    // ── Floor enforcement ─────────────────────────────────────────────────────
    // Zero out any derivative that would drive a floored concentration further
    // negative (mirrors the np.where guard in RateEquations.ode_system).
    // Restricted to the physical species; the accumulators are monotonic
    // non-negative integrals and must never be clamped.
    for (int k = 0; k < N_PHYS; ++k) {
        if (NV_Ith_S(y, k) <= C_floor && NV_Ith_S(ydot, k) < 0.0)
            NV_Ith_S(ydot, k) = 0.0;
    }

    // ── Operator-split QSSA: freeze the mobile species ───────────────────────
    // Two-time-scale coupling (steady mobile FEM solve supplies y0[0:4]). Zero
    // the four mobile derivatives LAST so the immobile / accumulator / rho_N
    // equations above still used the frozen Cv,Ci,C2i,C3i as parameters.
    if (P.freeze_mobile) {
        NV_Ith_S(ydot, 0) = 0.0;   // Cv
        NV_Ith_S(ydot, 1) = 0.0;   // Ci
        NV_Ith_S(ydot, 2) = 0.0;   // C2i
        NV_Ith_S(ydot, 3) = 0.0;   // C3i
    }

    return 0;
}
