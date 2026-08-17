/**
 * rate_equations_core.h — the ZrMicro ODE right-hand side, templated on the
 * scalar type.
 *
 * This is the SINGLE definition of the physics on the C++ side. It is
 * instantiated twice:
 *   * T = double    -> the residual evaluated by CVODE (rhs_zrmicro et al.)
 *   * T = Dual<N>   -> one sweep producing the exact Jacobian by forward AD
 *
 * Previously the residual lived in rate_equations.cpp and there was no Jacobian
 * at all (SUNDIALS fell back to difference quotients). Moving the body here
 * changes no arithmetic: the double instantiation is a line-for-line
 * transcription of the original rate_equations.cpp, INCLUDING the floating-
 * point association of every expression, so standalone runs stay bit-identical.
 * Where the original wrote `k1 * k2 * C * D` (i.e. ((k1*k2)*C)*D) this file
 * hoists `k1*k2` into a named double and keeps the remaining order — that is
 * the same sequence of operations, hence the same rounding.
 *
 * Mirrors: py_utils/rate_equations.py + py_utils/reaction_rates.py
 *
 * Full state layout (N_EQ = 19), see parameters.h:
 *   [0..3]   mobile        Cv Ci C2i C3i
 *   [4..11]  immobile      CiL CaiL CvL CavL CiL_i CaiL_i CvL_v CavL_v
 *   [12..17] accumulators  cum_prod_i cum_recomb_i cum_sink_i
 *                          cum_prod_v cum_recomb_v cum_sink_v
 *   [18]     rho_N
 *
 * The accumulators are PURE QUADRATURES: rows 12..17 are written but columns
 * 12..17 are never read by any other equation. That structural fact is what
 * lets the solver carry them as CVODES quadrature variables and shrink the
 * implicit system from 19x19 to 13x13 (standalone) or 9x9 (frozen mobile).
 */
#pragma once

#include "parameters.h"
#include "dual.h"

namespace zrcore {

static constexpr double PI = 3.14159265358979323846;

// Clamp to the concentration floor.
template <class T>
inline T fl(const T& c, double floor) {
    return ad_val(c) > floor ? c : T(floor);
}

/**
 * Evaluate the full 19-component right-hand side.
 *
 * @param y     full state, length N_EQ (unfloored — the floor is applied here)
 * @param ydot  output, length N_EQ
 * @param P     parameters (always plain double; they are not differentiated)
 */
template <class T>
void rhs_core(const T* y, T* ydot, const Parameters& P) {
    const double C_floor = P.C_floor;

    // ── Extract and floor concentrations ────────────────────────────────────
    T Cv     = fl(y[0],  C_floor);
    T Ci     = fl(y[1],  C_floor);
    T C2i    = fl(y[2],  C_floor);
    T C3i    = fl(y[3],  C_floor);
    T CiL    = fl(y[4],  C_floor);
    T CaiL   = fl(y[5],  C_floor);
    T CvL    = fl(y[6],  C_floor);
    T CavL   = fl(y[7],  C_floor);
    T CiL_i  = fl(y[8],  C_floor);
    T CaiL_i = fl(y[9],  C_floor);
    T CvL_v  = fl(y[10], C_floor);
    T CavL_v = fl(y[11], C_floor);

    // Evolving network dislocation density (state row IDX_RHO_N). Starts at the
    // grown-in seed P.rho_N and grows as loops are absorbed into the network;
    // floored to the seed so the sink strengths never see a non-positive value.
    T rho_N = ad_val(y[IDX_RHO_N]) > 0.0 ? y[IDX_RHO_N] : T(P.rho_N);

    // ── Loop radii (RateEquations.calculate_r_*) ─────────────────────────────
    // r = l_a * sqrt(CiL_i / CiL)  if CiL > floor, else 5 nm fallback
    T r_iL  = (ad_val(CiL)  > C_floor) ? P.l_a * ad_sqrt(CiL_i  / CiL)  : T(5e-9);
    T r_aiL = (ad_val(CaiL) > C_floor) ? P.l_a * ad_sqrt(CaiL_i / CaiL) : T(5e-9);
    T r_vL  = (ad_val(CvL)  > C_floor) ? P.l_c * ad_sqrt(CvL_v  / CvL)  : T(5e-9);
    T r_avL = (ad_val(CavL) > C_floor) ? P.l_c * ad_sqrt(CavL_v / CavL) : T(5e-9);
    (void) r_avL;   // computed for symmetry; only the a-loops are size-gated

    // ── Sink concentrations (Equations 28, 30) ───────────────────────────────
    // NETWORK-ONLY: loop absorption is handled explicitly via the loop-growth
    // coupling below (which debits the free pools), so loops must NOT also
    // appear in the point-defect sink strength or their absorption would be
    // double counted.
    const double a2_zc  = (P.a * P.a) / P.z_c;
    const double a2_zcZ = a2_zc * P.Z_N;
    T C_v_s = a2_zc  * rho_N;
    T C_i_s = a2_zcZ * rho_N;

    // ── Defect fluxes (ReactionRates.flux_v / flux_i) ────────────────────────
    T flux_v = P.omega_v * Cv;
    T flux_i = P.omega_i * (Ci + 2.0 * C2i + 3.0 * C3i);

    // ── Reaction rates ──────────────────────────────────────────────────────
    // Note: omega_3i == omega_2i (same migration energy assumed in Python)
    const double omega_3i = P.omega_2i;

    const double k_i_v   = P.recom * (P.omega_i + P.omega_v);
    const double k_2i_v  = P.omega_2i + P.omega_v;
    const double k_3i_v  = omega_3i   + P.omega_v;
    const double k_i_i   = 2.0 * P.omega_i;
    const double k_i_2i  = P.omega_i + P.omega_2i;
    const double k_i_3i  = P.omega_i + omega_3i;
    const double k_2i_2i = 4.0 * P.omega_2i;    // 2*(omega_2i+omega_2i)
    const double k_2i_3i = 2.0 * P.omega_2i;    // (omega_2i+omega_3i)

    T R_i_v   = k_i_v   * Ci  * Cv;
    T R_2i_v  = k_2i_v  * C2i * Cv;
    T R_3i_v  = k_3i_v  * C3i * Cv;
    T R_i_i   = k_i_i   * Ci  * Ci;
    T R_i_2i  = k_i_2i  * Ci  * C2i;
    T R_i_3i  = k_i_3i  * Ci  * C3i;
    T R_2i_2i = k_2i_2i * C2i * C2i;
    T R_2i_3i = k_2i_3i * C2i * C3i;
    T R_v_s   = P.omega_v * Cv  * C_v_s;
    T R_i_s   = P.omega_i * Ci  * C_i_s;
    T R_2i_s  = P.omega_i * C2i * C_i_s;    // uses omega_i (matches Python)
    T R_3i_s  = P.omega_i * C3i * C_i_s;    // uses omega_i (matches Python)

    // ── Thermal emission rates ──────────────────────────────────────────────
    const double ke_2i = P.omega_2i * P.e_2i;
    const double ke_3i = P.omega_2i * P.e_3i;   // omega_3i = omega_2i
    T emission_2i = ke_2i * C2i;
    T emission_3i = ke_3i * C3i;

    // ── Nucleation rates ────────────────────────────────────────────────────
    // Loop-number nucleation: each i+3i / 2i+2i reaction forms one loop, split
    // between non-aligned (f_na = 1 - f_a) and aligned (f_a).
    const double nuc_iL_frac  = P.f_na;
    const double nuc_aiL_frac = P.f_a;

    // Two a-loop nucleation sources: homogeneous clustering, plus the cascade
    // source G_iL/G_aiL per n_iL_nuc interstitials. G_iL/G_aiL already carry
    // the (1-f_a)/f_a split, so they are NOT re-scaled by nuc_*_frac.
    T nuc_iL  = nuc_iL_frac  * (R_i_3i + R_2i_2i) + P.G_iL  / P.n_iL_nuc;
    T nuc_aiL = nuc_aiL_frac * (R_i_3i + R_2i_2i) + P.G_aiL / P.n_iL_nuc;

    // Interstitial atoms consumed by the nucleation/loop reactions, deposited
    // into loop content so loops nucleate at finite size and mass is conserved.
    T nuc_content = 4.0 * R_i_3i + 2.0 * R_2i_2i + 3.0 * R_2i_3i;

    // Vacancy loops: G_vL/G_avL are cascade vacancy ATOM rates; loop NUMBER
    // nucleation = G_vL/n_vL_nuc (content seed G_vL added to dCvL_v below).
    const double nuc_vL  = P.G_vL  / P.n_vL_nuc;
    const double nuc_avL = P.G_avL / P.n_vL_nuc;

    // ── Thermal annealing rates ─────────────────────────────────────────────
    T ann_vL  = CvL  / P.tau_vL;     // loop number loss
    T ann_avL = CavL / P.tau_avL;
    // Vacancy content released to the free pool by annealing embryo loops.
    T ann_cont_vL  = P.n_vL_nuc * CvL  / P.tau_vL;
    T ann_cont_avL = P.n_vL_nuc * CavL / P.tau_avL;

    // ══ FAMILY SLOTS ═══════════════════════════════════════════════════════
    // Both loop models carry exactly four families with a (number, content)
    // pair each, so everything downstream -- coalescence, rho_N, the free-pool
    // debits, the accumulators -- is written once against these arrays. Only
    // the meaning of the four slots and the way their growth is formed differ.
    //
    //   loop_model 0 : iL , aiL , vL , avL      (aligned/non-aligned split)
    //   loop_model 1 : c  , a1  , a2  , a3      (one basal, three prismatic)
    //
    T f_num[4], f_cont[4];        // state
    T f_gain[4], f_loss[4];       // like- and opposite-polarity absorption, >=0
    T f_nucn[4], f_nucc[4];       // nucleation into number and content
    T f_annn[4], f_annc[4];       // thermal annealing out of number and content
    double f_lscale[4], f_cLL[4], f_cLN[4];
    T loop_abs_i, loop_abs_v, loop_recomb;
    // Legacy net-growth terms. Declared here rather than in the branch because
    // the legacy ydot assembly below is kept verbatim and refers to them.
    T growth_iL(0.0), growth_aiL(0.0), growth_vL(0.0), growth_avL(0.0);

    if (P.loop_model != 0) {
    // ═══════════════════════════════════════════════════════════════════════
    // SELF-CONSISTENT MODEL -- the same capture physics as the fast solve.
    //
    // State: y[4..11] = [n_c, n_a1, n_a2, n_a3, c_c, c_a1, c_a2, c_a3], which
    // is exactly MoDELib's CD immobile block, so the 0-D <-> 3-D bridge is an
    // identity in this mode instead of a lumping.
    //
    // Capture efficiencies are Woo's, generated from the SAME p_m that sets the
    // diffusion tensor (staging/anisotropy.py writes both from one source):
    //
    //     Z_basal(m)     = Z0_m p_m                    row 0, vacancy-type <c>
    //     Z_prismatic(m) = Z0_m (p_m + p_m^-2)/2       row 1, interstitial <a>
    //
    // identical to ClusterDynamicsParameters::loopDADbias. The decisive
    // difference from the legacy model is not the formula but the RESOLUTION:
    // every mobile species carries its OWN efficiency, where the legacy model
    // lumps Ci, C2i and C3i into one flux and applies a single Z_i_a to all
    // three. That is what makes an anisotropy given to the clusters alone
    // representable here and invisible there.
    //
    // The absorption rate of species m at family k, per unit volume, is
    //
    //     phi_km = S_k * Dbar_m * Z(row(k), m) * c_m * |m_m|
    //
    // with S_k the geometric sink strength. Dbar_m is the orientation-averaged
    // diffusivity (det D)^(1/3); because anisotropy.py splits the migration
    // energies at FIXED D_eff, that geometric mean is exactly the omega_m the
    // legacy model already uses, so no mobility is redefined here -- only its
    // directional weighting.
    // ═══════════════════════════════════════════════════════════════════════
        double Zrow[2][4];
        for (int m = 0; m < 4; ++m) {
            const double pm = P.dad_p[m] > 0.0 ? P.dad_p[m] : 1.0;
            Zrow[0][m] = P.dad_Z0[m] * pm;                          // basal
            Zrow[1][m] = P.dad_Z0[m] * 0.5 * (pm + 1.0 / (pm * pm)); // prismatic
        }
        const T      cm[4]  = {Cv, Ci, C2i, C3i};
        const double sz[4]  = {1.0, 1.0, 2.0, 3.0};   // |m_m|, atoms per cluster
        const double Dbar[4] = {P.omega_v, P.omega_i, P.omega_2i, P.omega_2i};

        f_num[0]  = CiL;  f_num[1]  = CaiL;  f_num[2]  = CvL;   f_num[3]  = CavL;
        f_cont[0] = CiL_i; f_cont[1] = CaiL_i; f_cont[2] = CvL_v; f_cont[3] = CavL_v;

        // Slot 0 is the basal <c> family, slots 1-3 the prismatic <a> variants.
        const int is_vac[4] = {1, 0, 0, 0};
        f_lscale[0] = P.l_c; f_cLL[0] = P.c_LL_c; f_cLN[0] = P.c_LN_c;
        for (int k = 1; k < 4; ++k) {
            f_lscale[k] = P.l_a; f_cLL[k] = P.c_LL_a; f_cLN[k] = P.c_LN_a;
        }

        loop_abs_i = T(0.0); loop_abs_v = T(0.0); loop_recomb = T(0.0);
        const double lc_l_sc = P.l_c / P.l;   // <c> uses the basal length scale
        const double la_l_sc = P.l_a / P.l;

        for (int k = 0; k < 4; ++k) {
            const int row = is_vac[k] ? 0 : 1;
            // Geometric sink strength, scaled per family exactly as
            // ImmobileSinks does with loopSinkScale. This replaces the legacy
            // Q, which scaled the <c> channel only.
            T pref = (is_vac[k] ? lc_l_sc : la_l_sc) * P.loop_sink_scale[k]
                     * ad_sqrt(f_num[k] * f_cont[k]);
            T gain(0.0), loss(0.0);
            for (int m = 0; m < 4; ++m) {
                const int m_is_vac = (m == 0);
                T rate = pref * (Zrow[row][m] * Dbar[m] * sz[m]) * cm[m];
                if (m_is_vac == is_vac[k]) gain = gain + rate;
                else                       loss = loss + rate;
            }
            // Minimum-stable-size gate, on the SHRINKING channel only, exactly
            // as ClusterDynamicsFEM applies it: a loop at r_min stops absorbing
            // the defect that would dissolve it, and the un-absorbed defects
            // stay in the free pool so the balance closes.
            if (!is_vac[k]) {
                T rk = f_lscale[k] * ad_sqrt(f_cont[k] / f_num[k]);
                if (P.r_min_a > 0.0) {
                    T x = (rk / P.r_min_a - 1.0) / P.w_rmin;
                    T g = ad_val(x) <= 0.0 ? T(0.0)
                        : (ad_val(x) >= 1.0 ? T(1.0) : x * x * (3.0 - 2.0 * x));
                    loss = loss * g;
                }
            }
            f_gain[k] = gain; f_loss[k] = loss;
            if (is_vac[k]) { loop_abs_v = loop_abs_v + gain; loop_abs_i = loop_abs_i + loss; }
            else           { loop_abs_i = loop_abs_i + gain; loop_abs_v = loop_abs_v + loss; }
            loop_recomb = loop_recomb + loss;   // opposite-polarity capture annihilates
        }

        // ── Nucleation, with no aligned/non-aligned split ───────────────────
        // The homogeneous clustering channel and the cascade source are shared
        // out over the three prismatic variants by variant_frac (equal thirds
        // at zero resolved stress), which is the same reduction
        // immobile_0d_to_modelib performs on the way to the 3-D code -- done
        // here at the source instead of on the way out.
        const double G_iL_tot = P.G_iL + P.G_aiL;
        const double G_vL_tot = P.G_vL + P.G_avL;
        T nuc_a_num  = (R_i_3i + R_2i_2i) + T(G_iL_tot / P.n_iL_nuc);
        T nuc_a_cont = nuc_content + T(G_iL_tot);
        f_nucn[0] = T(G_vL_tot / P.n_vL_nuc);
        f_nucc[0] = T(G_vL_tot);
        for (int k = 1; k < 4; ++k) {
            f_nucn[k] = P.variant_frac[k - 1] * nuc_a_num;
            f_nucc[k] = P.variant_frac[k - 1] * nuc_a_cont;
        }

        // Thermal annealing: vacancy loops only, one family instead of two.
        f_annn[0] = f_num[0] / P.tau_vL;
        f_annc[0] = P.n_vL_nuc * f_num[0] / P.tau_vL;
        for (int k = 1; k < 4; ++k) { f_annn[k] = T(0.0); f_annc[k] = T(0.0); }
    } else {
    // ── Loop growth rates (ReactionRates.loop_growth_rate_*) ────────────────
    // Decomposed into interstitial- and vacancy-absorption components so the
    // free pools can be debited consistently. Each component is >= 0.
    const double lc_l  = P.l_c / P.l;
    const double lc_lQ = lc_l * P.Q;

    T pref_iL  = lc_l  * ad_sqrt(CiL_i  * CiL);
    T pref_aiL = lc_l  * ad_sqrt(CaiL_i * CaiL);
    T pref_vL  = lc_lQ * ad_sqrt(CvL    * CvL_v);
    T pref_avL = lc_lQ * ad_sqrt(CavL   * CavL_v);

    // DAD: a-loops capture i with Z_i_a, v with Z_v_a; c-loops capture v with
    // Z_v_c, i with Z_i_c. With delta>0 both a-interstitial and c-vacancy loops
    // can grow simultaneously.
    T iL_i  = pref_iL  * P.Z_i_a * flux_i;   T iL_v  = pref_iL  * P.Z_v_a * flux_v;
    T aiL_i = pref_aiL * P.Z_i_a * flux_i;   T aiL_v = pref_aiL * P.Z_v_a * flux_v;
    T vL_v  = pref_vL  * P.Z_v_c * flux_v;   T vL_i  = pref_vL  * P.Z_i_c * flux_i;
    T avL_v = pref_avL * P.Z_v_c * flux_v;   T avL_i = pref_avL * P.Z_i_c * flux_i;

    // ── Minimum-stable-size gate on the a-loop vacancy-absorption channel ────
    // a-loops shrink by absorbing free vacancies. The gate g(r) -> 0 as
    // r -> r_min_a, so loops stop absorbing vacancies (and stop shrinking) at
    // the minimum stable radius. Gating iL_v/aiL_v themselves keeps the
    // point-defect balance closed: un-absorbed vacancies stay free.
    // C1-continuous smoothstep avoids solver stiffness.
    auto size_gate = [&](const T& r) -> T {
        if (P.r_min_a <= 0.0) return T(1.0);          // gate disabled
        T x = (r / P.r_min_a - 1.0) / P.w_rmin;
        if (ad_val(x) <= 0.0) return T(0.0);
        if (ad_val(x) >= 1.0) return T(1.0);
        return x * x * (3.0 - 2.0 * x);
    };
    iL_v  *= size_gate(r_iL);
    aiL_v *= size_gate(r_aiL);

    growth_iL  = iL_i  - iL_v;            // net dCiL_i/dt
    growth_aiL = aiL_i - aiL_v;
    growth_vL  = vL_v  - vL_i;            // net dCvL_v/dt
    growth_avL = avL_v - avL_i;

    // Free-pool depletion by loop absorption (all >= 0):
    loop_abs_i = iL_i + aiL_i + vL_i + avL_i;
    loop_abs_v = vL_v + avL_v + iL_v + aiL_v;
    // Defect-loop recombination: one i and one v annihilated per event.
    loop_recomb = iL_v + aiL_v + vL_i + avL_i;

    // Export the legacy quantities into the shared family slots. Assignments
    // only -- no legacy arithmetic is altered, so loop_model 0 reproduces the
    // fitted formulation operation for operation.
    f_num[0]  = CiL;   f_num[1]  = CaiL;   f_num[2]  = CvL;    f_num[3]  = CavL;
    f_cont[0] = CiL_i; f_cont[1] = CaiL_i; f_cont[2] = CvL_v;  f_cont[3] = CavL_v;
    f_gain[0] = iL_i;  f_gain[1] = aiL_i;  f_gain[2] = vL_v;   f_gain[3] = avL_v;
    f_loss[0] = iL_v;  f_loss[1] = aiL_v;  f_loss[2] = vL_i;   f_loss[3] = avL_i;
    f_nucn[0] = nuc_iL;  f_nucn[1] = nuc_aiL;  f_nucn[2] = T(nuc_vL); f_nucn[3] = T(nuc_avL);
    f_nucc[0] = nuc_iL_frac * nuc_content + P.G_iL;
    f_nucc[1] = nuc_aiL_frac * nuc_content + P.G_aiL;
    f_nucc[2] = T(P.G_vL);  f_nucc[3] = T(P.G_avL);
    f_annn[0] = T(0.0); f_annn[1] = T(0.0); f_annn[2] = ann_vL;      f_annn[3] = ann_avL;
    f_annc[0] = T(0.0); f_annc[1] = T(0.0); f_annc[2] = ann_cont_vL; f_annc[3] = ann_cont_avL;
    f_lscale[0] = P.l_a; f_cLL[0] = P.c_LL_a; f_cLN[0] = P.c_LN_a;
    f_lscale[1] = P.l_a; f_cLL[1] = P.c_LL_a; f_cLN[1] = P.c_LN_a;
    f_lscale[2] = P.l_c; f_cLL[2] = P.c_LL_c; f_cLN[2] = P.c_LN_c;
    f_lscale[3] = P.l_c; f_cLL[3] = P.c_LL_c; f_cLN[3] = P.c_LN_c;
    }

    // ── ODE right-hand side ─────────────────────────────────────────────────

    // dCv/dt  (Equations 34-35)
    ydot[0] = P.G_v + ann_cont_vL + ann_cont_avL
              - (R_i_v + R_2i_v + R_3i_v + R_v_s + loop_abs_v);

    // dCi/dt  (Equations 36-37)
    // 0.5*R_i_i: i+i -> 2i forms one di-interstitial per two monomers consumed.
    ydot[1] = P.G_i + R_2i_v
              + 2.0 * emission_2i + 3.0 * emission_3i
              - (R_i_v + R_i_i + R_i_2i + R_i_3i + R_i_s + loop_abs_i);

    // dC2i/dt  (Equations 38-39)
    ydot[2] = P.G_2i + 0.5 * R_i_i + R_3i_v
              - (R_2i_v + R_i_2i + R_2i_2i + R_2i_s) - emission_2i;

    // dC3i/dt  (Equations 40-41)
    ydot[3] = P.G_3i + R_i_2i
              - (R_3i_v + R_i_3i + R_3i_s + R_2i_3i) - emission_3i;

    // ── Loop number and content, from the family slots ──────────────────────
    // NUMBER changes only by nucleation, annealing and coalescence: growth
    // moves atoms into existing loops and never creates one. CONTENT carries
    // the net absorption (gain - loss), the atoms deposited at nucleation, and
    // the cascade content seed -- the last is essential, because without finite
    // content the sqrt(N*c) sink prefactor pins the family at the floor and it
    // can never grow.
    //
    // Legacy slots are (iL, aiL, vL, avL); self-consistent slots are
    // (c, a1, a2, a3). Equations 42-49 in the legacy numbering.
    if (P.loop_model == 0) {
        // VERBATIM the pre-existing expressions, including their association.
        // Regrouping them -- even into an algebraically identical form --
        // changes the rounding and the legacy result is no longer bit-identical,
        // which was measured: the 10th significant digit moves. The 28-parameter
        // fit was made against these exact operations.
        ydot[4] = nuc_iL;
        ydot[5] = nuc_aiL;
        ydot[6] = nuc_vL  - ann_vL;
        ydot[7] = nuc_avL - ann_avL;
        ydot[8] = growth_iL  + nuc_iL_frac  * nuc_content + P.G_iL;
        ydot[9] = growth_aiL + nuc_aiL_frac * nuc_content + P.G_aiL;
        ydot[10] = growth_vL  + P.G_vL  - ann_cont_vL;
        ydot[11] = growth_avL + P.G_avL - ann_cont_avL;
    } else {
        for (int k = 0; k < 4; ++k) {
            ydot[4 + k] = f_nucn[k] - f_annn[k];
            ydot[8 + k] = (f_gain[k] - f_loss[k]) + f_nucc[k] - f_annc[k];
        }
    }

    // ── Point-defect conservation accumulators (Eq. 12-17) ──────────────────
    // Monotonic time integrals of the production and physical loss channels for
    // the interstitial and vacancy atom balances. Internal transfers
    // (clustering, emission) cancel in the atom-weighted sums and are
    // intentionally excluded; the residual between stored atoms and
    // (production - losses) measured in post-processing quantifies the model's
    // conservation error.
    //   I_stored = Ci + 2*C2i + 3*C3i + CiL_i + CaiL_i
    //   V_stored = Cv + CvL_v + CavL_v
    const double prod_i = P.G_i + 2.0 * P.G_2i + 3.0 * P.G_3i + P.G_iL + P.G_aiL;
    const double prod_v = P.G_v + P.G_vL + P.G_avL;
    T recomb = R_i_v + R_2i_v + R_3i_v + loop_recomb;   // 1 i and 1 v per event
    T sink_i = R_i_s + 2.0 * R_2i_s + 3.0 * R_3i_s;     // network absorption
    T sink_v = R_v_s;

    ydot[12] = T(prod_i);   // cum_prod_i
    ydot[13] = recomb;      // cum_recomb_i
    ydot[14] = sink_i;      // cum_sink_i
    ydot[15] = T(prod_v);   // cum_prod_v
    ydot[16] = recomb;      // cum_recomb_v  (same Frenkel events)
    ydot[17] = sink_v;      // cum_sink_v

    // ── Geometric loop coalescence (absorbed-flux-climb driven) ─────────────
    //   (1) like-loop coarsening: number density drops, content conserved.
    //   (2) loop-network coalescence: number AND content removed; removed
    //       content booked as network sink absorption and the absorbed loop
    //       line length grows rho_N.
    // The rate scale is the loop climb velocity from the ABSORBED (gain-side)
    // flux, which stays positive at steady state, so coalescence persists.
    T sqrt_rhoN = ad_sqrt(rho_N);
    T d_N = 1.0 / sqrt_rhoN;   // loop-network spacing [m]; caps overlap radius

    const double cLL_geo = -P.kappa_LL * (4.0 / 3.0) * PI;
    const double cLN_geo = -P.kappa_LN * PI;

    auto coal = [&](const T& Cnum, const T& Ccont, double lscale, const T& gcont,
                    double c_LL, double c_LN,
                    T& num_loss, T& cont_loss, T& num_LN, T& r_out) {
        T r    = lscale * ad_sqrt(Ccont / Cnum);
        T N    = Cnum / P.Omega;
        // drdt from the gain-side absorption gcont gives the persistent
        // absorbed-flux climb speed v_abs = lscale*(l_c/l)*[Q]*Z_gain*flux/2.
        T drdt = lscale * gcont / (2.0 * ad_sqrt(Cnum * Ccont));
        T drdt_p = ad_val(drdt) > 0.0 ? drdt : T(0.0);
        // Bounded overlap gates: cap the radius at the relevant spacing so the
        // Avrami argument SATURATES instead of diverging as the number density
        // collapses. d_LL = N^(-1/3) (loop-loop), d_N = rho_N^(-1/2) (network).
        T d_LL = ad_cbrt(P.Omega / Cnum);
        T r_LL = ad_val(r) < ad_val(d_LL) ? r : d_LL;
        T r_LN = ad_val(r) < ad_val(d_N)  ? r : d_N;
        T phi_LL = 1.0 - ad_exp(cLL_geo * r_LL * r_LL * r_LL * N);
        T phi_LN = 1.0 - ad_exp(cLN_geo * r_LN * r_LN * rho_N);
        T nu_LL  = c_LL * drdt_p * ad_cbrt(N);
        T nu_LN  = c_LN * drdt_p * sqrt_rhoN;
        num_LN    = nu_LN * phi_LN * Cnum;
        num_loss  = nu_LL * phi_LL * Cnum + num_LN;
        cont_loss = nu_LN * phi_LN * Ccont;   // channel 1 conserves content
        r_out     = r;
    };

    // Coalescence is identical in both models -- the geometry does not care how
    // the families are labelled -- so it runs over the slots. The gain-side
    // absorption f_gain[k] sets the climb speed, which is why it stays positive
    // at steady state and coarsening persists.
    T coal_num[4], coal_cont[4], numLN[4], r_coal[4];
    for (int k = 0; k < 4; ++k)
        coal(f_num[k], f_cont[k], f_lscale[k], f_gain[k], f_cLL[k], f_cLN[k],
             coal_num[k], coal_cont[k], numLN[k], r_coal[k]);

    // The content removed by loop-network coalescence is booked as network sink
    // absorption on the polarity the family stores, so the atom balance closes.
    // In the legacy model slots 0,1 are interstitial and 2,3 vacancy; in the
    // self-consistent model slot 0 is vacancy and 1..3 interstitial.
    const int slot_is_vac[2][4] = {{0, 0, 1, 1}, {1, 0, 0, 0}};
    const int* sv = slot_is_vac[P.loop_model != 0 ? 1 : 0];
    for (int k = 0; k < 4; ++k) {
        ydot[4 + k] = ydot[4 + k] - coal_num[k];
        ydot[8 + k] = ydot[8 + k] - coal_cont[k];
    }
    if (P.loop_model == 0) {
        // Again verbatim: the paired sums associate differently from a loop.
        ydot[14] = ydot[14] + (coal_cont[0] + coal_cont[1]);   // cum_sink_i
        ydot[17] = ydot[17] + (coal_cont[2] + coal_cont[3]);   // cum_sink_v
    } else {
        for (int k = 0; k < 4; ++k) {
            if (sv[k]) ydot[17] = ydot[17] + coal_cont[k];
            else       ydot[14] = ydot[14] + coal_cont[k];
        }
    }

    // ── Evolving network dislocation density rho_N (index IDX_RHO_N) ────────
    // Source: each loop absorbed into the network contributes 2*pi*r of line
    // length per unit volume. Recovery: first-order relaxation of the
    // irradiation-grown excess toward the grown-in seed, so rho_N saturates.
    const double c_rho_geo = P.c_rhoN * (2.0 * PI / P.Omega);
    T rho_source = c_rho_geo *
        (r_coal[0] * numLN[0] + r_coal[1] * numLN[1] +
         r_coal[2] * numLN[2] + r_coal[3] * numLN[3]);
    T rho_recovery = P.k_rhoN_rec * (rho_N - P.rho_N);
    ydot[IDX_RHO_N] = rho_source - rho_recovery;

    // ── Floor enforcement ───────────────────────────────────────────────────
    // Zero any derivative that would drive a floored concentration further
    // negative. Restricted to the physical species; the accumulators are
    // monotonic non-negative integrals and must never be clamped.
    for (int k = 0; k < N_PHYS; ++k) {
        if (ad_val(y[k]) <= C_floor && ad_val(ydot[k]) < 0.0)
            ydot[k] = T(0.0);
    }

    // ── Operator-split QSSA: freeze the mobile species ──────────────────────
    // Two-time-scale coupling (the steady mobile FEM solve supplies y0[0:4]).
    // Zero the four mobile derivatives LAST so the immobile / accumulator /
    // rho_N equations above still used the frozen Cv,Ci,C2i,C3i as parameters.
    if (P.freeze_mobile) {
        ydot[0] = T(0.0);
        ydot[1] = T(0.0);
        ydot[2] = T(0.0);
        ydot[3] = T(0.0);
    }
}

}  // namespace zrcore
