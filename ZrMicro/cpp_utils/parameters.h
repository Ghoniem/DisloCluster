/**
 * parameters.h – ZrMicro solver parameter struct.
 *
 * All quantities that the ODE right-hand side needs are pre-computed on the
 * Python side (via InputData + ReactionRates) and forwarded to the solver as
 * --key=value command-line arguments.  build_parameters() unpacks those
 * arguments into this struct so the RHS only performs arithmetic, not lookups.
 *
 * Mirrors: py_utils/input_data.py, py_utils/reaction_rates.py
 */
#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <string>

// State vector layout:
//   [0..11]  physical species:
//            Cv Ci C2i C3i CiL CaiL CvL CavL CiL_i CaiL_i CvL_v CavL_v
//   [12..17] point-defect conservation accumulators (monotonic time integrals):
//            cum_prod_i cum_recomb_i cum_sink_i cum_prod_v cum_recomb_v cum_sink_v
//   The accumulators do not feed back into the physics; they integrate the
//   production / loss rates so post-processing can report the relative error in
//   the interstitial and vacancy atom balances (see rate_equations.cpp).
//   [18]     evolving network dislocation density rho_N [m^-2]: starts at the
//            grown-in seed (P.rho_N) and grows as loops are absorbed into the
//            network (loop-network coalescence); feeds back into the sink
//            strengths and coalescence gates.
static constexpr int N_PHYS = 12;   // physical species
static constexpr int N_ACC  = 6;    // conservation accumulators
static constexpr int N_RHO  = 1;    // evolving network density rho_N
static constexpr int IDX_RHO_N = N_PHYS + N_ACC;        // rho_N state index = 18
static constexpr int N_MOB   = 4;   // mobile species Cv Ci C2i C3i
static constexpr int N_IMMOB = 8;   // immobile species (loop numbers + contents)

// ── Step 1 of the implementation plan: the appended family slots ─────────────
// The plan raises the immobile family count from 4 to 8 -- adding the three
// prismatic VACANCY variants (v,a1..a3) and reserving one slot for the second
// basal state c_p that step 4 introduces.
//
// The four new families are APPENDED at 19..26 rather than inserted after the
// existing immobile block, so that every pre-existing state index keeps its
// meaning: mobile stays [0..3], the original four families stay [4..11], the
// accumulators stay [12..17] and rho_N stays 18. Every Python consumer that
// reads those columns -- coupling/field.py, post/coarsening.py, the 0-D figure
// suite, march_state.npz -- therefore keeps working against a longer state
// unchanged, which an insertion would have broken everywhere at once.
//
//   [19..22] numbers  n_a1v n_a2v n_a3v n_cp
//   [23..26] contents c_a1v c_a2v c_a3v c_cp
// Step 4 adds a NINTH family: the stacking-fault pyramid c_0, which is not a
// loop at all. It has no perimeter, no Burgers vector and no line density; its
// size is volumetric, R = (m Omega / sqrt 8)^(1/3), and it presents a capture
// cross-section rather than a sink line. Slot 8 carries it.
//
//   [19..23] numbers  n_a1v n_a2v n_a3v n_cp n_c0
//   [24..28] contents c_a1v c_a2v c_a3v c_cp c_c0
static constexpr int N_XFAM   = 5;              // appended family slots
static constexpr int N_XIMMOB = 2 * N_XFAM;     // their (number, content) pairs
static constexpr int IDX_XN   = 19;             // first appended NUMBER index
static constexpr int IDX_XC   = IDX_XN + N_XFAM;// first appended CONTENT index
static constexpr int N_FAM_MAX = 9;             // total family slots available
static constexpr int SLOT_SFP  = 8;             // the pyramid, c_0

static constexpr int N_EQ = N_PHYS + N_ACC + N_RHO + N_XIMMOB;   // total = 27

// State index of family k's number / content, for k in [0, N_FAM_MAX).
constexpr int fam_n_idx(int k) { return k < 4 ? 4 + k : IDX_XN + (k - 4); }
constexpr int fam_c_idx(int k) { return k < 4 ? 8 + k : IDX_XC + (k - 4); }

// ── Reduced (implicit-block) state layout ────────────────────────────────────
// The six accumulators are PURE QUADRATURES: rows 12..17 of the RHS are written
// but columns 12..17 are never read by any other equation. They therefore never
// belong in the Newton system, and CVODES can carry them as quadrature
// variables (CVodeQuadInit) integrated on the same step sequence but excluded
// from the implicit solve and, by default, from the error test.
//
// Under the operator split the four mobile species are additionally frozen, so
// the block the dense LU actually factorises is:
//     freeze_mobile : 8 immobile + rho_N             =  9
//     otherwise     : 4 mobile + 8 immobile + rho_N  = 13
// versus 19 before. Dense LU is O(n^3), so 9 vs 19 is ~9.4x fewer flops per
// Newton solve and the AD Jacobian sweep shrinks in proportion.
// The implicit block is sized by the ACTIVE family count, not by the slots
// available. Step 1 raises the ceiling from 4 families to 8, but a run that
// carries only the original 4 must keep factorising a 9x9, not a 17x17: dense
// LU is O(n^3), so paying for the empty slots would cost 6.7x per Newton solve
// for nothing. `n_fam` therefore selects among compile-time sizes at run time.
template <int NF> struct RedDims {
    static constexpr int frozen = 2 * NF + N_RHO;            // immobile + rho_N
    static constexpr int free_  = N_MOB + 2 * NF + N_RHO;    // + mobile
    static constexpr int rlx_frozen = frozen + N_ACC;
    static constexpr int rlx_free   = free_  + N_ACC;
};
static constexpr int N_RED_FROZEN = RedDims<4>::frozen;   //  9  (default)
static constexpr int N_RED_FREE   = RedDims<4>::free_;    // 13

// ── How the accumulators are carried (--acc_mode) ────────────────────────────
//   0 STATE_CTRL   legacy: in the implicit state AND in the error test.
//   1 QUADRATURE   CVODES quadrature variables. Smallest Newton block (9/13),
//                  but the quadrature right-hand side needs its own evaluation
//                  of the core at the accepted step point, and the memo that
//                  tries to reuse the residual's evaluation almost always
//                  misses (measured 1 hit in 2565), so it costs ~1 extra core
//                  sweep per step.
//   2 STATE_RELAX  in the implicit state, but with a per-component atol large
//                  enough that their error weights vanish. Larger Newton block
//                  (15/19) yet ZERO extra core evaluations, and it keeps the
//                  property that matters: the accumulators, which restart at
//                  zero every coupling substep, no longer drive the initial
//                  step down to the roundoff limit.
static constexpr int ACC_STATE_CTRL  = 0;
static constexpr int ACC_QUADRATURE  = 1;
static constexpr int ACC_STATE_RELAX = 2;

// Newton-block sizes for mode 2: the reduced set plus the six accumulators.
static constexpr int N_RLX_FROZEN = RedDims<4>::rlx_frozen;   // 15 (default)
static constexpr int N_RLX_FREE   = RedDims<4>::rlx_free;     // 19

// Largest reduced state any `n_fam` can produce. Stack buffers that hold a
// reduced vector are sized by this, never by the default: at n_fam = 8 the
// relaxed free block is 27, and sizing those buffers 19 would overflow them
// silently the first time the appended families were switched on.
static constexpr int N_RED_MAX = RedDims<N_FAM_MAX>::rlx_free;   // 27

// atol given to the accumulator components in mode 2. Large enough that
// 1/(rtol*|y| + atol) underflows the error weight to nothing, small enough to
// stay far from overflow.
static constexpr double ACC_ATOL_RELAXED = 1.0e300;

struct Parameters {
    // ── Pre-computed jump frequencies (ReactionRates.calculate_basic_frequencies) ──
    double omega_i;   // interstitial jump frequency  [s^-1]
    double omega_v;   // vacancy jump frequency        [s^-1]
    double omega_2i;  // di-interstitial jump freq     [s^-1]  (omega_3i = omega_2i)

    // ── Thermal emission probabilities (ReactionRates.calculate_thermal_emissions) ──
    double e_v;             // exp(-E_F_v / kT)
    double e_2i;            // exp(-E_b_2i / kT)
    double e_3i;            // exp(-E_b_3i / kT)
    double e_v_iL;          // exp(-5 eV / kT)  — approx for interstitial loops
    double e_v_vL;          // exp(-5 eV / kT)  — approx for vacancy loops
    // stress-modified emissions (ReactionRates.update_stress_effects)
    double e_sigma_v_iL;
    double e_sigma_v_vL;
    double e_sigma_v_avL;

    // ── Sink / material parameters ──────────────────────────────────────────
    double rho_N;   // network dislocation density  [m^-2]
    double a;       // lattice parameter             [m]
    double z_c;     // coordination number           [-]  (used as double: a^2/z_c)
    double Omega;   // atomic volume                 [m^3]
    double Z_N;     // bias factor – network dislocations (interstitials)
    // DAD orientation/species-resolved loop capture efficiencies
    // (InputData.derived, from delta_DAD). a-loops favour interstitials,
    // c-loops favour vacancies — see rate_equations.cpp / reaction_rates.py.
    double Z_i_a;   // a-loops (iL, aiL) capture interstitials   = 1 + delta_i
    double Z_v_a;   // a-loops (iL, aiL) capture vacancies        = 1 - delta_v
    double Z_i_c;   // c-loops (vL, avL) capture interstitials    = 1 - delta_i
    double Z_v_c;   // c-loops (vL, avL) capture vacancies        = 1 + delta_v
    double recom;   // recombination volume factor   [-]

    // ── Loop model selector ─────────────────────────────────────────────────
    // 0 = LEGACY (default). Four families split aligned/non-aligned --
    //     iL, aiL, vL, avL -- with the PHENOMENOLOGICAL efficiencies Z_i_a,
    //     Z_v_a, Z_i_c, Z_v_c above, taken from the fitted delta_DAD. The
    //     interstitial species are lumped into one flux, so all of Ci, C2i and
    //     C3i are captured with the same Z. This is the formulation the 28
    //     calibrated parameters were fitted against and it must stay
    //     reproducible bit-for-bit; see loop_model == 0 in
    //     rate_equations_core.h, which is untouched by the alternative.
    //
    // 1 = SELF-CONSISTENT. Four families matching the 3-D code -- one basal
    //     <c> and three prismatic <a> variants, NO aligned/non-aligned split --
    //     with Woo capture efficiencies generated from the SAME diffusion
    //     tensor the fast solve diffuses with, per mobile species:
    //
    //         Z_basal(m)     = Z0_m * p_m
    //         Z_prismatic(m) = Z0_m * (p_m + p_m^-2)/2
    //
    //     identical to ClusterDynamicsParameters::loopDADbias. Each mobile
    //     species then carries its own efficiency instead of sharing one, and
    //     the loop sink strength is scaled per family by loopSinkScale exactly
    //     as ImmobileSinks does.
    int    loop_model;

    // ── Step 1: how many immobile families are ACTIVE ────────────────────────
    // 4 (default) is the pre-step-1 set: <c> plus the three prismatic
    // interstitial variants. 8 adds the three prismatic VACANCY variants
    // (v,a1..a3) and the reserved second basal slot c_p that step 4 fills.
    //
    // This is the step's regression switch, and it is the COUNT rather than the
    // cascade yield eps_avL for a reason worth stating: a family carrying no
    // nucleation still sits at the concentration floor C_floor, and a family at
    // the floor still contributes floor-level terms to loop-network coalescence,
    // hence to rho_N and to the sink accumulators. eps_avL = 0 therefore
    // recovers the previous PHYSICS but not the previous BITS. n_fam = 4 skips
    // the slots entirely and recovers both.
    int    n_fam;

    // ── Step 2: the character factor X_{s,m} of Eq. (Zsk) ───────────────────
    // Every capture efficiency is a product Z^0_m * A_h(k)(p_m) * X_{s,m} * SIPA.
    // A_h breaks the ORIENTATION degeneracy (basal vs prismatic) and is already
    // carried; X breaks the CHARACTER one (what the loop stores), and without
    // it an interstitial and a vacancy loop on the same habit plane capture
    // identically.
    //
    // Thermal-drift capture radii make same-type capture the stronger one -- an
    // interstitial loop's dilatational field reaches interstitials further than
    // it reaches vacancies, and conversely -- so X is parameterised by the
    // single character-splitting factor
    //     chi = X_iI X_vV / (X_iV X_vI)
    // taking X = chi^(+1/4) for like pairs and chi^(-1/4) for unlike ones. That
    // reproduces chi exactly, leaves the geometric mean of the four at 1 (so no
    // overall capture magnitude is smuggled in), and gives X == 1.0 identically
    // at chi = 1, which is the step's regression.
    double chi;

    // ── Step 3: peripheral emission replaces the annealing lifetimes ────────
    // The pre-step-3 model removes vacancy loops through a fitted dissolution
    // lifetime tau_vL / tau_avL: a first-order number loss with no dependence on
    // the loop's size, on the stress, or on how far the matrix is from
    // equilibrium with it. It is a surrogate, and it cannot satisfy detailed
    // balance -- setting c^v to the loop's own equilibrium concentration does
    // not make the exchange vanish, because nothing in tau knows what that
    // concentration is.
    //
    // Step 3 replaces it with the physical channel. Every loop, of EITHER
    // character, exchanges vacancies with the matrix at the NET rate
    //
    //     phi_k = Z_k Dbar_v S_k [ c^v - c^{v,eq}_k(R_k, Sigma_k) ]
    //
    // which is identically zero at c^v = c^{v,eq}_k. The equilibrium
    // concentration comes from the work to detach one vacancy from the loop
    // periphery,
    //
    //     dE_k/dm = gamma_k Omega/(b.n)_k
    //             + Kbar_k |b|_k^2 lam_k^2 /(4R) [1 + ln(alpha R/|b|_k)]
    //     E_b     = E^f_v - varsigma_s(v) dE_k/dm - Sigma_k Omega
    //     c^{v,eq}= exp(-E_b / kT)
    //
    // with alpha = 8/e^2 and varsigma_s(v) = +1 for a vacancy family, -1 for an
    // interstitial one. All of it is Sec. 4.1-4.2 of the formulation, and the
    // Python reference is studies/loop_annealing.py, which these must reproduce.
    //
    // 0 = the legacy lifetimes (the default, so nothing that ran before step 3
    //     changes); 1 = the computed binding energy.
    //
    // THE PLAN SAYS THIS STEP HAS NO REGRESSION, because it removes a channel
    // rather than generalizing one. Keeping the old channel behind this switch
    // manufactures one: emission_model = 0 reproduces step 2 bit-for-bit, so
    // the irreversible step becomes A/B-comparable after all.
    int    emission_model;

    double kT_eV;        // k_B T in eV -- the solver otherwise never sees T
    double Ef_v;         // [eV] vacancy formation energy
    // Per family, in SI. Zero gamma means an unfaulted loop.
    double emis_gamma[N_FAM_MAX];   // [J/m^2] fault energy on the loop plane
    double emis_Kbar[N_FAM_MAX];    // [Pa]    orientation-averaged K
    double emis_bmag[N_FAM_MAX];    // [m]     |b|_k
    double emis_bdotn[N_FAM_MAX];   // [m]     (b.n)_k, the edge component
    double emis_lam[N_FAM_MAX];     // [m]     lam_k = sqrt(Omega/(pi (b.n)_k))
    double emis_sigma[N_FAM_MAX];   // [Pa]    resolved normal stress on the habit

    // ── Step 4: the basal chain c_0 -> c_f -> c_p ───────────────────────────
    // The stacking-fault pyramid is a COMPACT cluster, not a loop. Three things
    // follow and each is a special case in the family loop:
    //   * its size is volumetric, R = (m Omega/sqrt 8)^(1/3), Eq. (Rsfp) --
    //     there is no lambda_k and no Burgers vector to form R = lam sqrt(m);
    //   * it presents a capture cross-section rather than a sink line,
    //     Pi = alpha_sfp R n /(2 l), Eq. (Pisfp);
    //   * it has NO coalescence channel at all -- loop-loop and loop-network
    //     are defined on the loop families, and a compact cluster is neither.
    //
    // Its number equation is therefore linear and saturates in closed form,
    //     n* = J/(1/tau_sfp + nu_col),      f_col = nu_col/(1/tau_sfp + nu_col)
    // which is Eq. (sfp-saturation) and is validation goal (ii).
    //
    // The two transfers move number and content together at the SAME rate, so
    // nothing is created or destroyed by a conversion: col takes c_0 -> c_f and
    // uf takes c_f -> c_p. The distribution-fraction gates Phi^(j) of
    // Eq. (barrier) need the size distribution that step 5 supplies; until then
    // they are 1, which the formulation names as the barrier-limited limit.
    //
    // 0 = no basal chain (the default: the pyramid slot carries nothing and the
    //     model is exactly step 3); 1 = the chain runs.
    int    basal_chain;
    double eps_sfp;      // cascade yield into the pyramid, as a rate [1/s]
    double n_sfp_nuc;    // vacancies per cascade-nucleated pyramid
    double tau_sfp;      // [s] pyramid thermal dissolution lifetime
    double nu_col;       // [1/s] pyramid -> faulted basal loop
    double nu_uf;        // [1/s] faulted -> perfect basal loop
    double alpha_sfp;    // [-] pyramid capture prefactor, Eq. (partial_sink_bp)

    // Per mobile species (v, i, 2i, 3i). Used only when loop_model >= 1.
    double dad_p[4];    // p_m = (D_c/D_a)^(1/6), from the migration energies
    double dad_Z0[4];   // Z0_m, the isotropic-limit capture efficiency
    // Per family. loop_sink_scale replaces the legacy Q, which scaled the <c>
    // channel alone; here every family carries its own factor, as in the
    // material file's loopSinkScale. Slots 0..3 are (c, a1, a2, a3) and slots
    // 4..7 are (a1v, a2v, a3v, c_p).
    double loop_sink_scale[N_FAM_MAX];
    double variant_frac[3];   // how INTERSTITIAL <a> nucleation splits
    // How VACANCY <a> nucleation splits. Distinct from variant_frac because an
    // interstitial loop inserts material on its habit plane and a vacancy loop
    // removes it, so a resolved normal stress favours one exactly as much as it
    // disfavours the other: w^v_k is the Boltzmann weight with the sign flipped.
    // Defaults to variant_frac, which makes the two identical at zero
    // deviatoric stress -- where both are 1/3 -- and so changes nothing that ran
    // before step 1.
    double variant_frac_v[3];

    // ── Pre-computed length scales (InputData.calculate_derived_parameters) ──
    double l;    // z_c*Omega / (2*pi*a^2)
    double l_a;  // sqrt(Omega / (pi*b_a))
    double l_c;  // sqrt(Omega / (pi*b_c))

    // ── Loop growth ─────────────────────────────────────────────────────────
    double Q;    // absorption efficiency for <c> loops

    // ── Effective generation rates (InputData.derived, already divided) ────
    double G_v;    // vacancy generation rate
    double G_i;    // interstitial generation rate
    double G_2i;   // di-interstitial generation rate  (= derived['G_2i'] / 2)
    double G_3i;   // tri-interstitial generation rate (= derived['G_3i'] / 3)
    double G_iL;   // cascade interstitial a-loop atom rate (non-aligned)
    double G_aiL;  // cascade interstitial a-loop atom rate (aligned)
    double G_vL;   // vacancy loop generation rate
    double G_avL;  // aligned vacancy loop generation rate

    // ── Loop fractions (InputData.derived) ──────────────────────────────────
    double f_a;   // aligned fraction
    double f_na;  // non-aligned fraction  (= 1 - f_a in derived dict)

    // ── Thermal annealing lifetimes ──────────────────────────────────────────
    double tau_vL;   // vacancy loop annealing lifetime   [s]
    double tau_avL;  // aligned vacancy loop lifetime     [s]

    // Vacancies per nucleated cascade vacancy loop. G_vL/G_avL are vacancy ATOM
    // rates: loop NUMBER nucleation = G_vL/n_vL_nuc, loop CONTENT seed = G_vL.
    double n_vL_nuc;
    // Interstitials per nucleated cascade a-loop (mirror of n_vL_nuc). G_iL/G_aiL
    // are interstitial ATOM rates: loop NUMBER nucleation = G_iL/n_iL_nuc,
    // loop CONTENT seed = G_iL.
    double n_iL_nuc;

    // ── Geometric loop coalescence (growth-flux driven) ──────────────────────
    double c_LL;      // like-loop coalescence rate-scale coefficient   [-]
    double kappa_LL;  // loop-loop capture-zone factor                  [-]
    double c_LN;      // loop-network coalescence rate-scale coefficient[-]
    double kappa_LN;  // loop-network capture factor                    [-]
    // Orientation-resolved rate scales (a-loops = iL/aiL, c-loops = vL/avL).
    // Default to the shared c_LL/c_LN. Weaker c-loop coalescence -> larger
    // c-component vacancy loops, decoupled from the a-loop size/density.
    double c_LL_a;    // a-loop like-loop coalescence
    double c_LN_a;    // a-loop loop-network coalescence
    double c_LL_c;    // c-loop like-loop coalescence
    double c_LN_c;    // c-loop loop-network coalescence

    // ── Evolving network dislocation density rho_N ────────────────────────────
    // rho_N grows by the line length of loops absorbed into the network
    // (loop-network coalescence) and recovers first-order toward the grown-in
    // seed (P.rho_N), so it saturates rather than growing without bound.
    double c_rhoN;      // loop-network line-length transfer efficiency  [-]
    double k_rhoN_rec;  // first-order network recovery rate             [s^-1]

    // ── Minimum stable a-loop radius ──────────────────────────────────────────
    // Halts a-loop vacancy absorption (and hence shrinkage) as the loop radius
    // approaches r_min_a, so the mean a-loop diameter cannot collapse to zero at
    // high T / high dose. w_rmin sets the smoothstep ramp width (in units of
    // r_min_a). r_min_a <= 0 disables the gate.
    double r_min_a;   // minimum stable a-loop radius                   [m]
    double w_rmin;    // gate ramp width (fraction of r_min_a)          [-]

    // ── Concentration floor ──────────────────────────────────────────────────
    double C_floor;

    // ── Operator-split QSSA (two-time-scale ZrMicro <-> MoDELib2-NNL coupling) ─
    // When true, the four mobile species (Cv, Ci, C2i, C3i) are held FIXED at
    // their y0 values: their derivatives are zeroed last, so the immobile,
    // accumulator, and rho_N equations still read them as frozen parameters.
    // The spatially-resolved code supplies y0[0:4] from its steady mobile FEM
    // solve. Default false -> the standalone 0-D model is unchanged.
    bool   freeze_mobile;

    // ── Reduced implicit block ───────────────────────────────────────────────
    // One of ACC_STATE_CTRL / ACC_QUADRATURE / ACC_STATE_RELAX above. CVODE
    // only (ARKODE has no quadrature module). Default 0 -> the legacy
    // 19-equation implicit system, so existing runs stay bit-identical.
    int    acc_mode;

    // ── Analytic Jacobian by forward-mode AD (see dual.h) ────────────────────
    // Default false -> SUNDIALS' difference-quotient Jacobian. Dense linear
    // solver only; ignored for band/GMRES.
    bool   analytic_jac;

    // ── Fixed-step explicit Euler (reference integrator) ────────────────────
    // > 0 replaces CVODE with euler_nsub forward-Euler substeps per OUTPUT
    // interval, using the same rate_equations_core.h RHS. This is the scheme
    // MoDELib3's ClusterDynamicsFEM::solveImmobileClusters uses for the
    // immobile field (nSub = 20 substeps per dose step, nodal, no linear
    // algebra), so it lets the two integrators be compared on identical
    // equations, the same machine and the same units.
    int    euler_nsub;

    // Emit a "# STATS ..." comment line with the integrator counters. Ignored by
    // the Python row parsers (non-numeric line); read by the benchmark harness.
    bool   stats;

    // ── Initial concentrations y0[12] ────────────────────────────────────────
    double y0[N_EQ];

    // ── Solver settings ──────────────────────────────────────────────────────
    double t_begin;
    double t_end;
    int    n_points;
    bool   log_time;
    double rtol;
    double atol;

    // ── Integration method ────────────────────────────────────────────────────
    // backend   : 0=CVODE (default), 1=ARKODE ARKStep (implicit RK / DIRK)
    // lmm       : CVODE linear multistep — 2=CV_BDF (stiff, default), 1=CV_ADAMS
    // linsol    : linear solver — 0=dense (default), 1=band, 2=gmres
    // mu, ml    : upper/lower bandwidth for band solver (default = N_EQ-1 = full)
    // max_order : max solver order; 0 = solver default
    // ark_table : ARKODE_DIRKTableID integer (default 111 = ARK548L2SA_DIRK_8_4_5, 5th order)
    int    backend;
    int    lmm;
    int    linsol;
    int    mu;
    int    ml;
    int    max_order;
    int    ark_table;
};

// ── Reduced-state index mapping ──────────────────────────────────────────────

// True when the solver runs a reduced/reordered state rather than the plain
// 19-component legacy layout.
inline bool is_reduced(const Parameters& P) {
    return P.acc_mode == ACC_QUADRATURE || P.acc_mode == ACC_STATE_RELAX;
}

// Size of the implicit core (everything but the appended accumulators).
inline int red_core(const Parameters& P) {
    return (P.freeze_mobile ? 0 : N_MOB) + 2 * P.n_fam + N_RHO;
}

// Dimension of the implicit block actually solved.
inline int red_dim(const Parameters& P) {
    return red_core(P) + (P.acc_mode == ACC_STATE_RELAX ? N_ACC : 0);
}

// Reduced index j -> index into the full state.
//
// Ordered numbers-then-contents, which is what the original layout already was
// (4..7 are the four numbers, 8..11 the four contents), so at n_fam = 4 this
// reproduces the previous mapping index for index:
//   frozen : j = 0..3 -> 4..7, j = 4..7 -> 8..11, j = 8  -> 18
//   free   : j = 0..3 -> 0..3, then the same, and j = 12 -> 18
// At n_fam = 8 the appended families follow through fam_n_idx / fam_c_idx.
inline int red_idx(const Parameters& P, int j) {
    const int n_core = red_core(P);
    if (j >= n_core)            // modes 1/2: the six accumulators, appended
        return N_PHYS + (j - n_core);
    int t = j;
    if (!P.freeze_mobile) {
        if (t < N_MOB) return t;
        t -= N_MOB;
    }
    const int nf = P.n_fam;
    if (t < nf)      return fam_n_idx(t);
    t -= nf;
    if (t < nf)      return fam_c_idx(t);
    return IDX_RHO_N;
}

// Expand a reduced state into the full 19-vector the physics core expects.
// Accumulator slots are zeroed: the core writes rows 12..17 but never reads
// them, so whatever sits there cannot influence the result. Frozen mobile
// values come from y0[0:4], which is where the FEM fast solve deposits C_M*.
inline void red_scatter(const Parameters& P, const double* yr, double* yf) {
    // Accumulator slots start at zero; in mode 2 the loop below overwrites them
    // with the carried state, and in mode 1 they stay zero (the core writes
    // rows 12..17 but never reads them, so the value cannot matter).
    for (int k = 0; k < N_EQ; ++k) yf[k] = 0.0;
    if (P.freeze_mobile)
        for (int k = 0; k < N_MOB; ++k) yf[k] = P.y0[k];
    const int n = red_dim(P);
    for (int j = 0; j < n; ++j) yf[red_idx(P, j)] = yr[j];
}

// ── CLI argument helpers ─────────────────────────────────────────────────────

inline double require_param(const std::map<std::string, double>& m,
                            const std::string& key) {
    auto it = m.find(key);
    if (it == m.end()) {
        std::cerr << "Missing required parameter: " << key << "\n";
        exit(1);
    }
    return it->second;
}

inline double optional_param(const std::map<std::string, double>& m,
                              const std::string& key, double def) {
    auto it = m.find(key);
    return (it != m.end()) ? it->second : def;
}

// Unpack a CLI argument map into a Parameters struct.
inline Parameters build_parameters(const std::map<std::string, double>& p) {
    Parameters P{};

    // Jump frequencies
    P.omega_i  = require_param(p, "omega_i");
    P.omega_v  = require_param(p, "omega_v");
    P.omega_2i = require_param(p, "omega_2i");

    // Thermal emissions
    P.e_v            = require_param(p, "e_v");
    P.e_2i           = require_param(p, "e_2i");
    P.e_3i           = require_param(p, "e_3i");
    P.e_v_iL         = require_param(p, "e_v_iL");
    P.e_v_vL         = require_param(p, "e_v_vL");
    P.e_sigma_v_iL   = require_param(p, "e_sigma_v_iL");
    P.e_sigma_v_vL   = require_param(p, "e_sigma_v_vL");
    P.e_sigma_v_avL  = require_param(p, "e_sigma_v_avL");

    // Sink / material
    P.rho_N  = require_param(p, "rho_N");
    P.a      = require_param(p, "a");
    P.z_c    = require_param(p, "z_c");
    P.Omega  = require_param(p, "Omega");
    P.Z_N    = optional_param(p, "Z_N",   1.05);
    // DAD loop capture efficiencies (defaults match input_data.py delta_DAD=0.2)
    P.Z_i_a  = optional_param(p, "Z_i_a", 1.2);
    P.Z_v_a  = optional_param(p, "Z_v_a", 0.8);
    P.Z_i_c  = optional_param(p, "Z_i_c", 0.8);
    P.Z_v_c  = optional_param(p, "Z_v_c", 1.2);
    P.recom  = optional_param(p, "recom", 1.0);

    // Loop model. Absent => 0 => the legacy formulation, so every existing
    // command line, fit and stored result reproduces exactly.
    P.loop_model = static_cast<int>(optional_param(p, "loop_model", 0.0));
    {
        // Defaults are the isotropic limit: p_m = 1 makes both Woo rows equal
        // Z0_m, so a self-consistent run with no tensor anisotropy supplied is
        // unbiased rather than accidentally biased.
        const char* pk[4]  = {"dad_p_v",  "dad_p_i",  "dad_p_2i",  "dad_p_3i"};
        const char* z0k[4] = {"dad_Z0_v", "dad_Z0_i", "dad_Z0_2i", "dad_Z0_3i"};
        for (int m = 0; m < 4; ++m) {
            P.dad_p[m]  = optional_param(p, pk[m],  1.0);
            P.dad_Z0[m] = optional_param(p, z0k[m], 1.0);
        }
        // Slots 4..6 are the prismatic VACANCY variants, which share the <a>
        // habit and therefore default to the <a> sink scale rather than to 1:
        // the geometric factor belongs to the habit plane, not to the polarity.
        // Slot 7 is c_p, reserved for step 4, defaulting to the <c> value.
        const char* sk[N_FAM_MAX] = {
            "loop_sink_scale_c",   "loop_sink_scale_a1",
            "loop_sink_scale_a2",  "loop_sink_scale_a3",
            "loop_sink_scale_a1v", "loop_sink_scale_a2v",
            "loop_sink_scale_a3v", "loop_sink_scale_cp",
            "loop_sink_scale_c0"};
        for (int k = 0; k < 4; ++k)
            P.loop_sink_scale[k] = optional_param(p, sk[k], 1.0);
        for (int k = 4; k < 7; ++k)
            P.loop_sink_scale[k] = optional_param(p, sk[k],
                                                  P.loop_sink_scale[k - 3]);
        P.loop_sink_scale[7] = optional_param(p, sk[7], P.loop_sink_scale[0]);
        P.loop_sink_scale[8] = optional_param(p, sk[8], 1.0);
        const char* vk[3] = {"variant_frac_a1", "variant_frac_a2",
                             "variant_frac_a3"};
        for (int j = 0; j < 3; ++j)
            P.variant_frac[j] = optional_param(p, vk[j], 1.0 / 3.0);
        const char* vvk[3] = {"variant_frac_v_a1", "variant_frac_v_a2",
                              "variant_frac_v_a3"};
        for (int j = 0; j < 3; ++j)
            P.variant_frac_v[j] = optional_param(p, vvk[j], P.variant_frac[j]);
    }

    // Active family count. 4 is the pre-step-1 set and the default, so every
    // existing command line reproduces bit-for-bit; 8 switches on the prismatic
    // vacancy variants of step 1. Anything else is a typo, not a request.
    // ── Step 4 ──────────────────────────────────────────────────────────────
    P.basal_chain = static_cast<int>(optional_param(p, "basal_chain", 0.0));
    P.eps_sfp   = optional_param(p, "eps_sfp",   0.0);
    P.n_sfp_nuc = optional_param(p, "n_sfp_nuc", 1.0);
    P.tau_sfp   = optional_param(p, "tau_sfp",   0.0);
    P.nu_col    = optional_param(p, "nu_col",    0.0);
    P.nu_uf     = optional_param(p, "nu_uf",     0.0);
    P.alpha_sfp = optional_param(p, "alpha_sfp", 1.0);
    // The n_fam consistency check lives with n_fam's own parsing, below.
    // Checking it here read P.n_fam before it was set -- value-initialized to
    // zero -- so basal_chain=1 was rejected unconditionally.

    // ── Step 3 ──────────────────────────────────────────────────────────────
    P.emission_model = static_cast<int>(
        optional_param(p, "emission_model", 0.0));
    P.kT_eV = optional_param(p, "kT_eV", 0.0);
    P.Ef_v  = optional_param(p, "Ef_v", 0.0);
    {
        const char* fk[N_FAM_MAX] = {"c", "a1", "a2", "a3",
                                     "a1v", "a2v", "a3v", "cp", "c0"};
        for (int k = 0; k < N_FAM_MAX; ++k) {
            P.emis_gamma[k] = optional_param(p, std::string("emis_gamma_") + fk[k], 0.0);
            P.emis_Kbar[k]  = optional_param(p, std::string("emis_Kbar_")  + fk[k], 0.0);
            P.emis_bmag[k]  = optional_param(p, std::string("emis_bmag_")  + fk[k], 0.0);
            P.emis_bdotn[k] = optional_param(p, std::string("emis_bdotn_") + fk[k], 0.0);
            P.emis_lam[k]   = optional_param(p, std::string("emis_lam_")   + fk[k], 0.0);
            P.emis_sigma[k] = optional_param(p, std::string("emis_sigma_") + fk[k], 0.0);
        }
    }
    if (P.emission_model != 0) {
        // Refuse to run rather than silently emit at exp(-0/0). Every one of
        // these has to arrive from the material file; none has a sensible
        // default, and a wrong c^{v,eq} is not a small error -- it is
        // exponential in the quantity that is missing.
        if (P.kT_eV <= 0.0 || P.Ef_v <= 0.0) {
            std::cerr << "emission_model=1 requires kT_eV>0 and Ef_v>0\n";
            exit(1);
        }
        for (int k = 0; k < P.n_fam; ++k) {
            if (P.emis_bdotn[k] <= 0.0 || P.emis_bmag[k] <= 0.0
                || P.emis_lam[k] <= 0.0 || P.emis_Kbar[k] <= 0.0) {
                std::cerr << "emission_model=1: family " << k
                          << " is missing its geometry (bdotn/bmag/lam/Kbar)\n";
                exit(1);
            }
        }
    }

    // Character splitting. 1.0 is the character-degenerate model and the
    // default, so every command line written before step 2 is unchanged.
    P.chi = optional_param(p, "chi", 1.0);
    if (P.chi <= 0.0) {
        std::cerr << "chi must be positive, got " << P.chi << "\n";
        exit(1);
    }

    P.n_fam = static_cast<int>(optional_param(p, "n_fam", 4.0));
    if (P.n_fam != 4 && P.n_fam != 8 && P.n_fam != 9) {
        std::cerr << "n_fam must be 4, 8 or 9, got " << P.n_fam << "\n";
        exit(1);
    }
    if (P.basal_chain != 0 && P.n_fam < N_FAM_MAX) {
        std::cerr << "basal_chain=1 requires n_fam=" << N_FAM_MAX
                  << " (the pyramid occupies slot " << SLOT_SFP << ")\n";
        exit(1);
    }
    if (P.n_fam != 4 && P.loop_model == 0) {
        // The legacy slots are (iL, aiL, vL, avL) -- aligned/non-aligned, not
        // habit-resolved -- so an appended prismatic vacancy family has no
        // meaning there and would silently be added to a different population.
        std::cerr << "n_fam=8 requires loop_model>=1 "
                     "(the legacy slots are not habit-resolved)\n";
        exit(1);
    }

    // Length scales
    P.l   = require_param(p, "l");
    P.l_a = require_param(p, "l_a");
    P.l_c = require_param(p, "l_c");

    // Loop growth
    P.Q = require_param(p, "Q");

    // Generation rates
    P.G_v   = require_param(p, "G_v");
    P.G_i   = require_param(p, "G_i");
    P.G_2i  = require_param(p, "G_2i");
    P.G_3i  = require_param(p, "G_3i");
    P.G_iL  = require_param(p, "G_iL");
    P.G_aiL = require_param(p, "G_aiL");
    P.G_vL  = require_param(p, "G_vL");
    P.G_avL = require_param(p, "G_avL");

    // Fractions
    P.f_a  = require_param(p, "f_a");
    P.f_na = require_param(p, "f_na");

    // Annealing
    P.tau_vL   = require_param(p, "tau_vL");
    P.tau_avL  = require_param(p, "tau_avL");
    P.n_vL_nuc = optional_param(p, "n_vL_nuc", 20.0);
    P.n_iL_nuc = optional_param(p, "n_iL_nuc", 20.0);

    // Geometric loop coalescence (defaults match input_data.py)
    P.c_LL     = optional_param(p, "c_LL",     1.0);
    P.kappa_LL = optional_param(p, "kappa_LL", 1.0);
    P.c_LN     = optional_param(p, "c_LN",     1.0);
    P.kappa_LN = optional_param(p, "kappa_LN", 1.0);
    // Orientation-resolved scales; absent -> fall back to the shared c_LL/c_LN.
    P.c_LL_a   = optional_param(p, "c_LL_a", P.c_LL);
    P.c_LN_a   = optional_param(p, "c_LN_a", P.c_LN);
    P.c_LL_c   = optional_param(p, "c_LL_c", P.c_LL);
    P.c_LN_c   = optional_param(p, "c_LN_c", P.c_LN);

    // Evolving network density (defaults match reaction_rates.py rho_N_source)
    P.c_rhoN     = optional_param(p, "c_rhoN",     0.1);
    P.k_rhoN_rec = optional_param(p, "k_rhoN_rec", 1e-7);

    // Minimum stable a-loop radius (defaults match reaction_rates.py a_shrink_gate)
    P.r_min_a = optional_param(p, "r_min_a", 1.0e-9);
    P.w_rmin  = optional_param(p, "w_rmin",  0.3);

    // Floor
    P.C_floor = optional_param(p, "C_floor", 1e-20);

    // Operator-split QSSA flag (two-time-scale coupling); default off.
    P.freeze_mobile = (optional_param(p, "freeze_mobile", 0.0) > 0.5);

    // Reduced implicit block + analytic AD Jacobian; both default off so the
    // legacy path is untouched.
    // --acc_mode is the current spelling; --reduced=1 remains accepted as the
    // older name for the quadrature mode.
    P.acc_mode     = static_cast<int>(optional_param(
                         p, "acc_mode",
                         (optional_param(p, "reduced", 0.0) > 0.5) ? 1.0 : 0.0));
    P.analytic_jac = (optional_param(p, "analytic_jac", 0.0) > 0.5);
    P.stats        = (optional_param(p, "stats",        0.0) > 0.5);
    P.euler_nsub   = static_cast<int>(optional_param(p, "euler_nsub", 0.0));

    // Initial conditions.
    //
    // The first 19 are required, as they always were. The eight appended by
    // step 1 are OPTIONAL and default to zero, so every command line written
    // before step 1 -- and every caller that still builds a 19-component state,
    // which at this point is all of them -- keeps working unchanged and starts
    // the new families empty. Requiring them would have made the state length
    // a breaking change for the whole Python side at once, which is exactly
    // what appending the slots was meant to avoid.
    for (int k = 0; k < N_PHYS + N_ACC + N_RHO; ++k)
        P.y0[k] = require_param(p, "y0_" + std::to_string(k));
    for (int k = N_PHYS + N_ACC + N_RHO; k < N_EQ; ++k)
        P.y0[k] = optional_param(p, "y0_" + std::to_string(k), 0.0);

    // Solver settings
    P.t_begin  = require_param(p, "t_begin");
    P.t_end    = require_param(p, "t_end");
    P.n_points = static_cast<int>(require_param(p, "n_points"));
    P.log_time = (optional_param(p, "log_time", 1.0) > 0.5);
    P.rtol     = optional_param(p, "rtol",  1e-6);
    P.atol     = optional_param(p, "atol",  1e-20);

    // Integration method options
    P.backend   = static_cast<int>(optional_param(p, "backend",   0.0));  // 0=CVODE
    P.lmm       = static_cast<int>(optional_param(p, "lmm",       2.0));  // 2=BDF
    P.linsol    = static_cast<int>(optional_param(p, "linsol",    0.0));  // 0=dense
    P.mu        = static_cast<int>(optional_param(p, "mu",  static_cast<double>(N_EQ - 1)));
    P.ml        = static_cast<int>(optional_param(p, "ml",  static_cast<double>(N_EQ - 1)));
    P.max_order = static_cast<int>(optional_param(p, "max_order", 0.0));  // 0=default
    P.ark_table = static_cast<int>(optional_param(p, "ark_table", 111.0)); // ARK548L2SA 5th-order

    return P;
}
