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
static constexpr int N_EQ   = N_PHYS + N_ACC + N_RHO;   // total = 19
static constexpr int IDX_RHO_N = N_PHYS + N_ACC;        // rho_N state index = 18
static constexpr int N_MOB   = 4;   // mobile species Cv Ci C2i C3i
static constexpr int N_IMMOB = 8;   // immobile species (loop numbers + contents)

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
static constexpr int N_RED_FROZEN = N_EQ - N_ACC - N_MOB;   //  9
static constexpr int N_RED_FREE   = N_EQ - N_ACC;           // 13

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
static constexpr int N_RLX_FROZEN = N_RED_FROZEN + N_ACC;   // 15
static constexpr int N_RLX_FREE   = N_RED_FREE   + N_ACC;   // 19

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

// Dimension of the implicit block actually solved.
inline int red_dim(const Parameters& P) {
    if (P.acc_mode == ACC_STATE_RELAX)
        return P.freeze_mobile ? N_RLX_FROZEN : N_RLX_FREE;
    return P.freeze_mobile ? N_RED_FROZEN : N_RED_FREE;
}

// Reduced index j -> index into the full 19-component state.
//   frozen : j = 0..7  -> 4..11 (immobile),  j = 8  -> 18 (rho_N)
//   free   : j = 0..11 -> 0..11 (mobile + immobile), j = 12 -> 18
inline int red_idx(const Parameters& P, int j) {
    const int n_core = P.freeze_mobile ? N_RED_FROZEN : N_RED_FREE;
    if (j >= n_core)            // mode 2 only: the six accumulators, appended
        return N_PHYS + (j - n_core);
    if (P.freeze_mobile)
        return (j < N_IMMOB) ? (N_MOB + j) : IDX_RHO_N;
    return (j < N_PHYS) ? j : IDX_RHO_N;
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

    // Initial conditions
    for (int k = 0; k < N_EQ; ++k)
        P.y0[k] = require_param(p, "y0_" + std::to_string(k));

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
