/**
 * rate_equations.h – SUNDIALS callbacks for the ZrMicro system.
 *
 * The physics itself lives in rate_equations_core.h as a single scalar-type
 * templated function; everything here is a thin adapter between SUNDIALS' data
 * structures and that core.
 *
 * Full state vector y[19] (see parameters.h for the reduced layouts):
 *   [0]  Cv     – vacancy concentration
 *   [1]  Ci     – interstitial concentration
 *   [2]  C2i    – di-interstitial concentration
 *   [3]  C3i    – tri-interstitial concentration
 *   [4]  CiL    – interstitial loop density
 *   [5]  CaiL   – aligned interstitial loop density
 *   [6]  CvL    – vacancy loop density
 *   [7]  CavL   – aligned vacancy loop density
 *   [8]  CiL_i  – interstitials in interstitial loops
 *   [9]  CaiL_i – interstitials in aligned interstitial loops
 *   [10] CvL_v  – vacancies in vacancy loops
 *   [11] CavL_v – vacancies in aligned vacancy loops
 *   [12..17]    – point-defect conservation accumulators (pure quadratures)
 *   [18] rho_N  – evolving network dislocation density
 *
 * user_data always points to a Parameters struct.
 */
#pragma once

#include "parameters.h"
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sundials/sundials_types.h>

/**
 * Per-integration context passed to every SUNDIALS callback as user_data.
 *
 * It carries the parameters plus a one-entry memo of the accumulator rates.
 * The reduced RHS and the quadrature RHS need the SAME core evaluation: CVODES
 * calls the quadrature function at the accepted step point, which is normally
 * the last state the corrector evaluated the residual at. Memoising on the
 * exact (t, y) bit pattern turns that into a hit and avoids a second full core
 * sweep per step — without the memo the quadrature roughly doubled the RHS
 * cost of the reduced path and made it slower than the full system.
 *
 * Correctness never depends on the memo: a miss simply recomputes. One context
 * per integrate_one() call, so it is thread-safe under the OpenMP batch loop.
 */
struct SolverCtx {
    const Parameters* P;
    bool   acc_valid;          // memo holds a usable entry
    double acc_t;              // t it was taken at
    double acc_y[N_RED_MAX];   // reduced state it was taken at
    double acc[N_ACC];         // the six accumulator rates there
    long   nq_hit;             // quadrature memo hits
    long   nq_miss;            // quadrature memo misses (full core re-evaluated)
};

// ── Full 19-equation path (legacy; bit-identical to the original solver) ─────

/** CVODE/ARKODE RHS over the full 19-component state. */
int rhs_zrmicro(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);

/** Exact 19x19 dense Jacobian by forward-mode AD. */
int jac_zrmicro(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

// ── Reduced path (accumulators carried as CVODES quadrature variables) ───────

/** RHS over the reduced implicit block (dimension red_dim(P) = 9 or 13). */
int rhs_zrmicro_reduced(sunrealtype t, N_Vector y, N_Vector ydot,
                        void* user_data);

/** Quadrature RHS: the six accumulator rates, given the reduced state. */
int quad_zrmicro_reduced(sunrealtype t, N_Vector y, N_Vector yQdot,
                         void* user_data);

/** Exact dense Jacobian of the reduced block by forward-mode AD. */
int jac_zrmicro_reduced(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                        void* user_data, N_Vector tmp1, N_Vector tmp2,
                        N_Vector tmp3);
