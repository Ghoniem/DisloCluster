/**
 * rate_equations.cpp – SUNDIALS adapters around the templated physics core.
 *
 * The right-hand side used to be written out here; it now lives in
 * rate_equations_core.h templated on the scalar type, so that the residual
 * (T = double) and the exact Jacobian (T = Dual<N>, forward-mode AD) are two
 * instantiations of ONE definition and cannot drift apart. This file only
 * marshals between SUNDIALS' N_Vector/SUNMatrix and plain arrays.
 *
 * user_data is a SolverCtx* (see rate_equations.h), which carries the
 * Parameters plus the accumulator-rate memo shared by the reduced RHS and the
 * quadrature callback.
 */
#include "rate_equations.h"
#include "rate_equations_core.h"
#include "dual.h"

// ── Full 19-equation path ────────────────────────────────────────────────────

int rhs_zrmicro(sunrealtype /*t*/, N_Vector y, N_Vector ydot, void* user_data) {
    const Parameters& P = *static_cast<SolverCtx*>(user_data)->P;

    double yf[N_EQ], df[N_EQ];
    for (int k = 0; k < N_EQ; ++k) yf[k] = NV_Ith_S(y, k);

    zrcore::rhs_core<double>(yf, df, P);

    for (int k = 0; k < N_EQ; ++k) NV_Ith_S(ydot, k) = df[k];
    return 0;
}

int jac_zrmicro(sunrealtype /*t*/, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                void* user_data, N_Vector, N_Vector, N_Vector) {
    const Parameters& P = *static_cast<SolverCtx*>(user_data)->P;
    using D = Dual<N_EQ>;

    D yf[N_EQ], df[N_EQ];
    for (int k = 0; k < N_EQ; ++k) {
        yf[k] = D(NV_Ith_S(y, k));
        yf[k].seed(k);
    }

    zrcore::rhs_core<D>(yf, df, P);

    SUNMatZero(J);
    for (int i = 0; i < N_EQ; ++i)
        for (int j = 0; j < N_EQ; ++j)
            SM_ELEMENT_D(J, i, j) = df[i].d[j];
    return 0;
}

// ── Reduced path ─────────────────────────────────────────────────────────────

int rhs_zrmicro_reduced(sunrealtype t, N_Vector y, N_Vector ydot,
                        void* user_data) {
    SolverCtx& C = *static_cast<SolverCtx*>(user_data);
    const Parameters& P = *C.P;
    const int n = red_dim(P);

    double yr[N_RED_FREE], yf[N_EQ], df[N_EQ];
    for (int j = 0; j < n; ++j) yr[j] = NV_Ith_S(y, j);
    red_scatter(P, yr, yf);

    zrcore::rhs_core<double>(yf, df, P);

    for (int j = 0; j < n; ++j) NV_Ith_S(ydot, j) = df[red_idx(P, j)];

    // Memoise the accumulator rates: the quadrature callback is normally asked
    // for exactly this (t, y) at the end of the step.
    C.acc_valid = true;
    C.acc_t = static_cast<double>(t);
    for (int j = 0; j < n; ++j)       C.acc_y[j] = yr[j];
    for (int k = 0; k < N_ACC; ++k)   C.acc[k]   = df[N_PHYS + k];
    return 0;
}

int quad_zrmicro_reduced(sunrealtype t, N_Vector y, N_Vector yQdot,
                         void* user_data) {
    SolverCtx& C = *static_cast<SolverCtx*>(user_data);
    const Parameters& P = *C.P;
    const int n = red_dim(P);

    // Memo hit: the residual was already evaluated at this exact point, so the
    // accumulator rates it produced are the ones wanted here — bit for bit.
    if (C.acc_valid && C.acc_t == static_cast<double>(t)) {
        bool same = true;
        for (int j = 0; j < n && same; ++j)
            same = (C.acc_y[j] == NV_Ith_S(y, j));
        if (same) {
            ++C.nq_hit;
            for (int k = 0; k < N_ACC; ++k) NV_Ith_S(yQdot, k) = C.acc[k];
            return 0;
        }
    }

    ++C.nq_miss;
    double yr[N_RED_FREE], yf[N_EQ], df[N_EQ];
    for (int j = 0; j < n; ++j) yr[j] = NV_Ith_S(y, j);
    red_scatter(P, yr, yf);

    zrcore::rhs_core<double>(yf, df, P);

    for (int k = 0; k < N_ACC; ++k) NV_Ith_S(yQdot, k) = df[N_PHYS + k];
    return 0;
}

// Templated on the compile-time reduced dimension so the AD derivative loops
// are fully unrolled and no lanes are wasted: 9 seeds when the mobile species
// are frozen, 13 when they are not.
template <int NR>
static void jac_reduced_impl(const Parameters& P, N_Vector y, SUNMatrix J) {
    using D = Dual<NR>;

    D yf[N_EQ], df[N_EQ];
    for (int k = 0; k < N_EQ; ++k) yf[k] = D(0.0);
    if (P.freeze_mobile)
        for (int k = 0; k < N_MOB; ++k) yf[k] = D(P.y0[k]);

    for (int j = 0; j < NR; ++j) {
        const int k = red_idx(P, j);
        yf[k] = D(NV_Ith_S(y, j));
        yf[k].seed(j);
    }

    zrcore::rhs_core<D>(yf, df, P);

    SUNMatZero(J);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NR; ++j)
            SM_ELEMENT_D(J, i, j) = df[red_idx(P, i)].d[j];
}

int jac_zrmicro_reduced(sunrealtype /*t*/, N_Vector y, N_Vector /*fy*/,
                        SUNMatrix J, void* user_data,
                        N_Vector, N_Vector, N_Vector) {
    const Parameters& P = *static_cast<SolverCtx*>(user_data)->P;
    if (P.freeze_mobile) jac_reduced_impl<N_RED_FROZEN>(P, y, J);
    else                 jac_reduced_impl<N_RED_FREE>  (P, y, J);
    return 0;
}
