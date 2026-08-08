/**
 * jac_check.cpp — independent verification of the forward-AD Jacobian.
 *
 * Compares the exact Jacobian produced by one Dual<N> sweep through
 * zrcore::rhs_core against a central-difference Jacobian of the SAME core
 * evaluated in double. Because the two share the residual definition, any
 * disagreement beyond the finite-difference truncation floor is a bug in the
 * AD arithmetic (dual.h), not a modelling difference.
 *
 * Column j is differenced with a RELATIVE step h_j = eps_rel * |y_j|, because
 * the state spans ~34 decades (concentrations ~1e-20 up to rho_N ~1e14) and no
 * single absolute step can serve them all.
 *
 * Components sitting exactly on the concentration floor are reported but not
 * counted as failures: the floor makes the residual non-differentiable there,
 * so AD returns the one-sided derivative while a central difference straddles
 * the kink. That is a property of the model, not of the differentiation.
 *
 * Usage (same --key=value parameter set as solver.exe):
 *   jac_check.exe --omega_i=... --y0_0=... ... [--eps_rel=1e-6] [--verbose=1]
 *
 * Exit code 0 if every differentiable entry agrees to --tol (default 1e-6).
 */
#include "parameters.h"
#include "rate_equations_core.h"
#include "dual.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

static std::map<std::string, double> parse_args(int argc, char* argv[]) {
    std::map<std::string, double> p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0) a = a.substr(2);
        auto pos = a.find('=');
        if (pos == std::string::npos) continue;
        try { p[a.substr(0, pos)] = std::stod(a.substr(pos + 1)); } catch (...) {}
    }
    return p;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    const double eps_rel = args.count("eps_rel") ? args["eps_rel"] : 1e-6;
    const double tol     = args.count("tol")     ? args["tol"]     : 1e-6;
    const bool   verbose = args.count("verbose") && args["verbose"] > 0.5;

    Parameters P = build_parameters(args);

    double y[N_EQ];
    for (int k = 0; k < N_EQ; ++k) y[k] = P.y0[k];

    // ── AD Jacobian: one Dual<19> sweep ──────────────────────────────────────
    using D = Dual<N_EQ>;
    D yd[N_EQ], fd_[N_EQ];
    for (int k = 0; k < N_EQ; ++k) { yd[k] = D(y[k]); yd[k].seed(k); }
    zrcore::rhs_core<D>(yd, fd_, P);

    double Jad[N_EQ][N_EQ];
    for (int i = 0; i < N_EQ; ++i)
        for (int j = 0; j < N_EQ; ++j)
            Jad[i][j] = fd_[i].d[j];

    // ── Central-difference Jacobian of the same core ─────────────────────────
    double f0[N_EQ];
    zrcore::rhs_core<double>(y, f0, P);

    double Jfd[N_EQ][N_EQ];
    double fscale[N_EQ][N_EQ];   // magnitude of f_i near the perturbation
    double hcol[N_EQ];
    bool   at_floor[N_EQ];

    for (int j = 0; j < N_EQ; ++j) {
        at_floor[j] = (j < N_PHYS) && (std::fabs(y[j]) <= P.C_floor);

        double h = eps_rel * std::fabs(y[j]);
        if (h == 0.0) h = eps_rel;            // y_j == 0 -> absolute step
        hcol[j] = h;

        double yp[N_EQ], ym[N_EQ], fp[N_EQ], fm[N_EQ];
        for (int k = 0; k < N_EQ; ++k) { yp[k] = y[k]; ym[k] = y[k]; }
        yp[j] += h;  ym[j] -= h;
        zrcore::rhs_core<double>(yp, fp, P);
        zrcore::rhs_core<double>(ym, fm, P);
        for (int i = 0; i < N_EQ; ++i) {
            Jfd[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            fscale[i][j] = std::fmax(std::fabs(f0[i]),
                           std::fmax(std::fabs(fp[i]), std::fabs(fm[i])));
        }
    }

    // ── Compare ──────────────────────────────────────────────────────────────
    // Two classes of entry are excluded, for reasons that are properties of the
    // FINITE DIFFERENCE, not of the AD:
    //
    //  (a) FD-unresolvable. A central difference can only see an entry whose
    //      contribution J_ij * h_j rises above the floating-point granularity
    //      of f_i, i.e. above ~eps_mach * |f_i|. Many entries here sit 4-5
    //      decades below that: dCi/dt is pinned near G_i ~ 1e-7 while its Cv
    //      derivative moves it by ~1e-28. FD returns exactly 0 for those at ANY
    //      step size; AD returns the exact value. Requiring K_RES ULP of signal
    //      is the condition for the FD to carry ~1% of its own accuracy.
    //      (This is also precisely what SUNDIALS' difference-quotient Jacobian
    //      cannot see — the AD Jacobian is strictly more informative here.)
    //
    //  (b) On the concentration floor. fl() makes the residual non-smooth, so
    //      AD gives the one-sided derivative while the central difference
    //      straddles the kink. A model property, not a differentiation error.
    const double EPS_MACH = 2.220446049250313e-16;
    const double K_RES = 100.0;

    int nbad = 0, nfloor = 0, nunres = 0, ncmp = 0;
    double worst = 0.0;
    int    wi = -1, wj = -1;

    for (int i = 0; i < N_EQ; ++i) {
        double rowscale = 0.0;
        for (int j = 0; j < N_EQ; ++j) {
            rowscale = std::fmax(rowscale, std::fabs(Jad[i][j]));
            rowscale = std::fmax(rowscale, std::fabs(Jfd[i][j]));
        }
        if (rowscale == 0.0) continue;
        for (int j = 0; j < N_EQ; ++j) {
            if (at_floor[j]) { ++nfloor; continue; }

            const double signal = std::fabs(Jad[i][j]) * hcol[j];
            const double noise  = K_RES * EPS_MACH * fscale[i][j];
            if (signal < noise) { ++nunres; continue; }

            ++ncmp;
            double e = std::fabs(Jad[i][j] - Jfd[i][j]) / rowscale;
            if (e > worst) { worst = e; wi = i; wj = j; }
            if (e > tol) {
                ++nbad;
                if (verbose)
                    std::cout << "  MISMATCH J[" << i << "][" << j << "] "
                              << " AD=" << std::scientific << std::setprecision(8)
                              << Jad[i][j] << "  FD=" << Jfd[i][j]
                              << "  relrow=" << e
                              << "  signal/noise=" << signal / noise << "\n";
            }
        }
    }

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "JACCHECK eps_rel=" << eps_rel << " tol=" << tol
              << " worst_row_rel=" << worst
              << " at[" << wi << "][" << wj << "]"
              << " ncompared=" << ncmp
              << " nbad=" << nbad
              << " nskipped_floor=" << nfloor
              << " nskipped_unresolvable=" << nunres << "\n";
    return nbad == 0 ? 0 : 1;
}
