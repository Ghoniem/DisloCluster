/**
 * solver.cpp – ZrMicro main C++ ODE solver.
 *
 * Mirrors the role of py_utils/simulation.py (run_simulation / _ode_wrapper)
 * but uses SUNDIALS CVODE instead of scipy solve_ivp (LSODA).
 *
 * Two invocation modes
 * ────────────────────
 * 1. SINGLE CASE (default, unchanged interface) — one integration, args on CLI:
 *      solver.exe --omega_i=<val> ... --y0_0=<val> ... \
 *                 --t_begin=<val> --t_end=<val> --n_points=<val> \
 *                 --log_time=1 --rtol=1e-6 --atol=1e-20 \
 *                 --backend=0 --lmm=2 --linsol=0 --max_order=0 --ark_table=111
 *    Output: n_points rows × (1 + N_EQ) columns on stdout (see below).
 *
 * 2. BATCH (parameter identification) — many independent integrations solved
 *    concurrently with OpenMP, one subprocess instead of one-per-case:
 *      solver.exe --batch_file=<path>
 *    The file holds ONE case per line; each line is the same whitespace-
 *    separated key=value list a single-case invocation would receive (the
 *    leading "--" is optional). Each case is integrated on its own OpenMP
 *    thread with its own SUNContext / vectors / integrator memory (SUNDIALS
 *    objects are NOT shared across threads). Results are buffered per case and
 *    written back in input order, each block prefixed by:
 *        === CASE <i> status=<s> ===
 *    where status=0 means success; any non-zero value means that case failed
 *    and its rows (if any) should be discarded by the caller.
 *
 * Backend (--backend):
 *   0 = CVODE     — linear multistep BDF/Adams (default)
 *   1 = ARKODE    — implicit Runge-Kutta DIRK methods (ARKStep)
 *
 * Integration method — CVODE only (--lmm):
 *   2 = CV_BDF   — Backward Differentiation Formula, order 1-5 (stiff, default)
 *   1 = CV_ADAMS — Adams-Moulton,                   order 1-12 (non-stiff)
 *
 * DIRK table — ARKODE only (--ark_table, integer ARKODE_DIRKTableID):
 *   100 = SDIRK_2_1_2             order 2 (L-stable SDIRK)
 *   107 = SDIRK_5_3_4             order 4 (L-stable SDIRK)
 *   110 = KVAERNO_7_4_5           order 5 (A-stable)
 *   111 = ARK548L2SA_DIRK_8_4_5   order 5 (L-stable, default — Radau-class)
 *   121 = ESDIRK547L2SA_7_4_5     order 5 (ESDIRK, stiffly accurate)
 *
 * Linear solver — both backends (--linsol):
 *   0 = dense  — direct LU factorisation (default; best for N=12)
 *   1 = band   — banded direct solver; use --mu and --ml for bandwidth
 *   2 = gmres  — SPGMR iterative Krylov solver (matrix-free)
 *
 * Max solver order (--max_order):
 *   0 = solver default (BDF→5, Adams→12, ARKode→table order)
 *   positive integer → override
 *
 * Output: n_points rows × (1 + N_EQ) columns (space-separated, scientific notation):
 *   t  Cv  Ci  C2i  C3i  CiL  CaiL  CvL  CavL  CiL_i  CaiL_i  CvL_v  CavL_v
 *      cum_prod_i  cum_recomb_i  cum_sink_i  cum_prod_v  cum_recomb_v  cum_sink_v
 *   The trailing six columns are point-defect conservation accumulators.
 *
 * Build:
 *   cd ZrMicro/cpp_utils
 *   cmake -S . -B ../build -DCMAKE_BUILD_TYPE=Release
 *   cmake --build ../build --config Release
 */

#include "parameters.h"
#include "rate_equations.h"

#include <cvodes/cvodes.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_butcher_dirk.h>
#include <arkode/arkode_butcher_erk.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_band.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sundials/sundials_types.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ── CLI argument parser (single-case mode) ────────────────────────────────────

std::map<std::string, double> parse_args(int argc, char* argv[]) {
    std::map<std::string, double> props;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.size() < 3 || arg[0] != '-' || arg[1] != '-') {
            std::cerr << "Unexpected argument: " << arg << "\n";
            return {};
        }
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            std::cerr << "Invalid argument format (missing '='): " << arg << "\n";
            return {};
        }
        std::string key = arg.substr(2, pos - 2);
        double val = 0.0;
        try {
            val = std::stod(arg.substr(pos + 1));
        } catch (...) {
            std::cerr << "Invalid numeric value for " << key << "\n";
            return {};
        }
        props[key] = val;
    }
    return props;
}

// Parse ONE whitespace-separated "key=value [key=value ...]" line (batch mode).
// A leading "--" on each token is accepted but optional. Malformed tokens are
// skipped silently; build_parameters() reports any genuinely missing keys.
std::map<std::string, double> parse_kv_line(const std::string& line) {
    std::map<std::string, double> props;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
        if (tok.size() >= 2 && tok[0] == '-' && tok[1] == '-')
            tok = tok.substr(2);
        auto pos = tok.find('=');
        if (pos == std::string::npos)
            continue;
        std::string key = tok.substr(0, pos);
        try {
            props[key] = std::stod(tok.substr(pos + 1));
        } catch (...) {
            // ignore non-numeric token
        }
    }
    return props;
}

// ── One integration ───────────────────────────────────────────────────────────
//
// Integrate a single parameter set and append the result rows to `out`.
// Self-contained: creates and frees its OWN SUNContext, state vector, linear
// solver and integrator memory, so it is safe to call concurrently from
// independent OpenMP threads (each call touches only its own objects + the
// thread-local `out` stream and the read-only `P`).
//
// Returns 0 on success; a non-zero status code on any setup/integration error.

static int integrate_one(const Parameters& P, std::ostream& out) {

    // ── Time evaluation grid ──────────────────────────────────────────────────
    std::vector<double> t_eval(P.n_points);
    if (P.log_time) {
        double log_t0 = std::log10(P.t_begin);
        double log_tf = std::log10(P.t_end);
        double step   = (log_tf - log_t0) / (P.n_points - 1);
        for (int i = 0; i < P.n_points; ++i)
            t_eval[i] = std::pow(10.0, log_t0 + i * step);
    } else {
        double step = (P.t_end - P.t_begin) / (P.n_points - 1);
        for (int i = 0; i < P.n_points; ++i)
            t_eval[i] = P.t_begin + i * step;
    }

    // ── SUNDIALS context (one per call → thread-safe) ─────────────────────────
    SUNContext sunctx;
    if (SUNContext_Create(SUN_COMM_NULL, &sunctx) != 0)
        return 101;

    // ── State vector — initialised from Python-computed y0 ────────────────────
    N_Vector y = N_VNew_Serial(N_EQ, sunctx);
    if (!y) { SUNContext_Free(&sunctx); return 102; }
    for (int k = 0; k < N_EQ; ++k)
        NV_Ith_S(y, k) = P.y0[k];

    SUNMatrix       A  = nullptr;
    SUNLinearSolver LS = nullptr;

    auto make_linear_solver = [&](void* solver_mem, bool is_arkode) -> bool {
        if (P.linsol == 1) {
            A  = SUNBandMatrix(N_EQ, P.mu, P.ml, sunctx);
            if (!A)  return false;
            LS = SUNLinSol_Band(y, A, sunctx);
            if (!LS) return false;
        } else if (P.linsol == 2) {
            LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
            if (!LS) return false;
            A = nullptr;
        } else {
            A  = SUNDenseMatrix(N_EQ, N_EQ, sunctx);
            if (!A)  return false;
            LS = SUNLinSol_Dense(y, A, sunctx);
            if (!LS) return false;
        }
        int r = is_arkode ? ARKodeSetLinearSolver(solver_mem, LS, A)
                          : CVodeSetLinearSolver(solver_mem, LS, A);
        return r == 0;
    };

    auto cleanup = [&]() {
        if (LS) SUNLinSolFree(LS);
        if (A)  SUNMatDestroy(A);
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
    };

    out << std::scientific << std::setprecision(10);
    int flag, status = 0;
    void* user_data = const_cast<Parameters*>(&P);

    if (P.backend == 1) {
        // ── ARKODE ARKStep — implicit Runge-Kutta (DIRK) ──────────────────────
        void* ark_mem = ARKStepCreate(nullptr, rhs_zrmicro, t_eval[0], y, sunctx);
        if (!ark_mem) { cleanup(); return 110; }

        if (ARKStepSetImplicit(ark_mem) != ARK_SUCCESS ||
            ARKStepSetTableNum(ark_mem,
                               static_cast<ARKODE_DIRKTableID>(P.ark_table),
                               ARKODE_ERK_NONE) != ARK_SUCCESS ||
            ARKodeSetUserData(ark_mem, user_data) != ARK_SUCCESS ||
            ARKodeSStolerances(ark_mem, P.rtol, P.atol) != ARK_SUCCESS ||
            ARKodeSetMaxNumSteps(ark_mem, 500000) != ARK_SUCCESS) {
            ARKodeFree(&ark_mem); cleanup(); return 111;
        }
        if (P.max_order > 0 &&
            ARKodeSetOrder(ark_mem, P.max_order) != ARK_SUCCESS) {
            ARKodeFree(&ark_mem); cleanup(); return 111;
        }
        if (!make_linear_solver(ark_mem, true)) {
            ARKodeFree(&ark_mem); cleanup(); return 112;
        }

        out << t_eval[0];
        for (int k = 0; k < N_EQ; ++k) out << ' ' << NV_Ith_S(y, k);
        out << '\n';

        sunrealtype t_current = t_eval[0];
        for (int i = 1; i < P.n_points; ++i) {
            flag = ARKodeEvolve(ark_mem, t_eval[i], y, &t_current, ARK_NORMAL);
            if (flag < 0) { status = 120; break; }
            out << t_eval[i];
            for (int k = 0; k < N_EQ; ++k) out << ' ' << NV_Ith_S(y, k);
            out << '\n';
        }
        ARKodeFree(&ark_mem);

    } else {
        // ── CVODE — linear multistep BDF or Adams ─────────────────────────────
        int lmm_flag = (P.lmm == 1) ? CV_ADAMS : CV_BDF;

        void* cvode_mem = CVodeCreate(lmm_flag, sunctx);
        if (!cvode_mem) { cleanup(); return 130; }

        if (CVodeSetUserData(cvode_mem, user_data) != CV_SUCCESS ||
            CVodeInit(cvode_mem, rhs_zrmicro, t_eval[0], y) != CV_SUCCESS ||
            CVodeSStolerances(cvode_mem, P.rtol, P.atol) != CV_SUCCESS ||
            CVodeSetMaxNumSteps(cvode_mem, 500000) != CV_SUCCESS) {
            CVodeFree(&cvode_mem); cleanup(); return 131;
        }
        if (P.max_order > 0 &&
            CVodeSetMaxOrd(cvode_mem, P.max_order) != CV_SUCCESS) {
            CVodeFree(&cvode_mem); cleanup(); return 131;
        }
        if (!make_linear_solver(cvode_mem, false)) {
            CVodeFree(&cvode_mem); cleanup(); return 132;
        }

        out << t_eval[0];
        for (int k = 0; k < N_EQ; ++k) out << ' ' << NV_Ith_S(y, k);
        out << '\n';

        sunrealtype t_current = t_eval[0];
        for (int i = 1; i < P.n_points; ++i) {
            flag = CVode(cvode_mem, t_eval[i], y, &t_current, CV_NORMAL);
            if (flag < 0) { status = 140; break; }
            out << t_eval[i];
            for (int k = 0; k < N_EQ; ++k) out << ' ' << NV_Ith_S(y, k);
            out << '\n';
        }
        CVodeFree(&cvode_mem);
    }

    cleanup();
    return status;
}

// ── Batch mode ─────────────────────────────────────────────────────────────────

static int run_batch(const std::string& batch_file) {
    std::ifstream f(batch_file);
    if (!f) {
        std::cerr << "Cannot open batch file: " << batch_file << "\n";
        return 1;
    }

    std::vector<Parameters> cases;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find('=') == std::string::npos)
            continue;   // skip blank / comment lines
        cases.push_back(build_parameters(parse_kv_line(line)));
    }
    const int ncases = static_cast<int>(cases.size());
    if (ncases == 0) {
        std::cerr << "Batch file contained no cases: " << batch_file << "\n";
        return 1;
    }

    std::vector<std::string> outbuf(ncases);
    std::vector<int>         status(ncases, 0);

    // Force any one-time SUNDIALS global initialisation to happen on a single
    // thread BEFORE the parallel region, so the first concurrent
    // SUNContext_Create cannot race on first-touch global state.
    {
        SUNContext warm;
        if (SUNContext_Create(SUN_COMM_NULL, &warm) == 0)
            SUNContext_Free(&warm);
    }

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads > ncases) nthreads = ncases;
    if (nthreads < 1)      nthreads = 1;
    omp_set_num_threads(nthreads);
    std::cerr << "Batch: " << ncases << " cases on " << nthreads
              << " OpenMP thread(s)\n";
#else
    std::cerr << "Batch: " << ncases << " cases (serial; built without OpenMP)\n";
#endif

    // Cases differ in stiffness / number of internal steps, so use dynamic
    // scheduling to keep all threads busy until the work is drained.
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < ncases; ++c) {
        std::ostringstream oss;
        int s = integrate_one(cases[c], oss);
        status[c] = s;
        outbuf[c] = oss.str();
    }

    // Ordered, race-free output: one thread wrote each buffer; main emits them.
    for (int c = 0; c < ncases; ++c) {
        std::cout << "=== CASE " << c << " status=" << status[c] << " ===\n";
        std::cout << outbuf[c];
    }
    return 0;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {

    // Batch mode is selected by a --batch_file=<path> argument. Detect it before
    // the numeric CLI parser (a path is not parseable as a double).
    const std::string batch_key = "--batch_file=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(batch_key, 0) == 0)
            return run_batch(a.substr(batch_key.size()));
    }

    // ── Single-case mode (unchanged interface) ────────────────────────────────
    auto args = parse_args(argc, argv);
    if (args.empty() && argc > 1) return 1;

    Parameters P = build_parameters(args);

    std::ostringstream oss;
    int status = integrate_one(P, oss);
    std::cout << oss.str();
    if (status != 0)
        std::cerr << "Integration failed (status=" << status << ")\n";
    return status == 0 ? 0 : 1;
}
