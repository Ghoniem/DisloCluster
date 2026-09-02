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
#include "rate_equations_core.h"

// How many state components a row of output carries.
//
// THE SOLVER ALWAYS INTEGRATES N_EQ; WHAT IT PRINTS IS A DIFFERENT QUESTION.
// The grain-boundary ledger added two accumulators at the end, and emitting
// them unconditionally silently widened every row from 39 to 41 columns --
// which every caller that assumed 38 state components then mis-read. They are
// pure diagnostics and identically zero unless the channel is on, so a run
// without it prints exactly what it always did and every existing consumer,
// artifact and regression keeps parsing.
static inline int n_out_state(const Parameters& P) {
    return P.gb_absorption ? N_EQ : N_EQ - N_GBACC;
}


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

#include <chrono>
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

// ── Reusable per-thread solver workspace ──────────────────────────────────────
//
// Building a SUNContext, state vector, dense matrix, linear solver and CVODE
// memory, then tearing them all down again, costs FAR more than a short
// integration: measured on this machine a 256-case frozen-mobile batch spends
// ~1.3 s of thread time on construction/destruction versus ~0.2 s on the actual
// solves. The coupling march runs exactly that shape of workload — one short
// integration per quadrature point per substep — so the per-case setup was the
// dominant cost of the whole 0-D side.
//
// A Workspace is created once per OpenMP thread and reused across every case
// that thread handles, via CVodeReInit / CVodeQuadReInit. It is rebuilt only if
// a case needs a different shape (dimension, linear solver, LMM, Jacobian mode).
// Nothing is shared between threads, so the batch stays thread-safe.
struct Workspace {
    SUNContext      sunctx    = nullptr;
    void*           cvode_mem = nullptr;
    N_Vector        y         = nullptr;
    N_Vector        yQ        = nullptr;
    SUNMatrix       A         = nullptr;
    SUNLinearSolver LS        = nullptr;
    SolverCtx       ctx{};                 // stable address -> set user_data once

    // Signature of what is currently built.
    N_Vector        atolv     = nullptr;   // mode 2: per-component atol
    int  neq = -1, linsol = -1, lmm = -1, max_order = -1, acc_mode = -1;
    bool reduced = false, ajac = false, built = false;

    long n_build = 0, n_reuse = 0;
};

static void ws_teardown_solver(Workspace& ws) {
    if (ws.cvode_mem) { CVodeFree(&ws.cvode_mem); ws.cvode_mem = nullptr; }
    if (ws.LS) { SUNLinSolFree(ws.LS); ws.LS = nullptr; }
    if (ws.A)  { SUNMatDestroy(ws.A);  ws.A  = nullptr; }
    if (ws.atolv) { N_VDestroy(ws.atolv); ws.atolv = nullptr; }
    if (ws.yQ) { N_VDestroy(ws.yQ);    ws.yQ = nullptr; }
    if (ws.y)  { N_VDestroy(ws.y);     ws.y  = nullptr; }
    ws.built = false;
}

static void ws_release(Workspace& ws) {
    ws_teardown_solver(ws);
    if (ws.sunctx) { SUNContext_Free(&ws.sunctx); ws.sunctx = nullptr; }
}

// Ensure the workspace matches the requested shape. Returns 0 on success.
static int ws_ensure(Workspace& ws, const Parameters& P,
                     int neq, bool reduced, bool ajac) {
    if (ws.built && ws.neq == neq && ws.reduced == reduced && ws.ajac == ajac &&
        ws.acc_mode == P.acc_mode &&
        ws.linsol == P.linsol && ws.lmm == P.lmm && ws.max_order == P.max_order) {
        ++ws.n_reuse;
        return 0;
    }
    ws_teardown_solver(ws);

    // The context itself is shape-independent and survives a rebuild.
    if (!ws.sunctx && SUNContext_Create(SUN_COMM_NULL, &ws.sunctx) != 0)
        return 101;

    ws.y = N_VNew_Serial(neq, ws.sunctx);
    if (!ws.y) return 102;
    if (P.acc_mode == ACC_QUADRATURE) {
        ws.yQ = N_VNew_Serial(n_acc(P), ws.sunctx);
        if (!ws.yQ) return 103;
    }
    if (P.acc_mode == ACC_STATE_RELAX) {
        ws.atolv = N_VNew_Serial(neq, ws.sunctx);
        if (!ws.atolv) return 103;
    }

    if (P.linsol == 1) {
        int mu = P.mu < neq - 1 ? P.mu : neq - 1;
        int ml = P.ml < neq - 1 ? P.ml : neq - 1;
        ws.A  = SUNBandMatrix(neq, mu, ml, ws.sunctx);
        if (!ws.A)  return 104;
        ws.LS = SUNLinSol_Band(ws.y, ws.A, ws.sunctx);
    } else if (P.linsol == 2) {
        ws.LS = SUNLinSol_SPGMR(ws.y, SUN_PREC_NONE, 0, ws.sunctx);
    } else {
        ws.A  = SUNDenseMatrix(neq, neq, ws.sunctx);
        if (!ws.A)  return 104;
        ws.LS = SUNLinSol_Dense(ws.y, ws.A, ws.sunctx);
    }
    if (!ws.LS) return 105;

    const int lmm_flag = (P.lmm == 1) ? CV_ADAMS : CV_BDF;
    ws.cvode_mem = CVodeCreate(lmm_flag, ws.sunctx);
    if (!ws.cvode_mem) return 130;

    CVRhsFn rhs_fn = reduced ? rhs_zrmicro_reduced : rhs_zrmicro;
    // t0 here is a placeholder; every case sets its own through CVodeReInit.
    if (CVodeInit(ws.cvode_mem, rhs_fn, 0.0, ws.y) != CV_SUCCESS) return 131;
    if (CVodeSetUserData(ws.cvode_mem, &ws.ctx) != CV_SUCCESS) return 131;
    if (CVodeSetLinearSolver(ws.cvode_mem, ws.LS, ws.A) != CV_SUCCESS) return 132;
    if (ajac) {
        CVLsJacFn jf = reduced ? jac_zrmicro_reduced : jac_zrmicro;
        if (CVodeSetJacFn(ws.cvode_mem, jf) != CV_SUCCESS) return 132;
    }
    if (P.acc_mode == ACC_QUADRATURE &&
        CVodeQuadInit(ws.cvode_mem, quad_zrmicro_reduced, ws.yQ) != CV_SUCCESS)
        return 133;
    if (P.max_order > 0 &&
        CVodeSetMaxOrd(ws.cvode_mem, P.max_order) != CV_SUCCESS) return 131;

    ws.neq = neq; ws.reduced = reduced; ws.ajac = ajac;
    ws.acc_mode = P.acc_mode;
    ws.linsol = P.linsol; ws.lmm = P.lmm; ws.max_order = P.max_order;
    ws.built = true;
    ++ws.n_build;
    return 0;
}

// ── One integration ───────────────────────────────────────────────────────────
//
// Integrate a single parameter set and append the result rows to `out`, reusing
// the caller-supplied per-thread workspace. Each call touches only that
// workspace, the thread-local `out` stream and the read-only `P`, so it remains
// safe to call concurrently from independent OpenMP threads.
//
// Returns 0 on success; a non-zero status code on any setup/integration error.

static int integrate_one(const Parameters& P, std::ostream& out, Workspace& ws) {

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

    // ── Fixed-step explicit Euler ─────────────────────────────────────────────
    // The reference scheme: forward Euler on the SAME rate_equations_core.h
    // right-hand side, euler_nsub substeps per output interval, with the
    // per-substep positivity clamp MoDELib applies to its immobile field.
    // No Jacobian, no linear solve, no error control -- exactly what
    // ClusterDynamicsFEM::solveImmobileClusters does nodally.
    if (P.euler_nsub > 0) {
        out << std::scientific << std::setprecision(10);
        const auto tE0 = std::chrono::steady_clock::now();
        double y[N_EQ], dy[N_EQ];
        for (int k = 0; k < N_EQ; ++k) y[k] = P.y0[k];

        auto emit = [&](double t) {
            out << t;
            for (int k = 0; k < n_out_state(P); ++k) out << ' ' << y[k];
            out << '\n';
        };
        emit(t_eval[0]);

        long nfe = 0;
        bool blew_up = false;
        for (int i = 1; i < P.n_points && !blew_up; ++i) {
            const double h = (t_eval[i] - t_eval[i - 1]) / P.euler_nsub;
            for (int s = 0; s < P.euler_nsub; ++s) {
                zrcore::rhs_core<double>(y, dy, P);
                ++nfe;
                for (int k = 0; k < N_EQ; ++k) y[k] += h * dy[k];
                // MoDELib clamps n and c to their floors after every substep.
                for (int k = 0; k < N_PHYS; ++k)
                    if (y[k] < P.C_floor) y[k] = P.C_floor;
                for (int k = 0; k < N_EQ; ++k)
                    if (!std::isfinite(y[k])) { blew_up = true; break; }
                if (blew_up) break;
            }
            if (!blew_up) emit(t_eval[i]);
        }
        const double tE = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tE0).count();
        if (P.stats) {
            out << "# STATS neq=" << N_EQ << " nst=" << (nfe)
                << " nfe=" << nfe << " nje=0 nni=0 netf=0 nfeLS=0 nsetups=0"
                << " ncfn=" << (blew_up ? 1 : 0) << " nqe=0 nqhit=0 nqmiss=0"
                << " nbuild=0 nreuse=0 t_int=" << tE << '\n';
        }
        return blew_up ? 150 : 0;
    }

    // ── Reduced implicit block ────────────────────────────────────────────────
    // The six accumulators are pure quadratures (rows written, columns never
    // read), so CVODES can integrate them with CVodeQuadInit on the same step
    // sequence while keeping them out of the Newton system and the error test.
    // With the mobile species also frozen by the operator split, the block the
    // dense LU factorises drops from 19x19 to 9x9. ARKODE has no quadrature
    // module, so the reduction is CVODE-only; requesting it with --backend=1
    // silently falls back to the full system (the emitted stats line reports
    // the dimension actually used, so a benchmark cannot be misled).
    const bool reduced   = is_reduced(P) && P.backend == 0;
    const int  NEQ_SOLVE = reduced ? red_dim(P) : N_EQ;
    const bool use_quad  = reduced && P.acc_mode == ACC_QUADRATURE;

    // Analytic AD Jacobian: dense direct solver only. Band would need a
    // different element accessor and GMRES needs no Jacobian at all; both keep
    // SUNDIALS' difference quotients.
    const bool use_ajac = P.analytic_jac && P.linsol == 0;

    out << std::scientific << std::setprecision(10);
    int flag, status = 0;

    // Integration-only wall time, so a benchmark can separate solver cost from
    // the process-spawn and text-parsing overhead of the batch interface.
    const auto t_wall0 = std::chrono::steady_clock::now();

    // ── ARKODE: not reused (no quadrature module, not the coupling path) ──────
    if (P.backend == 1) {
        SUNContext sunctx;
        if (SUNContext_Create(SUN_COMM_NULL, &sunctx) != 0) return 101;

        N_Vector y = N_VNew_Serial(N_EQ, sunctx);
        if (!y) { SUNContext_Free(&sunctx); return 102; }
        for (int k = 0; k < N_EQ; ++k) NV_Ith_S(y, k) = P.y0[k];

        SUNMatrix       A  = nullptr;
        SUNLinearSolver LS = nullptr;
        if (P.linsol == 1) {
            A  = SUNBandMatrix(N_EQ, P.mu, P.ml, sunctx);
            LS = A ? SUNLinSol_Band(y, A, sunctx) : nullptr;
        } else if (P.linsol == 2) {
            LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
        } else {
            A  = SUNDenseMatrix(N_EQ, N_EQ, sunctx);
            LS = A ? SUNLinSol_Dense(y, A, sunctx) : nullptr;
        }
        auto ark_cleanup = [&]() {
            if (LS) SUNLinSolFree(LS);
            if (A)  SUNMatDestroy(A);
            N_VDestroy(y);
            SUNContext_Free(&sunctx);
        };
        if (!LS) { ark_cleanup(); return 105; }

        SolverCtx actx{};
        actx.P = &P; actx.acc_valid = false; actx.nq_hit = 0; actx.nq_miss = 0;

        void* ark_mem = ARKStepCreate(nullptr, rhs_zrmicro, t_eval[0], y, sunctx);
        if (!ark_mem) { ark_cleanup(); return 110; }

        if (ARKStepSetImplicit(ark_mem) != ARK_SUCCESS ||
            ARKStepSetTableNum(ark_mem,
                               static_cast<ARKODE_DIRKTableID>(P.ark_table),
                               ARKODE_ERK_NONE) != ARK_SUCCESS ||
            ARKodeSetUserData(ark_mem, &actx) != ARK_SUCCESS ||
            ARKodeSStolerances(ark_mem, P.rtol, P.atol) != ARK_SUCCESS ||
            ARKodeSetMaxNumSteps(ark_mem, 500000) != ARK_SUCCESS ||
            ARKodeSetLinearSolver(ark_mem, LS, A) != ARK_SUCCESS) {
            ARKodeFree(&ark_mem); ark_cleanup(); return 111;
        }
        if (P.max_order > 0 &&
            ARKodeSetOrder(ark_mem, P.max_order) != ARK_SUCCESS) {
            ARKodeFree(&ark_mem); ark_cleanup(); return 111;
        }
        if (use_ajac && ARKodeSetJacFn(ark_mem, jac_zrmicro) != ARK_SUCCESS) {
            ARKodeFree(&ark_mem); ark_cleanup(); return 112;
        }

        auto emit_ark = [&](double t) {
            out << t;
            for (int k = 0; k < n_out_state(P); ++k) out << ' ' << NV_Ith_S(y, k);
            out << '\n';
        };
        emit_ark(t_eval[0]);

        sunrealtype t_current = t_eval[0];
        for (int i = 1; i < P.n_points; ++i) {
            flag = ARKodeEvolve(ark_mem, t_eval[i], y, &t_current, ARK_NORMAL);
            if (flag < 0) { status = 120; break; }
            emit_ark(t_eval[i]);
        }
        const double t_int_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_wall0).count();
        if (P.stats) {
            // ARKODE is not the benchmarked path (the reduction is CVODE-only),
            // so only the portable step/nonlinear counters are reported here.
            long nst = 0, nni = 0, netf = 0;
            ARKodeGetNumSteps(ark_mem, &nst);
            ARKodeGetNumNonlinSolvIters(ark_mem, &nni);
            ARKodeGetNumErrTestFails(ark_mem, &netf);
            out << "# STATS neq=" << N_EQ << " nst=" << nst
                << " nfe=-1 nje=-1"
                << " nni=" << nni << " netf=" << netf
                << " nfeLS=-1 nsetups=-1 ncfn=-1 nqe=-1"
                << " nqhit=0 nqmiss=0 nbuild=1 nreuse=0"
                << " t_int=" << t_int_s << '\n';
        }
        ARKodeFree(&ark_mem);
        ark_cleanup();
        return status;
    }

    // ── CVODE, on the reusable workspace ──────────────────────────────────────
    const int st = ws_ensure(ws, P, NEQ_SOLVE, reduced, use_ajac);
    if (st != 0) return st;

    ws.ctx.P = &P;
    ws.ctx.acc_valid = false;
    ws.ctx.nq_hit = 0;
    ws.ctx.nq_miss = 0;

    N_Vector y  = ws.y;
    N_Vector yQ = ws.yQ;
    void* cvode_mem = ws.cvode_mem;

    if (reduced)
        for (int j = 0; j < NEQ_SOLVE; ++j) NV_Ith_S(y, j) = P.y0[red_idx(P, j)];
    else
        for (int k = 0; k < N_EQ; ++k)      NV_Ith_S(y, k) = P.y0[k];
    if (use_quad)
        for (int k = 0; k < n_acc(P); ++k)
            NV_Ith_S(yQ, k) = P.y0[acc_full_idx(P, k)];

    if (CVodeReInit(cvode_mem, t_eval[0], y) != CV_SUCCESS) return 134;
    if (use_quad && CVodeQuadReInit(cvode_mem, yQ) != CV_SUCCESS) return 135;
    if (P.acc_mode == ACC_STATE_RELAX) {
        // Per-component absolute tolerance: the accumulators get one so large
        // that their error weight 1/(rtol|y|+atol) is numerically zero, so they
        // ride the step sequence the state equations demand instead of setting
        // it. Same effect on the error test as taking them out of the system,
        // at no extra core evaluation.
        for (int j = 0; j < NEQ_SOLVE; ++j)
            NV_Ith_S(ws.atolv, j) = (red_idx(P, j) >= N_PHYS &&
                                     red_idx(P, j) < IDX_RHO_N)
                                    ? ACC_ATOL_RELAXED : P.atol;
        if (CVodeSVtolerances(cvode_mem, P.rtol, ws.atolv) != CV_SUCCESS) return 131;
    } else if (CVodeSStolerances(cvode_mem, P.rtol, P.atol) != CV_SUCCESS) return 131;
    if (CVodeSetMaxNumSteps(cvode_mem, 500000) != CV_SUCCESS) return 131;

    // Emit one output row in the unchanged 19-column contract, reassembling the
    // full state from the reduced block, the frozen mobile values and the
    // quadrature accumulators when running reduced.
    auto emit_row = [&](double t) {
        // Zeroed, not merely declared. The reduced path fills only the slots its
        // block covers, so any component outside it -- which from step 1 of the
        // plan includes the eight appended family slots whenever n_fam = 4 --
        // would otherwise be emitted as whatever was on the stack. It printed
        // denormals and a stray 1e+08, which read as physical values and would
        // have been carried into march_state.npz without anything complaining.
        double full[N_EQ] = {0.0};
        if (reduced) {
            for (int k = 0; k < N_MOB; ++k) full[k] = P.y0[k];   // frozen or seeded
            for (int j = 0; j < NEQ_SOLVE; ++j) full[red_idx(P, j)] = NV_Ith_S(y, j);
            if (use_quad)
                for (int k = 0; k < n_acc(P); ++k)
                    full[acc_full_idx(P, k)] = NV_Ith_S(yQ, k);
        } else {
            for (int k = 0; k < N_EQ; ++k) full[k] = NV_Ith_S(y, k);
        }
        out << t;
        for (int k = 0; k < n_out_state(P); ++k) out << ' ' << full[k];
        out << '\n';
    };

    emit_row(t_eval[0]);

    sunrealtype t_current = t_eval[0];
    for (int i = 1; i < P.n_points; ++i) {
        flag = CVode(cvode_mem, t_eval[i], y, &t_current, CV_NORMAL);
        if (flag < 0) { status = 140; break; }
        // `use_quad`, NOT `reduced`: the quadrature module is only ever
        // initialised when acc_mode == ACC_QUADRATURE, so a reduced solve in
        // any other accumulator mode -- acc_mode 2 is what the coupling march
        // uses -- called CVodeGetQuad on a module that was never activated.
        // SUNDIALS answered with one error line per case: 20 000 of
        // "[CVodeGetQuadDky] Quadrature integration not activated." per batch,
        // enough to bury a real solver error. Harmless to the result (yQ is
        // only read under use_quad) and, measured, not a cost either.
        if (use_quad) {
            sunrealtype tq;
            CVodeGetQuad(cvode_mem, &tq, yQ);
        }
        emit_row(t_eval[i]);
    }

    const double t_int_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_wall0).count();
    if (P.stats) {
        long nst = 0, nfe = 0, nsetups = 0, netf = 0, nni = 0, ncfn = 0,
             nje = 0, nfeLS = 0, nqe = 0;
        CVodeGetNumSteps(cvode_mem, &nst);
        CVodeGetNumRhsEvals(cvode_mem, &nfe);
        CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
        CVodeGetNumErrTestFails(cvode_mem, &netf);
        CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
        CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
        if (P.linsol != 2) {
            CVodeGetNumJacEvals(cvode_mem, &nje);
            CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
        }
        if (use_quad) CVodeGetQuadNumRhsEvals(cvode_mem, &nqe);
        out << "# STATS neq=" << NEQ_SOLVE << " nst=" << nst
            << " nfe=" << nfe << " nje=" << nje
            << " nni=" << nni << " netf=" << netf
            << " nfeLS=" << nfeLS << " nsetups=" << nsetups
            << " ncfn=" << ncfn << " nqe=" << nqe
            << " nqhit=" << ws.ctx.nq_hit << " nqmiss=" << ws.ctx.nq_miss
            << " nbuild=" << ws.n_build << " nreuse=" << ws.n_reuse
            << " t_int=" << t_int_s << '\n';
    }
    return status;
}

// Convenience overload for the single-case path: private workspace, torn down
// on return, so behaviour is exactly as before.
static int integrate_one(const Parameters& P, std::ostream& out) {
    Workspace ws;
    int s = integrate_one(P, out, ws);
    ws_release(ws);
    return s;
}

// ── Batch mode ─────────────────────────────────────────────────────────────────

static int run_batch(const std::string& batch_file) {
    std::ifstream f(batch_file);
    if (!f) {
        std::cerr << "Cannot open batch file: " << batch_file << "\n";
        return 1;
    }

    // ── Case-file protocol ────────────────────────────────────────────────────
    // Historically every line repeated the FULL parameter set -- ~150 key=value
    // tokens -- although only the 19 y0 values and the time window differ from
    // case to case. At q = 24115 that is ~3.6M redundant string-to-double
    // conversions and map inserts per substep, which measurement showed
    // dominating the batch once the solver itself was made cheap.
    //
    // A line beginning with "@BASE " now supplies the shared defaults once;
    // every later line carries only its overrides. The old format still works:
    // with no @BASE line each case is parsed standalone exactly as before.
    std::vector<Parameters> cases;
    std::map<std::string, double> base_map;
    bool have_base = false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("@BASE", 0) == 0) {
            base_map = parse_kv_line(line.substr(5));
            have_base = true;
            continue;
        }
        if (line.find('=') == std::string::npos)
            continue;   // skip blank / comment lines
        if (have_base) {
            std::map<std::string, double> m(base_map);
            for (const auto& kv : parse_kv_line(line))
                m[kv.first] = kv.second;      // per-case override wins
            cases.push_back(build_parameters(m));
        } else {
            cases.push_back(build_parameters(parse_kv_line(line)));
        }
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

    // One workspace per thread, reused across every case that thread handles.
    // Cases differ in stiffness / number of internal steps, so dynamic
    // scheduling keeps all threads busy until the work is drained.
    long tot_build = 0, tot_reuse = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+ : tot_build, tot_reuse)
#endif
    {
        Workspace ws;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int c = 0; c < ncases; ++c) {
            std::ostringstream oss;
            int s = integrate_one(cases[c], oss, ws);
            status[c] = s;
            outbuf[c] = oss.str();
        }
        tot_build += ws.n_build;
        tot_reuse += ws.n_reuse;
        ws_release(ws);
    }
    std::cerr << "Batch: " << tot_build << " solver build(s), "
              << tot_reuse << " reuse(s)\n";

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
