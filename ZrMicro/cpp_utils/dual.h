/**
 * dual.h — forward-mode automatic differentiation for the ZrMicro RHS.
 *
 * Dual<N> carries a value plus N directional derivatives. Evaluating the
 * templated RHS core (rate_equations_core.h) once with Dual<N> seeded on the N
 * state components produces the EXACT Jacobian column-by-column in a single
 * sweep — the transcendental functions (sqrt, exp, cbrt) are evaluated once and
 * their derivatives propagated by a multiply, rather than N times as a
 * difference-quotient Jacobian would.
 *
 * Why AD rather than a hand-written analytic Jacobian:
 *   * exactness — no perturbation size to choose, no truncation error, so
 *     Newton converges in fewer iterations and CVODE reuses the Jacobian for
 *     more steps;
 *   * it cannot drift out of sync with the RHS. The physics lives in exactly
 *     one place (the templated core) and both the residual and the Jacobian are
 *     instantiations of it. Hand-differentiating the coalescence Avrami gates
 *     and the smoothstep size gate would be a second, silently-divergent copy.
 *
 * Non-smooth points (concentration floors, min(), max(0,·), the gate cut-offs)
 * yield the one-sided derivative. That is the correct local linearisation on
 * each side of the kink and is no worse than what a difference quotient does
 * when its perturbation straddles one.
 */
#pragma once

#include <cmath>
#include <limits>

template <int N>
struct Dual {
    double v;       // value
    double d[N];    // derivatives w.r.t. the N seeded directions

    Dual() : v(0.0) { for (int i = 0; i < N; ++i) d[i] = 0.0; }
    Dual(double x) : v(x) { for (int i = 0; i < N; ++i) d[i] = 0.0; }

    // Seed direction k with unit derivative (used to build the identity seed).
    void seed(int k) { d[k] = 1.0; }
};

// ── Arithmetic ───────────────────────────────────────────────────────────────

template <int N>
inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] + b.d[i];
    return r;
}
template <int N>
inline Dual<N> operator+(const Dual<N>& a, double b) {
    Dual<N> r; r.v = a.v + b;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i];
    return r;
}
template <int N>
inline Dual<N> operator+(double a, const Dual<N>& b) { return b + a; }

template <int N>
inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] - b.d[i];
    return r;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a, double b) {
    Dual<N> r; r.v = a.v - b;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i];
    return r;
}
template <int N>
inline Dual<N> operator-(double a, const Dual<N>& b) {
    Dual<N> r; r.v = a - b.v;
    for (int i = 0; i < N; ++i) r.d[i] = -b.d[i];
    return r;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a) {
    Dual<N> r; r.v = -a.v;
    for (int i = 0; i < N; ++i) r.d[i] = -a.d[i];
    return r;
}

template <int N>
inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return r;
}
template <int N>
inline Dual<N> operator*(const Dual<N>& a, double b) {
    Dual<N> r; r.v = a.v * b;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b;
    return r;
}
template <int N>
inline Dual<N> operator*(double a, const Dual<N>& b) { return b * a; }

template <int N>
inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v / b.v;
    const double inv = 1.0 / b.v;
    const double inv2 = inv * inv;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * inv - a.v * b.d[i] * inv2;
    return r;
}
template <int N>
inline Dual<N> operator/(const Dual<N>& a, double b) {
    Dual<N> r; const double inv = 1.0 / b;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * inv;
    return r;
}
template <int N>
inline Dual<N> operator/(double a, const Dual<N>& b) {
    Dual<N> r; const double inv = 1.0 / b.v;
    r.v = a * inv;
    const double c = -a * inv * inv;
    for (int i = 0; i < N; ++i) r.d[i] = c * b.d[i];
    return r;
}

template <int N> inline Dual<N>& operator+=(Dual<N>& a, const Dual<N>& b) { a = a + b; return a; }
template <int N> inline Dual<N>& operator-=(Dual<N>& a, const Dual<N>& b) { a = a - b; return a; }
template <int N> inline Dual<N>& operator*=(Dual<N>& a, const Dual<N>& b) { a = a * b; return a; }
template <int N> inline Dual<N>& operator*=(Dual<N>& a, double b)         { a = a * b; return a; }

// ── Comparisons (value-only; branches pick a smooth piece) ───────────────────

template <int N> inline bool operator> (const Dual<N>& a, double b) { return a.v >  b; }
template <int N> inline bool operator< (const Dual<N>& a, double b) { return a.v <  b; }
template <int N> inline bool operator>=(const Dual<N>& a, double b) { return a.v >= b; }
template <int N> inline bool operator<=(const Dual<N>& a, double b) { return a.v <= b; }
template <int N> inline bool operator> (const Dual<N>& a, const Dual<N>& b) { return a.v >  b.v; }
template <int N> inline bool operator< (const Dual<N>& a, const Dual<N>& b) { return a.v <  b.v; }

// ── Elementary functions ─────────────────────────────────────────────────────
// Free functions in the global namespace so the templated core can call them
// unqualified (`using std::sqrt; sqrt(x)`) and have ADL resolve double -> std
// and Dual -> these.

template <int N>
inline Dual<N> sqrt(const Dual<N>& a) {
    Dual<N> r; r.v = std::sqrt(a.v);
    // d/dx sqrt(x) = 1/(2 sqrt(x)); guarded so a floored zero does not produce
    // a NaN derivative (the value is already floored by the caller).
    const double c = (r.v > 0.0) ? 0.5 / r.v : 0.0;
    for (int i = 0; i < N; ++i) r.d[i] = c * a.d[i];
    return r;
}

template <int N>
inline Dual<N> exp(const Dual<N>& a) {
    Dual<N> r; r.v = std::exp(a.v);
    for (int i = 0; i < N; ++i) r.d[i] = r.v * a.d[i];
    return r;
}

template <int N>
inline Dual<N> cbrt(const Dual<N>& a) {
    Dual<N> r; r.v = std::cbrt(a.v);
    // d/dx x^(1/3) = 1/(3 x^(2/3))
    const double denom = 3.0 * r.v * r.v;
    const double c = (denom != 0.0) ? 1.0 / denom : 0.0;
    for (int i = 0; i < N; ++i) r.d[i] = c * a.d[i];
    return r;
}

template <int N>
inline Dual<N> log(const Dual<N>& a) {
    Dual<N> r; r.v = std::log(a.v);
    // d/dx ln(x) = 1/x, guarded the way sqrt is: the capillary term of the
    // step-3 emission channel evaluates ln(alpha R/|b|) on a radius the caller
    // has already clamped to r_min, but a floored zero must not produce a NaN
    // derivative and poison the whole Jacobian column.
    const double c = (a.v > 0.0) ? 1.0 / a.v : 0.0;
    for (int i = 0; i < N; ++i) r.d[i] = c * a.d[i];
    return r;
}

template <int N>
inline Dual<N> erfc(const Dual<N>& a) {
    Dual<N> r; r.v = std::erfc(a.v);
    // d/dx erfc(x) = -2/sqrt(pi) exp(-x^2). No guard is needed: the derivative
    // is finite everywhere and underflows to zero in the same tail where erfc
    // itself does, which is the correct limit rather than a clamped one.
    const double c = -1.1283791670955126 * std::exp(-a.v * a.v);
    for (int i = 0; i < N; ++i) r.d[i] = c * a.d[i];
    return r;
}

// ── erfc^-1, needed to ask "which size is the top f of this family?" ─────────
// There is no std::erfcinv. The value comes from the normal quantile via
// erfc(x) = 2 Phi(-x sqrt2), i.e. erfcinv(y) = -ndtri(y/2)/sqrt2, with Acklam's
// rational approximation refined by two Newton steps on erfc itself -- which
// converges to machine precision because erfc is smooth and monotone.
inline double _ndtri(double p) {
    // Acklam's inverse normal CDF; |relative error| < 1.15e-9 before refinement.
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                 2.445134137142996e+00,  3.754408661907416e+00};
    const double pl = 0.02425;
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return  std::numeric_limits<double>::infinity();
    double q, r, x;
    if (p < pl) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 1.0 - pl) {
        q = p - 0.5; r = q * q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
            / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    return x;
}

inline double erfcinv(double y) {
    if (y <= 0.0) return  std::numeric_limits<double>::infinity();
    if (y >= 2.0) return -std::numeric_limits<double>::infinity();
    double x = -_ndtri(0.5 * y) / 1.4142135623730951;
    for (int it = 0; it < 2; ++it) {          // Newton on erfc(x) - y = 0
        const double f = std::erfc(x) - y;
        const double df = -1.1283791670955126 * std::exp(-x * x);
        if (df == 0.0) break;
        x -= f / df;
    }
    return x;
}

template <int N>
inline Dual<N> erfcinv(const Dual<N>& a) {
    // The VALUE is the Newton solve above; the DERIVATIVE is the inverse
    // function's, which is exact and needs no iteration:
    //     d/dy erfc^-1(y) = -sqrt(pi)/2 exp( (erfc^-1 y)^2 ).
    Dual<N> r; r.v = erfcinv(a.v);
    const double c = -0.8862269254527580 * std::exp(r.v * r.v);
    for (int i = 0; i < N; ++i) r.d[i] = c * a.d[i];
    return r;
}

// ── Type-dispatching wrappers used by the templated core ─────────────────────
// The core calls ad_sqrt / ad_exp / ad_cbrt rather than unqualified sqrt/exp/
// cbrt. A block-scope `using std::sqrt` would HIDE the Dual overloads above
// (block-scope using-declarations hide outer-scope names), and relying on ADL
// alone is fragile for the double instantiation. These wrappers are explicit
// and resolve unambiguously for both T = double and T = Dual<N>.

inline double ad_sqrt(double x) { return std::sqrt(x); }
inline double ad_exp (double x) { return std::exp(x);  }
inline double ad_cbrt(double x) { return std::cbrt(x); }
inline double ad_log (double x) { return std::log(x);  }
inline double ad_erfc(double x) { return std::erfc(x); }
inline double ad_erfcinv(double x) { return erfcinv(x); }

template <int N> inline Dual<N> ad_sqrt(const Dual<N>& x) { return sqrt(x); }
template <int N> inline Dual<N> ad_exp (const Dual<N>& x) { return exp(x);  }
template <int N> inline Dual<N> ad_cbrt(const Dual<N>& x) { return cbrt(x); }
template <int N> inline Dual<N> ad_log (const Dual<N>& x) { return log(x);  }
template <int N> inline Dual<N> ad_erfc(const Dual<N>& x) { return erfc(x); }
template <int N> inline Dual<N> ad_erfcinv(const Dual<N>& x) { return erfcinv(x); }

// Value extraction (for comparisons written generically).
inline double ad_val(double x) { return x; }
template <int N> inline double ad_val(const Dual<N>& x) { return x.v; }
