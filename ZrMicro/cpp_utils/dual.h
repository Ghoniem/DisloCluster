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

// ── Type-dispatching wrappers used by the templated core ─────────────────────
// The core calls ad_sqrt / ad_exp / ad_cbrt rather than unqualified sqrt/exp/
// cbrt. A block-scope `using std::sqrt` would HIDE the Dual overloads above
// (block-scope using-declarations hide outer-scope names), and relying on ADL
// alone is fragile for the double instantiation. These wrappers are explicit
// and resolve unambiguously for both T = double and T = Dual<N>.

inline double ad_sqrt(double x) { return std::sqrt(x); }
inline double ad_exp (double x) { return std::exp(x);  }
inline double ad_cbrt(double x) { return std::cbrt(x); }

template <int N> inline Dual<N> ad_sqrt(const Dual<N>& x) { return sqrt(x); }
template <int N> inline Dual<N> ad_exp (const Dual<N>& x) { return exp(x);  }
template <int N> inline Dual<N> ad_cbrt(const Dual<N>& x) { return cbrt(x); }

// Value extraction (for comparisons written generically).
inline double ad_val(double x) { return x; }
template <int N> inline double ad_val(const Dual<N>& x) { return x.v; }
