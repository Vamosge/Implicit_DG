#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace dg {

#if defined(_MSC_VER)
#define DG_BLOCK_AD_FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define DG_BLOCK_AD_FORCE_INLINE inline __attribute__((always_inline))
#else
#define DG_BLOCK_AD_FORCE_INLINE inline
#endif

template <std::size_t DerivativeCount>
struct BlockAD {
    double v = 0.0;
    std::array<double, DerivativeCount> d = {};

    BlockAD() = default;
    BlockAD(double value) : v(value) {}

    DG_BLOCK_AD_FORCE_INLINE BlockAD& operator+=(const BlockAD& other)
    {
        v += other.v;
        for (std::size_t p = 0; p < DerivativeCount; ++p) d[p] += other.d[p];
        return *this;
    }

    DG_BLOCK_AD_FORCE_INLINE BlockAD& operator-=(const BlockAD& other)
    {
        v -= other.v;
        for (std::size_t p = 0; p < DerivativeCount; ++p) d[p] -= other.d[p];
        return *this;
    }
};

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> make_seed(double value, std::size_t p)
{
    BlockAD<N> result(value);
    result.d[p] = 1.0;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator+(const BlockAD<N>& a, const BlockAD<N>& b)
{
    BlockAD<N> result(a.v + b.v);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = a.d[p] + b.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator-(const BlockAD<N>& a, const BlockAD<N>& b)
{
    BlockAD<N> result(a.v - b.v);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = a.d[p] - b.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator-(const BlockAD<N>& a)
{
    BlockAD<N> result(-a.v);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = -a.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator*(const BlockAD<N>& a, const BlockAD<N>& b)
{
    BlockAD<N> result(a.v * b.v);
    for (std::size_t p = 0; p < N; ++p) {
        result.d[p] = a.d[p] * b.v + a.v * b.d[p];
    }
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator/(const BlockAD<N>& a, const BlockAD<N>& b)
{
    const double inv = 1.0 / b.v;
    BlockAD<N> result(a.v * inv);
    const double inv2 = inv * inv;
    for (std::size_t p = 0; p < N; ++p) {
        result.d[p] = (a.d[p] * b.v - a.v * b.d[p]) * inv2;
    }
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator+(double a, const BlockAD<N>& b)
{
    BlockAD<N> result = b;
    result.v += a;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator+(const BlockAD<N>& a, double b)
{
    BlockAD<N> result = a;
    result.v += b;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator-(double a, const BlockAD<N>& b)
{
    BlockAD<N> result(a - b.v);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = -b.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator-(const BlockAD<N>& a, double b)
{
    BlockAD<N> result = a;
    result.v -= b;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator*(double a, const BlockAD<N>& b)
{
    BlockAD<N> result(a * b.v);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = a * b.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator*(const BlockAD<N>& a, double b)
{
    BlockAD<N> result(a.v * b);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = a.d[p] * b;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator/(double a, const BlockAD<N>& b)
{
    const double inv = 1.0 / b.v;
    BlockAD<N> result(a * inv);
    const double inv2 = inv * inv;
    for (std::size_t p = 0; p < N; ++p) result.d[p] = (-a * b.d[p]) * inv2;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> operator/(const BlockAD<N>& a, double b)
{
    const double inv = 1.0 / b;
    BlockAD<N> result(a.v * inv);
    const double inv2 = inv * inv;
    for (std::size_t p = 0; p < N; ++p) result.d[p] = (a.d[p] * b) * inv2;
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_sqrt(const BlockAD<N>& a)
{
    const double value = std::sqrt(a.v);
    BlockAD<N> result(value);
    const double factor = 0.5 / std::max(value, 1e-300);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = factor * a.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_exp(const BlockAD<N>& a)
{
    const double value = std::exp(a.v);
    BlockAD<N> result(value);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = value * a.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_pow(const BlockAD<N>& a, double exponent)
{
    const double value = std::pow(a.v, exponent);
    BlockAD<N> result(value);
    const double factor = (a.v == 0.0 && exponent < 1.0)
        ? 0.0 : exponent * std::pow(a.v, exponent - 1.0);
    for (std::size_t p = 0; p < N; ++p) result.d[p] = factor * a.d[p];
    return result;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_abs(const BlockAD<N>& a)
{
    return a.v >= 0.0 ? a : -a;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_max(const BlockAD<N>& a, const BlockAD<N>& b)
{
    return a.v >= b.v ? a : b;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_min(const BlockAD<N>& a, const BlockAD<N>& b)
{
    return a.v <= b.v ? a : b;
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_max(const BlockAD<N>& a, double b)
{
    return a.v >= b ? a : BlockAD<N>(b);
}

template <std::size_t N>
DG_BLOCK_AD_FORCE_INLINE BlockAD<N> ad_min(const BlockAD<N>& a, double b)
{
    return a.v <= b ? a : BlockAD<N>(b);
}

// Math-policy adapter used by shared scalar/AD physics kernels.  Keeping this
// next to BlockAD prevents the residual and its Jacobian from drifting into
// separate hand-maintained implementations.
template <std::size_t N>
struct BlockADMath {
    static double value(const BlockAD<N>& x) { return x.v; }
    static BlockAD<N> max(const BlockAD<N>& a, const BlockAD<N>& b) {
        return ad_max(a, b);
    }
    static BlockAD<N> min(const BlockAD<N>& a, const BlockAD<N>& b) {
        return ad_min(a, b);
    }
    static BlockAD<N> abs(const BlockAD<N>& x) { return ad_abs(x); }
    static BlockAD<N> exp(const BlockAD<N>& x) { return ad_exp(x); }
    static BlockAD<N> pow(const BlockAD<N>& x, double exponent) {
        return ad_pow(x, exponent);
    }
    static BlockAD<N> freeze(const BlockAD<N>& x) { return BlockAD<N>(x.v); }
};

#undef DG_BLOCK_AD_FORCE_INLINE

} // namespace dg
