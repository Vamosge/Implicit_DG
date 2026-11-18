#pragma once

#include "dg/block_ad.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace dg::adphysics {

struct GasProperties {
    double gamma = 1.4;
    double gamma_minus_one = 0.4;
    double density_floor = 1e-12;
    double pressure_floor = 1e-12;
    double freestream_density = 1.0;
    double freestream_u = 0.0;
    double freestream_v = 0.0;
    double freestream_pressure = 1.0;
    double freestream_nu_tilde = 0.0;
    double freestream_mach = 0.0;
};

template <std::size_t N>
struct PrimitiveState {
    BlockAD<N> rho;
    BlockAD<N> u;
    BlockAD<N> v;
    BlockAD<N> p;
    BlockAD<N> nut;
};

template <std::size_t N>
inline PrimitiveState<N> freestream_primitive(const GasProperties& gas)
{
    PrimitiveState<N> state;
    state.rho = gas.freestream_density;
    state.u = gas.freestream_u;
    state.v = gas.freestream_v;
    state.p = gas.freestream_pressure;
    state.nut = gas.freestream_nu_tilde;
    return state;
}

template <std::size_t N>
inline PrimitiveState<N> conservative_to_primitive(
    const BlockAD<N>* conservative,
    const GasProperties& gas)
{
    PrimitiveState<N> state;
    state.rho = ad_max(conservative[0], gas.density_floor);
    state.u = conservative[1] / state.rho;
    state.v = conservative[2] / state.rho;
    state.p = ad_max(
        gas.gamma_minus_one
            * (conservative[3]
                - 0.5 * state.rho * (state.u * state.u + state.v * state.v)),
        gas.pressure_floor);
    state.nut = conservative[4] / state.rho;
    return state;
}

template <std::size_t N>
inline void primitive_to_conservative(
    const PrimitiveState<N>& state,
    BlockAD<N>* conservative,
    const GasProperties& gas)
{
    const BlockAD<N> rho = ad_max(state.rho, gas.density_floor);
    const BlockAD<N> p = ad_max(state.p, gas.pressure_floor);
    conservative[0] = rho;
    conservative[1] = rho * state.u;
    conservative[2] = rho * state.v;
    conservative[3] = p / gas.gamma_minus_one
        + 0.5 * rho * (state.u * state.u + state.v * state.v);
    conservative[4] = rho * state.nut;
}

template <std::size_t N>
inline BlockAD<N> entropy_measure(
    const PrimitiveState<N>& state,
    const GasProperties& gas)
{
    return ad_max(state.p, gas.pressure_floor)
        / ad_pow(ad_max(state.rho, gas.density_floor), gas.gamma);
}

template <std::size_t N>
inline BlockAD<N> sound_speed(
    const PrimitiveState<N>& state,
    const GasProperties& gas)
{
    return ad_sqrt(
        gas.gamma * ad_max(state.p, gas.pressure_floor)
        / ad_max(state.rho, gas.density_floor));
}

template <std::size_t N>
inline void velocity_to_normal_tangent(
    const BlockAD<N>& u,
    const BlockAD<N>& v,
    double nx,
    double ny,
    BlockAD<N>& un,
    BlockAD<N>& ut)
{
    un = u * nx + v * ny;
    ut = -u * ny + v * nx;
}

template <std::size_t N>
inline void normal_tangent_to_velocity(
    const BlockAD<N>& un,
    const BlockAD<N>& ut,
    double nx,
    double ny,
    BlockAD<N>& u,
    BlockAD<N>& v)
{
    u = un * nx - ut * ny;
    v = un * ny + ut * nx;
}

template <std::size_t N>
inline PrimitiveState<N> characteristic_farfield_boundary(
    const PrimitiveState<N>& interior,
    const PrimitiveState<N>& exterior,
    double nx,
    double ny,
    const GasProperties& gas)
{
    PrimitiveState<N> boundary = interior;
    const double gm1 = gas.gamma_minus_one;
    const BlockAD<N> ai = sound_speed(interior, gas);
    const BlockAD<N> ae = sound_speed(exterior, gas);
    BlockAD<N> uni, uti, une, ute;
    velocity_to_normal_tangent(interior.u, interior.v, nx, ny, uni, uti);
    velocity_to_normal_tangent(exterior.u, exterior.v, nx, ny, une, ute);

    if (uni.v <= -ai.v) return exterior;
    if (uni.v >= ai.v) return interior;

    const BlockAD<N> Rp = uni + 2.0 * ai / gm1;
    const BlockAD<N> Rm = une - 2.0 * ae / gm1;
    const bool inflow = (uni.v < 0.0);
    const BlockAD<N> Sb = inflow
        ? entropy_measure(exterior, gas)
        : entropy_measure(interior, gas);
    const BlockAD<N> utb = inflow ? ute : uti;
    const BlockAD<N> nutb = inflow ? exterior.nut : interior.nut;
    const BlockAD<N> unb = 0.5 * (Rp + Rm);
    const BlockAD<N> ab = ad_max(0.25 * gm1 * (Rp - Rm), 1e-8);
    const BlockAD<N> rhob = ad_pow(
        ad_max((ab * ab) / (gas.gamma * ad_max(Sb, 1e-30)), 1e-30),
        1.0 / gm1);
    const BlockAD<N> pb = ad_max(
        Sb * ad_pow(rhob, gas.gamma), gas.pressure_floor);
    BlockAD<N> ub, vb;
    normal_tangent_to_velocity(unb, utb, nx, ny, ub, vb);
    boundary.rho = rhob;
    boundary.u = ub;
    boundary.v = vb;
    boundary.p = pb;
    boundary.nut = nutb;
    return boundary;
}

template <std::size_t N>
inline PrimitiveState<N> characteristic_pressure_outlet_boundary(
    const PrimitiveState<N>& interior,
    double back_pressure,
    double nx,
    double ny,
    const GasProperties& gas)
{
    const PrimitiveState<N> exterior = freestream_primitive<N>(gas);
    PrimitiveState<N> boundary = interior;
    const BlockAD<N> ai = sound_speed(interior, gas);
    BlockAD<N> uni, uti;
    velocity_to_normal_tangent(interior.u, interior.v, nx, ny, uni, uti);

    if (uni.v >= ai.v) return interior;
    if (uni.v <= -ai.v) {
        PrimitiveState<N> inflow = exterior;
        inflow.p = back_pressure;
        return characteristic_farfield_boundary(interior, inflow, nx, ny, gas);
    }

    const BlockAD<N> Sb = entropy_measure(interior, gas);
    const BlockAD<N> pb = ad_max(BlockAD<N>(back_pressure), gas.pressure_floor);
    const BlockAD<N> rhob = ad_pow(
        ad_max(pb / ad_max(Sb, 1e-30), 1e-30), 1.0 / gas.gamma);
    const BlockAD<N> ab = ad_sqrt(gas.gamma * pb / rhob);
    const BlockAD<N> Rp = uni + 2.0 * ai / gas.gamma_minus_one;
    const BlockAD<N> unb = Rp - 2.0 * ab / gas.gamma_minus_one;
    BlockAD<N> ub, vb;
    normal_tangent_to_velocity(unb, uti, nx, ny, ub, vb);
    boundary.rho = rhob;
    boundary.u = ub;
    boundary.v = vb;
    boundary.p = pb;
    boundary.nut = interior.nut;
    return boundary;
}

template <std::size_t N>
inline PrimitiveState<N> characteristic_total_inflow_boundary(
    const PrimitiveState<N>& interior,
    double total_pressure,
    double total_temperature,
    double flow_angle,
    double nx,
    double ny,
    const GasProperties& gas)
{
    const double dir_x = std::cos(flow_angle);
    const double dir_y = std::sin(flow_angle);
    const double dir_n = dir_x * nx + dir_y * ny;
    if (dir_n >= -1e-12) {
        return characteristic_farfield_boundary(
            interior, freestream_primitive<N>(gas), nx, ny, gas);
    }

    const BlockAD<N> ai = sound_speed(interior, gas);
    BlockAD<N> uni, uti;
    velocity_to_normal_tangent(interior.u, interior.v, nx, ny, uni, uti);
    const BlockAD<N> Rplus = uni + 2.0 * ai / gas.gamma_minus_one;
    const double a0 = std::sqrt(
        gas.gamma * std::max(total_temperature, 1e-30));

    auto invariant_from_mach = [&](double mach) {
        const double denominator = std::sqrt(std::max(
            1.0 + 0.5 * gas.gamma_minus_one * mach * mach, 1e-30));
        const double a = a0 / denominator;
        const double speed = mach * a;
        return speed * dir_n + 2.0 * a / gas.gamma_minus_one;
    };

    double lo = 0.0;
    double hi = 0.999;
    double flo = invariant_from_mach(lo) - Rplus.v;
    double fhi = invariant_from_mach(hi) - Rplus.v;
    if (flo * fhi > 0.0) {
        hi = 5.0;
        fhi = invariant_from_mach(hi) - Rplus.v;
    }
    for (int iteration = 0; iteration < 80 && flo * fhi <= 0.0; ++iteration) {
        const double mid = 0.5 * (lo + hi);
        const double fm = invariant_from_mach(mid) - Rplus.v;
        if (flo * fm <= 0.0) {
            hi = mid;
            fhi = fm;
        }
        else {
            lo = mid;
            flo = fm;
        }
    }
    const double mach = (flo * fhi <= 0.0)
        ? 0.5 * (lo + hi)
        : gas.freestream_mach;
    const double delta_mach = 1e-6 * std::max(1.0, std::abs(mach));
    const double lower_mach = std::max(mach - delta_mach, 0.0);
    const double d_invariant_d_mach =
        (invariant_from_mach(mach + delta_mach)
            - invariant_from_mach(lower_mach))
        / (mach + delta_mach - lower_mach);
    const double d_mach_d_rplus =
        (std::isfinite(d_invariant_d_mach)
            && std::abs(d_invariant_d_mach) > 1e-12)
        ? (1.0 / d_invariant_d_mach)
        : 0.0;
    const BlockAD<N> mach_ad = mach
        + d_mach_d_rplus * (Rplus - Rplus.v);
    const BlockAD<N> total_temperature_ratio = 1.0
        + 0.5 * gas.gamma_minus_one * mach_ad * mach_ad;
    const BlockAD<N> temperature = total_temperature
        / ad_max(total_temperature_ratio, 1e-30);
    const BlockAD<N> pressure = total_pressure
        / ad_pow(ad_max(total_temperature_ratio, 1e-30),
            gas.gamma / gas.gamma_minus_one);
    const BlockAD<N> density = pressure / ad_max(temperature, 1e-30);
    const BlockAD<N> a = ad_sqrt(
        gas.gamma * pressure / ad_max(density, gas.density_floor));
    const BlockAD<N> speed = mach_ad * a;

    PrimitiveState<N> boundary;
    boundary.rho = density;
    boundary.u = speed * dir_x;
    boundary.v = speed * dir_y;
    boundary.p = pressure;
    boundary.nut = gas.freestream_nu_tilde;
    return boundary;
}

} // namespace dg::adphysics
