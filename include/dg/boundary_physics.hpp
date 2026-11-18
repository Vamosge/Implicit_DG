#pragma once

#include "dg/physics.hpp"

#include <algorithm>
#include <cmath>

namespace dg::boundary_physics {

struct GasProperties {
    double gamma = 1.4;
    double gamma_minus_one = 0.4;
    double freestream_mach = 0.0;
    PrimitiveState freestream;
};

inline PrimitiveState characteristic_farfield_boundary(
    const PrimitiveState& interior,
    const PrimitiveState& exterior,
    double nx,
    double ny,
    const GasProperties& gas)
{
    PrimitiveState boundary = interior;
    const double gm1 = gas.gamma_minus_one;
    const double ai = primitive_sound_speed(interior, gas.gamma);
    const double ae = primitive_sound_speed(exterior, gas.gamma);
    double uni = 0.0, uti = 0.0, une = 0.0, ute = 0.0;
    velocity_to_normal_tangent(interior.u, interior.v, nx, ny, uni, uti);
    velocity_to_normal_tangent(exterior.u, exterior.v, nx, ny, une, ute);

    if (uni <= -ai) return exterior;
    if (uni >= ai) return interior;

    const double Rp = uni + 2.0 * ai / gm1;
    const double Rm = une - 2.0 * ae / gm1;
    const bool inflow = (uni < 0.0);
    const double Sb = inflow
        ? primitive_entropy(exterior, gas.gamma)
        : primitive_entropy(interior, gas.gamma);
    const double utb = inflow ? ute : uti;
    const double nutb = inflow ? exterior.nut : interior.nut;
    const double unb = 0.5 * (Rp + Rm);
    const double ab = std::max(0.25 * gm1 * (Rp - Rm), 1e-8);
    const double rhob = std::pow(std::max(
        (ab * ab) / (gas.gamma * std::max(Sb, 1e-30)), 1e-30),
        1.0 / gm1);
    const double pb = std::max(
        Sb * std::pow(rhob, gas.gamma), 1e-12);
    double ub = 0.0, vb = 0.0;
    normal_tangent_to_velocity(unb, utb, nx, ny, ub, vb);
    boundary.rho = rhob;
    boundary.u = ub;
    boundary.v = vb;
    boundary.p = pb;
    boundary.nut = nutb;
    return boundary;
}

inline PrimitiveState characteristic_pressure_outlet_boundary(
    const PrimitiveState& interior,
    double back_pressure,
    double nx,
    double ny,
    const GasProperties& gas)
{
    PrimitiveState boundary = interior;
    const double ai = primitive_sound_speed(interior, gas.gamma);
    double uni = 0.0, uti = 0.0;
    velocity_to_normal_tangent(interior.u, interior.v, nx, ny, uni, uti);

    if (uni >= ai) return interior;
    if (uni <= -ai) {
        PrimitiveState inflow = gas.freestream;
        inflow.p = back_pressure;
        return characteristic_farfield_boundary(interior, inflow, nx, ny, gas);
    }

    const double Sb = primitive_entropy(interior, gas.gamma);
    const double pb = std::max(back_pressure, 1e-12);
    const double rhob = std::pow(
        std::max(pb / std::max(Sb, 1e-30), 1e-30), 1.0 / gas.gamma);
    const double ab = std::sqrt(gas.gamma * pb / rhob);
    const double Rp = uni + 2.0 * ai / gas.gamma_minus_one;
    const double unb = Rp - 2.0 * ab / gas.gamma_minus_one;
    double ub = 0.0, vb = 0.0;
    normal_tangent_to_velocity(unb, uti, nx, ny, ub, vb);
    boundary.rho = rhob;
    boundary.u = ub;
    boundary.v = vb;
    boundary.p = pb;
    boundary.nut = interior.nut;
    return boundary;
}

inline PrimitiveState characteristic_total_inflow_boundary(
    const PrimitiveState& interior,
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
            interior, gas.freestream, nx, ny, gas);
    }

    const double ai = primitive_sound_speed(interior, gas.gamma);
    double uni = 0.0, uti = 0.0;
    velocity_to_normal_tangent(interior.u, interior.v, nx, ny, uni, uti);
    const double Rplus = uni + 2.0 * ai / gas.gamma_minus_one;
    const double a0 = std::sqrt(
        gas.gamma * std::max(total_temperature, 1e-30));

    auto invariant_from_mach = [&](double mach) {
        const double denominator = std::sqrt(std::max(
            1.0 + 0.5 * gas.gamma_minus_one * mach * mach, 1e-30));
        const double a = a0 / denominator;
        const double speed = mach * a;
        return speed * dir_n + 2.0 * a / gas.gamma_minus_one;
    };

    double lo = 0.0, hi = 0.999;
    double flo = invariant_from_mach(lo) - Rplus;
    double fhi = invariant_from_mach(hi) - Rplus;
    if (flo * fhi > 0.0) {
        hi = 5.0;
        fhi = invariant_from_mach(hi) - Rplus;
    }
    for (int iteration = 0; iteration < 80 && flo * fhi <= 0.0; ++iteration) {
        const double mid = 0.5 * (lo + hi);
        const double fm = invariant_from_mach(mid) - Rplus;
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
    const double temperature_ratio =
        1.0 + 0.5 * gas.gamma_minus_one * mach * mach;
    const double temperature = std::max(
        total_temperature / std::max(temperature_ratio, 1e-30), 1e-30);
    const double pressure = std::max(
        total_pressure / std::pow(temperature_ratio,
            gas.gamma / gas.gamma_minus_one), 1e-12);
    const double density = std::max(pressure / temperature, 1e-12);
    const double a = std::sqrt(gas.gamma * pressure / density);
    const double speed = mach * a;

    PrimitiveState boundary;
    boundary.rho = density;
    boundary.u = speed * dir_x;
    boundary.v = speed * dir_y;
    boundary.p = pressure;
    boundary.nut = gas.freestream.nut;
    return boundary;
}

} // namespace dg::boundary_physics
