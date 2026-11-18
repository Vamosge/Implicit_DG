#include "dg/boundary_physics.hpp"

#include <cassert>
#include <cmath>

namespace {

bool close(double a, double b, double tolerance = 1e-10)
{
    return std::abs(a - b) <= tolerance * (1.0 + std::abs(a) + std::abs(b));
}

} // namespace

int main()
{
    constexpr double gamma = 1.4;
    constexpr double gamma_minus_one = gamma - 1.0;
    constexpr double mach = 0.2;

    dg::boundary_physics::GasProperties gas;
    gas.gamma = gamma;
    gas.gamma_minus_one = gamma_minus_one;
    gas.freestream_mach = mach;
    gas.freestream.rho = 1.0;
    gas.freestream.u = mach;
    gas.freestream.v = 0.0;
    gas.freestream.p = 1.0 / gamma;
    gas.freestream.nut = 3.0e-6;

    const double static_temperature = gas.freestream.p / gas.freestream.rho;
    const double total_temperature = static_temperature
        * (1.0 + 0.5 * gamma_minus_one * mach * mach);
    const double total_pressure = gas.freestream.p
        * std::pow(total_temperature / static_temperature,
            gamma / gamma_minus_one);

    const dg::PrimitiveState inflow =
        dg::boundary_physics::characteristic_total_inflow_boundary(
            gas.freestream, total_pressure, total_temperature, 0.0,
            -1.0, 0.0, gas);
    assert(close(inflow.rho, gas.freestream.rho));
    assert(close(inflow.u, gas.freestream.u));
    assert(close(inflow.v, gas.freestream.v));
    assert(close(inflow.p, gas.freestream.p));
    assert(close(inflow.nut, gas.freestream.nut));

    const dg::PrimitiveState outlet =
        dg::boundary_physics::characteristic_pressure_outlet_boundary(
            gas.freestream, 0.95 * gas.freestream.p, 1.0, 0.0, gas);
    assert(close(outlet.p, 0.95 * gas.freestream.p));
    assert(std::isfinite(outlet.rho));
    assert(std::isfinite(outlet.u));

    const dg::PrimitiveState wrong_facing =
        dg::boundary_physics::characteristic_total_inflow_boundary(
            gas.freestream, total_pressure, total_temperature, 0.0,
            1.0, 0.0, gas);
    const dg::PrimitiveState farfield =
        dg::boundary_physics::characteristic_farfield_boundary(
            gas.freestream, gas.freestream, 1.0, 0.0, gas);
    assert(close(wrong_facing.rho, farfield.rho));
    assert(close(wrong_facing.u, farfield.u));
    assert(close(wrong_facing.p, farfield.p));
}
