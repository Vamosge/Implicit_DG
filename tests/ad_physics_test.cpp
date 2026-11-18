#include "dg/ad_physics.hpp"

#include <array>
#include <cassert>
#include <cmath>

namespace {

bool close(double a, double b, double tolerance = 1e-12)
{
    return std::abs(a - b) <= tolerance * (1.0 + std::abs(a) + std::abs(b));
}

using AD = dg::BlockAD<1>;
using State = dg::adphysics::PrimitiveState<1>;

dg::adphysics::GasProperties gas_properties()
{
    dg::adphysics::GasProperties gas;
    gas.gamma = 1.4;
    gas.gamma_minus_one = 0.4;
    gas.freestream_density = 1.0;
    gas.freestream_u = 0.2;
    gas.freestream_v = 0.0;
    gas.freestream_pressure = 1.0;
    gas.freestream_nu_tilde = 3.0e-6;
    gas.freestream_mach = 0.2;
    return gas;
}

State state(double rho, double u, double v, double p, double nut)
{
    State result;
    result.rho = rho;
    result.u = u;
    result.v = v;
    result.p = p;
    result.nut = nut;
    return result;
}

} // namespace

int main()
{
    const auto gas = gas_properties();

    std::array<AD, 5> conservative = {
        dg::make_seed<1>(2.0, 0), AD(6.0), AD(-2.0), AD(20.0), AD(0.4)};
    const State primitive =
        dg::adphysics::conservative_to_primitive<1>(conservative.data(), gas);

    assert(close(primitive.rho.v, 2.0));
    assert(close(primitive.u.v, 3.0));
    assert(close(primitive.v.v, -1.0));
    assert(close(primitive.p.v, 4.0));
    assert(close(primitive.nut.v, 0.2));
    assert(close(primitive.rho.d[0], 1.0));
    assert(close(primitive.u.d[0], -1.5));
    assert(close(primitive.v.d[0], 0.5));
    assert(close(primitive.p.d[0], 2.0));
    assert(close(primitive.nut.d[0], -0.1));

    std::array<AD, 5> roundtrip;
    dg::adphysics::primitive_to_conservative<1>(primitive, roundtrip.data(), gas);
    for (std::size_t equation = 0; equation < roundtrip.size(); ++equation) {
        assert(close(roundtrip[equation].v, conservative[equation].v));
        assert(close(roundtrip[equation].d[0], conservative[equation].d[0]));
    }

    const State exterior = dg::adphysics::freestream_primitive<1>(gas);
    const State supersonic_outflow = state(1.1, 3.0, 0.1, 1.0, 4.0e-6);
    const State outflow_boundary =
        dg::adphysics::characteristic_farfield_boundary<1>(
            supersonic_outflow, exterior, 1.0, 0.0, gas);
    assert(close(outflow_boundary.u.v, supersonic_outflow.u.v));
    assert(close(outflow_boundary.rho.v, supersonic_outflow.rho.v));

    const State supersonic_inflow = state(1.1, -3.0, 0.1, 1.0, 4.0e-6);
    const State inflow_boundary =
        dg::adphysics::characteristic_farfield_boundary<1>(
            supersonic_inflow, exterior, 1.0, 0.0, gas);
    assert(close(inflow_boundary.u.v, exterior.u.v));
    assert(close(inflow_boundary.rho.v, exterior.rho.v));

    State subsonic = state(1.0, 0.3, -0.1, 1.0, 5.0e-6);
    subsonic.rho = dg::make_seed<1>(1.0, 0);
    const State outlet =
        dg::adphysics::characteristic_pressure_outlet_boundary<1>(
            subsonic, 0.9, 1.0, 0.0, gas);
    assert(close(outlet.p.v, 0.9));
    assert(close(outlet.p.d[0], 0.0));
    assert(close(outlet.nut.v, subsonic.nut.v));
    assert(std::isfinite(outlet.rho.v));
    assert(std::isfinite(outlet.u.v));
    assert(std::isfinite(outlet.rho.d[0]));
    assert(std::isfinite(outlet.u.d[0]));

    const double total_temperature = 1.0 * (1.0 + 0.5 * 0.4 * 0.2 * 0.2);
    const double total_pressure = 1.0 * std::pow(total_temperature, 1.4 / 0.4);
    State inflow_interior = state(1.0, 0.2, 0.0, 1.0, 5.0e-6);
    inflow_interior.rho = dg::make_seed<1>(1.0, 0);
    const State total_inflow =
        dg::adphysics::characteristic_total_inflow_boundary<1>(
            inflow_interior, total_pressure, total_temperature, 0.0,
            -1.0, 0.0, gas);
    assert(total_inflow.rho.v > 0.0);
    assert(total_inflow.p.v > 0.0);
    assert(close(total_inflow.nut.v, gas.freestream_nu_tilde));
    assert(std::isfinite(total_inflow.rho.d[0]));
    assert(std::isfinite(total_inflow.u.d[0]));
    assert(std::isfinite(total_inflow.p.d[0]));

    const double epsilon = 1e-6;
    const auto inflow_at_density = [&](double density) {
        State sample = state(density, 0.2, 0.0, 1.0, 5.0e-6);
        return dg::adphysics::characteristic_total_inflow_boundary<1>(
            sample, total_pressure, total_temperature, 0.0,
            -1.0, 0.0, gas);
    };
    const State inflow_plus = inflow_at_density(1.0 + epsilon);
    const State inflow_minus = inflow_at_density(1.0 - epsilon);
    const double rho_derivative_fd =
        (inflow_plus.rho.v - inflow_minus.rho.v) / (2.0 * epsilon);
    const double u_derivative_fd =
        (inflow_plus.u.v - inflow_minus.u.v) / (2.0 * epsilon);
    const double p_derivative_fd =
        (inflow_plus.p.v - inflow_minus.p.v) / (2.0 * epsilon);
    assert(close(total_inflow.rho.d[0], rho_derivative_fd, 2e-4));
    assert(close(total_inflow.u.d[0], u_derivative_fd, 2e-4));
    assert(close(total_inflow.p.d[0], p_derivative_fd, 2e-4));

    AD un, ut, u, v;
    dg::adphysics::velocity_to_normal_tangent<1>(AD(2.0), AD(-1.0), 0.6, 0.8, un, ut);
    dg::adphysics::normal_tangent_to_velocity<1>(un, ut, 0.6, 0.8, u, v);
    assert(close(u.v, 2.0));
    assert(close(v.v, -1.0));
}
