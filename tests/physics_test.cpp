#include "dg/physics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

bool close(double a, double b, double tolerance = 1e-13)
{
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

int main()
{
    constexpr double gamma = 1.4;
    const dg::PrimitiveState primitive{1.2, 0.7, -0.2, 0.9, 3.0e-5};
    double conservative[5] = {};
    dg::primitive_to_conservative(primitive, gamma, conservative);
    const dg::PrimitiveState recovered =
        dg::conservative_to_primitive(conservative, gamma);
    assert(close(recovered.rho, primitive.rho));
    assert(close(recovered.u, primitive.u));
    assert(close(recovered.v, primitive.v));
    assert(close(recovered.p, primitive.p));
    assert(close(recovered.nut, primitive.nut));

    double reflected[5] = {};
    dg::reflect_normal(conservative, 1.0, 0.0, reflected);
    assert(close(reflected[1], -conservative[1]));
    assert(close(reflected[2], conservative[2]));
    assert(close(reflected[3], conservative[3]));
    assert(close(reflected[4], conservative[4]));

    double flux[4] = {};
    dg::hllc_euler_flux(conservative, conservative, 1.0, 0.0, gamma, flux);
    assert(close(flux[0], conservative[0] * primitive.u));
    assert(close(flux[1], conservative[1] * primitive.u + primitive.p));
    assert(close(flux[2], conservative[2] * primitive.u));
    assert(close(flux[3], (conservative[3] + primitive.p) * primitive.u));

    double un = 0.0, ut = 0.0, u = 0.0, v = 0.0;
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    dg::velocity_to_normal_tangent(primitive.u, primitive.v,
        inv_sqrt_two, inv_sqrt_two, un, ut);
    dg::normal_tangent_to_velocity(un, ut,
        inv_sqrt_two, inv_sqrt_two, u, v);
    assert(close(u, primitive.u));
    assert(close(v, primitive.v));
}
