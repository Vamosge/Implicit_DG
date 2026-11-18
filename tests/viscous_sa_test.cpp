#include "dg/physics.hpp"
#include "dg/block_ad.hpp"
#include "dg/viscous_sa.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

bool close(double a, double b, double tolerance = 1e-13)
{
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

dg::ViscousSAParameters parameters()
{
    constexpr double cb1 = 0.1355;
    constexpr double cb2 = 0.622;
    constexpr double sigma = 2.0 / 3.0;
    constexpr double kappa = 0.41;
    constexpr double cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;
    return {1.4, 1.0 / 1.4, 2.0e-5, 0.72, 0.9, 110.4 / 300.0,
        sigma, cb1, cb2, kappa, cw1, 0.3, 2.0, 7.1, 1.2, 0.5, 16.0};
}

} // namespace

int main()
{
    const dg::ViscousSAParameters model = parameters();
    const dg::PrimitiveState primitive{1.0, 0.2, 0.0, 1.0 / 1.4, 6.0e-5};
    double U[5] = {};
    dg::primitive_to_conservative(primitive, model.gamma, U);

    const double mu = dg::laminar_viscosity<double, dg::DoubleMath>(U, model);
    assert(close(mu, model.mu_reference));
    const double mu_t = dg::sa_eddy_viscosity<double, dg::DoubleMath>(
        primitive.rho, primitive.nut, mu, model);
    assert(mu_t > 0.0);

    const double zero_gradient[5] = {};
    double Fv[5] = {}, Gv[5] = {};
    dg::viscous_flux<double, dg::DoubleMath>(
        U, zero_gradient, zero_gradient, model, Fv, Gv);
    for (int equation = 0; equation < 5; ++equation) {
        assert(close(Fv[equation], 0.0));
        assert(close(Gv[equation], 0.0));
    }

    const double dUx[5] = {0.01, 0.03, -0.02, 0.04, 2.0e-6};
    const double dUy[5] = {-0.02, 0.01, 0.05, -0.03, -1.0e-6};
    dg::viscous_flux<double, dg::DoubleMath>(U, dUx, dUy, model, Fv, Gv);
    for (int equation = 0; equation < 5; ++equation) {
        assert(std::isfinite(Fv[equation]));
        assert(std::isfinite(Gv[equation]));
    }

    const auto positive = dg::sa_source_terms<double, dg::DoubleMath>(
        U, dUx, dUy, 0.02, 0.02, model);
    assert(std::isfinite(positive.total));
    assert(close(positive.total,
        positive.production - positive.destruction + positive.cross_diffusion));

    // The AD kernel must differentiate the same positive-branch production
    // used by the nonlinear residual.  A historical solver shortcut froze
    // this derivative and produced an inexact "AD Jacobian".
    using AD1 = dg::BlockAD<1>;
    AD1 Uad[5] = {};
    AD1 dUxad[5] = {};
    AD1 dUyad[5] = {};
    for (int equation = 0; equation < 5; ++equation) {
        Uad[equation] = U[equation];
        dUxad[equation] = dUx[equation];
        dUyad[equation] = dUy[equation];
    }
    Uad[4] = dg::make_seed<1>(U[4], 0);
    const auto positive_ad = dg::sa_source_terms<AD1,
        dg::BlockADMath<1>>(Uad, dUxad, dUyad, 0.02, 0.02, model);
    const double perturbation = std::abs(U[4]) * 1e-5;
    double Uplus[5] = {}, Uminus[5] = {};
    std::copy(U, U + 5, Uplus);
    std::copy(U, U + 5, Uminus);
    Uplus[4] += perturbation;
    Uminus[4] -= perturbation;
    const double production_fd =
        (dg::sa_source_terms<double, dg::DoubleMath>(Uplus, dUx, dUy,
            0.02, 0.02, model).production
        - dg::sa_source_terms<double, dg::DoubleMath>(Uminus, dUx, dUy,
            0.02, 0.02, model).production) / (2.0 * perturbation);
    assert(std::abs(positive_ad.production.d[0]) > 1e-12);
    assert(std::abs(positive_ad.production.d[0] - production_fd)
        <= 2e-6 * std::max(1e-12, std::abs(production_fd)));
    assert(close(positive_ad.total.v, positive.total));

    U[4] = -std::abs(U[4]);
    const auto negative = dg::sa_source_terms<double, dg::DoubleMath>(
        U, dUx, dUy, 0.02, 0.02, model);
    assert(std::isfinite(negative.total));
    assert(close(negative.total,
        negative.production - negative.destruction + negative.cross_diffusion));
}
