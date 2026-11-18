#include "dg/block_ad.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

bool close(double a, double b)
{
    return std::abs(a - b) < 1e-13;
}

} // namespace

int main()
{
    using AD = dg::BlockAD<3>;
    const AD x = dg::make_seed<3>(2.0, 0);
    const AD y = dg::make_seed<3>(3.0, 1);
    const AD z = dg::make_seed<3>(0.5, 2);

    // f = x*y + x/y - 2*z + exp(z) + sqrt(x) + y^2
    const AD f = x * y + x / y - 2.0 * z + dg::ad_exp(z)
        + dg::ad_sqrt(x) + dg::ad_pow(y, 2.0);
    const double expected_value = 6.0 + 2.0 / 3.0 - 1.0
        + std::exp(0.5) + std::sqrt(2.0) + 9.0;
    const double dfdx = 3.0 + 1.0 / 3.0 + 0.5 / std::sqrt(2.0);
    const double dfdy = 2.0 - 2.0 / 9.0 + 6.0;
    const double dfdz = -2.0 + std::exp(0.5);

    if (!close(f.v, expected_value) || !close(f.d[0], dfdx)
        || !close(f.d[1], dfdy) || !close(f.d[2], dfdz)) {
        std::cerr << "BlockAD chain-rule check failed\n";
        return EXIT_FAILURE;
    }

    AD accumulated(1.0);
    accumulated += x;
    accumulated -= y;
    accumulated += 4.0 * z;
    if (!close(accumulated.v, 2.0) || !close(accumulated.d[0], 1.0)
        || !close(accumulated.d[1], -1.0) || !close(accumulated.d[2], 4.0)) {
        std::cerr << "BlockAD accumulation check failed\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
