#include "dg/reference_element.hpp"

#include <cassert>
#include <cmath>

static bool close(double a, double b, double tolerance = 2e-14)
{
    return std::abs(a - b) <= tolerance;
}

int main()
{
    double points[5] = {}, weights[5] = {};
    dg::gauss_legendre_1d<5>(points, weights);
    for (int degree = 0; degree <= 9; ++degree) {
        double integral = 0.0;
        for (int q = 0; q < 5; ++q) {
            integral += weights[q] * std::pow(points[q], degree);
        }
        const double exact = degree % 2 == 0 ? 2.0 / (degree + 1) : 0.0;
        assert(close(integral, exact));
    }

    double phi[9] = {}, dr[9] = {}, ds[9] = {};
    dg::tensor_lagrange_2d<2>(0.173, -0.421, phi, dr, ds);
    double sum_phi = 0.0, sum_dr = 0.0, sum_ds = 0.0;
    for (int k = 0; k < 9; ++k) {
        sum_phi += phi[k]; sum_dr += dr[k]; sum_ds += ds[k];
    }
    assert(close(sum_phi, 1.0));
    assert(close(sum_dr, 0.0));
    assert(close(sum_ds, 0.0));

    for (int node = 0; node < 9; ++node) {
        double r = 0.0, s = 0.0;
        dg::reference_node_coords_2d<2>(node, r, s);
        dg::tensor_lagrange_2d<2>(r, s, phi);
        for (int k = 0; k < 9; ++k) {
            assert(close(phi[k], k == node ? 1.0 : 0.0));
        }
    }
}
