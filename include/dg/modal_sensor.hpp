#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace dg {

struct Q2ModalEnergy {
    double mean = 0.0;
    double fluctuation_energy = 0.0;
    double highest_order_energy = 0.0;
    double highest_order_fraction = 0.0;
    double highest_order_rms_over_mean = 0.0;
};

inline std::array<double, 3> q2_nodal_to_legendre_1d(
    const std::array<double, 3>& values)
{
    return {
        (values[0] + 4.0 * values[1] + values[2]) / 6.0,
        0.5 * (values[2] - values[0]),
        (values[0] - 2.0 * values[1] + values[2]) / 3.0
    };
}

inline std::array<double, 9> q2_remove_degree_two_modes(
    const double* nodal_values)
{
    double r_coefficients[3][3] = {};
    for (int b = 0; b < 3; ++b) {
        const auto row = q2_nodal_to_legendre_1d({
            nodal_values[3 * b], nodal_values[3 * b + 1],
            nodal_values[3 * b + 2] });
        for (int a = 0; a < 3; ++a) r_coefficients[b][a] = row[a];
    }

    double coefficients[3][3] = {};
    for (int a = 0; a < 3; ++a) {
        const auto column = q2_nodal_to_legendre_1d({
            r_coefficients[0][a], r_coefficients[1][a],
            r_coefficients[2][a] });
        for (int b = 0; b < 3; ++b) coefficients[b][a] = column[b];
    }
    for (int index = 0; index < 3; ++index) {
        coefficients[2][index] = 0.0;
        coefficients[index][2] = 0.0;
    }

    constexpr double nodes[3] = {-1.0, 0.0, 1.0};
    std::array<double, 9> filtered{};
    for (int b_node = 0; b_node < 3; ++b_node) {
        const double s = nodes[b_node];
        const double ps[3] = {1.0, s, 0.5 * (3.0 * s * s - 1.0)};
        for (int a_node = 0; a_node < 3; ++a_node) {
            const double r = nodes[a_node];
            const double pr[3] = {1.0, r, 0.5 * (3.0 * r * r - 1.0)};
            double value = 0.0;
            for (int b = 0; b < 3; ++b) {
                for (int a = 0; a < 3; ++a) {
                    value += coefficients[b][a] * pr[a] * ps[b];
                }
            }
            filtered[3 * b_node + a_node] = value;
        }
    }
    return filtered;
}

// Reference-cell sensor for a Q2 tensor polynomial. Nodal values use the
// solver ordering k = 3*b + a at r,s = {-1,0,1}. Modes with degree two in
// either direction are classified as highest order. Legendre orthogonality
// makes the reported energy independent of the nodal basis scaling.
inline Q2ModalEnergy q2_modal_energy(const double* nodal_values)
{
    double r_coefficients[3][3] = {};
    for (int b = 0; b < 3; ++b) {
        const auto row = q2_nodal_to_legendre_1d({
            nodal_values[3 * b], nodal_values[3 * b + 1],
            nodal_values[3 * b + 2] });
        for (int a = 0; a < 3; ++a) r_coefficients[b][a] = row[a];
    }

    double coefficients[3][3] = {};
    for (int a = 0; a < 3; ++a) {
        const auto column = q2_nodal_to_legendre_1d({
            r_coefficients[0][a], r_coefficients[1][a],
            r_coefficients[2][a] });
        for (int b = 0; b < 3; ++b) coefficients[b][a] = column[b];
    }

    Q2ModalEnergy result;
    result.mean = coefficients[0][0];
    for (int b = 0; b < 3; ++b) {
        for (int a = 0; a < 3; ++a) {
            if (a == 0 && b == 0) continue;
            const double norm = 4.0
                / ((2.0 * a + 1.0) * (2.0 * b + 1.0));
            const double energy = coefficients[b][a]
                * coefficients[b][a] * norm;
            result.fluctuation_energy += energy;
            if (a == 2 || b == 2) result.highest_order_energy += energy;
        }
    }
    if (result.fluctuation_energy > 0.0) {
        result.highest_order_fraction = std::clamp(
            result.highest_order_energy / result.fluctuation_energy,
            0.0, 1.0);
    }
    const double mean_l2 = 2.0 * std::abs(result.mean);
    result.highest_order_rms_over_mean = std::sqrt(
        std::max(0.0, result.highest_order_energy))
        / std::max(mean_l2, 1e-300);
    return result;
}

} // namespace dg
