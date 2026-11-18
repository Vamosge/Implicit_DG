#include "dg/modal_sensor.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(double a, double b, double tolerance = 1e-13)
{
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

int main()
{
    double constant[9];
    for (double& value : constant) value = 3.0;
    const auto constant_energy = dg::q2_modal_energy(constant);
    if (!close(constant_energy.mean, 3.0)
        || !close(constant_energy.fluctuation_energy, 0.0)
        || !close(constant_energy.highest_order_fraction, 0.0)) {
        std::cerr << "constant Q2 modal decomposition failed\n";
        return 1;
    }

    const double nodes[3] = { -1.0, 0.0, 1.0 };
    double linear[9] = {};
    double mixed[9] = {};
    for (int b = 0; b < 3; ++b) {
        for (int a = 0; a < 3; ++a) {
            linear[3 * b + a] = 1.0 + 2.0 * nodes[a] - 3.0 * nodes[b];
            const double p2 = 0.5 * (3.0 * nodes[a] * nodes[a] - 1.0);
            mixed[3 * b + a] = 1.0 + 0.5 * nodes[a] + 2.0 * p2;
        }
    }
    const auto linear_energy = dg::q2_modal_energy(linear);
    if (!close(linear_energy.highest_order_energy, 0.0)
        || !close(linear_energy.highest_order_fraction, 0.0)) {
        std::cerr << "linear field leaked into Q2 modes\n";
        return 1;
    }

    const auto mixed_energy = dg::q2_modal_energy(mixed);
    const double expected_linear_energy = 1.0 / 3.0;
    const double expected_q2_energy = 16.0 / 5.0;
    const double expected_fraction = expected_q2_energy
        / (expected_linear_energy + expected_q2_energy);
    if (!close(mixed_energy.mean, 1.0)
        || !close(mixed_energy.highest_order_energy, expected_q2_energy)
        || !close(mixed_energy.highest_order_fraction, expected_fraction)) {
        std::cerr << "mixed Q1/Q2 modal energy is incorrect\n";
        return 1;
    }
    const auto filtered = dg::q2_remove_degree_two_modes(mixed);
    const auto filtered_energy = dg::q2_modal_energy(filtered.data());
    if (!close(filtered_energy.mean, 1.0)
        || !close(filtered_energy.highest_order_energy, 0.0)
        || !close(filtered[0], 0.5)
        || !close(filtered[2], 1.5)) {
        std::cerr << "Q2 degree-two removal failed\n";
        return 1;
    }
    return 0;
}
