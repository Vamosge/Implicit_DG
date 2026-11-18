#pragma once

#include <cmath>
#include <stdexcept>

namespace dg {

// Equispaced nodal coordinates on the reference interval [-1, 1]. The
// explicit low-order values preserve the arithmetic of the original solver.
template <int Order>
inline void reference_nodes_1d(double* nodes)
{
    static_assert(Order >= 1, "A reference element needs positive order");
    if constexpr (Order == 1) {
        nodes[0] = -1.0; nodes[1] = 1.0;
    }
    else if constexpr (Order == 2) {
        nodes[0] = -1.0; nodes[1] = 0.0; nodes[2] = 1.0;
    }
    else if constexpr (Order == 3) {
        nodes[0] = -1.0; nodes[1] = -1.0 / 3.0;
        nodes[2] = 1.0 / 3.0; nodes[3] = 1.0;
    }
    else {
        for (int n = 0; n <= Order; ++n) {
            nodes[n] = -1.0 + 2.0 * static_cast<double>(n) /
                static_cast<double>(Order);
        }
    }
}

template <int PointCount>
inline void gauss_legendre_1d(double* points, double* weights)
{
    static_assert(PointCount >= 2 && PointCount <= 5,
        "Implemented Gauss-Legendre rules contain 2 through 5 points");
    if constexpr (PointCount == 2) {
        const double a = 1.0 / std::sqrt(3.0);
        points[0] = -a; points[1] = a;
        weights[0] = 1.0; weights[1] = 1.0;
    }
    else if constexpr (PointCount == 3) {
        points[0] = -0.7745966692414834; points[1] = 0.0;
        points[2] = 0.7745966692414834;
        weights[0] = 0.5555555555555556; weights[1] = 0.8888888888888888;
        weights[2] = 0.5555555555555556;
    }
    else if constexpr (PointCount == 4) {
        points[0] = -0.8611363115940526; points[1] = -0.3399810435848563;
        points[2] = 0.3399810435848563; points[3] = 0.8611363115940526;
        weights[0] = 0.3478548451374539; weights[1] = 0.6521451548625461;
        weights[2] = 0.6521451548625461; weights[3] = 0.3478548451374539;
    }
    else if constexpr (PointCount == 5) {
        points[0] = -0.9061798459386640; points[1] = -0.5384693101056831;
        points[2] = 0.0; points[3] = 0.5384693101056831;
        points[4] = 0.9061798459386640;
        weights[0] = 0.2369268850561891; weights[1] = 0.4786286704993665;
        weights[2] = 0.5688888888888889; weights[3] = 0.4786286704993665;
        weights[4] = 0.2369268850561891;
    }
}

template <int NodeCount>
inline void lagrange_1d(double x, const double* nodes, double* values,
    double* derivatives)
{
    static_assert(NodeCount >= 2, "At least two interpolation nodes are required");
    for (int i = 0; i < NodeCount; ++i) {
        double value = 1.0;
        for (int j = 0; j < NodeCount; ++j) {
            if (j != i) value *= (x - nodes[j]) / (nodes[i] - nodes[j]);
        }
        values[i] = value;

        double derivative = 0.0;
        for (int m = 0; m < NodeCount; ++m) {
            if (m == i) continue;
            double term = 1.0 / (nodes[i] - nodes[m]);
            for (int j = 0; j < NodeCount; ++j) {
                if (j != i && j != m) {
                    term *= (x - nodes[j]) / (nodes[i] - nodes[j]);
                }
            }
            derivative += term;
        }
        derivatives[i] = derivative;
    }
}

template <int Order>
inline void tensor_lagrange_2d(double r, double s, double* values,
    double* derivatives_r = nullptr, double* derivatives_s = nullptr)
{
    constexpr int node_count = Order + 1;
    double nodes[node_count] = {};
    double lr[node_count] = {}, dr[node_count] = {};
    double ls[node_count] = {}, ds[node_count] = {};
    reference_nodes_1d<Order>(nodes);
    lagrange_1d<node_count>(r, nodes, lr, dr);
    lagrange_1d<node_count>(s, nodes, ls, ds);
    int k = 0;
    for (int b = 0; b < node_count; ++b) {
        for (int a = 0; a < node_count; ++a, ++k) {
            values[k] = lr[a] * ls[b];
            if (derivatives_r) derivatives_r[k] = dr[a] * ls[b];
            if (derivatives_s) derivatives_s[k] = lr[a] * ds[b];
        }
    }
}

template <int Order>
inline void reference_node_coords_2d(int index, double& r, double& s)
{
    constexpr int node_count = Order + 1;
    if (index < 0 || index >= node_count * node_count) {
        throw std::out_of_range("reference node index is outside the tensor element");
    }
    double nodes[node_count] = {};
    reference_nodes_1d<Order>(nodes);
    r = nodes[index % node_count];
    s = nodes[index / node_count];
}

} // namespace dg
