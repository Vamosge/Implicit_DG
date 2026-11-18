#include "dg/cgns_reader.hpp"
#include "dg/cartesian_topology.hpp"
#include "dg/physics.hpp"
#include "dg/reference_element.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

struct FaceMetric {
    double x = 0.0;
    double y = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    double measure = 0.0;
};

// Reconstruct the solver's order-four geometry on the j=0 face of a coarse
// cell.  This deliberately duplicates the face-metric convention used by the
// solver so the test audits the complete curved wake seam, not only vertices.
FaceMetric lower_face_metric(const dg::StructuredGrid2D& fine, int coarse_i,
    double r)
{
    constexpr int geometry_order = 4;
    constexpr int node_count = geometry_order + 1;
    double nodes[node_count] = {};
    double basis[node_count] = {};
    double derivative[node_count] = {};
    dg::reference_nodes_1d<geometry_order>(nodes);
    dg::lagrange_1d<node_count>(r, nodes, basis, derivative);

    FaceMetric metric;
    double xr = 0.0;
    double yr = 0.0;
    for (int a = 0; a < node_count; ++a) {
        const std::size_t point = static_cast<std::size_t>(
            geometry_order * coarse_i + a);
        metric.x += fine.x[point] * basis[a];
        metric.y += fine.y[point] * basis[a];
        xr += fine.x[point] * derivative[a];
        yr += fine.y[point] * derivative[a];
    }
    metric.measure = std::hypot(xr, yr);
    if (!(metric.measure > 0.0) || !std::isfinite(metric.measure)) {
        throw std::runtime_error("degenerate curved face on the C-grid wake cut");
    }
    // Solver face 0 convention: (nx ds, ny ds) = (yr, -xr).
    metric.nx = yr / metric.measure;
    metric.ny = -xr / metric.measure;
    return metric;
}

double relative_error(double a, double b)
{
    return std::abs(a - b) / std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: naca_cgns_mesh_test TOPOLOGY.cgns GEOMETRY.cgns");
        }
        constexpr int coarse_ni = 113;
        constexpr int coarse_nj = 33;
        constexpr int refinement = 4;
        constexpr int fine_ni = refinement * (coarse_ni - 1) + 1;
        constexpr int fine_nj = refinement * (coarse_nj - 1) + 1;
        const dg::StructuredGrid2D coarse =
            dg::read_structured_cgns_2d(argv[1], coarse_ni, coarse_nj);
        const dg::StructuredGrid2D fine =
            dg::read_structured_cgns_2d(argv[2], fine_ni, fine_nj);

        if (coarse.wall_i_begin != 24 || coarse.wall_i_end != 88) {
            throw std::runtime_error("unexpected NACA viscous-wall index range");
        }
        double max_nested_error = 0.0;
        for (int j = 0; j < coarse_nj; ++j) {
            for (int i = 0; i < coarse_ni; ++i) {
                const size_t c = static_cast<size_t>(j) * coarse_ni + i;
                const size_t f = static_cast<size_t>(refinement * j) * fine_ni
                    + refinement * i;
                max_nested_error = std::max(max_nested_error,
                    std::hypot(coarse.x[c] - fine.x[f], coarse.y[c] - fine.y[f]));
            }
        }
        if (max_nested_error > 1e-9) {
            throw std::runtime_error("topology and Q4 geometry grids are not nested");
        }

        // The two sides of the 24-cell C-grid wake cut must coincide with
        // reversed logical orientation so the structured wake pairing is valid.
        double max_wake_error = 0.0;
        for (int i = 0; i <= refinement * 24; ++i) {
            const int opposite = fine_ni - 1 - i;
            max_wake_error = std::max(max_wake_error,
                std::hypot(fine.x[static_cast<size_t>(i)] - fine.x[static_cast<size_t>(opposite)],
                    fine.y[static_cast<size_t>(i)] - fine.y[static_cast<size_t>(opposite)]));
        }
        if (max_wake_error > 1e-9) {
            throw std::runtime_error("opposite sides of the C-grid wake cut do not coincide");
        }

        // Audit the actual curved-face quadrature convention used by the DG
        // residual.  A valid C-grid seam has coincident points and measures,
        // opposite outward normals, and equal-and-opposite numerical fluxes.
        constexpr int coarse_nx = coarse_ni - 1;
        constexpr int coarse_ny = coarse_nj - 1;
        constexpr int wake_pairs = 24;
        constexpr int face_quadrature = 5;
        const dg::CartesianTopology topology(coarse_nx, coarse_ny, wake_pairs);
        double gauss[face_quadrature] = {};
        double weights[face_quadrature] = {};
        dg::gauss_legendre_1d<face_quadrature>(gauss, weights);

        const dg::PrimitiveState primitive_left{1.07, 0.19, -0.04, 0.91, 2e-5};
        const dg::PrimitiveState primitive_right{0.96, 0.11, 0.03, 1.04, 7e-6};
        double state_left[5] = {};
        double state_right[5] = {};
        dg::primitive_to_conservative(primitive_left, 1.4, state_left);
        dg::primitive_to_conservative(primitive_right, 1.4, state_right);

        double max_face_coordinate_error = 0.0;
        double max_face_measure_error = 0.0;
        double max_opposite_normal_error = 0.0;
        double max_flux_cancellation_error = 0.0;
        double max_lifting_moment_error = 0.0;
        for (int i = 0; i < wake_pairs; ++i) {
            const auto neighbor = topology.face_neighbor(i, 0, 0);
            if (!neighbor.interior || neighbor.face != 0
                || !neighbor.reverse_quadrature
                || neighbor.i != coarse_nx - 1 - i || neighbor.j != 0) {
                throw std::runtime_error("unexpected C-grid wake topology mapping");
            }
            // BR2 lifting RHS for a nonconstant physical jump.  The right
            // trace has the opposite jump and normal, while its local r basis
            // is reversed.  Corresponding modal moments must therefore agree.
            std::array<double, 9> lifting_left_x = {};
            std::array<double, 9> lifting_left_y = {};
            std::array<double, 9> lifting_right_x = {};
            std::array<double, 9> lifting_right_y = {};
            for (int g = 0; g < face_quadrature; ++g) {
                const FaceMetric left = lower_face_metric(fine, i, gauss[g]);
                const FaceMetric right = lower_face_metric(
                    fine, neighbor.i, gauss[face_quadrature - 1 - g]);
                max_face_coordinate_error = std::max(max_face_coordinate_error,
                    std::hypot(left.x - right.x, left.y - right.y));
                max_face_measure_error = std::max(max_face_measure_error,
                    relative_error(left.measure, right.measure));
                max_opposite_normal_error = std::max(max_opposite_normal_error,
                    std::hypot(left.nx + right.nx, left.ny + right.ny));

                double left_flux[4] = {};
                double right_flux[4] = {};
                dg::hllc_euler_flux(state_left, state_right,
                    left.nx, left.ny, 1.4, left_flux);
                dg::hllc_euler_flux(state_right, state_left,
                    right.nx, right.ny, 1.4, right_flux);
                for (int equation = 0; equation < 4; ++equation) {
                    const double scaled_left = left_flux[equation] * left.measure;
                    const double scaled_right = right_flux[equation] * right.measure;
                    const double cancellation = std::abs(scaled_left + scaled_right)
                        / std::max({1.0, std::abs(scaled_left), std::abs(scaled_right)});
                    max_flux_cancellation_error = std::max(
                        max_flux_cancellation_error, cancellation);
                }

                double basis_left[9] = {};
                double basis_right[9] = {};
                dg::tensor_lagrange_2d<2>(gauss[g], -1.0, basis_left);
                dg::tensor_lagrange_2d<2>(
                    gauss[face_quadrature - 1 - g], -1.0, basis_right);
                const double physical_jump = 0.17 + 0.31 * left.x
                    - 0.23 * left.y + 0.07 * left.x * left.x;
                for (int k = 0; k < 9; ++k) {
                    lifting_left_x[k] += weights[g] * physical_jump
                        * left.nx * basis_left[k] * left.measure;
                    lifting_left_y[k] += weights[g] * physical_jump
                        * left.ny * basis_left[k] * left.measure;
                    lifting_right_x[k] += weights[g] * (-physical_jump)
                        * right.nx * basis_right[k] * right.measure;
                    lifting_right_y[k] += weights[g] * (-physical_jump)
                        * right.ny * basis_right[k] * right.measure;
                }
            }
            for (int b = 0; b < 3; ++b) {
                for (int a = 0; a < 3; ++a) {
                    const int left_k = 3 * b + a;
                    const int right_k = 3 * b + (2 - a);
                    max_lifting_moment_error = std::max(max_lifting_moment_error,
                        relative_error(lifting_left_x[left_k], lifting_right_x[right_k]));
                    max_lifting_moment_error = std::max(max_lifting_moment_error,
                        relative_error(lifting_left_y[left_k], lifting_right_y[right_k]));
                }
            }
        }
        if (max_face_coordinate_error > 1e-9) {
            throw std::runtime_error("wake quadrature points are not coincident");
        }
        if (max_face_measure_error > 1e-10) {
            throw std::runtime_error("wake face measures do not match");
        }
        if (max_opposite_normal_error > 1e-10) {
            throw std::runtime_error("wake face normals are not opposite");
        }
        if (max_flux_cancellation_error > 1e-10) {
            throw std::runtime_error("wake HLLC flux is not conservative");
        }
        if (max_lifting_moment_error > 1e-10) {
            throw std::runtime_error("wake BR2 lifting orientation is inconsistent");
        }

        const auto x_bounds = std::minmax_element(coarse.x.begin(), coarse.x.end());
        const auto y_bounds = std::minmax_element(coarse.y.begin(), coarse.y.end());
        if (*x_bounds.first > -400.0 || *x_bounds.second < 400.0
            || *y_bounds.first > -400.0 || *y_bounds.second < 400.0) {
            throw std::runtime_error("NASA farfield extent was not recovered");
        }
        std::cout << "coarse=" << coarse.ni << "x" << coarse.nj
            << " fine=" << fine.ni << "x" << fine.nj
            << " wall_i=" << coarse.wall_i_begin << ".." << coarse.wall_i_end
            << " nested_error=" << max_nested_error
            << " wake_error=" << max_wake_error
            << " wake_quad_error=" << max_face_coordinate_error
            << " wake_measure_error=" << max_face_measure_error
            << " wake_normal_error=" << max_opposite_normal_error
            << " wake_flux_error=" << max_flux_cancellation_error
            << " wake_lifting_error=" << max_lifting_moment_error << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "naca_cgns_mesh_test: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
