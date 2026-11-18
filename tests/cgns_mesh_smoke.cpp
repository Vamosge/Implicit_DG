#include "dg/cgns_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    try {
        if (argc != 5) throw std::runtime_error("usage: cgns_mesh_smoke GRID NI NJ bump|plate");
        const int expected_ni = std::stoi(argv[2]);
        const int expected_nj = std::stoi(argv[3]);
        const std::string case_name = argv[4];
        const dg::StructuredGrid2D grid = dg::read_structured_cgns_2d(
            argv[1], expected_ni, expected_nj);
        if (grid.ni != expected_ni || grid.nj != expected_nj) {
            throw std::runtime_error("unexpected extracted grid dimensions");
        }
        if (grid.x.size() != grid.y.size() || grid.x.empty()) {
            throw std::runtime_error("coordinate arrays are empty or inconsistent");
        }
        const auto x_bounds = std::minmax_element(grid.x.begin(), grid.x.end());
        const auto y_bounds = std::minmax_element(grid.y.begin(), grid.y.end());
        const bool bump = case_name == "bump";
        const bool plate = case_name == "plate";
        if (!bump && !plate) throw std::runtime_error("case must be bump or plate");
        const double expected_x_min = bump ? -25.0 : -0.33333;
        const double expected_x_max = bump ? 26.5 : 2.0;
        const double expected_y_max = bump ? 5.0 : 1.0;
        if (std::abs(*x_bounds.first - expected_x_min) > 1e-10
            || std::abs(*x_bounds.second - expected_x_max) > 1e-10
            || std::abs(*y_bounds.first) > 1e-10
            || std::abs(*y_bounds.second - expected_y_max) > 1e-10) {
            throw std::runtime_error("unexpected grid coordinate bounds: x="
                + std::to_string(*x_bounds.first) + ".." + std::to_string(*x_bounds.second)
                + " y=" + std::to_string(*y_bounds.first) + ".."
                + std::to_string(*y_bounds.second));
        }
        if (bump && (grid.wall_i_begin < 0 || grid.wall_i_end < grid.wall_i_begin)) {
            throw std::runtime_error("viscous wall boundary patch was not found");
        }
        int wall_begin = grid.wall_i_begin;
        int wall_end = grid.wall_i_end;
        if (plate) {
            wall_begin = 0;
            double closest_to_zero = std::abs(grid.x[0]);
            for (int i = 1; i < grid.ni; ++i) {
                const double distance = std::abs(grid.x[static_cast<std::size_t>(i)]);
                if (distance < closest_to_zero) {
                    wall_begin = i;
                    closest_to_zero = distance;
                }
            }
            wall_end = grid.ni - 1;
        }
        const double wall_x_begin = grid.x[static_cast<std::size_t>(wall_begin)];
        const double wall_x_end = grid.x[static_cast<std::size_t>(wall_end)];
        const double expected_wall_end = bump ? 1.5 : 2.0;
        if (std::abs(wall_x_begin) > 1e-10
            || std::abs(wall_x_end - expected_wall_end) > 1e-10) {
            throw std::runtime_error("unexpected viscous-wall extent");
        }
        double bump_height = 0.0;
        for (int i = 0; i < grid.ni; ++i) bump_height = std::max(bump_height, grid.y[i]);
        const double expected_height = bump ? 0.05 : 0.0;
        if (std::abs(bump_height - expected_height) > 1e-10) {
            throw std::runtime_error("unexpected wall height");
        }
        std::cout << "mesh=" << grid.ni << "x" << grid.nj
                  << " wall_i=" << wall_begin << ".." << wall_end
                  << " wall_x=" << wall_x_begin << ".." << wall_x_end
                  << " case=" << case_name
                  << " wall_height=" << bump_height << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "cgns_mesh_smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
