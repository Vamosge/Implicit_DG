#include "dg/gmsh_reader.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) return EXIT_FAILURE;
    try {
        const auto grid = dg::read_logically_structured_gmsh_2d(argv[1], 225, 65);
        if (grid.ni != 225 || grid.nj != 65) return EXIT_FAILURE;
        if (grid.wall_i_begin != 48 || grid.wall_i_end != 176) return EXIT_FAILURE;
        const auto index = [](int i, int j) { return static_cast<std::size_t>(j * 225 + i); };
        if (std::abs(grid.x[index(48, 0)] - 1.0) > 1e-10) return EXIT_FAILURE;
        if (std::abs(grid.x[index(112, 0)]) > 1e-10) return EXIT_FAILURE;
        if (std::abs(grid.x[index(176, 0)] - 1.0) > 1e-10) return EXIT_FAILURE;
        // The cut has separate logical vertices with coincident coordinates.
        if (std::abs(grid.x[index(0, 0)] - grid.x[index(224, 0)]) > 1e-10) return EXIT_FAILURE;

        const auto coarse = dg::read_logically_structured_gmsh_2d(
            argv[1], 113, 33, 2, 225, 65);
        if (coarse.ni != 113 || coarse.nj != 33
            || coarse.wall_i_begin != 24 || coarse.wall_i_end != 88) return EXIT_FAILURE;
        const auto coarse_index = [](int i, int j) {
            return static_cast<std::size_t>(j * 113 + i);
        };
        if (std::abs(coarse.x[coarse_index(24, 0)] - 1.0) > 1e-10) return EXIT_FAILURE;
        if (std::abs(coarse.x[coarse_index(56, 0)]) > 1e-10) return EXIT_FAILURE;
        if (std::abs(coarse.x[coarse_index(88, 0)] - 1.0) > 1e-10) return EXIT_FAILURE;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
