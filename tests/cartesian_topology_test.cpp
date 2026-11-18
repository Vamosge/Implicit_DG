#include "dg/cartesian_topology.hpp"

#include <algorithm>
#include <cstdlib>

int main()
{
    const dg::CartesianTopology topology(8, 6);
    const int center = topology.cell_id(3, 3);
    const auto interior = topology.manhattan_neighbors(center, 2);
    if (interior.size() != 13U) return EXIT_FAILURE;
    if (std::find(interior.begin(), interior.end(), center) == interior.end()) return EXIT_FAILURE;

    const auto corner = topology.manhattan_neighbors(topology.cell_id(0, 0), 2);
    if (corner.size() != 6U) return EXIT_FAILURE;
    if (topology.cell_i(topology.cell_id(7, 5)) != 7) return EXIT_FAILURE;
    if (topology.cell_j(topology.cell_id(7, 5)) != 5) return EXIT_FAILURE;

    const dg::CartesianTopology c_grid(224, 64, 48);
    const auto lower_wake = c_grid.face_neighbor(0, 0, 0);
    if (!lower_wake.interior || lower_wake.i != 223 || lower_wake.j != 0
        || lower_wake.face != 0 || !lower_wake.reverse_quadrature) return EXIT_FAILURE;
    const auto upper_wake = c_grid.face_neighbor(223, 0, 0);
    if (!upper_wake.interior || upper_wake.i != 0 || !upper_wake.reverse_quadrature)
        return EXIT_FAILURE;
    if (c_grid.face_neighbor(48, 0, 0).interior) return EXIT_FAILURE;
    const auto wake_stencil = c_grid.manhattan_neighbors(c_grid.cell_id(0, 0), 1);
    if (std::find(wake_stencil.begin(), wake_stencil.end(),
            c_grid.cell_id(223, 0)) == wake_stencil.end()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
