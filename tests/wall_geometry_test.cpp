#include "dg/wall_geometry.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

int main()
{
    dg::WallGeometry wall;
    wall.set_profile({-1.0, 0.0, 0.5, 1.0, 2.0},
        {0.0, 0.0, 0.1, 0.0, 0.0}, 1, 3);

    if (std::abs(wall.x_begin()) > 1e-14 || std::abs(wall.x_end() - 1.0) > 1e-14) {
        return EXIT_FAILURE;
    }
    if (std::abs(wall.height(0.25) - 0.05) > 1e-14) return EXIT_FAILURE;
    if (std::abs(wall.distance(0.5, 0.2) - 0.1) > 1e-14) return EXIT_FAILURE;
    if (std::abs(wall.distance(-0.3, 0.4) - 0.5) > 1e-14) return EXIT_FAILURE;
    if (wall.distance(0.5, 0.1, 1e-6) != 1e-6) return EXIT_FAILURE;

    dg::WallGeometry airfoil;
    airfoil.set_polyline({1.0, 0.0, 1.0}, {-0.05, 0.0, 0.05});
    if (std::abs(airfoil.distance(0.0, 0.0, 1e-6) - 1e-6) > 1e-14)
        return EXIT_FAILURE;
    if (std::abs(airfoil.distance(0.5, 0.0)
            - std::sqrt(0.025 * 0.025 / 1.0025)) > 1e-12)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
