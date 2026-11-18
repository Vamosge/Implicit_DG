#include "dg/stencil_coloring.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    for (int di = -4; di <= 4; ++di) {
        for (int dj = -4; dj <= 4; ++dj) {
            if (di == 0 && dj == 0) continue;
            if (std::abs(di) + std::abs(dj) > 4) continue;
            if (dg::radius_two_color(0, 0) == dg::radius_two_color(di, dj)) {
                std::cerr << "color collision at offset (" << di << "," << dj << ")\n";
                return EXIT_FAILURE;
            }
        }
    }

    // A synthetic C-grid wake row. Cells 1 and 110 have the same fixed
    // Cartesian color, so the general topology-derived coloring must separate
    // them when they occur in one residual stencil.
    std::vector<std::vector<int>> stencils(112);
    for (int cell = 0; cell < 112; ++cell) stencils[cell] = {cell};
    stencils[0] = {0, 1, 110, 111};
    dg::StencilColoring coloring = dg::greedy_stencil_coloring(stencils);
    dg::audit_stencil_coloring(stencils, coloring);
    if (coloring.collision_rows != 0 || coloring.collision_pairs != 0)
        return EXIT_FAILURE;
    if (coloring.cell_color[1] == coloring.cell_color[110])
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
