#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dg {

inline constexpr int radius_two_color_count = 13;

// Perfect Lee-code coloring for a two-dimensional Manhattan-radius-two
// stencil. Equal colors are separated by at least five grid steps.
inline constexpr int radius_two_color(int i, int j)
{
    const int residue = (i + 5 * j) % radius_two_color_count;
    return residue < 0 ? residue + radius_two_color_count : residue;
}

struct StencilColoring {
    std::vector<int> cell_color;
    int color_count = 0;
    int collision_rows = 0;
    int collision_pairs = 0;
};

// Build the column-conflict graph induced by residual-row stencils. Two
// columns conflict when they occur together in any row stencil, because their
// compressed forward-AD derivatives would otherwise be added together.
inline StencilColoring greedy_stencil_coloring(
    const std::vector<std::vector<int>>& row_stencils)
{
    const int cell_count = static_cast<int>(row_stencils.size());
    std::vector<std::vector<int>> conflicts(static_cast<std::size_t>(cell_count));
    for (const auto& stencil : row_stencils) {
        for (std::size_t a = 0; a < stencil.size(); ++a) {
            for (std::size_t b = a + 1; b < stencil.size(); ++b) {
                const int left = stencil[a];
                const int right = stencil[b];
                if (left < 0 || left >= cell_count || right < 0 || right >= cell_count
                    || left == right) continue;
                conflicts[static_cast<std::size_t>(left)].push_back(right);
                conflicts[static_cast<std::size_t>(right)].push_back(left);
            }
        }
    }
    for (auto& row : conflicts) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    StencilColoring result;
    result.cell_color.assign(static_cast<std::size_t>(cell_count), -1);
    std::vector<unsigned char> forbidden;
    for (int cell = 0; cell < cell_count; ++cell) {
        forbidden.assign(static_cast<std::size_t>(result.color_count + 1), 0u);
        for (int other : conflicts[static_cast<std::size_t>(cell)]) {
            if (other >= cell) continue;
            const int color = result.cell_color[static_cast<std::size_t>(other)];
            if (color >= 0 && color < static_cast<int>(forbidden.size())) {
                forbidden[static_cast<std::size_t>(color)] = 1u;
            }
        }
        int color = 0;
        while (color < static_cast<int>(forbidden.size())
            && forbidden[static_cast<std::size_t>(color)] != 0u) ++color;
        result.cell_color[static_cast<std::size_t>(cell)] = color;
        result.color_count = std::max(result.color_count, color + 1);
    }
    return result;
}

inline void audit_stencil_coloring(const std::vector<std::vector<int>>& row_stencils,
    StencilColoring& coloring)
{
    coloring.collision_rows = 0;
    coloring.collision_pairs = 0;
    for (const auto& stencil : row_stencils) {
        bool row_collision = false;
        for (std::size_t a = 0; a < stencil.size(); ++a) {
            for (std::size_t b = a + 1; b < stencil.size(); ++b) {
                const int left = stencil[a];
                const int right = stencil[b];
                if (left < 0 || right < 0
                    || left >= static_cast<int>(coloring.cell_color.size())
                    || right >= static_cast<int>(coloring.cell_color.size())) continue;
                if (coloring.cell_color[static_cast<std::size_t>(left)]
                    == coloring.cell_color[static_cast<std::size_t>(right)]) {
                    ++coloring.collision_pairs;
                    row_collision = true;
                }
            }
        }
        if (row_collision) ++coloring.collision_rows;
    }
}

} // namespace dg
