#pragma once

#include "dg/linear_solver.hpp"

#include <utility>
#include <vector>

#ifndef DG_HAS_OPENMP
#define DG_HAS_OPENMP 0
#endif

namespace dg::ad {

template <class Field>
struct JacobianWorkspace {
    Field state;
    Field residual;
    Field rhs;
    Field gradient_x;
    Field gradient_y;

    template <class EnsureField>
    void ensure_allocated(EnsureField&& ensure_field) {
        ensure_field(state);
        ensure_field(residual);
        ensure_field(rhs);
        ensure_field(gradient_x);
        ensure_field(gradient_y);
    }
};

// Assemble a sparse block Jacobian with compressed forward-AD seeding.
// Cells sharing a color must have disjoint affected-row sets. The physics
// callback owns residual evaluation; this function owns color orchestration,
// sparse-block lookup, and parallel seed/extraction passes.
template <int BlockSize,
    class CellColor,
    class SeedCell,
    class EvaluateResidual,
    class AffectedRows,
    class ExtractBlock>
bool assemble_colored_block_jacobian(
    linear::BlockRows<BlockSize>& rows,
    int cell_count,
    int color_count,
    CellColor&& cell_color,
    SeedCell&& seed_cell,
    EvaluateResidual&& evaluate_residual,
    AffectedRows&& affected_rows,
    ExtractBlock&& extract_block)
{
    for (auto& row : rows) {
        for (linear::BlockEntry<BlockSize>& entry : row) {
            linear::zero_block<BlockSize>(entry.a);
        }
    }

    for (int color = 0; color < color_count; ++color) {
#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int cell = 0; cell < cell_count; ++cell) {
            seed_cell(cell, cell_color(cell) == color);
        }

        if (!evaluate_residual()) return false;

#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int column_cell = 0; column_cell < cell_count; ++column_cell) {
            if (cell_color(column_cell) != color) continue;
            const auto affected = affected_rows(column_cell);
            for (int row_cell : affected) {
                linear::BlockEntry<BlockSize>* block =
                    linear::find_block<BlockSize>(rows[row_cell], column_cell);
                if (block) extract_block(row_cell, column_cell, block->a);
            }
        }
    }
    return true;
}

} // namespace dg::ad
