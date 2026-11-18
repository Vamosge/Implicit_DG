#include "dg/ad_jacobian.hpp"

#include <iostream>
#include <vector>

namespace {

struct FakeField {
    bool allocated = false;
};

} // namespace

int main() {
    dg::ad::JacobianWorkspace<FakeField> workspace;
    workspace.ensure_allocated([](FakeField& field) { field.allocated = true; });
    if (!workspace.state.allocated || !workspace.residual.allocated
        || !workspace.rhs.allocated || !workspace.gradient_x.allocated
        || !workspace.gradient_y.allocated) {
        std::cerr << "AD Jacobian workspace allocation callback was incomplete\n";
        return 1;
    }

    constexpr int block_size = 2;
    constexpr int cell_count = 4;
    dg::linear::BlockRows<block_size> rows(cell_count);
    for (int cell = 0; cell < cell_count; ++cell) {
        rows[cell].push_back({ cell, {} });
    }

    std::vector<bool> seeded(cell_count, false);
    int residual_evaluations = 0;
    const bool assembled = dg::ad::assemble_colored_block_jacobian<block_size>(
        rows,
        cell_count,
        2,
        [](int cell) { return cell % 2; },
        [&](int cell, bool active) { seeded[cell] = active; },
        [&]() {
            ++residual_evaluations;
            return true;
        },
        [](int column_cell) { return std::vector<int>{ column_cell }; },
        [&](int row_cell, int column_cell, dg::linear::DenseBlock<2>& block) {
            if (!seeded[column_cell] || row_cell != column_cell) return;
            dg::linear::block_at<2>(block, 0, 0) = 10.0 + column_cell;
            dg::linear::block_at<2>(block, 1, 1) = 20.0 + column_cell;
        });

    if (!assembled || residual_evaluations != 2) {
        std::cerr << "colored assembler did not evaluate once per color\n";
        return 1;
    }
    for (int cell = 0; cell < cell_count; ++cell) {
        const auto& block = rows[cell][0].a;
        if (dg::linear::block_at_const<2>(block, 0, 0) != 10.0 + cell
            || dg::linear::block_at_const<2>(block, 1, 1) != 20.0 + cell) {
            std::cerr << "colored assembler extracted the wrong block for cell " << cell << '\n';
            return 1;
        }
    }
    return 0;
}
