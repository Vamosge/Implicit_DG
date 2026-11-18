#include "dg/linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int block_size = 2;
using Entry = dg::linear::BlockEntry<block_size>;
using Block = dg::linear::DenseBlock<block_size>;

struct TestMatrix {
    dg::linear::BlockRows<block_size> rows;
};

Block scaled_identity(double scale) {
    Block block = {};
    for (int i = 0; i < block_size; ++i) {
        dg::linear::block_at<block_size>(block, i, i) = scale;
    }
    return block;
}

bool vectors_close(const std::vector<double>& first,
    const std::vector<double>& second,
    double tolerance)
{
    if (first.size() != second.size()) return false;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (std::abs(first[i] - second[i]) > tolerance) return false;
    }
    return true;
}

} // namespace

int main() {
    TestMatrix matrix;
    matrix.rows.resize(3);
    for (int row = 0; row < 3; ++row) {
        if (row > 0) matrix.rows[row].push_back(Entry{ row - 1, scaled_identity(-1.0) });
        matrix.rows[row].push_back(Entry{ row, scaled_identity(4.0) });
        if (row + 1 < 3) matrix.rows[row].push_back(Entry{ row + 1, scaled_identity(-1.0) });
    }

    const std::vector<double> expected = { 1.0, -2.0, 3.0, 0.5, -1.0, 4.0 };
    std::vector<double> rhs;
    dg::linear::block_sparse_matvec<block_size>(matrix.rows, expected, rhs);

    dg::linear::BlockILU0Preconditioner<block_size> ilu;
    if (!ilu.build_from_matrix(matrix, 0.0)) {
        std::cerr << "ILU(0) construction failed\n";
        return 1;
    }
    std::vector<double> ilu_solution;
    ilu.apply(rhs, ilu_solution);
    if (!vectors_close(ilu_solution, expected, 1e-12)) {
        std::cerr << "level-scheduled ILU solve does not reproduce the exact block-tridiagonal solution\n";
        return 1;
    }

    auto apply_matrix = [&](const std::vector<double>& input, std::vector<double>& output) {
        dg::linear::block_sparse_matvec<block_size>(matrix.rows, input, output);
    };
    auto apply_identity = [](const std::vector<double>& input, std::vector<double>& output) {
        output = input;
    };
    std::vector<double> gmres_solution;
    const dg::linear::GMRESResult result = dg::linear::gmres_solve_right_preconditioned(
        apply_matrix, apply_identity, rhs, gmres_solution,
        6, 12, 1e-12, 0.0, 1.0);
    if (!result.converged || !vectors_close(gmres_solution, expected, 1e-10)) {
        std::cerr << "GMRES failed: iterations=" << result.iterations
            << " relative_residual=" << result.rel_res << '\n';
        return 1;
    }

    TestMatrix restricted = matrix;
    if (!dg::linear::constrain_to_equation<block_size, 2>(restricted.rows, 1)) {
        std::cerr << "equation restriction failed\n";
        return 1;
    }
    const std::vector<double> mixed_input = { 2.0, -2.0, 3.0, 0.5, -4.0, 4.0 };
    std::vector<double> restricted_product;
    dg::linear::block_sparse_matvec<block_size>(
        restricted.rows, mixed_input, restricted_product);
    const std::vector<double> restricted_expected = { 2.0, -8.5, 3.0, 0.0, -4.0, 15.5 };
    if (!vectors_close(restricted_product, restricted_expected, 1e-12)) {
        std::cerr << "equation restriction did not produce identity inactive rows and the active scalar operator\n";
        return 1;
    }

    return 0;
}
