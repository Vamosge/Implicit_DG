#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#ifndef DG_HAS_OPENMP
#define DG_HAS_OPENMP 0
#endif

namespace dg::linear {

template <int BlockSize>
using DenseBlock = std::array<double, BlockSize * BlockSize>;

template <int BlockSize>
struct BlockEntry {
    int col = -1;
    DenseBlock<BlockSize> a = {};
};

template <int BlockSize>
using BlockRows = std::vector<std::vector<BlockEntry<BlockSize>>>;

template <int BlockSize>
inline double& block_at(DenseBlock<BlockSize>& a, int row, int col) {
    return a[row * BlockSize + col];
}

template <int BlockSize>
inline double block_at_const(const DenseBlock<BlockSize>& a, int row, int col) {
    return a[row * BlockSize + col];
}

template <int BlockSize>
inline void zero_block(DenseBlock<BlockSize>& a) {
    a.fill(0.0);
}

template <int BlockSize>
inline BlockEntry<BlockSize>* find_block(std::vector<BlockEntry<BlockSize>>& row, int col) {
    for (BlockEntry<BlockSize>& entry : row) {
        if (entry.col == col) return &entry;
    }
    return nullptr;
}

// Restrict an interleaved block system to one equation. Inactive equations
// receive identity rows and have all couplings to the active equation removed.
// This makes a full-system preconditioner safe for a scalar diagnostic solve.
template <int BlockSize, int EquationCount>
inline bool constrain_to_equation(BlockRows<BlockSize>& rows,
    int active_equation)
{
    static_assert(BlockSize % EquationCount == 0,
        "BlockSize must contain complete equation groups");
    if (active_equation < 0 || active_equation >= EquationCount) return false;
    for (int block_row = 0; block_row < static_cast<int>(rows.size()); ++block_row) {
        BlockEntry<BlockSize>* diagonal = nullptr;
        for (BlockEntry<BlockSize>& entry : rows[block_row]) {
            if (entry.col == block_row) diagonal = &entry;
            for (int row = 0; row < BlockSize; ++row) {
                const bool active_row = row % EquationCount == active_equation;
                for (int col = 0; col < BlockSize; ++col) {
                    const bool active_col = col % EquationCount == active_equation;
                    if (!active_row || !active_col) {
                        block_at<BlockSize>(entry.a, row, col) = 0.0;
                    }
                }
            }
        }
        if (!diagonal) return false;
        for (int dof = 0; dof < BlockSize; ++dof) {
            if (dof % EquationCount != active_equation) {
                block_at<BlockSize>(diagonal->a, dof, dof) = 1.0;
            }
        }
    }
    return true;
}

template <int BlockSize>
inline bool invert_dense_block(const DenseBlock<BlockSize>& a, DenseBlock<BlockSize>& inverse) {
    std::vector<double> augmented(BlockSize * 2 * BlockSize, 0.0);
    for (int row = 0; row < BlockSize; ++row) {
        for (int col = 0; col < BlockSize; ++col) {
            augmented[row * (2 * BlockSize) + col] = block_at_const<BlockSize>(a, row, col);
        }
        augmented[row * (2 * BlockSize) + BlockSize + row] = 1.0;
    }

    for (int col = 0; col < BlockSize; ++col) {
        int pivot = col;
        double best = std::abs(augmented[col * (2 * BlockSize) + col]);
        for (int row = col + 1; row < BlockSize; ++row) {
            const double value = std::abs(augmented[row * (2 * BlockSize) + col]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (!std::isfinite(best) || best < 1e-14) return false;
        if (pivot != col) {
            for (int j = 0; j < 2 * BlockSize; ++j) {
                std::swap(augmented[col * (2 * BlockSize) + j],
                    augmented[pivot * (2 * BlockSize) + j]);
            }
        }
        const double diagonal = augmented[col * (2 * BlockSize) + col];
        for (int j = 0; j < 2 * BlockSize; ++j) {
            augmented[col * (2 * BlockSize) + j] /= diagonal;
        }
        for (int row = 0; row < BlockSize; ++row) {
            if (row == col) continue;
            const double factor = augmented[row * (2 * BlockSize) + col];
            if (factor == 0.0) continue;
            for (int j = 0; j < 2 * BlockSize; ++j) {
                augmented[row * (2 * BlockSize) + j] -=
                    factor * augmented[col * (2 * BlockSize) + j];
            }
        }
    }

    for (int row = 0; row < BlockSize; ++row) {
        for (int col = 0; col < BlockSize; ++col) {
            block_at<BlockSize>(inverse, row, col) =
                augmented[row * (2 * BlockSize) + BlockSize + col];
        }
    }
    return true;
}

template <int BlockSize>
inline void block_right_multiply(DenseBlock<BlockSize>& a, const DenseBlock<BlockSize>& b) {
    DenseBlock<BlockSize> result = {};
    for (int row = 0; row < BlockSize; ++row) {
        for (int col = 0; col < BlockSize; ++col) {
            double sum = 0.0;
            for (int middle = 0; middle < BlockSize; ++middle) {
                sum += block_at_const<BlockSize>(a, row, middle)
                    * block_at_const<BlockSize>(b, middle, col);
            }
            block_at<BlockSize>(result, row, col) = sum;
        }
    }
    a = result;
}

template <int BlockSize>
inline void block_multiply_subtract(DenseBlock<BlockSize>& result,
    const DenseBlock<BlockSize>& a,
    const DenseBlock<BlockSize>& b)
{
    for (int row = 0; row < BlockSize; ++row) {
        for (int col = 0; col < BlockSize; ++col) {
            double sum = 0.0;
            for (int middle = 0; middle < BlockSize; ++middle) {
                sum += block_at_const<BlockSize>(a, row, middle)
                    * block_at_const<BlockSize>(b, middle, col);
            }
            block_at<BlockSize>(result, row, col) -= sum;
        }
    }
}

template <int BlockSize>
inline void block_matvec(const DenseBlock<BlockSize>& a,
    const double* x,
    double* y,
    double alpha = 1.0)
{
    for (int row = 0; row < BlockSize; ++row) {
        double sum = 0.0;
        for (int col = 0; col < BlockSize; ++col) {
            sum += block_at_const<BlockSize>(a, row, col) * x[col];
        }
        y[row] += alpha * sum;
    }
}

template <int BlockSize>
inline void block_matvec_assign(const DenseBlock<BlockSize>& a, const double* x, double* y) {
    for (int row = 0; row < BlockSize; ++row) {
        double sum = 0.0;
        for (int col = 0; col < BlockSize; ++col) {
            sum += block_at_const<BlockSize>(a, row, col) * x[col];
        }
        y[row] = sum;
    }
}

template <int BlockSize>
inline void block_matvec_subtract(const DenseBlock<BlockSize>& a, const double* x, double* y) {
    for (int row = 0; row < BlockSize; ++row) {
        double sum = 0.0;
        for (int col = 0; col < BlockSize; ++col) {
            sum += block_at_const<BlockSize>(a, row, col) * x[col];
        }
        y[row] -= sum;
    }
}

template <int BlockSize>
inline void block_sparse_matvec(const BlockRows<BlockSize>& rows,
    const std::vector<double>& x,
    std::vector<double>& y)
{
    const int cell_count = static_cast<int>(rows.size());
    y.assign(static_cast<std::size_t>(cell_count) * BlockSize, 0.0);
#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row_cell = 0; row_cell < cell_count; ++row_cell) {
        double* output = &y[static_cast<std::size_t>(row_cell) * BlockSize];
        for (const BlockEntry<BlockSize>& entry : rows[row_cell]) {
            block_matvec<BlockSize>(entry.a,
                &x[static_cast<std::size_t>(entry.col) * BlockSize], output);
        }
    }
}

template <int BlockSize>
class BlockILU0Preconditioner {
public:
    template <class Matrix>
    bool build_from_matrix(const Matrix& matrix, double diagonal_shift) {
        rows_ = matrix.rows;
        const int cell_count = static_cast<int>(rows_.size());
        diagonal_inverse_.assign(cell_count, {});
        solve_workspace_.resize(static_cast<std::size_t>(cell_count) * BlockSize);
        build_solve_levels();
        ready_ = false;
        return factorize(diagonal_shift);
    }

    void invalidate() { ready_ = false; }
    bool ready() const { return ready_; }

    void apply(const std::vector<double>& b, std::vector<double>& x) const {
        x.resize(b.size());
        if (!ready_) {
            x = b;
            return;
        }

        std::vector<double>& y = solve_workspace_;
#if DG_HAS_OPENMP
#pragma omp parallel
#endif
        {
            for (int level = 0; level < static_cast<int>(forward_levels_.size()); ++level) {
                const std::vector<int>& cells = forward_levels_[level];
#if DG_HAS_OPENMP
#pragma omp for schedule(static)
#endif
                for (int k = 0; k < static_cast<int>(cells.size()); ++k) {
                    const int cell = cells[k];
                    const int base = cell * BlockSize;
                    double rhs[BlockSize];
                    for (int p = 0; p < BlockSize; ++p) rhs[p] = b[base + p];
                    for (const BlockEntry<BlockSize>& entry : rows_[cell]) {
                        if (entry.col >= cell) continue;
                        block_matvec_subtract<BlockSize>(entry.a,
                            &y[entry.col * BlockSize], rhs);
                    }
                    for (int p = 0; p < BlockSize; ++p) y[base + p] = rhs[p];
                }
            }

            for (int level = 0; level < static_cast<int>(backward_levels_.size()); ++level) {
                const std::vector<int>& cells = backward_levels_[level];
#if DG_HAS_OPENMP
#pragma omp for schedule(static)
#endif
                for (int k = 0; k < static_cast<int>(cells.size()); ++k) {
                    const int cell = cells[k];
                    const int base = cell * BlockSize;
                    double rhs[BlockSize];
                    for (int p = 0; p < BlockSize; ++p) rhs[p] = y[base + p];
                    for (const BlockEntry<BlockSize>& entry : rows_[cell]) {
                        if (entry.col <= cell) continue;
                        block_matvec_subtract<BlockSize>(entry.a,
                            &x[entry.col * BlockSize], rhs);
                    }
                    block_matvec_assign<BlockSize>(diagonal_inverse_[cell], rhs, &x[base]);
                }
            }
        }
    }

private:
    void build_solve_levels() {
        const int cell_count = static_cast<int>(rows_.size());
        std::vector<int> cell_level(cell_count, 0);
        int maximum_level = 0;
        for (int cell = 0; cell < cell_count; ++cell) {
            int level = 0;
            for (const BlockEntry<BlockSize>& entry : rows_[cell]) {
                if (entry.col < cell) level = std::max(level, cell_level[entry.col] + 1);
            }
            cell_level[cell] = level;
            maximum_level = std::max(maximum_level, level);
        }
        forward_levels_.assign(maximum_level + 1, {});
        for (int cell = 0; cell < cell_count; ++cell) {
            forward_levels_[cell_level[cell]].push_back(cell);
        }

        std::fill(cell_level.begin(), cell_level.end(), 0);
        maximum_level = 0;
        for (int cell = cell_count - 1; cell >= 0; --cell) {
            int level = 0;
            for (const BlockEntry<BlockSize>& entry : rows_[cell]) {
                if (entry.col > cell) level = std::max(level, cell_level[entry.col] + 1);
            }
            cell_level[cell] = level;
            maximum_level = std::max(maximum_level, level);
        }
        backward_levels_.assign(maximum_level + 1, {});
        for (int cell = cell_count - 1; cell >= 0; --cell) {
            backward_levels_[cell_level[cell]].push_back(cell);
        }
    }

    bool factorize(double diagonal_shift) {
        const int cell_count = static_cast<int>(rows_.size());
        for (int row = 0; row < cell_count; ++row) {
            for (BlockEntry<BlockSize>& lower : rows_[row]) {
                const int pivot_row = lower.col;
                if (pivot_row >= row) continue;
                block_right_multiply<BlockSize>(lower.a, diagonal_inverse_[pivot_row]);
                for (const BlockEntry<BlockSize>& upper : rows_[pivot_row]) {
                    const int col = upper.col;
                    if (col <= pivot_row) continue;
                    BlockEntry<BlockSize>* target = find_block<BlockSize>(rows_[row], col);
                    if (target) {
                        block_multiply_subtract<BlockSize>(target->a, lower.a, upper.a);
                    }
                }
            }

            BlockEntry<BlockSize>* diagonal = find_block<BlockSize>(rows_[row], row);
            if (!diagonal) return false;
            for (int p = 0; p < BlockSize; ++p) {
                block_at<BlockSize>(diagonal->a, p, p) += diagonal_shift;
            }
            if (!invert_dense_block<BlockSize>(diagonal->a, diagonal_inverse_[row])) {
                for (int p = 0; p < BlockSize; ++p) {
                    block_at<BlockSize>(diagonal->a, p, p) += 100.0 * diagonal_shift;
                }
                if (!invert_dense_block<BlockSize>(diagonal->a, diagonal_inverse_[row])) {
                    return false;
                }
            }
        }
        ready_ = true;
        return true;
    }

    BlockRows<BlockSize> rows_;
    std::vector<DenseBlock<BlockSize>> diagonal_inverse_;
    mutable std::vector<double> solve_workspace_;
    std::vector<std::vector<int>> forward_levels_;
    std::vector<std::vector<int>> backward_levels_;
    bool ready_ = false;
};

class IdentityPreconditioner {
public:
    template <class Matrix>
    bool build_from_matrix(const Matrix&, double) {
        ready_ = true;
        return true;
    }
    void invalidate() { ready_ = false; }
    bool ready() const { return ready_; }
    void apply(const std::vector<double>& b, std::vector<double>& x) const { x = b; }

private:
    bool ready_ = true;
};

template <int BlockSize>
class BlockLUSGSPreconditioner {
public:
    template <class Matrix>
    bool build_from_matrix(const Matrix& matrix, double diagonal_shift) {
        rows_ = matrix.rows;
        const int cell_count = static_cast<int>(rows_.size());
        diagonal_inverse_.assign(cell_count, {});
        solve_workspace_.resize(static_cast<std::size_t>(cell_count) * BlockSize);
        for (int cell = 0; cell < cell_count; ++cell) {
            BlockEntry<BlockSize>* diagonal = find_block<BlockSize>(rows_[cell], cell);
            if (!diagonal) return false;
            for (int p = 0; p < BlockSize; ++p) {
                block_at<BlockSize>(diagonal->a, p, p) += diagonal_shift;
            }
            if (!invert_dense_block<BlockSize>(diagonal->a, diagonal_inverse_[cell])) {
                for (int p = 0; p < BlockSize; ++p) {
                    block_at<BlockSize>(diagonal->a, p, p) += 100.0 * diagonal_shift;
                }
                if (!invert_dense_block<BlockSize>(diagonal->a, diagonal_inverse_[cell])) {
                    return false;
                }
            }
        }
        ready_ = true;
        return true;
    }

    void invalidate() { ready_ = false; }
    bool ready() const { return ready_; }

    void apply(const std::vector<double>& b, std::vector<double>& x) const {
        x.resize(b.size());
        if (!ready_) {
            x = b;
            return;
        }

        const int cell_count = static_cast<int>(rows_.size());
        std::vector<double>& y = solve_workspace_;
        double rhs[BlockSize];
        for (int cell = 0; cell < cell_count; ++cell) {
            const int base = cell * BlockSize;
            for (int p = 0; p < BlockSize; ++p) rhs[p] = b[base + p];
            for (const BlockEntry<BlockSize>& entry : rows_[cell]) {
                if (entry.col >= cell) continue;
                block_matvec_subtract<BlockSize>(entry.a, &y[entry.col * BlockSize], rhs);
            }
            block_matvec_assign<BlockSize>(diagonal_inverse_[cell], rhs, &y[base]);
        }
        for (int cell = cell_count - 1; cell >= 0; --cell) {
            const int base = cell * BlockSize;
            for (int p = 0; p < BlockSize; ++p) rhs[p] = y[base + p];
            for (const BlockEntry<BlockSize>& entry : rows_[cell]) {
                if (entry.col <= cell) continue;
                block_matvec_subtract<BlockSize>(entry.a, &x[entry.col * BlockSize], rhs);
            }
            block_matvec_assign<BlockSize>(diagonal_inverse_[cell], rhs, &x[base]);
        }
    }

private:
    BlockRows<BlockSize> rows_;
    std::vector<DenseBlock<BlockSize>> diagonal_inverse_;
    mutable std::vector<double> solve_workspace_;
    bool ready_ = false;
};

struct GMRESResult {
    bool converged = false;
    int iterations = 0;
    double rel_res = std::numeric_limits<double>::infinity();
    double rel_update = std::numeric_limits<double>::infinity();
};

namespace detail {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

inline double norm(const std::vector<double>& values) {
    return std::sqrt(dot(values, values));
}

inline void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    for (std::size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

inline void apply_givens(double cosine, double sine, double& first, double& second) {
    const double new_first = cosine * first + sine * second;
    const double new_second = -sine * first + cosine * second;
    first = new_first;
    second = new_second;
}

inline void make_givens(double first, double second, double& cosine, double& sine) {
    if (std::abs(second) < 1e-300) {
        cosine = 1.0;
        sine = 0.0;
    }
    else {
        const double radius = std::hypot(first, second);
        cosine = first / radius;
        sine = second / radius;
    }
}

inline void update_solution(int used,
    const std::vector<std::vector<double>>& hessenberg,
    const std::vector<double>& transformed_rhs,
    const std::vector<std::vector<double>>& preconditioned_basis,
    std::vector<double>& solution)
{
    std::vector<double> coefficients(used, 0.0);
    for (int row = used - 1; row >= 0; --row) {
        double sum = transformed_rhs[row];
        for (int col = row + 1; col < used; ++col) {
            sum -= hessenberg[row][col] * coefficients[col];
        }
        coefficients[row] = sum / hessenberg[row][row];
    }
    for (int col = 0; col < used; ++col) {
        axpy(coefficients[col], preconditioned_basis[col], solution);
    }
}

} // namespace detail

template <class ApplyMatrix, class ApplyPreconditioner>
GMRESResult gmres_solve_right_preconditioned(
    ApplyMatrix&& apply_matrix,
    ApplyPreconditioner&& apply_preconditioner,
    const std::vector<double>& b,
    std::vector<double>& x,
    int restart,
    int maximum_iterations,
    double linear_tolerance,
    double update_tolerance,
    double update_residual_gate)
{
    GMRESResult result;
    const int dof_count = static_cast<int>(b.size());
    x.assign(dof_count, 0.0);
    const double rhs_norm = std::max(detail::norm(b), 1e-300);
    std::vector<double> matrix_x(dof_count, 0.0), residual(dof_count, 0.0);

    while (result.iterations < maximum_iterations) {
        apply_matrix(x, matrix_x);
        for (int i = 0; i < dof_count; ++i) residual[i] = b[i] - matrix_x[i];
        const double beta = detail::norm(residual);
        result.rel_res = beta / rhs_norm;
        if (result.rel_res < linear_tolerance) {
            result.rel_update = 0.0;
            result.converged = true;
            return result;
        }

        const int basis_size = std::min(restart, maximum_iterations - result.iterations);
        std::vector<std::vector<double>> basis(basis_size + 1);
        std::vector<std::vector<double>> preconditioned_basis(basis_size);
        std::vector<std::vector<double>> hessenberg(
            basis_size + 1, std::vector<double>(basis_size, 0.0));
        std::vector<double> cosine(basis_size, 0.0), sine(basis_size, 0.0);
        std::vector<double> transformed_rhs(basis_size + 1, 0.0);
        basis[0].resize(dof_count);
        for (int i = 0; i < dof_count; ++i) basis[0][i] = residual[i] / beta;
        transformed_rhs[0] = beta;

        int used = 0;
        std::vector<double> previous_inner_solution = x;
        std::vector<double> candidate_solution(dof_count);
        std::vector<double> work;
        for (int col = 0; col < basis_size; ++col) {
            preconditioned_basis[col].resize(dof_count);
            apply_preconditioner(basis[col], preconditioned_basis[col]);
            apply_matrix(preconditioned_basis[col], work);

            for (int row = 0; row <= col; ++row) {
                hessenberg[row][col] = detail::dot(work, basis[row]);
                detail::axpy(-hessenberg[row][col], basis[row], work);
            }
            hessenberg[col + 1][col] = detail::norm(work);
            const bool happy_breakdown = hessenberg[col + 1][col] <= 1e-300;
            if (!happy_breakdown) {
                basis[col + 1].resize(dof_count);
                for (int i = 0; i < dof_count; ++i) {
                    basis[col + 1][i] = work[i] / hessenberg[col + 1][col];
                }
            }

            for (int row = 0; row < col; ++row) {
                detail::apply_givens(cosine[row], sine[row],
                    hessenberg[row][col], hessenberg[row + 1][col]);
            }
            detail::make_givens(hessenberg[col][col], hessenberg[col + 1][col],
                cosine[col], sine[col]);
            detail::apply_givens(cosine[col], sine[col],
                hessenberg[col][col], hessenberg[col + 1][col]);
            detail::apply_givens(cosine[col], sine[col],
                transformed_rhs[col], transformed_rhs[col + 1]);

            used = col + 1;
            ++result.iterations;
            result.rel_res = std::abs(transformed_rhs[col + 1]) / rhs_norm;
            candidate_solution = x;
            detail::update_solution(used, hessenberg, transformed_rhs,
                preconditioned_basis, candidate_solution);
            double update_norm_squared = 0.0;
            for (int i = 0; i < dof_count; ++i) {
                const double update = candidate_solution[i] - previous_inner_solution[i];
                update_norm_squared += update * update;
            }
            result.rel_update = std::sqrt(update_norm_squared)
                / std::max(detail::norm(candidate_solution), 1e-300);
            previous_inner_solution.swap(candidate_solution);
            const bool residual_converged = result.rel_res < linear_tolerance;
            const bool update_converged = update_tolerance > 0.0
                && result.rel_update < update_tolerance
                && result.rel_res < update_residual_gate;
            if (residual_converged || update_converged || happy_breakdown) {
                detail::update_solution(used, hessenberg, transformed_rhs,
                    preconditioned_basis, x);
                result.converged = true;
                return result;
            }
        }
        detail::update_solution(used, hessenberg, transformed_rhs,
            preconditioned_basis, x);
    }

    apply_matrix(x, matrix_x);
    for (int i = 0; i < dof_count; ++i) residual[i] = b[i] - matrix_x[i];
    result.rel_res = detail::norm(residual) / rhs_norm;
    result.converged = result.rel_res < linear_tolerance
        || (update_tolerance > 0.0
            && result.rel_update < update_tolerance
            && result.rel_res < update_residual_gate);
    return result;
}

} // namespace dg::linear
