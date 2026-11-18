#pragma once

namespace dg {

// Global-reduction seam used by Krylov solvers and convergence diagnostics.
// The serial implementation is intentionally trivial; an MPI implementation
// will map these operations to MPI_Allreduce without changing numerical code.
class CollectiveOperations {
public:
    virtual ~CollectiveOperations() = default;
    [[nodiscard]] virtual int rank() const noexcept = 0;
    [[nodiscard]] virtual int size() const noexcept = 0;
    [[nodiscard]] virtual double sum(double local_value) const = 0;
    [[nodiscard]] virtual double minimum(double local_value) const = 0;
    [[nodiscard]] virtual double maximum(double local_value) const = 0;
};

class SerialCollectives final : public CollectiveOperations {
public:
    [[nodiscard]] int rank() const noexcept override { return 0; }
    [[nodiscard]] int size() const noexcept override { return 1; }
    [[nodiscard]] double sum(double local_value) const override { return local_value; }
    [[nodiscard]] double minimum(double local_value) const override { return local_value; }
    [[nodiscard]] double maximum(double local_value) const override { return local_value; }
};

} // namespace dg
