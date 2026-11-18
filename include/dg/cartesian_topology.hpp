#pragma once

#include <utility>
#include <vector>

namespace dg {

// Serial ownership/topology description. Solver kernels use this interface
// instead of embedding global-index arithmetic; a distributed implementation
// can later provide rank-local ownership and halo mappings behind the same seam.
class CartesianTopology {
public:
    struct FaceNeighbor {
        bool interior = false;
        int i = -1;
        int j = -1;
        int face = -1;
        bool reverse_quadrature = false;
    };

    // wake_pairs joins that many cells from each end of the j=0 boundary.
    // This represents a structured C-grid cut without converting the solver
    // to a general unstructured data layout.
    CartesianTopology(int nx, int ny, int wake_pairs = 0);

    [[nodiscard]] int nx() const noexcept { return nx_; }
    [[nodiscard]] int ny() const noexcept { return ny_; }
    [[nodiscard]] int cell_count() const noexcept { return nx_ * ny_; }
    [[nodiscard]] int cell_id(int i, int j) const noexcept { return i * ny_ + j; }
    [[nodiscard]] int cell_i(int id) const noexcept { return id / ny_; }
    [[nodiscard]] int cell_j(int id) const noexcept { return id % ny_; }
    [[nodiscard]] bool contains(int i, int j) const noexcept;
    [[nodiscard]] FaceNeighbor face_neighbor(int i, int j, int face) const;
    [[nodiscard]] std::vector<int> manhattan_neighbors(int id, int radius) const;

private:
    int nx_;
    int ny_;
    int wake_pairs_;
};

} // namespace dg
