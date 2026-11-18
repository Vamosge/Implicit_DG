#include "dg/cartesian_topology.hpp"

#include <algorithm>
#include <deque>
#include <stdexcept>

namespace dg {

CartesianTopology::CartesianTopology(int nx, int ny, int wake_pairs)
    : nx_(nx), ny_(ny), wake_pairs_(wake_pairs)
{
    if (nx <= 0 || ny <= 0) throw std::invalid_argument("Cartesian topology extents must be positive");
    if (wake_pairs < 0 || 2 * wake_pairs > nx) {
        throw std::invalid_argument("C-grid wake-pair count is outside the topology");
    }
}

CartesianTopology::FaceNeighbor CartesianTopology::face_neighbor(
    int i, int j, int face) const
{
    if (!contains(i, j)) throw std::out_of_range("cell is outside topology");
    if (face < 0 || face > 3) throw std::out_of_range("face must be in [0,3]");

    int ni = i + (face == 1 ? 1 : (face == 3 ? -1 : 0));
    int nj = j + (face == 2 ? 1 : (face == 0 ? -1 : 0));
    if (contains(ni, nj)) {
        return {true, ni, nj, (face + 2) % 4, false};
    }
    if (face == 0 && j == 0
        && (i < wake_pairs_ || i >= nx_ - wake_pairs_)) {
        return {true, nx_ - 1 - i, 0, 0, true};
    }
    return {};
}

bool CartesianTopology::contains(int i, int j) const noexcept
{
    return i >= 0 && i < nx_ && j >= 0 && j < ny_;
}

std::vector<int> CartesianTopology::manhattan_neighbors(int id, int radius) const
{
    if (id < 0 || id >= cell_count()) throw std::out_of_range("cell id is outside topology");
    if (radius < 0) throw std::invalid_argument("stencil radius cannot be negative");

    std::vector<int> distance(static_cast<std::size_t>(cell_count()), -1);
    std::deque<int> pending;
    distance[static_cast<std::size_t>(id)] = 0;
    pending.push_back(id);
    while (!pending.empty()) {
        const int current = pending.front();
        pending.pop_front();
        if (distance[static_cast<std::size_t>(current)] == radius) continue;
        const int i = cell_i(current);
        const int j = cell_j(current);
        for (int face = 0; face < 4; ++face) {
            const FaceNeighbor link = face_neighbor(i, j, face);
            if (!link.interior) continue;
            const int next = cell_id(link.i, link.j);
            if (distance[static_cast<std::size_t>(next)] < 0) {
                distance[static_cast<std::size_t>(next)] =
                    distance[static_cast<std::size_t>(current)] + 1;
                pending.push_back(next);
            }
        }
    }
    std::vector<int> neighbors;
    for (int cell = 0; cell < cell_count(); ++cell) {
        if (distance[static_cast<std::size_t>(cell)] >= 0) neighbors.push_back(cell);
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    return neighbors;
}

} // namespace dg
