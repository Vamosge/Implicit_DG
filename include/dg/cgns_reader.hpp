#pragma once

#include <array>
#include <string>
#include <vector>

namespace dg {

enum class BoundaryKind {
    Unknown,
    Inflow,
    Outflow,
    Symmetry,
    ViscousWall,
    InviscidWall
};

struct BoundaryPatch {
    std::string name;
    BoundaryKind kind = BoundaryKind::Unknown;
    std::array<long long, 3> begin = {1, 1, 1};
    std::array<long long, 3> end = {1, 1, 1};
};

struct StructuredGrid2D {
    int ni = 0;
    int nj = 0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<BoundaryPatch> boundaries;
    int wall_i_begin = -1;
    int wall_i_end = -1;
};

StructuredGrid2D read_structured_cgns_2d(const std::string& path,
    int expected_ni = 0, int expected_nj = 0);

} // namespace dg
