#include "dg/cgns_reader.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(DG_USE_CGNS)
#include <cgnslib.h>
#endif

namespace dg {

#if defined(DG_USE_CGNS)
namespace {

[[noreturn]] void throw_cgns(const std::string& operation)
{
    throw std::runtime_error(operation + ": " + cg_get_error());
}

void check_cgns(int status, const std::string& operation)
{
    if (status != CG_OK) throw_cgns(operation);
}

struct CgnsFile {
    int id = 0;

    explicit CgnsFile(const std::string& path)
    {
        check_cgns(cg_open(path.c_str(), CG_MODE_READ, &id), "cg_open(" + path + ")");
    }

    ~CgnsFile()
    {
        if (id != 0) cg_close(id);
    }

    CgnsFile(const CgnsFile&) = delete;
    CgnsFile& operator=(const CgnsFile&) = delete;
};

size_t linear_index(const std::array<cgsize_t, 3>& dims,
    const std::array<cgsize_t, 3>& index)
{
    return static_cast<size_t>(index[0])
        + static_cast<size_t>(dims[0]) *
        (static_cast<size_t>(index[1])
            + static_cast<size_t>(dims[1]) * static_cast<size_t>(index[2]));
}

double coordinate_axis_variation(const std::vector<double>& values,
    const std::array<cgsize_t, 3>& dims, int axis)
{
    double variation = 0.0;
    size_t samples = 0;
    for (int a = 0; a < 3; ++a) {
        if (a == axis) continue;
        for (int side = 0; side < 2; ++side) {
            std::array<cgsize_t, 3> lo = {0, 0, 0};
            std::array<cgsize_t, 3> hi = {0, 0, 0};
            const int other = 3 - axis - a;
            lo[a] = hi[a] = side == 0 ? 0 : dims[a] - 1;
            lo[other] = hi[other] = dims[other] / 2;
            hi[axis] = dims[axis] - 1;
            variation += std::abs(values[linear_index(dims, hi)]
                - values[linear_index(dims, lo)]);
            ++samples;
        }
    }
    return samples == 0 ? 0.0 : variation / static_cast<double>(samples);
}

BoundaryKind classify_boundary(CGNS_ENUMT(BCType_t) type)
{
    switch (type) {
    case CGNS_ENUMV(BCTunnelInflow):
    case CGNS_ENUMV(BCInflow):
    case CGNS_ENUMV(BCInflowSubsonic):
    case CGNS_ENUMV(BCInflowSupersonic):
        return BoundaryKind::Inflow;
    case CGNS_ENUMV(BCTunnelOutflow):
    case CGNS_ENUMV(BCOutflow):
    case CGNS_ENUMV(BCOutflowSubsonic):
    case CGNS_ENUMV(BCOutflowSupersonic):
    case CGNS_ENUMV(BCExtrapolate):
        return BoundaryKind::Outflow;
    case CGNS_ENUMV(BCSymmetryPlane):
    case CGNS_ENUMV(BCSymmetryPolar):
        return BoundaryKind::Symmetry;
    case CGNS_ENUMV(BCWallViscous):
    case CGNS_ENUMV(BCWallViscousHeatFlux):
    case CGNS_ENUMV(BCWallViscousIsothermal):
        return BoundaryKind::ViscousWall;
    case CGNS_ENUMV(BCWall):
    case CGNS_ENUMV(BCWallInviscid):
        return BoundaryKind::InviscidWall;
    default:
        return BoundaryKind::Unknown;
    }
}

} // namespace
#endif

StructuredGrid2D read_structured_cgns_2d(const std::string& path,
    int expected_ni, int expected_nj)
{
#if !defined(DG_USE_CGNS)
    (void)path;
    (void)expected_ni;
    (void)expected_nj;
    throw std::runtime_error("CGNS support was not enabled at build time");
#else
    CgnsFile file(path);
    constexpr int base = 1;
    constexpr int zone = 1;

    char base_name[33] = {};
    int cell_dim = 0;
    int physical_dim = 0;
    check_cgns(cg_base_read(file.id, base, base_name, &cell_dim, &physical_dim),
        "cg_base_read");
    if (cell_dim != 3 || physical_dim != 3) {
        throw std::runtime_error("expected the extruded 3-D structured CGNS representation");
    }

    char zone_name[33] = {};
    cgsize_t zone_size[9] = {};
    check_cgns(cg_zone_read(file.id, base, zone, zone_name, zone_size), "cg_zone_read");
    CGNS_ENUMT(ZoneType_t) zone_type = CGNS_ENUMV(ZoneTypeNull);
    check_cgns(cg_zone_type(file.id, base, zone, &zone_type), "cg_zone_type");
    if (zone_type == CGNS_ENUMV(Unstructured)) {
        if (expected_ni <= 1 || expected_nj <= 1) {
            throw std::runtime_error(
                "an unstructured CGNS zone requires expected logical ni,nj dimensions");
        }
        const size_t plane_points = static_cast<size_t>(expected_ni) * expected_nj;
        const size_t vertex_count = static_cast<size_t>(zone_size[0]);
        if (vertex_count == 0 || vertex_count % plane_points != 0) {
            throw std::runtime_error("unstructured CGNS vertex count is incompatible with logical ni,nj");
        }
        const size_t span_count = vertex_count / plane_points;

        int coordinate_count = 0;
        check_cgns(cg_ncoords(file.id, base, zone, &coordinate_count), "cg_ncoords");
        if (coordinate_count < 2) {
            throw std::runtime_error("CGNS zone has fewer than two coordinates");
        }
        struct Coordinate {
            std::string name;
            std::vector<double> values;
            double range = 0.0;
        };
        std::vector<Coordinate> coordinates;
        const cgsize_t range_min = 1;
        const cgsize_t range_max = zone_size[0];
        for (int coordinate = 1; coordinate <= coordinate_count; ++coordinate) {
            char coordinate_name[33] = {};
            CGNS_ENUMT(DataType_t) data_type = CGNS_ENUMV(DataTypeNull);
            check_cgns(cg_coord_info(file.id, base, zone, coordinate,
                &data_type, coordinate_name), "cg_coord_info");
            Coordinate item;
            item.name = coordinate_name;
            item.values.resize(vertex_count);
            check_cgns(cg_coord_read(file.id, base, zone, coordinate_name,
                CGNS_ENUMV(RealDouble), &range_min, &range_max, item.values.data()),
                "cg_coord_read(" + item.name + ")");
            const auto bounds = std::minmax_element(item.values.begin(), item.values.end());
            item.range = *bounds.second - *bounds.first;
            coordinates.push_back(std::move(item));
        }
        auto x_it = std::find_if(coordinates.begin(), coordinates.end(),
            [](const Coordinate& coordinate) { return coordinate.name == "CoordinateX"; });
        if (x_it == coordinates.end()) {
            x_it = std::max_element(coordinates.begin(), coordinates.end(),
                [](const Coordinate& a, const Coordinate& b) { return a.range < b.range; });
        }
        struct Layout {
            bool interleaved_span = true;
            bool i_major = true;
        };
        const std::array<Layout, 4> layouts = {{{true, true}, {true, false},
            {false, true}, {false, false}}};
        const auto source_index = [=](const Layout& layout, int i, int j) {
            const size_t point = layout.i_major
                ? static_cast<size_t>(i) * expected_nj + j
                : static_cast<size_t>(j) * expected_ni + i;
            return layout.interleaved_span ? point * span_count : point;
        };
        double best_score = -std::numeric_limits<double>::infinity();
        Layout best_layout;
        auto vertical_it = coordinates.end();
        for (auto vertical_candidate = coordinates.begin();
            vertical_candidate != coordinates.end(); ++vertical_candidate) {
            if (vertical_candidate == x_it || vertical_candidate->range <= 0.0) continue;
            for (const Layout& layout : layouts) {
            double forward = 0.0;
            double reverse = 0.0;
            double x_min = std::numeric_limits<double>::infinity();
            double x_max = -std::numeric_limits<double>::infinity();
            double y_min = std::numeric_limits<double>::infinity();
            double y_max = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < expected_nj; ++j) {
                for (int i = 0; i < expected_ni; ++i) {
                    const size_t source = source_index(layout, i, j);
                    x_min = std::min(x_min, x_it->values[source]);
                    x_max = std::max(x_max, x_it->values[source]);
                    y_min = std::min(y_min, vertical_candidate->values[source]);
                    y_max = std::max(y_max, vertical_candidate->values[source]);
                }
            }
            if (x_max - x_min < 0.9 * x_it->range
                || y_max - y_min < 0.9 * vertical_candidate->range) {
                continue;
            }
            for (int j = 0; j < expected_nj; ++j) {
                for (int i = 0; i + 1 < expected_ni; ++i) {
                    const double delta = x_it->values[source_index(layout, i + 1, j)]
                        - x_it->values[source_index(layout, i, j)];
                    forward += std::max(delta, 0.0);
                    reverse += std::max(-delta, 0.0);
                }
            }
            for (int j = 0; j + 1 < expected_nj; ++j) {
                for (int i = 0; i < expected_ni; ++i) {
                    const double delta = vertical_candidate->values[source_index(layout, i, j + 1)]
                        - vertical_candidate->values[source_index(layout, i, j)];
                    forward += std::max(delta, 0.0);
                    reverse += std::max(-delta, 0.0);
                }
            }
            const double score = forward - 1000.0 * reverse;
            if (score > best_score) {
                best_score = score;
                best_layout = layout;
                vertical_it = vertical_candidate;
            }
            }
        }
        if (vertical_it == coordinates.end()
            || !std::isfinite(best_score) || best_score <= 0.0) {
            throw std::runtime_error("could not recover a monotone logical grid from unstructured coordinates");
        }

        StructuredGrid2D result;
        result.ni = expected_ni;
        result.nj = expected_nj;
        result.x.resize(plane_points);
        result.y.resize(plane_points);
        for (int j = 0; j < expected_nj; ++j) {
            for (int i = 0; i < expected_ni; ++i) {
                const size_t source = source_index(best_layout, i, j);
                const size_t destination = static_cast<size_t>(j) * expected_ni + i;
                result.x[destination] = x_it->values[source];
                result.y[destination] = vertical_it->values[source];
            }
        }
        return result;
    }
    if (zone_type != CGNS_ENUMV(Structured)) {
        throw std::runtime_error("unsupported CGNS zone type");
    }
    const std::array<cgsize_t, 3> dims = {zone_size[0], zone_size[1], zone_size[2]};
    const size_t point_count = static_cast<size_t>(dims[0])
        * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]);
    if (point_count == 0) throw std::runtime_error("CGNS zone contains no vertices");

    int coordinate_count = 0;
    check_cgns(cg_ncoords(file.id, base, zone, &coordinate_count), "cg_ncoords");
    if (coordinate_count < 2) throw std::runtime_error("CGNS zone has fewer than two coordinates");

    struct Coordinate {
        std::string name;
        std::vector<double> values;
        double range = 0.0;
    };
    std::vector<Coordinate> coordinates;
    const cgsize_t range_min[3] = {1, 1, 1};
    const cgsize_t range_max[3] = {dims[0], dims[1], dims[2]};
    for (int coordinate = 1; coordinate <= coordinate_count; ++coordinate) {
        char coordinate_name[33] = {};
        CGNS_ENUMT(DataType_t) data_type = CGNS_ENUMV(DataTypeNull);
        check_cgns(cg_coord_info(file.id, base, zone, coordinate, &data_type, coordinate_name),
            "cg_coord_info");
        Coordinate item;
        item.name = coordinate_name;
        item.values.resize(point_count);
        check_cgns(cg_coord_read(file.id, base, zone, coordinate_name,
            CGNS_ENUMV(RealDouble), range_min, range_max, item.values.data()),
            "cg_coord_read(" + item.name + ")");
        const auto bounds = std::minmax_element(item.values.begin(), item.values.end());
        item.range = *bounds.second - *bounds.first;
        coordinates.push_back(std::move(item));
    }

    auto x_it = std::find_if(coordinates.begin(), coordinates.end(), [](const Coordinate& c) {
        return c.name == "CoordinateX";
    });
    if (x_it == coordinates.end()) {
        x_it = std::max_element(coordinates.begin(), coordinates.end(),
            [](const Coordinate& a, const Coordinate& b) { return a.range < b.range; });
    }
    auto vertical_it = coordinates.end();
    for (auto it = coordinates.begin(); it != coordinates.end(); ++it) {
        if (it == x_it) continue;
        if (vertical_it == coordinates.end() || it->range > vertical_it->range) vertical_it = it;
    }
    if (vertical_it == coordinates.end()) throw std::runtime_error("could not identify vertical coordinate");

    int stream_axis = -1;
    int cross_axis = -1;
    // Logical C-grids are not monotone in physical x/y, so coordinate
    // variation cannot reliably identify their i/j axes.  When the caller
    // supplies the expected logical dimensions, use those dimensions as the
    // authoritative mapping.  The variation heuristic remains a fallback for
    // callers that do not know the logical sizes.
    if (expected_ni > 1 && expected_nj > 1) {
        for (int i_axis = 0; i_axis < 3 && stream_axis < 0; ++i_axis) {
            if (dims[i_axis] != static_cast<cgsize_t>(expected_ni)) continue;
            for (int j_axis = 0; j_axis < 3; ++j_axis) {
                if (j_axis != i_axis
                    && dims[j_axis] == static_cast<cgsize_t>(expected_nj)) {
                    stream_axis = i_axis;
                    cross_axis = j_axis;
                    break;
                }
            }
        }
        if (stream_axis < 0) {
            throw std::runtime_error("structured CGNS dimensions do not contain the requested logical ni,nj");
        }
    }
    else {
        stream_axis = 0;
        cross_axis = 0;
        for (int axis = 1; axis < 3; ++axis) {
            if (coordinate_axis_variation(x_it->values, dims, axis)
                > coordinate_axis_variation(x_it->values, dims, stream_axis)) stream_axis = axis;
            if (coordinate_axis_variation(vertical_it->values, dims, axis)
                > coordinate_axis_variation(vertical_it->values, dims, cross_axis)) cross_axis = axis;
        }
    }
    if (stream_axis == cross_axis) {
        throw std::runtime_error("could not identify independent streamwise and cross-stream axes");
    }
    int span_axis = 0;
    while (span_axis == stream_axis || span_axis == cross_axis) ++span_axis;

    StructuredGrid2D result;
    result.ni = static_cast<int>(dims[stream_axis]);
    result.nj = static_cast<int>(dims[cross_axis]);
    result.x.resize(static_cast<size_t>(result.ni) * result.nj);
    result.y.resize(static_cast<size_t>(result.ni) * result.nj);
    for (int j = 0; j < result.nj; ++j) {
        for (int i = 0; i < result.ni; ++i) {
            std::array<cgsize_t, 3> index = {0, 0, 0};
            index[stream_axis] = i;
            index[cross_axis] = j;
            index[span_axis] = 0;
            const size_t source = linear_index(dims, index);
            const size_t destination = static_cast<size_t>(j) * result.ni + i;
            result.x[destination] = x_it->values[source];
            result.y[destination] = vertical_it->values[source];
        }
    }

    int boundary_count = 0;
    check_cgns(cg_nbocos(file.id, base, zone, &boundary_count), "cg_nbocos");
    for (int boundary = 1; boundary <= boundary_count; ++boundary) {
        char boundary_name[33] = {};
        CGNS_ENUMT(BCType_t) boundary_type = CGNS_ENUMV(BCTypeNull);
        CGNS_ENUMT(PointSetType_t) point_set_type = CGNS_ENUMV(PointSetTypeNull);
        cgsize_t point_count_in_set = 0;
        int normal_index[3] = {};
        cgsize_t normal_list_size = 0;
        CGNS_ENUMT(DataType_t) normal_data_type = CGNS_ENUMV(DataTypeNull);
        int data_set_count = 0;
        check_cgns(cg_boco_info(file.id, base, zone, boundary, boundary_name,
            &boundary_type, &point_set_type, &point_count_in_set, normal_index,
            &normal_list_size, &normal_data_type, &data_set_count), "cg_boco_info");

        BoundaryPatch patch;
        patch.name = boundary_name;
        patch.kind = classify_boundary(boundary_type);
        if (point_set_type == CGNS_ENUMV(PointRange) && point_count_in_set == 2) {
            cgsize_t points[6] = {};
            check_cgns(cg_boco_read(file.id, base, zone, boundary, points, nullptr),
                "cg_boco_read(" + patch.name + ")");
            for (int axis = 0; axis < 3; ++axis) {
                patch.begin[axis] = static_cast<long long>(points[axis]);
                patch.end[axis] = static_cast<long long>(points[3 + axis]);
            }
            if (patch.kind == BoundaryKind::ViscousWall) {
                const int begin = static_cast<int>(std::min(points[stream_axis], points[3 + stream_axis]) - 1);
                const int end = static_cast<int>(std::max(points[stream_axis], points[3 + stream_axis]) - 1);
                result.wall_i_begin = result.wall_i_begin < 0 ? begin : std::min(result.wall_i_begin, begin);
                result.wall_i_end = std::max(result.wall_i_end, end);
            }
        }
        result.boundaries.push_back(std::move(patch));
    }

    return result;
#endif
}

} // namespace dg
