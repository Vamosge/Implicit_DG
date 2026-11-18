#include "dg/gmsh_reader.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dg {
namespace {

struct Point { double x = 0.0; double y = 0.0; };

void require_line(std::ifstream& input, const char* expected)
{
    std::string line;
    if (!std::getline(input, line) || line != expected) {
        throw std::runtime_error(std::string("invalid Gmsh section: expected ") + expected);
    }
}

} // namespace

StructuredGrid2D read_logically_structured_gmsh_2d(const std::string& path,
    int expected_ni, int expected_nj, int wall_physical_tag,
    int source_ni, int source_nj)
{
    if (expected_ni < 2 || expected_nj < 2) {
        throw std::invalid_argument("expected Gmsh logical dimensions must exceed one");
    }
    if (source_ni == 0) source_ni = expected_ni;
    if (source_nj == 0) source_nj = expected_nj;
    if (source_ni < expected_ni || source_nj < expected_nj
        || (source_ni - 1) % (expected_ni - 1) != 0
        || (source_nj - 1) % (expected_nj - 1) != 0) {
        throw std::invalid_argument("Gmsh source and requested grids are not nested");
    }
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open Gmsh mesh: " + path);

    std::unordered_map<long long, Point> nodes;
    std::vector<std::array<long long, 4>> quads;
    std::unordered_set<long long> wall_nodes;
    std::string line;
    while (std::getline(input, line)) {
        if (line == "$MeshFormat") {
            std::string format;
            if (!std::getline(input, format) || format.rfind("2.2 0 ", 0) != 0) {
                throw std::runtime_error("only Gmsh 2.2 ASCII meshes are supported");
            }
            require_line(input, "$EndMeshFormat");
        }
        else if (line == "$Nodes") {
            std::size_t count = 0;
            if (!(input >> count)) throw std::runtime_error("invalid Gmsh node count");
            nodes.reserve(count);
            for (std::size_t n = 0; n < count; ++n) {
                long long id = 0;
                Point point;
                double z = 0.0;
                if (!(input >> id >> point.x >> point.y >> z)) {
                    throw std::runtime_error("invalid Gmsh node record");
                }
                nodes.emplace(id, point);
            }
            input.ignore(1024, '\n');
            require_line(input, "$EndNodes");
        }
        else if (line == "$Elements") {
            std::size_t count = 0;
            if (!(input >> count)) throw std::runtime_error("invalid Gmsh element count");
            quads.reserve(static_cast<std::size_t>((source_ni - 1) * (source_nj - 1)));
            for (std::size_t n = 0; n < count; ++n) {
                long long id = 0;
                int type = 0, tag_count = 0;
                if (!(input >> id >> type >> tag_count)) {
                    throw std::runtime_error("invalid Gmsh element record");
                }
                int physical_tag = 0;
                for (int tag = 0; tag < tag_count; ++tag) {
                    int value = 0;
                    input >> value;
                    if (tag == 0) physical_tag = value;
                }
                if (type == 1) {
                    long long a = 0, b = 0;
                    input >> a >> b;
                    if (physical_tag == wall_physical_tag) {
                        wall_nodes.insert(a);
                        wall_nodes.insert(b);
                    }
                }
                else if (type == 3) {
                    std::array<long long, 4> quad{};
                    input >> quad[0] >> quad[1] >> quad[2] >> quad[3];
                    quads.push_back(quad);
                }
                else {
                    std::string remainder;
                    std::getline(input, remainder);
                }
            }
            input.ignore(1024, '\n');
            require_line(input, "$EndElements");
        }
    }

    const int nx = source_ni - 1;
    const int ny = source_nj - 1;
    if (quads.size() != static_cast<std::size_t>(nx * ny)) {
        throw std::runtime_error("Gmsh quad count does not match expected logical dimensions");
    }
    std::vector<long long> logical_nodes(static_cast<std::size_t>(source_ni * source_nj), -1);
    auto set_logical = [&](int i, int j, long long node) {
        const std::size_t index = static_cast<std::size_t>(j * source_ni + i);
        if (logical_nodes[index] >= 0 && logical_nodes[index] != node) {
            throw std::runtime_error("Gmsh quads do not form the expected structured ordering");
        }
        logical_nodes[index] = node;
    };
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const auto& q = quads[static_cast<std::size_t>(j * nx + i)];
            set_logical(i, j, q[0]);
            set_logical(i + 1, j, q[1]);
            set_logical(i + 1, j + 1, q[2]);
            set_logical(i, j + 1, q[3]);
        }
    }

    StructuredGrid2D result;
    result.ni = expected_ni;
    result.nj = expected_nj;
    result.x.resize(static_cast<std::size_t>(expected_ni * expected_nj));
    result.y.resize(result.x.size());
    const int stride_i = (source_ni - 1) / (expected_ni - 1);
    const int stride_j = (source_nj - 1) / (expected_nj - 1);
    for (int j = 0; j < expected_nj; ++j) {
        for (int i = 0; i < expected_ni; ++i) {
            const long long node = logical_nodes[static_cast<std::size_t>(
                (j * stride_j) * source_ni + i * stride_i)];
            const auto found = nodes.find(node);
            if (found == nodes.end()) {
                throw std::runtime_error("Gmsh element references a missing node");
            }
            const std::size_t target = static_cast<std::size_t>(j * expected_ni + i);
            result.x[target] = found->second.x;
            result.y[target] = found->second.y;
        }
    }

    for (int i = 0; i < expected_ni; ++i) {
        const long long node = logical_nodes[static_cast<std::size_t>(i * stride_i)];
        if (wall_nodes.count(node) != 0U) {
            if (result.wall_i_begin < 0) result.wall_i_begin = i;
            result.wall_i_end = i;
        }
    }
    if (result.wall_i_begin < 0 || result.wall_i_end <= result.wall_i_begin) {
        throw std::runtime_error("Gmsh airfoil wall was not found on the logical j=0 boundary");
    }
    return result;
}

} // namespace dg
