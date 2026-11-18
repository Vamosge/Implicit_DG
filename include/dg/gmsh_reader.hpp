#pragma once

#include "dg/cgns_reader.hpp"

#include <string>

namespace dg {

// Read a Gmsh 2.2 ASCII quadrilateral mesh whose elements form a logical
// structured grid. Coincident nodes on a C-grid cut may share a node id; the
// returned coordinate arrays retain separate logical vertices.
StructuredGrid2D read_logically_structured_gmsh_2d(const std::string& path,
    int expected_ni, int expected_nj, int wall_physical_tag = 2,
    int source_ni = 0, int source_nj = 0);

} // namespace dg
