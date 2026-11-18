#include "dg/residual_workspace.hpp"

#include <cassert>

int main()
{
    dg::CellResidualWorkspace<double, 9, 5, 5> workspace;
    workspace.cell(3, 2) = 7.0;
    workspace.face(workspace.face_left, 4, 1) = 8.0;
    workspace.lifting(workspace.lifting_x, 8, 4) = 9.0;
    workspace.strict_wall_viscous[2] = true;

    workspace.clear_cell();
    assert(workspace.cell(3, 2) == 0.0);
    assert(workspace.face(workspace.face_left, 4, 1) == 8.0);

    workspace.clear_face_data();
    assert(workspace.face(workspace.face_left, 4, 1) == 0.0);
    // Lifting outputs are assigned in full by the kernel and need no clearing.
    assert(workspace.lifting(workspace.lifting_x, 8, 4) == 9.0);
    assert(!workspace.strict_wall_viscous[2]);

    workspace.gradient_x[0] = 4.0;
    workspace.numerical_flux[3] = 5.0;
    workspace.clear_face_point();
    assert(workspace.gradient_x[0] == 0.0);
    // Flux outputs are assigned in full and intentionally retain old storage.
    assert(workspace.numerical_flux[3] == 5.0);
}
