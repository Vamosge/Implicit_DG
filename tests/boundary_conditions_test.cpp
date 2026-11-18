#include "dg/boundary_conditions.hpp"

#include <cassert>
#include <stdexcept>

int main()
{
    const dg::BoundaryConditionSet boundaries =
        dg::make_channel_boundary_conditions(0.7, 0.8, 0.9, 0.1);

    assert(boundaries.face(0).type == dg::BoundaryType::SlipWall);
    assert(boundaries.face(1).type == dg::BoundaryType::PressureOutlet);
    assert(boundaries.face(1).static_pressure == 0.7);
    assert(boundaries.face(2).type == dg::BoundaryType::Farfield);
    assert(boundaries.face(3).type == dg::BoundaryType::TotalInflow);
    assert(boundaries.face(3).total_pressure == 0.8);
    assert(boundaries.face(3).total_temperature == 0.9);
    assert(boundaries.face(3).flow_angle == 0.1);

    assert(boundaries.resolve(0, true).type == dg::BoundaryType::ViscousWall);
    assert(boundaries.resolve(0, false).type == dg::BoundaryType::SlipWall);
    assert(dg::is_wall(dg::BoundaryType::ViscousWall));
    assert(dg::is_wall(dg::BoundaryType::SlipWall));
    assert(dg::has_strict_viscous_trace(dg::BoundaryType::ViscousWall));
    assert(!dg::has_zero_viscous_flux(dg::BoundaryType::ViscousWall));
    assert(dg::has_zero_viscous_flux(dg::BoundaryType::Farfield));

    const dg::BoundaryConditionSet bump_boundaries =
        dg::make_channel_boundary_conditions(0.7, 0.8, 0.9, 0.1,
            dg::BoundaryType::SlipWall);
    assert(bump_boundaries.face(2).type == dg::BoundaryType::SlipWall);

    bool rejected_invalid_upper_boundary = false;
    try {
        (void)dg::make_channel_boundary_conditions(0.7, 0.8, 0.9, 0.1,
            dg::BoundaryType::ViscousWall);
    }
    catch (const std::invalid_argument&) {
        rejected_invalid_upper_boundary = true;
    }
    assert(rejected_invalid_upper_boundary);

    bool rejected_invalid_face = false;
    try {
        (void)boundaries.face(4);
    }
    catch (const std::out_of_range&) {
        rejected_invalid_face = true;
    }
    assert(rejected_invalid_face);
}
