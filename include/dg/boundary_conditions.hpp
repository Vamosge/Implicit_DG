#pragma once

#include <array>
#include <stdexcept>

namespace dg {

enum class BoundaryType {
    Farfield,
    PressureOutlet,
    TotalInflow,
    SlipWall,
    ViscousWall
};

struct BoundaryCondition {
    BoundaryType type = BoundaryType::Farfield;
    double static_pressure = 0.0;
    double total_pressure = 0.0;
    double total_temperature = 0.0;
    double flow_angle = 0.0;
};

class BoundaryConditionSet {
public:
    static constexpr int face_count = 4;

    BoundaryConditionSet()
    {
        faces_.fill(BoundaryCondition{});
        solid_wall_.type = BoundaryType::ViscousWall;
    }

    BoundaryCondition& face(int face)
    {
        validate_face(face);
        return faces_[static_cast<std::size_t>(face)];
    }

    const BoundaryCondition& face(int face) const
    {
        validate_face(face);
        return faces_[static_cast<std::size_t>(face)];
    }

    BoundaryCondition& solid_wall() { return solid_wall_; }
    const BoundaryCondition& solid_wall() const { return solid_wall_; }

    const BoundaryCondition& resolve(int face, bool on_solid_wall) const
    {
        return on_solid_wall ? solid_wall_ : this->face(face);
    }

private:
    static void validate_face(int face)
    {
        if (face < 0 || face >= face_count) {
            throw std::out_of_range("boundary face index must be in [0, 3]");
        }
    }

    std::array<BoundaryCondition, face_count> faces_;
    BoundaryCondition solid_wall_;
};

inline BoundaryConditionSet make_channel_boundary_conditions(
    double outlet_pressure,
    double inflow_total_pressure,
    double inflow_total_temperature,
    double inflow_angle,
    BoundaryType upper_boundary = BoundaryType::Farfield)
{
    if (upper_boundary != BoundaryType::Farfield
        && upper_boundary != BoundaryType::SlipWall) {
        throw std::invalid_argument("upper channel boundary must be farfield or slip wall");
    }
    BoundaryConditionSet boundaries;
    boundaries.face(0).type = BoundaryType::SlipWall;
    boundaries.face(1).type = BoundaryType::PressureOutlet;
    boundaries.face(1).static_pressure = outlet_pressure;
    boundaries.face(2).type = upper_boundary;
    boundaries.face(3).type = BoundaryType::TotalInflow;
    boundaries.face(3).total_pressure = inflow_total_pressure;
    boundaries.face(3).total_temperature = inflow_total_temperature;
    boundaries.face(3).flow_angle = inflow_angle;
    boundaries.solid_wall().type = BoundaryType::ViscousWall;
    return boundaries;
}

inline bool is_wall(BoundaryType type)
{
    return type == BoundaryType::SlipWall || type == BoundaryType::ViscousWall;
}

inline bool has_strict_viscous_trace(BoundaryType type)
{
    return type == BoundaryType::ViscousWall;
}

inline bool has_zero_viscous_flux(BoundaryType type)
{
    return type != BoundaryType::ViscousWall;
}

} // namespace dg
