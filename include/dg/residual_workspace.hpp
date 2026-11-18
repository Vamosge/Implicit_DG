#pragma once

#include <array>
#include <cstddef>

namespace dg {

template <class Scalar, std::size_t BasisDofs, std::size_t FacePoints,
    std::size_t Equations>
struct CellResidualWorkspace {
    static constexpr std::size_t cell_values = BasisDofs * Equations;
    static constexpr std::size_t face_values = FacePoints * Equations;

    std::array<Scalar, cell_values> cell_rhs = {};

    std::array<Scalar, Equations> state = {};
    std::array<Scalar, Equations> gradient_x = {};
    std::array<Scalar, Equations> gradient_y = {};
    std::array<Scalar, Equations> convective_x = {};
    std::array<Scalar, Equations> convective_y = {};
    std::array<Scalar, Equations> viscous_x = {};
    std::array<Scalar, Equations> viscous_y = {};

    std::array<Scalar, face_values> face_left = {};
    std::array<Scalar, face_values> face_right_convective = {};
    std::array<Scalar, face_values> face_right_viscous = {};
    std::array<Scalar, face_values> face_jump = {};
    std::array<Scalar, face_values> face_auxiliary = {};
    std::array<bool, FacePoints> strict_wall_viscous = {};
    std::array<bool, FacePoints> zero_viscous_flux = {};

    std::array<Scalar, cell_values> lifting_x = {};
    std::array<Scalar, cell_values> lifting_y = {};
    std::array<Scalar, cell_values> adjacent_lifting_x = {};
    std::array<Scalar, cell_values> adjacent_lifting_y = {};

    std::array<Scalar, Equations> right_convective = {};
    std::array<Scalar, Equations> right_viscous = {};
    std::array<Scalar, Equations> right_gradient_x = {};
    std::array<Scalar, Equations> right_gradient_y = {};
    std::array<Scalar, Equations> averaged_state = {};
    std::array<Scalar, Equations> averaged_gradient_x = {};
    std::array<Scalar, Equations> averaged_gradient_y = {};
    std::array<Scalar, Equations> numerical_flux = {};
    std::array<Scalar, Equations> viscous_normal = {};
    std::array<Scalar, BasisDofs> mass_rhs = {};

    Scalar& cell(std::size_t basis, std::size_t equation)
    {
        return cell_rhs[basis * Equations + equation];
    }

    Scalar& face(std::array<Scalar, face_values>& values,
        std::size_t point, std::size_t equation)
    {
        return values[point * Equations + equation];
    }

    const Scalar& face(const std::array<Scalar, face_values>& values,
        std::size_t point, std::size_t equation) const
    {
        return values[point * Equations + equation];
    }

    Scalar& lifting(std::array<Scalar, cell_values>& values,
        std::size_t basis, std::size_t equation)
    {
        return values[basis * Equations + equation];
    }

    void clear_cell()
    {
        cell_rhs.fill(Scalar{});
    }

    void clear_volume_point()
    {
        state.fill(Scalar{});
        gradient_x.fill(Scalar{});
        gradient_y.fill(Scalar{});
    }

    void clear_face_data()
    {
        face_left.fill(Scalar{});
        face_right_convective.fill(Scalar{});
        face_right_viscous.fill(Scalar{});
        face_jump.fill(Scalar{});
        strict_wall_viscous.fill(false);
        zero_viscous_flux.fill(false);
    }

    void clear_face_point()
    {
        gradient_x.fill(Scalar{});
        gradient_y.fill(Scalar{});
        right_gradient_x.fill(Scalar{});
        right_gradient_y.fill(Scalar{});
    }
};

} // namespace dg
