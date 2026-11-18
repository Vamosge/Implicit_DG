#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace dg {

enum class SASampleKind {
    none,
    cell_average,
    nodal_dof,
    volume_quadrature,
    face_quadrature
};

inline const char* sa_sample_kind_name(SASampleKind kind)
{
    switch (kind) {
    case SASampleKind::none: return "none";
    case SASampleKind::cell_average: return "cell_average";
    case SASampleKind::nodal_dof: return "nodal_dof";
    case SASampleKind::volume_quadrature: return "volume_quadrature";
    case SASampleKind::face_quadrature: return "face_quadrature";
    }
    return "unknown";
}

struct SALimiterReport {
    int limited_cells = 0;
    int average_corrections = 0;
    int point_corrections = 0;
    int worst_i = -1;
    int worst_j = -1;
    SASampleKind worst_kind = SASampleKind::none;
    int worst_local_index = -1;
    int worst_face = -1;
    double worst_x = std::numeric_limits<double>::quiet_NaN();
    double worst_y = std::numeric_limits<double>::quiet_NaN();
    double worst_rho = std::numeric_limits<double>::quiet_NaN();
    double worst_nu_lam = std::numeric_limits<double>::quiet_NaN();
    double worst_nu_tilde = std::numeric_limits<double>::quiet_NaN();
    double worst_chi = std::numeric_limits<double>::quiet_NaN();
    double worst_floor = std::numeric_limits<double>::quiet_NaN();
    double minimum_theta = 1.0;
    double sa_change_l2_squared = 0.0;
    double sa_change_max_abs = 0.0;

    bool activated() const
    {
        return average_corrections > 0 || point_corrections > 0;
    }

    bool severe(double minimum_allowed_theta) const
    {
        return average_corrections > 0
            || (point_corrections > 0
                && minimum_theta < minimum_allowed_theta);
    }

    double sa_change_l2() const
    {
        return std::sqrt(sa_change_l2_squared);
    }

    void record_change(double change)
    {
        sa_change_l2_squared += change * change;
        sa_change_max_abs = std::max(sa_change_max_abs, std::abs(change));
    }
};

} // namespace dg
