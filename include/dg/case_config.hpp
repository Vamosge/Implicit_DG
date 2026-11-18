#pragma once

#include <cmath>

namespace dg::case_config {

#if defined(DG_CASE_BUMP_CHANNEL)
inline constexpr const char* name = "smooth_bump_channel";
inline constexpr const char* checkpoint_id = "smooth_bump_channel";
inline constexpr const char* short_name = "bump";
inline constexpr const char* study_focus = "NASA_TMR_2D_bump_in_channel_RANS_SA";
inline constexpr const char* grid_family = "NASA_TMR_structured_bump";
inline constexpr double x_min = -25.0;
inline constexpr double x_max = 26.5;
inline constexpr double y_min = 0.0;
inline constexpr double y_max = 5.0;
inline constexpr double reynolds_number = 3.0e6;
inline constexpr double mach_number = 0.2;
inline constexpr double angle_of_attack_degrees = 0.0;
inline constexpr double reference_temperature_kelvin = 300.0;
inline constexpr double viscous_wall_x_min = 0.0;
inline constexpr double viscous_wall_x_max = 1.5;
inline constexpr double wall_reference_x = 0.75;
inline constexpr bool mesh_wall_profile = true;
inline constexpr bool upper_boundary_is_symmetry = true;
inline constexpr bool use_analytic_wall_correction = true;
inline constexpr bool gmsh_logical_c_grid = false;
inline constexpr int wake_pairs = 0;

inline double wall_height(double x)
{
    constexpr double active_x_min = 0.3;
    constexpr double active_x_max = 1.2;
    constexpr double height = 0.05;
    constexpr double pi = 3.141592653589793238462643383279502884;
    if (x <= active_x_min || x >= active_x_max) return 0.0;
    const double s = std::sin(pi * x / 0.9 - pi / 3.0);
    return height * s * s * s * s;
}
#elif defined(DG_CASE_NACA0012)
inline constexpr const char* name = "NACA0012";
// Keep the historical identifier in checkpoint signatures so checkpoints
// written before the grid-family label was corrected remain restartable.
inline constexpr const char* checkpoint_id = "NACA0012_FamilyII";
inline constexpr const char* short_name = "naca0012";
inline constexpr const char* study_focus = "NASA_TMR_NACA0012_RANS_SA";
#if defined(DG_NACA_GRID_FAMILY_II)
inline constexpr const char* grid_family = "NASA_TMR_Family_II";
#elif defined(DG_NACA_GRID_FAMILY_ORIGINAL)
inline constexpr const char* grid_family = "NASA_TMR_original_validation";
#else
inline constexpr const char* grid_family = "unspecified_NACA0012_grid_family";
#endif
inline constexpr const char* reference_comparison =
    "NASA_TMR_SA_without_point_vortex";
inline constexpr const char* farfield_correction = "none";
inline constexpr const char* moment_convention =
    "positive_counterclockwise_about_quarter_chord";
inline constexpr double x_min = -501.0;
inline constexpr double x_max = 501.0;
inline constexpr double y_min = -501.0;
inline constexpr double y_max = 501.0;
inline constexpr double reynolds_number = 6.0e6;
inline constexpr double mach_number = 0.15;
inline constexpr double angle_of_attack_degrees = 10.0;
inline constexpr double reference_temperature_kelvin = 300.0;
inline constexpr double viscous_wall_x_min = 0.0;
inline constexpr double viscous_wall_x_max = 1.0;
inline constexpr double wall_reference_x = 0.5;
inline constexpr bool mesh_wall_profile = true;
inline constexpr bool upper_boundary_is_symmetry = false;
inline constexpr bool use_analytic_wall_correction = false;
inline constexpr bool gmsh_logical_c_grid = true;
#ifndef DG_NACA_WAKE_PAIRS
#define DG_NACA_WAKE_PAIRS 48
#endif
inline constexpr int wake_pairs = DG_NACA_WAKE_PAIRS;

inline double wall_height(double)
{
    return 0.0;
}
#else
inline constexpr const char* name = "VERIF_2DZP_flat_plate";
inline constexpr const char* checkpoint_id = "VERIF_2DZP_flat_plate";
inline constexpr const char* short_name = "plate";
inline constexpr const char* study_focus = "NASA_TMR_2D_ZPG_flat_plate_RANS_SA";
inline constexpr const char* grid_family = "NASA_TMR_VERIF_2DZP_structured";
inline constexpr double x_min = -0.33333;
inline constexpr double x_max = 2.0;
inline constexpr double y_min = 0.0;
inline constexpr double y_max = 1.0;
inline constexpr double reynolds_number = 5.0e6;
inline constexpr double mach_number = 0.2;
inline constexpr double angle_of_attack_degrees = 0.0;
inline constexpr double reference_temperature_kelvin = 300.0;
inline constexpr double viscous_wall_x_min = 0.0;
inline constexpr double viscous_wall_x_max = 2.0;
inline constexpr double wall_reference_x = 1.0;
inline constexpr bool mesh_wall_profile = false;
inline constexpr bool upper_boundary_is_symmetry = false;
inline constexpr bool use_analytic_wall_correction = false;
inline constexpr bool gmsh_logical_c_grid = false;
inline constexpr int wake_pairs = 0;

inline double wall_height(double)
{
    return 0.0;
}
#endif

} // namespace dg::case_config
