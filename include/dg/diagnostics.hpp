#pragma once

#include "dg/sa_limiter_diagnostics.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace dg::io {

class TextOutputFile {
public:
    TextOutputFile() = default;
    explicit TextOutputFile(const std::string& path) { open(path); }

    void open(const std::string& path) { stream_.open(path); }
    bool is_open() const { return stream_.is_open(); }
    void flush() { if (stream_.is_open()) stream_.flush(); }
    void close() { if (stream_.is_open()) stream_.close(); }

    template <class Value>
    TextOutputFile& operator<<(const Value& value) {
        if (stream_.is_open()) stream_ << value;
        return *this;
    }

    TextOutputFile& operator<<(std::ostream& (*manipulator)(std::ostream&)) {
        if (stream_.is_open()) manipulator(stream_);
        return *this;
    }

private:
    std::ofstream stream_;
};

inline std::string sanitize_token(std::string value) {
    std::replace(value.begin(), value.end(), ' ', '_');
    std::replace(value.begin(), value.end(), '\t', '_');
    std::replace(value.begin(), value.end(), '\n', '_');
    return value;
}

struct EventRecord {
    int step = 0;
    double pseudo_time = 0.0;
    double solver_cpu_seconds = 0.0;
    std::string type;
    double old_cfl = 0.0;
    double new_cfl = 0.0;
    double dtau_min = 0.0;
    double dtau_max = 0.0;
    double max_l2_scaled = 0.0;
    double max_ns_l2_scaled = 0.0;
    double sa_l2_scaled = 0.0;
    double rho_min = 0.0;
    double p_min = 0.0;
    double nut_min = 0.0;
    std::string detail;
};

class EventHistoryWriter {
public:
    void open(const std::string& path,
        const std::string& case_name,
        const std::string& robust_solver_group,
        const std::string& robust_cfl_mode)
    {
        output_.open(path);
        output_ << "# case_name " << case_name << "\n";
        output_ << "# robust_solver_group " << robust_solver_group << "\n";
        output_ << "# robust_cfl_mode " << robust_cfl_mode << "\n";
        output_ << "# step pseudo_time solver_cpu_seconds event_type old_cfl new_cfl "
            << "dtau_min dtau_max maxL2scaled maxNSL2scaled saL2scaled "
            << "rho_min p_min nut_min detail\n";
    }

    void write(const EventRecord& event) {
        output_ << event.step << " " << std::setprecision(16) << event.pseudo_time
            << " " << event.solver_cpu_seconds
            << " " << sanitize_token(event.type)
            << " " << event.old_cfl
            << " " << event.new_cfl
            << " " << event.dtau_min
            << " " << event.dtau_max
            << " " << event.max_l2_scaled
            << " " << event.max_ns_l2_scaled
            << " " << event.sa_l2_scaled
            << " " << event.rho_min
            << " " << event.p_min
            << " " << event.nut_min
            << " " << sanitize_token(event.detail) << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    TextOutputFile output_;
};

class SALimiterHistoryWriter {
public:
    void open(const std::string& path)
    {
        output_.open(path);
        output_ << "# step pseudo_time solver_cpu_seconds cfl limited_cells "
            << "average_corrections point_corrections min_theta "
            << "change_l2 change_max_abs cell_i cell_j sample_kind "
            << "local_index face x y rho nu_lam nu_tilde chi floor\n";
    }

    void write(int step, double pseudo_time, double solver_cpu_seconds,
        double cfl, const dg::SALimiterReport& report)
    {
        output_ << step << " " << std::setprecision(16) << pseudo_time
            << " " << solver_cpu_seconds
            << " " << cfl
            << " " << report.limited_cells
            << " " << report.average_corrections
            << " " << report.point_corrections
            << " " << report.minimum_theta
            << " " << report.sa_change_l2()
            << " " << report.sa_change_max_abs
            << " " << report.worst_i
            << " " << report.worst_j
            << " " << dg::sa_sample_kind_name(report.worst_kind)
            << " " << report.worst_local_index
            << " " << report.worst_face
            << " " << report.worst_x
            << " " << report.worst_y
            << " " << report.worst_rho
            << " " << report.worst_nu_lam
            << " " << report.worst_nu_tilde
            << " " << report.worst_chi
            << " " << report.worst_floor << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    TextOutputFile output_;
};

struct SAResidualDiagnosticRecord {
    int local_step = 0;
    long long total_step = 0;
    double pseudo_time = 0.0;
    double solver_cpu_seconds = 0.0;
    double cfl = 0.0;
    double global_l2_scaled = 0.0;
    double maximum_abs = 0.0;
    double maximum_scaled = 0.0;
    int maximum_i = -1;
    int maximum_j = -1;
    int maximum_quadrature = -1;
    double maximum_x = 0.0;
    double maximum_y = 0.0;
    double maximum_wall_distance = 0.0;
    double maximum_rho = 0.0;
    double maximum_nu_tilde = 0.0;
    double maximum_chi = 0.0;
    double maximum_mut_over_mu_inf = 0.0;
    int maximum_cell_i = -1;
    int maximum_cell_j = -1;
    double maximum_cell_l2_scaled = 0.0;
    double upstream_l2_scaled = 0.0;
    double wall_patch_l2_scaled = 0.0;
    double downstream_l2_scaled = 0.0;
    double airfoil_band_l2_scaled = 0.0;
    double wake_band_l2_scaled = 0.0;
    double wake_pair_common_l2_scaled = 0.0;
    double wake_pair_antisymmetric_l2_scaled = 0.0;
    double near_wall_l2_scaled = 0.0;
    double outer_l2_scaled = 0.0;
    double wall_begin_outside_l2_scaled = 0.0;
    double wall_begin_inside_l2_scaled = 0.0;
    double wall_end_inside_l2_scaled = 0.0;
    double wall_end_outside_l2_scaled = 0.0;
    double wall_core_band_l2_scaled = 0.0;
};

class SAResidualDiagnosticWriter {
public:
    void open(const std::string& path, double near_wall_distance,
        int wall_i_begin = -1, int wall_i_end = -1,
        int junction_band_rows = 4) {
        output_.open(path);
        output_ << "# near_wall_distance " << std::setprecision(16)
            << near_wall_distance << "\n";
        output_ << "# viscous_wall_cell_index_range " << wall_i_begin
            << " " << wall_i_end << " (half-open)\n";
        output_ << "# boundary_junction_band_rows "
            << junction_band_rows << "\n";
        output_ << "# local_step total_step pseudo_time solver_cpu_seconds cfl "
            << "saL2scaled max_abs max_scaled max_i max_j max_q x y wall_distance "
            << "rho nu_tilde chi mut_over_mu_inf max_cell_i max_cell_j "
            << "max_cell_L2scaled upstream_L2scaled wall_patch_L2scaled "
            << "downstream_L2scaled airfoil_band_L2scaled wake_band_L2scaled "
            << "wake_pair_common_L2scaled wake_pair_antisymmetric_L2scaled "
            << "near_wall_L2scaled outer_L2scaled "
            << "wall_begin_outside_L2scaled wall_begin_inside_L2scaled "
            << "wall_end_inside_L2scaled wall_end_outside_L2scaled "
            << "wall_core_band_L2scaled\n";
    }

    void write(const SAResidualDiagnosticRecord& record) {
        output_ << record.local_step
            << " " << record.total_step
            << " " << std::setprecision(16) << record.pseudo_time
            << " " << record.solver_cpu_seconds
            << " " << record.cfl
            << " " << record.global_l2_scaled
            << " " << record.maximum_abs
            << " " << record.maximum_scaled
            << " " << record.maximum_i
            << " " << record.maximum_j
            << " " << record.maximum_quadrature
            << " " << record.maximum_x
            << " " << record.maximum_y
            << " " << record.maximum_wall_distance
            << " " << record.maximum_rho
            << " " << record.maximum_nu_tilde
            << " " << record.maximum_chi
            << " " << record.maximum_mut_over_mu_inf
            << " " << record.maximum_cell_i
            << " " << record.maximum_cell_j
            << " " << record.maximum_cell_l2_scaled
            << " " << record.upstream_l2_scaled
            << " " << record.wall_patch_l2_scaled
            << " " << record.downstream_l2_scaled
            << " " << record.airfoil_band_l2_scaled
            << " " << record.wake_band_l2_scaled
            << " " << record.wake_pair_common_l2_scaled
            << " " << record.wake_pair_antisymmetric_l2_scaled
            << " " << record.near_wall_l2_scaled
            << " " << record.outer_l2_scaled
            << " " << record.wall_begin_outside_l2_scaled
            << " " << record.wall_begin_inside_l2_scaled
            << " " << record.wall_end_inside_l2_scaled
            << " " << record.wall_end_outside_l2_scaled
            << " " << record.wall_core_band_l2_scaled << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    TextOutputFile output_;
};

inline constexpr std::size_t sa_term_balance_component_count = 9;

// The eight entries use the residual convention
//   convection + volume viscosity + interior-face consistent viscosity
//   + interior-face BR2 lifting correction
//   + wall-face viscosity + other-boundary viscosity
//   + production - destruction + cross-diffusion.
// Destruction is stored as a positive sink so that an unexpectedly large
// destruction term remains easy to recognize in the output.
struct SATermBalanceRecord {
    int local_step = 0;
    long long total_step = 0;
    double pseudo_time = 0.0;
    double solver_cpu_seconds = 0.0;
    double cfl = 0.0;
    int hotspot_i = -1;
    int hotspot_j = -1;
    int hotspot_quadrature = -1;
    double hotspot_x = 0.0;
    double hotspot_y = 0.0;
    double hotspot_wall_distance = 0.0;
    double hotspot_rho = 0.0;
    double hotspot_nu_tilde = 0.0;
    double hotspot_chi = 0.0;
    double hotspot_vorticity = 0.0;
    double hotspot_mu_lam_over_mu_inf = 0.0;
    double hotspot_mu_t_over_mu_inf = 0.0;
    double hotspot_mu_effective_over_mu_inf = 0.0;
    int hotspot_touches_wall = 0;
    int hotspot_touches_wake_interface = 0;
    int hotspot_touches_open_boundary = 0;
    std::array<double, sa_term_balance_component_count> hotspot_terms{};
    double hotspot_viscosity_total = 0.0;
    int hotspot_dominant_component = -1;
    double hotspot_dominant_abs_fraction = 0.0;
    double hotspot_cancellation_ratio = 0.0;
    int hotspot_dominant_interior_face = -1;
    int hotspot_dominant_face_quadrature = -1;
    double hotspot_dominant_face_x = 0.0;
    double hotspot_dominant_face_y = 0.0;
    double hotspot_dominant_face_consistent = 0.0;
    double hotspot_dominant_face_lifting = 0.0;
    double hotspot_dominant_face_total = 0.0;
    double hotspot_dominant_face_qsa_jump = 0.0;
    double hotspot_dominant_face_nu_tilde_jump = 0.0;
    double hotspot_dominant_face_normal_gradient_jump = 0.0;
    double hotspot_sa_q2_fraction = 0.0;
    double hotspot_sa_q2_rms_over_mean = 0.0;
    int dominant_neighbor_i = -1;
    int dominant_neighbor_j = -1;
    double dominant_neighbor_sa_q2_fraction = 0.0;
    double dominant_neighbor_sa_q2_rms_over_mean = 0.0;
    double hotspot_total_rhs = 0.0;
    double hotspot_reconstruction = 0.0;
    double hotspot_reconstruction_error = 0.0;
    double point_source_production = 0.0;
    double point_source_destruction = 0.0;
    double point_source_cross_diffusion = 0.0;
    double point_source_net = 0.0;
    std::array<double, sa_term_balance_component_count> global_l2_scaled{};
    double global_viscosity_total_l2_scaled = 0.0;
    double global_reconstruction_error_l2_scaled = 0.0;
    std::array<double, sa_term_balance_component_count> airfoil_band_l2_scaled{};
    double airfoil_band_viscosity_total_l2_scaled = 0.0;
    std::array<double, sa_term_balance_component_count> wake_band_l2_scaled{};
    double wake_band_viscosity_total_l2_scaled = 0.0;
    // Conservation audit over the paired C-grid wake faces.  Each value is
    // max |left + right| / (|left| + |right|), evaluated pair by pair from
    // the actual assembled constant-test-function face residual.  A
    // conservative paired interface should be near roundoff.
    double wake_interface_convection_cancel_rel_max = 0.0;
    double wake_interface_consistent_cancel_rel_max = 0.0;
    double wake_interface_lifting_cancel_rel_max = 0.0;
    double wake_interface_viscosity_cancel_rel_max = 0.0;
    double wake_interface_total_cancel_rel_max = 0.0;
    double wake_interface_nu_tilde_jump_abs_max = 0.0;
    int global_dominant_component = -1;
    int global_fastest_growth_component = -1;
    double global_fastest_growth_ratio = 0.0;
};

class SATermBalanceWriter {
public:
    void open(const std::string& path) {
        output_.open(path);
        output_ << "# component_sign reconstruction=convection+viscosity_volume+viscosity_interior_consistent+viscosity_interior_lifting+viscosity_wall_face+viscosity_other_boundary+production-destruction+cross_diffusion\n";
        output_ << "# destruction is reported as a positive sink\n";
        output_ << "# component_id 0=convection 1=viscosity_volume 2=viscosity_interior_consistent 3=viscosity_interior_lifting 4=viscosity_wall_face 5=viscosity_other_boundary 6=production 7=destruction 8=cross_diffusion\n";
        output_ << "# local_step total_step pseudo_time solver_cpu_seconds cfl "
            << "hotspot_i hotspot_j hotspot_q x y wall_distance "
            << "hotspot_rho hotspot_nu_tilde hotspot_chi hotspot_vorticity "
            << "hotspot_mu_lam_over_mu_inf hotspot_mu_t_over_mu_inf "
            << "hotspot_mu_effective_over_mu_inf hotspot_touches_wall "
            << "hotspot_touches_wake_interface hotspot_touches_open_boundary "
            << "hotspot_convection hotspot_viscosity_volume "
            << "hotspot_viscosity_interior_consistent hotspot_viscosity_interior_lifting "
            << "hotspot_viscosity_wall_face "
            << "hotspot_viscosity_other_boundary hotspot_production "
            << "hotspot_destruction_sink hotspot_cross_diffusion "
            << "hotspot_viscosity_total hotspot_dominant_component "
            << "hotspot_dominant_abs_fraction hotspot_cancellation_ratio "
            << "dominant_interior_face dominant_face_q dominant_face_x dominant_face_y "
            << "dominant_face_consistent dominant_face_lifting dominant_face_total "
            << "dominant_face_qsa_jump dominant_face_nu_tilde_jump "
            << "dominant_face_normal_gradient_jump "
            << "hotspot_sa_q2_fraction hotspot_sa_q2_rms_over_mean "
            << "dominant_neighbor_i dominant_neighbor_j "
            << "dominant_neighbor_sa_q2_fraction "
            << "dominant_neighbor_sa_q2_rms_over_mean "
            << "hotspot_total_rhs hotspot_reconstruction hotspot_reconstruction_error "
            << "point_source_production point_source_destruction_sink "
            << "point_source_cross_diffusion point_source_net "
            << "global_convection_L2scaled global_viscosity_volume_L2scaled "
            << "global_viscosity_interior_consistent_L2scaled "
            << "global_viscosity_interior_lifting_L2scaled global_viscosity_wall_face_L2scaled "
            << "global_viscosity_other_boundary_L2scaled global_production_L2scaled "
            << "global_destruction_L2scaled global_cross_diffusion_L2scaled "
            << "global_viscosity_total_L2scaled global_reconstruction_error_L2scaled "
            << "airfoil_convection_L2scaled airfoil_viscosity_volume_L2scaled "
            << "airfoil_viscosity_interior_consistent_L2scaled "
            << "airfoil_viscosity_interior_lifting_L2scaled airfoil_viscosity_wall_face_L2scaled "
            << "airfoil_viscosity_other_boundary_L2scaled airfoil_production_L2scaled "
            << "airfoil_destruction_L2scaled airfoil_cross_diffusion_L2scaled "
            << "airfoil_viscosity_total_L2scaled wake_convection_L2scaled "
            << "wake_viscosity_volume_L2scaled wake_viscosity_interior_consistent_L2scaled "
            << "wake_viscosity_interior_lifting_L2scaled "
            << "wake_viscosity_wall_face_L2scaled wake_viscosity_other_boundary_L2scaled "
            << "wake_production_L2scaled wake_destruction_L2scaled "
            << "wake_cross_diffusion_L2scaled wake_viscosity_total_L2scaled "
            << "wake_convection_cancel_rel_max "
            << "wake_consistent_cancel_rel_max wake_lifting_cancel_rel_max "
            << "wake_viscosity_cancel_rel_max wake_total_cancel_rel_max "
            << "wake_nu_tilde_jump_abs_max global_dominant_component "
            << "global_fastest_growth_component global_fastest_growth_ratio\n";
    }

    void write(const SATermBalanceRecord& record) {
        output_ << record.local_step << " " << record.total_step
            << " " << std::setprecision(16) << record.pseudo_time
            << " " << record.solver_cpu_seconds << " " << record.cfl
            << " " << record.hotspot_i << " " << record.hotspot_j
            << " " << record.hotspot_quadrature << " " << record.hotspot_x
            << " " << record.hotspot_y << " " << record.hotspot_wall_distance
            << " " << record.hotspot_rho
            << " " << record.hotspot_nu_tilde
            << " " << record.hotspot_chi
            << " " << record.hotspot_vorticity
            << " " << record.hotspot_mu_lam_over_mu_inf
            << " " << record.hotspot_mu_t_over_mu_inf
            << " " << record.hotspot_mu_effective_over_mu_inf
            << " " << record.hotspot_touches_wall
            << " " << record.hotspot_touches_wake_interface
            << " " << record.hotspot_touches_open_boundary;
        for (double value : record.hotspot_terms) output_ << " " << value;
        output_ << " " << record.hotspot_viscosity_total
            << " " << record.hotspot_dominant_component
            << " " << record.hotspot_dominant_abs_fraction
            << " " << record.hotspot_cancellation_ratio
            << " " << record.hotspot_dominant_interior_face
            << " " << record.hotspot_dominant_face_quadrature
            << " " << record.hotspot_dominant_face_x
            << " " << record.hotspot_dominant_face_y
            << " " << record.hotspot_dominant_face_consistent
            << " " << record.hotspot_dominant_face_lifting
            << " " << record.hotspot_dominant_face_total
            << " " << record.hotspot_dominant_face_qsa_jump
            << " " << record.hotspot_dominant_face_nu_tilde_jump
            << " " << record.hotspot_dominant_face_normal_gradient_jump
            << " " << record.hotspot_sa_q2_fraction
            << " " << record.hotspot_sa_q2_rms_over_mean
            << " " << record.dominant_neighbor_i
            << " " << record.dominant_neighbor_j
            << " " << record.dominant_neighbor_sa_q2_fraction
            << " " << record.dominant_neighbor_sa_q2_rms_over_mean
            << " " << record.hotspot_total_rhs
            << " " << record.hotspot_reconstruction
            << " " << record.hotspot_reconstruction_error
            << " " << record.point_source_production
            << " " << record.point_source_destruction
            << " " << record.point_source_cross_diffusion
            << " " << record.point_source_net;
        for (double value : record.global_l2_scaled) output_ << " " << value;
        output_ << " " << record.global_viscosity_total_l2_scaled
            << " " << record.global_reconstruction_error_l2_scaled;
        for (double value : record.airfoil_band_l2_scaled) output_ << " " << value;
        output_ << " " << record.airfoil_band_viscosity_total_l2_scaled;
        for (double value : record.wake_band_l2_scaled) output_ << " " << value;
        output_ << " " << record.wake_band_viscosity_total_l2_scaled
            << " " << record.wake_interface_convection_cancel_rel_max
            << " " << record.wake_interface_consistent_cancel_rel_max
            << " " << record.wake_interface_lifting_cancel_rel_max
            << " " << record.wake_interface_viscosity_cancel_rel_max
            << " " << record.wake_interface_total_cancel_rel_max
            << " " << record.wake_interface_nu_tilde_jump_abs_max
            << " " << record.global_dominant_component
            << " " << record.global_fastest_growth_component
            << " " << record.global_fastest_growth_ratio << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    TextOutputFile output_;
};

struct PhysicalConvergenceRecord {
    int local_step = 0;
    long long total_step = 0;
    double pseudo_time = 0.0;
    double solver_cpu_seconds = 0.0;
    double cfl = 0.0;
    double cf_x1 = 0.0;
    double cf_x2 = 0.0;
    double cf_x3 = 0.0;
    double cf_wall_mean = 0.0;
    double cf_relative_change = 0.0;
    double cd = 0.0;
    double cl = 0.0;
    double cm_c4 = 0.0;
    double aerodynamic_relative_change = 0.0;
    double convergence_relative_change = 0.0;
    int convergence_window_intervals = 0;
    double cf_x1_window_relative_change = 0.0;
    double cf_x2_window_relative_change = 0.0;
    double cf_x3_window_relative_change = 0.0;
    double cf_mean_window_relative_change = 0.0;
    double cd_window_relative_change = 0.0;
    double cl_window_relative_change = 0.0;
    double cm_window_relative_change = 0.0;
};

class PhysicalConvergenceWriter {
public:
    void open(const std::string& path, double x1, double x2, double x3) {
        output_.open(path);
        output_ << "# cf_sample_x " << std::setprecision(16)
            << x1 << " " << x2 << " " << x3 << "\n";
        output_ << "# local_step total_step pseudo_time solver_cpu_seconds cfl "
            << "cf_x1 cf_x2 cf_x3 cf_wall_mean cf_relative_change "
            << "Cd Cl Cm_c4 aerodynamic_relative_change "
            << "convergence_relative_change convergence_window_intervals "
            << "cf_x1_window_relative_change cf_x2_window_relative_change "
            << "cf_x3_window_relative_change cf_mean_window_relative_change "
            << "Cd_window_relative_change Cl_window_relative_change "
            << "Cm_window_relative_change\n";
    }

    void write(const PhysicalConvergenceRecord& record) {
        output_ << record.local_step
            << " " << record.total_step
            << " " << std::setprecision(16) << record.pseudo_time
            << " " << record.solver_cpu_seconds
            << " " << record.cfl
            << " " << record.cf_x1
            << " " << record.cf_x2
            << " " << record.cf_x3
            << " " << record.cf_wall_mean
            << " " << record.cf_relative_change
            << " " << record.cd
            << " " << record.cl
            << " " << record.cm_c4
            << " " << record.aerodynamic_relative_change
            << " " << record.convergence_relative_change
            << " " << record.convergence_window_intervals
            << " " << record.cf_x1_window_relative_change
            << " " << record.cf_x2_window_relative_change
            << " " << record.cf_x3_window_relative_change
            << " " << record.cf_mean_window_relative_change
            << " " << record.cd_window_relative_change
            << " " << record.cl_window_relative_change
            << " " << record.cm_window_relative_change << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    TextOutputFile output_;
};

struct ResidualHistoryRecord {
    int step = 0;
    double pseudo_time = 0.0;
    double solver_cpu_seconds = 0.0;
    double step_cpu_seconds = 0.0;
    double cfl = 0.0;
    double dtau_min = 0.0;
    double dtau_max = 0.0;
    double rho_update_l2 = 0.0;
    double log10_rho_update_l2 = 0.0;
    std::vector<double> l2;
    std::vector<double> l2_scaled;
    std::vector<double> linf;
    std::vector<double> linf_scaled;
    double max_l2_scaled = 0.0;
    double max_linf_scaled = 0.0;
    double sa_l2_scaled_initial = 0.0;
    double sa_l2_scaled_boundary_layer = 0.0;
    double sa_l2_relative_initial = 0.0;
};

class ResidualHistoryWriter {
public:
    void open(const std::string& path) {
        output_.open(path);
        output_ << "# step pseudo_time solver_cpu_seconds step_cpu_seconds cfl dtau_min dtau_max "
            << "rho_update_l2 lg_rho_update_l2 "
            << "L2_rho L2_rhou L2_rhov L2_E L2_q "
            << "L2scaled_rho L2scaled_rhou L2scaled_rhov L2scaled_E L2scaled_q_init "
            << "Linf_rho Linf_rhou Linf_rhov Linf_E Linf_q "
            << "Linfscaled_rho Linfscaled_rhou Linfscaled_rhov Linfscaled_E Linfscaled_q_init "
            << "maxL2scaled maxLinfscaled saL2scaled_init saL2scaled_bl saL2rel_initial\n";
    }

    void write(const ResidualHistoryRecord& record) {
        output_ << record.step << " " << std::setprecision(16) << record.pseudo_time
            << " " << record.solver_cpu_seconds
            << " " << record.step_cpu_seconds
            << " " << record.cfl
            << " " << record.dtau_min
            << " " << record.dtau_max
            << " " << record.rho_update_l2
            << " " << record.log10_rho_update_l2 << " ";
        write_values(record.l2);
        write_values(record.l2_scaled);
        write_values(record.linf);
        write_values(record.linf_scaled);
        output_ << record.max_l2_scaled
            << " " << record.max_linf_scaled
            << " " << record.sa_l2_scaled_initial
            << " " << record.sa_l2_scaled_boundary_layer
            << " " << record.sa_l2_relative_initial << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    void write_values(const std::vector<double>& values) {
        for (double value : values) output_ << value << " ";
    }

    TextOutputFile output_;
};

struct NonlinearHistoryRecord {
    int steady_step = 0;
    int newton_iteration = 0;
    double pseudo_time = 0.0;
    double solver_cpu_seconds = 0.0;
    double step_cpu_seconds = 0.0;
    double cfl = 0.0;
    double dtau_min = 0.0;
    double dtau_max = 0.0;
    std::vector<double> residual_l2;
    std::vector<double> residual_l2_scaled;
    double maximum_scaled_residual = 0.0;
    double relative_residual = 0.0;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double gmres_relative_update = 0.0;
    double gmres_tolerance = 0.0;
    double accepted_step = 0.0;
    bool accepted = false;
};

class NonlinearHistoryWriter {
public:
    void open(const std::string& path) {
        output_.open(path);
        output_ << "# steady_step newton_iter pseudo_time solver_cpu_seconds step_cpu_seconds cfl dtau_min dtau_max "
            << "beL2_rho beL2_rhou beL2_rhov beL2_E beL2_q "
            << "beL2scaled_rho beL2scaled_rhou beL2scaled_rhov beL2scaled_E beL2scaled_q "
            << "maxBEscaled relBE gmres_iters gmres_rel gmres_update gmres_tol alpha accepted\n";
    }

    void write(const NonlinearHistoryRecord& record) {
        output_ << record.steady_step
            << " " << record.newton_iteration
            << " " << std::setprecision(16) << record.pseudo_time
            << " " << record.solver_cpu_seconds
            << " " << record.step_cpu_seconds
            << " " << record.cfl
            << " " << record.dtau_min
            << " " << record.dtau_max << " ";
        write_values(record.residual_l2);
        write_values(record.residual_l2_scaled);
        output_ << record.maximum_scaled_residual
            << " " << record.relative_residual
            << " " << record.gmres_iterations
            << " " << record.gmres_relative_residual
            << " " << record.gmres_relative_update
            << " " << record.gmres_tolerance
            << " " << record.accepted_step
            << " " << (record.accepted ? 1 : 0) << "\n";
        output_.flush();
    }

    void close() { output_.close(); }

private:
    void write_values(const std::vector<double>& values) {
        for (double value : values) output_ << value << " ";
    }

    TextOutputFile output_;
};

struct SolutionPoint {
    double x = 0.0;
    double y = 0.0;
    double density = 0.0;
    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double pressure = 0.0;
    double nu_tilde = 0.0;
    double turbulent_viscosity = 0.0;
    double turbulent_viscosity_ratio = 0.0;
};

class TecplotQuadrilateralWriter {
public:
    void open(const std::string& path,
        bool write_connectivity,
        int node_count,
        int element_count)
    {
        write_connectivity_ = write_connectivity;
        output_.open(path);
        if (write_connectivity_) {
            output_ << "Variables=\"X\",\"Y\",\"Rho\",\"U\",\"V\",\"P\",\"NuTilde\",\"Mu_t\",\"MuT_over_MuInf\"\n";
            output_ << "ZONE T=\"SolutionGaussPoints\" N=" << node_count
                << " E=" << element_count
                << " DATAPACKING=POINT ZONETYPE=FEQUADRILATERAL\n";
        }
        else {
            output_ << "# X Y Rho U V P NuTilde Mu_t MuT_over_MuInf\n";
            output_ << "# Origin-friendly point data only; no Tecplot zone or connectivity rows.\n";
        }
    }

    void write_point(const SolutionPoint& point) {
        output_ << std::scientific << std::setprecision(16)
            << point.x << " " << point.y
            << " " << point.density
            << " " << point.velocity_x
            << " " << point.velocity_y
            << " " << point.pressure
            << " " << point.nu_tilde
            << " " << point.turbulent_viscosity
            << " " << point.turbulent_viscosity_ratio << "\n";
    }

    void write_quadrilateral(int first, int second, int third, int fourth) {
        if (!write_connectivity_) return;
        output_ << first << " " << second << " " << third << " " << fourth << "\n";
    }

private:
    TextOutputFile output_;
    bool write_connectivity_ = false;
};

inline void write_study_guidance(TextOutputFile& output,
    const std::string& outcome,
    const std::string& failure_reason,
    int rejected_steps)
{
    output << "# study_notes_if_failure\n";
    if (outcome == "failed") {
        if (failure_reason.find("GMRES") != std::string::npos
            || failure_reason.find("ILU") != std::string::npos
            || failure_reason.find("Jacobian") != std::string::npos) {
            output << "# study_focus linear_solve_and_preconditioner_robustness\n";
            output << "# compare GMRES iterations, AD Jacobian rebuilds, ILU0 failures, and CFL at failure.\n";
        }
        else if (failure_reason.find("line search") != std::string::npos
            || failure_reason.find("Newton") != std::string::npos) {
            output << "# study_focus Newton_update_globalization\n";
            output << "# compare full Newton, residual line search, valid-state line search, and Jacobian retry.\n";
        }
        else if (failure_reason.find("invalid") != std::string::npos) {
            output << "# study_focus physicality_constraints\n";
            output << "# inspect rho/p/SA validity checks and limiter activations at the stopping step.\n";
        }
        else {
            output << "# study_focus unclassified_failure_mode\n";
            output << "# inspect nonlinear_residual_history.dat and residual_history.dat near failure_step.\n";
        }
    }
    else if (rejected_steps > 0) {
        output << "# study_focus recovered_nonlinear_failures\n";
        output << "# run did not terminate in failure, but rejected steps show which safeguards were active.\n";
    }
    else {
        output << "# study_focus baseline_success_without_detected_failure\n";
        output << "# use this run as a stable reference for comparing weaker solver variants.\n";
    }
    output << "# suggested_figures residual_history nonlinear_residual_history event_history "
        << "CFL_evolution failure_counts limiter_counts\n";
}

} // namespace dg::io
