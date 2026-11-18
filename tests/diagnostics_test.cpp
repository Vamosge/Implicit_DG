#include "dg/diagnostics.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main() {
    if (dg::io::sanitize_token("line search\tfailure") != "line_search_failure") {
        std::cerr << "diagnostic token sanitization failed\n";
        return 1;
    }

    const char* path = "diagnostics_test_event.dat";
    {
        dg::io::EventHistoryWriter writer;
        writer.open(path, "test_case", "test_group", "test_cfl");
        writer.write({ 3, 1.25, 0.5, "cfl grow", 5.0, 6.0, 0.1, 0.2,
            1.0, 0.5, 0.25, 0.9, 0.7, 0.0, "healthy step" });
    }

    std::ifstream input(path);
    std::string contents((std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    input.close();
    std::remove(path);
    if (contents.find("cfl_grow") == std::string::npos
        || contents.find("healthy_step") == std::string::npos) {
        std::cerr << "event history record was not written correctly\n";
        return 1;
    }

    const char* residual_path = "diagnostics_test_residual.dat";
    {
        dg::io::ResidualHistoryWriter writer;
        writer.open(residual_path);
        writer.write({ 1, 0.1, 0.2, 0.3, 5.0, 0.01, 0.02, 1e-3, -3.0,
            { 1, 2, 3, 4, 5 }, { 2, 3, 4, 5, 6 },
            { 3, 4, 5, 6, 7 }, { 4, 5, 6, 7, 8 },
            6, 8, 6, 0.5, 0.25 });
    }
    std::ifstream residual_input(residual_path);
    contents.assign(std::istreambuf_iterator<char>(residual_input),
        std::istreambuf_iterator<char>());
    residual_input.close();
    std::remove(residual_path);
    if (contents.find("L2scaled_rho") == std::string::npos
        || contents.find("1 0.1") == std::string::npos) {
        std::cerr << "residual history record was not written correctly\n";
        return 1;
    }

    const char* nonlinear_path = "diagnostics_test_nonlinear.dat";
    {
        dg::io::NonlinearHistoryWriter writer;
        writer.open(nonlinear_path);
        writer.write({ 2, 3, 0.4, 0.5, 0.6, 7.0, 0.02, 0.03,
            { 1, 2, 3, 4, 5 }, { 2, 3, 4, 5, 6 },
            6, 0.1, 4, 1e-4, 1e-5, 1e-3, 0.5, true });
    }
    std::ifstream nonlinear_input(nonlinear_path);
    contents.assign(std::istreambuf_iterator<char>(nonlinear_input),
        std::istreambuf_iterator<char>());
    nonlinear_input.close();
    std::remove(nonlinear_path);
    if (contents.find("gmres_iters") == std::string::npos
        || contents.find("2 3 0.4") == std::string::npos) {
        std::cerr << "nonlinear history record was not written correctly\n";
        return 1;
    }

    const char* sa_limiter_path = "diagnostics_test_sa_limiter.dat";
    {
        dg::SALimiterReport report;
        report.limited_cells = 1;
        report.point_corrections = 1;
        report.minimum_theta = 0.25;
        report.worst_i = 4;
        report.worst_j = 5;
        report.worst_kind = dg::SASampleKind::face_quadrature;
        report.worst_local_index = 2;
        report.worst_face = 0;
        report.worst_x = 0.75;
        report.worst_y = 0.01;
        report.worst_rho = 1.0;
        report.worst_nu_lam = 2e-5;
        report.worst_nu_tilde = -4e-4;
        report.worst_chi = -20.0;
        report.worst_floor = -2.88e-4;
        report.record_change(3e-6);
        if (!report.severe(0.95)) {
            std::cerr << "strong SA limiter activation was not classified as severe\n";
            return 1;
        }
        dg::io::SALimiterHistoryWriter writer;
        writer.open(sa_limiter_path);
        writer.write(10, 0.5, 1.5, 100.0, report);
    }
    std::ifstream sa_limiter_input(sa_limiter_path);
    contents.assign(std::istreambuf_iterator<char>(sa_limiter_input),
        std::istreambuf_iterator<char>());
    sa_limiter_input.close();
    std::remove(sa_limiter_path);
    if (contents.find("face_quadrature") == std::string::npos
        || contents.find("10 0.5 1.5 100") == std::string::npos) {
        std::cerr << "SA limiter history record was not written correctly\n";
        return 1;
    }

    {
        dg::SALimiterReport mild_report;
        mild_report.point_corrections = 3;
        mild_report.minimum_theta = 0.998;
        if (mild_report.severe(0.95)) {
            std::cerr << "small SA limiter correction was classified as severe\n";
            return 1;
        }
        mild_report.average_corrections = 1;
        if (!mild_report.severe(0.95)) {
            std::cerr << "SA cell-average correction was not classified as severe\n";
            return 1;
        }
    }

    const char* sa_residual_path = "diagnostics_test_sa_residual.dat";
    {
        dg::io::SAResidualDiagnosticWriter writer;
        writer.open(sa_residual_path, 0.05);
        dg::io::SAResidualDiagnosticRecord record;
        record.local_step = 50;
        record.total_step = 3050;
        record.pseudo_time = 0.25;
        record.cfl = 300.0;
        record.global_l2_scaled = 0.125;
        record.maximum_i = 7;
        record.maximum_j = 2;
        record.maximum_x = 0.75;
        record.maximum_y = 0.01;
        record.wall_patch_l2_scaled = 0.1;
        record.wall_begin_outside_l2_scaled = 0.2;
        writer.write(record);
    }
    std::ifstream sa_residual_input(sa_residual_path);
    contents.assign(std::istreambuf_iterator<char>(sa_residual_input),
        std::istreambuf_iterator<char>());
    sa_residual_input.close();
    std::remove(sa_residual_path);
    if (contents.find("wall_patch_L2scaled") == std::string::npos
        || contents.find("airfoil_band_L2scaled") == std::string::npos
        || contents.find("wake_band_L2scaled") == std::string::npos
        || contents.find("wake_pair_common_L2scaled") == std::string::npos
        || contents.find("wake_pair_antisymmetric_L2scaled") == std::string::npos
        || contents.find("wall_begin_outside_L2scaled") == std::string::npos
        || contents.find("wall_core_band_L2scaled") == std::string::npos
        || contents.find("50 3050 0.25") == std::string::npos) {
        std::cerr << "SA residual diagnostic was not written correctly\n";
        return 1;
    }

    const char* sa_balance_path = "diagnostics_test_sa_balance.dat";
    {
        dg::io::SATermBalanceWriter writer;
        writer.open(sa_balance_path);
        dg::io::SATermBalanceRecord record;
        record.local_step = 50;
        record.total_step = 1550;
        record.hotspot_terms = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
        record.hotspot_reconstruction_error = 1e-14;
        record.hotspot_vorticity = 12.5;
        record.hotspot_dominant_component = 3;
        record.global_fastest_growth_component = 6;
        record.global_fastest_growth_ratio = 1.25;
        record.wake_interface_total_cancel_rel_max = 2e-15;
        record.wake_interface_nu_tilde_jump_abs_max = 3e-6;
        writer.write(record);
    }
    std::ifstream sa_balance_input(sa_balance_path);
    contents.assign(std::istreambuf_iterator<char>(sa_balance_input),
        std::istreambuf_iterator<char>());
    sa_balance_input.close();
    std::remove(sa_balance_path);
    if (contents.find("hotspot_viscosity_interior_consistent") == std::string::npos
        || contents.find("hotspot_viscosity_interior_lifting") == std::string::npos
        || contents.find("hotspot_viscosity_wall_face") == std::string::npos
        || contents.find("dominant_face_nu_tilde_jump") == std::string::npos
        || contents.find("hotspot_sa_q2_fraction") == std::string::npos
        || contents.find("dominant_neighbor_sa_q2_fraction") == std::string::npos
        || contents.find("hotspot_destruction_sink") == std::string::npos
        || contents.find("hotspot_vorticity") == std::string::npos
        || contents.find("hotspot_dominant_component") == std::string::npos
        || contents.find("global_fastest_growth_component") == std::string::npos
        || contents.find("global_fastest_growth_ratio") == std::string::npos
        || contents.find("global_reconstruction_error_L2scaled") == std::string::npos
        || contents.find("wake_convection_cancel_rel_max") == std::string::npos
        || contents.find("wake_total_cancel_rel_max") == std::string::npos
        || contents.find("50 1550") == std::string::npos) {
        std::cerr << "SA term-balance diagnostic was not written correctly\n";
        return 1;
    }

    const char* physical_path = "diagnostics_test_physical.dat";
    {
        dg::io::PhysicalConvergenceWriter writer;
        writer.open(physical_path, 0.6321975, 0.75, 0.8678025);
        dg::io::PhysicalConvergenceRecord record;
        record.local_step = 50;
        record.total_step = 3050;
        record.pseudo_time = 0.25;
        record.solver_cpu_seconds = 1.0;
        record.cfl = 300.0;
        record.cf_x1 = 0.0051;
        record.cf_x2 = 0.0061;
        record.cf_x3 = 0.0027;
        record.cf_wall_mean = 0.0045;
        record.cf_relative_change = 1e-4;
        record.cd = 0.012;
        record.cl = 1.05;
        record.cm_c4 = -0.02;
        record.aerodynamic_relative_change = 2e-4;
        record.convergence_relative_change = 2e-4;
        writer.write(record);
    }
    std::ifstream physical_input(physical_path);
    contents.assign(std::istreambuf_iterator<char>(physical_input),
        std::istreambuf_iterator<char>());
    physical_input.close();
    std::remove(physical_path);
    if (contents.find("aerodynamic_relative_change") == std::string::npos
        || contents.find("convergence_relative_change") == std::string::npos
        || contents.find("Cd_window_relative_change") == std::string::npos
        || contents.find("50 3050 0.25") == std::string::npos) {
        std::cerr << "physical convergence record was not written correctly\n";
        return 1;
    }

    const char* solution_path = "diagnostics_test_solution.dat";
    {
        dg::io::TecplotQuadrilateralWriter writer;
        writer.open(solution_path, true, 4, 1);
        writer.write_point({ 0, 0, 1, 2, 3, 4, 5, 6, 7 });
        writer.write_point({ 1, 0, 1, 2, 3, 4, 5, 6, 7 });
        writer.write_point({ 1, 1, 1, 2, 3, 4, 5, 6, 7 });
        writer.write_point({ 0, 1, 1, 2, 3, 4, 5, 6, 7 });
        writer.write_quadrilateral(1, 2, 3, 4);
    }
    std::ifstream solution_input(solution_path);
    contents.assign(std::istreambuf_iterator<char>(solution_input),
        std::istreambuf_iterator<char>());
    solution_input.close();
    std::remove(solution_path);
    if (contents.find("ZONETYPE=FEQUADRILATERAL") == std::string::npos
        || contents.find("1 2 3 4") == std::string::npos) {
        std::cerr << "Tecplot solution snapshot was not written correctly\n";
        return 1;
    }

    const char* guidance_path = "diagnostics_test_guidance.dat";
    {
        dg::io::TextOutputFile output(guidance_path);
        dg::io::write_study_guidance(
            output, "failed", "GMRES did not converge", 1);
    }
    std::ifstream guidance_input(guidance_path);
    contents.assign(std::istreambuf_iterator<char>(guidance_input),
        std::istreambuf_iterator<char>());
    guidance_input.close();
    std::remove(guidance_path);
    if (contents.find("linear_solve_and_preconditioner_robustness")
        == std::string::npos) {
        std::cerr << "run-summary study guidance is incorrect\n";
        return 1;
    }
    return 0;
}
