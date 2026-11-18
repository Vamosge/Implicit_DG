#include "dg/steady_driver.hpp"

#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const double recovery_growth = dg::steady::cfl_accepted_growth_factor(
        1.65, 1.65, 500);
    if (!(recovery_growth > 1.0)
        || std::abs(std::pow(recovery_growth, 500) - 2.0) > 1e-12) {
        std::cerr << "CFL recovery growth was disabled at the ceiling\n";
        return 1;
    }
    const double planned_growth = dg::steady::cfl_accepted_growth_factor(
        1.0, 4.0, 2);
    if (std::abs(planned_growth - 2.0) > 1e-15) {
        std::cerr << "CFL planned ramp did not preserve the larger span\n";
        return 1;
    }
    dg::steady::CflController recovery_controller(
        { 1.65, 1e-3, 1.65, 0.5, recovery_growth, 0.1 });
    const auto recovery_cut = recovery_controller.reject();
    const auto recovered = recovery_controller.accept(true, 0.01);
    if (recovery_cut.new_value != 0.825
        || !recovered.effectively_increased()
        || recovered.new_value <= recovery_cut.new_value) {
        std::cerr << "CFL controller could not recover after a ceiling cut\n";
        return 1;
    }

    dg::steady::CflController controller({ 5.0, 1.0, 20.0, 0.5, 2.0, 0.1 });
    const auto grown = controller.accept(true, 0.05);
    if (!grown.effectively_increased() || grown.new_value != 10.0
        || std::string(grown.action) != "fcfl-grow") {
        std::cerr << "healthy accepted step did not grow CFL\n";
        return 1;
    }
    const auto held = controller.accept(false, 0.05);
    if (held.changed() || std::string(held.action) != "fcfl-hold") {
        std::cerr << "unhealthy accepted step did not hold CFL\n";
        return 1;
    }
    const auto shrunk = controller.reject();
    if (!shrunk.decreased() || shrunk.new_value != 5.0) {
        std::cerr << "rejected step did not shrink CFL\n";
        return 1;
    }
    const auto residual_shrunk = controller.accept(true, 0.01, false, true);
    if (!residual_shrunk.decreased()
        || std::string(residual_shrunk.action) != "residual-shrink") {
        std::cerr << "worsening residual did not shrink CFL\n";
        return 1;
    }
    const auto growth_blocked = controller.accept(true, 0.01, false, false);
    if (growth_blocked.changed()
        || std::string(growth_blocked.action) != "fcfl-hold") {
        std::cerr << "residual growth guard did not hold CFL\n";
        return 1;
    }

    const dg::steady::SAResidualShrinkConfig sa_policy{ 1.5, 2.0, 50 };
    const auto cooldown_hold = dg::steady::sa_residual_shrink_decision(
        1.7, 1.0, 1.1, false, 20, sa_policy);
    if (cooldown_hold.request || cooldown_hold.emergency) {
        std::cerr << "ordinary SA rebound bypassed shrink cooldown\n";
        return 1;
    }
    const auto ordinary_request = dg::steady::sa_residual_shrink_decision(
        1.7, 1.0, 1.1, false, 50, sa_policy);
    if (!ordinary_request.request || ordinary_request.emergency) {
        std::cerr << "ordinary SA rebound did not shrink after cooldown\n";
        return 1;
    }
    const auto emergency_request = dg::steady::sa_residual_shrink_decision(
        2.1, 1.0, 1.1, false, 20, sa_policy);
    if (!emergency_request.request || !emergency_request.emergency) {
        std::cerr << "large SA rebound did not bypass shrink cooldown\n";
        return 1;
    }
    const auto non_rising_hold = dg::steady::sa_residual_shrink_decision(
        2.1, 1.0, 0.9, false, 20, sa_policy);
    if (non_rising_hold.request) {
        std::cerr << "falling SA residual requested emergency shrink\n";
        return 1;
    }
    const auto severe_after_cooldown = dg::steady::sa_residual_shrink_decision(
        1.0, 1.0, 1.0, true, 50, sa_policy);
    if (!severe_after_cooldown.request || severe_after_cooldown.emergency) {
        std::cerr << "severe limiter event did not honor completed cooldown\n";
        return 1;
    }

    dg::steady::ResidualGrowthGuard trend_guard({ 3, 1.05, 1.05, 1.0, 1.0 });
    if (trend_guard.observe(1.0, 1.0, 1.0, 1.0).history_ready
        || trend_guard.observe(0.9, 0.9, 0.9, 0.9).allow_growth
        || trend_guard.observe(0.8, 0.8, 0.8, 0.8).allow_growth) {
        std::cerr << "residual trend guard grew before its history was ready\n";
        return 1;
    }
    const auto healthy_trend = trend_guard.observe(0.7, 0.7, 0.7, 0.7);
    if (!healthy_trend.history_ready || !healthy_trend.allow_growth
        || std::abs(healthy_trend.ns_window_ratio - 0.7) > 1e-15
        || std::abs(healthy_trend.sa_window_ratio - 0.7) > 1e-15) {
        std::cerr << "decreasing NS/SA trend did not allow CFL growth\n";
        return 1;
    }
    const auto sa_drift = trend_guard.observe(0.6, 0.95, 0.6, 0.7);
    if (sa_drift.allow_growth) {
        std::cerr << "adverse SA window trend did not block CFL growth\n";
        return 1;
    }
    const auto ns_drift = trend_guard.observe(0.95, 0.6, 0.6, 0.6);
    if (ns_drift.allow_growth) {
        std::cerr << "adverse NS window trend did not block CFL growth\n";
        return 1;
    }
    const auto far_from_best = trend_guard.observe(0.7, 0.7, 0.5, 0.5);
    if (far_from_best.allow_growth) {
        std::cerr << "residuals far from their running best allowed CFL growth\n";
        return 1;
    }
    const auto invalid_trend = trend_guard.observe(
        std::numeric_limits<double>::quiet_NaN(), 0.5, 0.5, 0.5);
    if (invalid_trend.history_ready || invalid_trend.allow_growth) {
        std::cerr << "invalid residual did not reset the trend guard\n";
        return 1;
    }

    dg::steady::ResidualGrowthGuard restored_guard(
        { 3, 1.05, 1.05, 1.0, 1.0 });
    restored_guard.restore_history({
        { 9.0, 9.0 }, { 1.0, 1.0 }, { 0.9, 0.9 }, { 0.8, 0.8 }
    });
    const auto restored_history = restored_guard.history();
    if (restored_history.size() != 4
        || restored_history.front().first != 9.0) {
        std::cerr << "residual trend history did not round-trip\n";
        return 1;
    }
    const auto restored_decision = restored_guard.observe(
        0.7, 0.7, 0.7, 0.7);
    if (!restored_decision.history_ready
        || !restored_decision.allow_growth
        || std::abs(restored_decision.ns_window_ratio - 0.7) > 1e-15) {
        std::cerr << "restored trend window did not continue correctly\n";
        return 1;
    }

    dg::steady::PhysicalConvergenceGuard physical_guard({ 2, 7 });
    const dg::steady::PhysicalConvergenceSample physical0 = {
        0.003, 0.0031, 0.0032, 0.0031, 0.014, 1.1, -0.004
    };
    auto physical1 = physical0;
    auto physical2 = physical0;
    physical1[4] *= 1.0005;
    physical2[4] *= 1.002;
    if (physical_guard.observe(physical0).history_ready
        || physical_guard.observe(physical1).history_ready) {
        std::cerr << "physical convergence window was ready too early\n";
        return 1;
    }
    const auto physical_drift = physical_guard.observe(physical2);
    if (!physical_drift.history_ready
        || physical_drift.maximum_relative_change < 0.0019
        || physical_drift.component_relative_change[4] < 0.0019) {
        std::cerr << "long-window drag drift was not detected\n";
        return 1;
    }
    dg::steady::PhysicalConvergenceGuard restored_physical({ 2, 7 });
    restored_physical.restore_history(physical_guard.history());
    const auto physical_roundtrip = restored_physical.observe(physical2);
    if (!physical_roundtrip.history_ready
        || physical_roundtrip.component_relative_change[4] <= 0.0) {
        std::cerr << "physical convergence history did not round-trip\n";
        return 1;
    }

    dg::steady::RunStatistics run;
    run.record_nonlinear_failure("GMRES did not converge");
    run.record_nonlinear_failure("Newton line search failed");
    run.record_nonlinear_failure("Newton did not converge");
    if (run.gmres_failures != 1 || run.line_search_failures != 1
        || run.newton_failures != 1) {
        std::cerr << "nonlinear failure classification is incorrect\n";
        return 1;
    }

    dg::steady::ProfileStatistics profile;
    profile.gmres_seconds = 10.0;
    profile.gmres_matvec_seconds = 3.0;
    profile.gmres_preconditioner_apply_seconds = 4.0;
    if (std::abs(profile.gmres_vector_overhead_seconds() - 3.0) > 1e-15) {
        std::cerr << "GMRES derived timing is incorrect\n";
        return 1;
    }
    return 0;
}
