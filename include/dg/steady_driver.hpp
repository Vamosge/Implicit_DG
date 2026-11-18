#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dg::steady {

struct CflConfig {
    double initial = 1.0;
    double minimum = 1e-3;
    double maximum = 1e5;
    double rejection_factor = 0.5;
    double accepted_growth_factor = 1.0;
    double accepted_update_limit = 0.1;
};

struct CflUpdate {
    double old_value = 0.0;
    double new_value = 0.0;
    const char* action = "hold";

    bool changed() const { return new_value != old_value; }
    bool effectively_increased() const {
        return std::isfinite(new_value)
            && new_value > old_value * (1.0 + 1e-12);
    }
    bool decreased() const { return new_value < old_value; }
};

// Plan the normal CFL ramp while retaining the ability to recover after a
// residual-triggered cut.  If a continuation starts exactly at its ceiling,
// maximum/initial is one; using that ratio alone would permanently disable
// all later recovery growth.
inline double cfl_accepted_growth_factor(double initial, double maximum,
    int adaptation_steps, double minimum_recovery_span = 2.0)
{
    if (!std::isfinite(initial) || !std::isfinite(maximum)
        || !std::isfinite(minimum_recovery_span) || initial <= 0.0
        || maximum <= 0.0 || adaptation_steps <= 0
        || minimum_recovery_span < 1.0) {
        return 1.0;
    }
    const double planned_span = std::max(maximum / initial,
        minimum_recovery_span);
    return std::pow(planned_span, 1.0 / adaptation_steps);
}

struct SAResidualShrinkConfig {
    double repeat_factor = 1.5;
    double emergency_factor = 2.0;
    int cooldown_steps = 50;
};

struct SAResidualShrinkDecision {
    bool request = false;
    bool emergency = false;
};

struct ResidualGrowthConfig {
    int trend_window_steps = 12;
    double ns_best_factor = 1.05;
    double sa_best_factor = 1.05;
    double ns_window_ratio_limit = 1.0;
    double sa_window_ratio_limit = 1.0;
};

struct ResidualGrowthDecision {
    bool allow_growth = false;
    bool history_ready = false;
    double ns_window_ratio = std::numeric_limits<double>::infinity();
    double sa_window_ratio = std::numeric_limits<double>::infinity();
};

inline constexpr std::size_t physical_convergence_component_count = 7;
using PhysicalConvergenceSample =
    std::array<double, physical_convergence_component_count>;

struct PhysicalConvergenceConfig {
    int window_intervals = 10;
    int active_components = 7;
    std::array<double, physical_convergence_component_count> scale_floors = {
        1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-2, 1e-3
    };
};

struct PhysicalConvergenceDecision {
    bool history_ready = false;
    double maximum_relative_change =
        std::numeric_limits<double>::infinity();
    PhysicalConvergenceSample component_relative_change = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    };
};

// Compare each monitored output with the same output several diagnostic
// intervals earlier. A component-wise maximum prevents a large coefficient
// such as lift from hiding a meaningful slow drift in drag, moment, or skin
// friction.
class PhysicalConvergenceGuard {
public:
    explicit PhysicalConvergenceGuard(PhysicalConvergenceConfig config)
        : config_(config) {}

    PhysicalConvergenceDecision observe(
        const PhysicalConvergenceSample& sample)
    {
        const int active = std::clamp(config_.active_components, 1,
            static_cast<int>(physical_convergence_component_count));
        for (int component = 0; component < active; ++component) {
            if (!std::isfinite(sample[component])) {
                history_.clear();
                return {};
            }
        }
        history_.push_back(sample);
        const std::size_t required = static_cast<std::size_t>(
            std::max(config_.window_intervals, 1)) + 1;
        while (history_.size() > required) history_.pop_front();
        if (history_.size() < required) return {};

        PhysicalConvergenceDecision decision;
        decision.history_ready = true;
        decision.maximum_relative_change = 0.0;
        for (int component = 0; component < active; ++component) {
            const double current = sample[component];
            const double previous = history_.front()[component];
            const double scale = std::max({ std::abs(current),
                std::abs(previous), config_.scale_floors[component] });
            const double relative = std::abs(current - previous) / scale;
            decision.component_relative_change[component] = relative;
            decision.maximum_relative_change = std::max(
                decision.maximum_relative_change, relative);
        }
        return decision;
    }

    std::vector<PhysicalConvergenceSample> history() const {
        return { history_.begin(), history_.end() };
    }

    void restore_history(
        const std::vector<PhysicalConvergenceSample>& history)
    {
        history_.clear();
        const std::size_t maximum_size = static_cast<std::size_t>(
            std::max(config_.window_intervals, 1)) + 1;
        const std::size_t first = history.size() > maximum_size
            ? history.size() - maximum_size : 0;
        const int active = std::clamp(config_.active_components, 1,
            static_cast<int>(physical_convergence_component_count));
        for (std::size_t index = first; index < history.size(); ++index) {
            bool valid = true;
            for (int component = 0;
                component < active; ++component) {
                valid = valid && std::isfinite(history[index][component]);
            }
            if (valid) history_.push_back(history[index]);
        }
    }

private:
    PhysicalConvergenceConfig config_;
    std::deque<PhysicalConvergenceSample> history_;
};

// A one-step ratio does not detect a small adverse drift that persists for
// many accepted steps. Keep a short residual history and permit adaptive CFL
// growth only while both equation groups remain near their running best and
// have not risen over the complete trend window.
class ResidualGrowthGuard {
public:
    using ResidualPair = std::pair<double, double>;

    explicit ResidualGrowthGuard(ResidualGrowthConfig config)
        : config_(config) {}

    ResidualGrowthDecision observe(double ns_residual, double sa_residual,
        double best_ns_residual, double best_sa_residual)
    {
        if (!valid_residual(ns_residual) || !valid_residual(sa_residual)
            || !valid_residual(best_ns_residual)
            || !valid_residual(best_sa_residual)) {
            history_.clear();
            return {};
        }

        history_.emplace_back(ns_residual, sa_residual);
        const std::size_t required = static_cast<std::size_t>(
            std::max(config_.trend_window_steps, 1)) + 1;
        while (history_.size() > required) history_.pop_front();
        if (history_.size() < required) return {};

        ResidualGrowthDecision decision;
        decision.history_ready = true;
        decision.ns_window_ratio = ns_residual / history_.front().first;
        decision.sa_window_ratio = sa_residual / history_.front().second;
        const bool near_best =
            ns_residual <= config_.ns_best_factor * best_ns_residual
            && sa_residual <= config_.sa_best_factor * best_sa_residual;
        const bool trend_healthy =
            decision.ns_window_ratio <= config_.ns_window_ratio_limit
            && decision.sa_window_ratio <= config_.sa_window_ratio_limit;
        decision.allow_growth = near_best && trend_healthy;
        return decision;
    }

    std::vector<ResidualPair> history() const {
        return { history_.begin(), history_.end() };
    }

    void restore_history(const std::vector<ResidualPair>& history) {
        history_.clear();
        const std::size_t maximum_size = static_cast<std::size_t>(
            std::max(config_.trend_window_steps, 1)) + 1;
        const std::size_t first = history.size() > maximum_size
            ? history.size() - maximum_size : 0;
        for (std::size_t index = first; index < history.size(); ++index) {
            if (valid_residual(history[index].first)
                && valid_residual(history[index].second)) {
                history_.push_back(history[index]);
            }
        }
    }

private:
    static bool valid_residual(double value) {
        return std::isfinite(value) && value > 0.0;
    }

    ResidualGrowthConfig config_;
    std::deque<ResidualPair> history_;
};

// Ordinary residual/limiter requests respect the cooldown, preventing repeated
// CFL cuts for the same transient. A much larger, still-growing SA rebound may
// bypass it so the controller does not wait while a localized mode amplifies.
inline SAResidualShrinkDecision sa_residual_shrink_decision(
    double residual, double reference, double step_ratio,
    bool severe_limiter_event, int steps_since_shrink,
    const SAResidualShrinkConfig& config)
{
    if (!std::isfinite(residual) || !std::isfinite(reference)
        || !std::isfinite(step_ratio) || reference <= 0.0) {
        return {};
    }
    const bool rising = step_ratio > 1.0;
    const bool ordinary_residual_request = rising
        && residual > config.repeat_factor * reference;
    const bool cooldown_elapsed = steps_since_shrink >= config.cooldown_steps;
    if (cooldown_elapsed
        && (severe_limiter_event || ordinary_residual_request)) {
        return { true, false };
    }
    const bool emergency_request = rising
        && residual > config.emergency_factor * reference;
    return { emergency_request, emergency_request };
}

class CflController {
public:
    explicit CflController(CflConfig config)
        : config_(config), current_(config.initial) {}

    double current() const { return current_; }
    double& current_reference() { return current_; }

    CflUpdate reject() {
        const double old_value = current_;
        current_ = std::max(config_.minimum, current_ * config_.rejection_factor);
        return { old_value, current_, current_ < old_value ? "retry-shrink" : "hold" };
    }

    CflUpdate accept(bool nonlinear_step_healthy, double relative_update,
        bool allow_growth = true, bool request_shrink = false) {
        const double old_value = current_;
        if (request_shrink) {
            current_ = std::max(config_.minimum,
                current_ * config_.rejection_factor);
            return { old_value, current_,
                current_ < old_value ? "residual-shrink" : "fcfl-hold" };
        }
        const bool may_grow = nonlinear_step_healthy
            && std::isfinite(relative_update)
            && relative_update <= config_.accepted_update_limit
            && allow_growth;
        if (may_grow) {
            current_ = std::min(config_.maximum,
                current_ * config_.accepted_growth_factor);
            return { old_value, current_, current_ > old_value ? "fcfl-grow" : "hold" };
        }
        return { old_value, current_, "fcfl-hold" };
    }

private:
    CflConfig config_;
    double current_;
};

struct RunStatistics {
    long long total_newton_iterations = 0;
    long long total_gmres_iterations = 0;
    int rejected_steps = 0;
    int line_search_failures = 0;
    int gmres_failures = 0;
    int newton_failures = 0;
    int invalid_raw_failures = 0;
    int invalid_post_failures = 0;
    int cfl_grows = 0;
    int cfl_shrinks = 0;

    void record_nonlinear_failure(const std::string& reason) {
        if (reason.find("GMRES") != std::string::npos) ++gmres_failures;
        else if (reason.find("line search") != std::string::npos) ++line_search_failures;
        else ++newton_failures;
    }
};

struct ProfileStatistics {
    double dtau_seconds = 0.0;
    double jacobian_seconds = 0.0;
    double preconditioner_build_seconds = 0.0;
    double gmres_seconds = 0.0;
    double gmres_matvec_seconds = 0.0;
    double gmres_preconditioner_apply_seconds = 0.0;
    double line_search_seconds = 0.0;
    double stabilization_seconds = 0.0;
    long long jacobian_builds = 0;
    long long gmres_calls = 0;
    long long gmres_matvec_calls = 0;
    long long gmres_preconditioner_apply_calls = 0;
    long long line_search_trials = 0;
    int attempted_steps = 0;
    int accepted_steps = 0;

    double gmres_vector_overhead_seconds() const {
        return std::max(0.0,
            gmres_seconds - gmres_matvec_seconds - gmres_preconditioner_apply_seconds);
    }
};

} // namespace dg::steady
