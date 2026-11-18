#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace dg::nonlinear {

struct SteadyResidualConvergencePolicy {
    double ns_absolute_tolerance = 1e-6;
    double sa_absolute_tolerance = 1e-6;

    bool ns_converged(double ns_residual) const {
        return std::isfinite(ns_residual)
            && ns_residual < ns_absolute_tolerance;
    }

    bool sa_converged(double sa_residual) const {
        return std::isfinite(sa_residual)
            && sa_residual < sa_absolute_tolerance;
    }

    bool converged(double ns_residual, double sa_residual) const
    {
        return ns_converged(ns_residual)
            && sa_converged(sa_residual);
    }
};

struct NewtonConvergencePolicy {
    double absolute_tolerance = 1e-6;
    double reduction_tolerance = 1e-1;

    bool converged(int iteration,
        double residual,
        double relative_residual,
        bool outer_residual_converged) const
    {
        const bool allow_iteration_zero_absolute = iteration > 0 || outer_residual_converged;
        return (allow_iteration_zero_absolute && residual < absolute_tolerance)
            || (iteration > 0 && relative_residual < reduction_tolerance);
    }

    bool converged_after_update(double residual, double relative_residual) const {
        return residual < absolute_tolerance || relative_residual < reduction_tolerance;
    }
};

enum class JacobianRebuildReason {
    none,
    not_built,
    preconditioner_not_ready,
    newton_stagnation,
    state_change,
    age,
    cfl_change,
    dtau_change
};

inline const char* rebuild_reason_name(JacobianRebuildReason reason) {
    switch (reason) {
    case JacobianRebuildReason::none: return "none";
    case JacobianRebuildReason::not_built: return "not_built";
    case JacobianRebuildReason::preconditioner_not_ready: return "preconditioner_not_ready";
    case JacobianRebuildReason::newton_stagnation: return "newton_stagnation";
    case JacobianRebuildReason::state_change: return "state_change";
    case JacobianRebuildReason::age: return "age";
    case JacobianRebuildReason::cfl_change: return "cfl_change";
    case JacobianRebuildReason::dtau_change: return "dtau_change";
    }
    return "unknown";
}

struct JacobianReusePolicy {
    int maximum_age = 500;
    double cfl_relative_tolerance = 1.0;
    double dtau_relative_tolerance = 1.0;
    double state_relative_tolerance = 1e-2;
    double stagnation_ratio = 0.8;
};

struct JacobianReuseState {
    bool matrix_needed = true;
    bool matrix_available = false;
    bool preconditioner_ready = false;
    int steady_step = 0;
    int newton_iteration = 0;
    int last_build_step = -1;
    double current_cfl = 0.0;
    double build_cfl = 0.0;
    double current_dtau_min = 0.0;
    double current_dtau_max = 0.0;
    double build_dtau_min = 0.0;
    double build_dtau_max = 0.0;
    double state_relative_change = 0.0;
    double current_residual = std::numeric_limits<double>::infinity();
    double previous_residual = std::numeric_limits<double>::infinity();
};

struct JacobianReuseDecision {
    bool rebuild = false;
    JacobianRebuildReason reason = JacobianRebuildReason::none;
};

inline double relative_change(double current, double reference) {
    if (reference <= 0.0 || !std::isfinite(reference) || !std::isfinite(current)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::abs(current - reference) / std::max(std::abs(reference), 1e-300);
}

inline JacobianReuseDecision decide_jacobian_rebuild(
    const JacobianReusePolicy& policy,
    const JacobianReuseState& state)
{
    if (!state.matrix_needed) return {};

    const bool first_newton_iteration = state.newton_iteration == 0;
    const bool age_due = first_newton_iteration
        && (state.steady_step - state.last_build_step) >= policy.maximum_age;
    const bool cfl_changed = first_newton_iteration
        && relative_change(state.current_cfl, state.build_cfl) > policy.cfl_relative_tolerance;
    const bool dtau_changed = first_newton_iteration
        && (relative_change(state.current_dtau_min, state.build_dtau_min)
                > policy.dtau_relative_tolerance
            || relative_change(state.current_dtau_max, state.build_dtau_max)
                > policy.dtau_relative_tolerance);
    const bool solution_changed = first_newton_iteration
        && state.matrix_available
        && state.state_relative_change > policy.state_relative_tolerance;
    const bool stagnated = state.newton_iteration > 1
        && std::isfinite(state.previous_residual)
        && state.current_residual > state.previous_residual * policy.stagnation_ratio;

    if (!state.matrix_available) return { true, JacobianRebuildReason::not_built };
    if (!state.preconditioner_ready) {
        return { true, JacobianRebuildReason::preconditioner_not_ready };
    }
    if (stagnated) return { true, JacobianRebuildReason::newton_stagnation };
    if (solution_changed) return { true, JacobianRebuildReason::state_change };
    if (age_due) return { true, JacobianRebuildReason::age };
    if (cfl_changed) return { true, JacobianRebuildReason::cfl_change };
    if (dtau_changed) return { true, JacobianRebuildReason::dtau_change };
    return {};
}

struct InexactNewtonPolicy {
    double minimum_linear_tolerance = 1e-4;
    double maximum_linear_tolerance = 2e-2;
    double turbulence_dominant_maximum = 1e-2;
    double forcing_factor = 0.05;

    double linear_tolerance(double relative_nonlinear_residual,
        bool turbulence_dominant) const
    {
        const double maximum = turbulence_dominant
            ? turbulence_dominant_maximum
            : maximum_linear_tolerance;
        return std::min(maximum,
            std::max(minimum_linear_tolerance,
                forcing_factor * relative_nonlinear_residual));
    }

    double update_residual_gate(bool turbulence_dominant) const {
        return turbulence_dominant
            ? turbulence_dominant_maximum
            : maximum_linear_tolerance;
    }
};

struct BacktrackingConfig {
    int maximum_trials = 12;
    double minimum_step = 1.0 / 4096.0;
    double contraction = 0.5;
    double armijo_factor = 1e-4;
    double relaxed_step_threshold = 0.25;
};

template <class State>
struct BacktrackingResult {
    bool accepted = false;
    double step = 0.0;
    double residual = std::numeric_limits<double>::infinity();
    double best_rejected_ratio = std::numeric_limits<double>::infinity();
    int trials = 0;
    State state;
};

template <class State, class ApplyUpdate, class IsValid, class EvaluateResidual>
BacktrackingResult<State> backtracking_line_search(
    const State& base_state,
    double base_residual,
    const BacktrackingConfig& config,
    ApplyUpdate&& apply_update,
    IsValid&& is_valid,
    EvaluateResidual&& evaluate_residual)
{
    BacktrackingResult<State> result;
    result.residual = base_residual;
    result.state = base_state;
    double step = 1.0;
    for (int trial = 0; trial < config.maximum_trials; ++trial) {
        ++result.trials;
        State candidate = base_state;
        apply_update(candidate, step);
        if (!is_valid(candidate)) {
            step *= config.contraction;
            if (step < config.minimum_step) break;
            continue;
        }

        const double candidate_residual = evaluate_residual(candidate);
        const bool sufficient_decrease =
            candidate_residual <= base_residual * (1.0 - config.armijo_factor * step)
            || (step <= config.relaxed_step_threshold
                && candidate_residual < base_residual);
        if (std::isfinite(candidate_residual) && sufficient_decrease) {
            result.accepted = true;
            result.step = step;
            result.residual = candidate_residual;
            result.state = std::move(candidate);
            return result;
        }
        if (std::isfinite(candidate_residual)) {
            result.best_rejected_ratio = std::min(result.best_rejected_ratio,
                candidate_residual / std::max(base_residual, 1e-300));
        }
        step *= config.contraction;
        if (step < config.minimum_step) break;
    }
    return result;
}

} // namespace dg::nonlinear
