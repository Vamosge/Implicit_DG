#include "dg/nonlinear_solver.hpp"
#include "dg/solver.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace dg::nonlinear;

    if (dg::SolverOptions{}.limit_negative_sa) {
        std::cerr << "legacy SA floor unexpectedly became the default\n";
        return 1;
    }
    if (dg::SolverOptions{}.steady_ns_tolerance != 1e-6
        || dg::SolverOptions{}.steady_sa_tolerance != 1e-6) {
        std::cerr << "steady residual defaults are not the documented 1e-6\n";
        return 1;
    }

    NewtonConvergencePolicy convergence{ 1e-6, 1e-2 };
    if (convergence.converged(0, 1e-8, 1.0, false)) {
        std::cerr << "iteration-zero convergence bypassed the outer-residual gate\n";
        return 1;
    }
    if (!convergence.converged(1, 1.0, 1e-3, false)) {
        std::cerr << "relative Newton convergence was not detected\n";
        return 1;
    }

    SteadyResidualConvergencePolicy steady{ 1e-3, 1e-3 };
    if (steady.converged(9e-4, 1e-2)) {
        std::cerr << "NS-only convergence incorrectly bypassed the SA criterion\n";
        return 1;
    }
    if (!steady.converged(9e-4, 9e-4)) {
        std::cerr << "coupled NS/SA absolute convergence was not detected\n";
        return 1;
    }
    if (steady.converged(9e-4, 9e-3)) {
        std::cerr << "relative SA reduction incorrectly bypassed the absolute criterion\n";
        return 1;
    }

    JacobianReusePolicy reuse;
    JacobianReuseState reuse_state;
    JacobianReuseDecision decision = decide_jacobian_rebuild(reuse, reuse_state);
    if (!decision.rebuild || decision.reason != JacobianRebuildReason::not_built) {
        std::cerr << "missing Jacobian did not request a rebuild\n";
        return 1;
    }
    reuse_state.matrix_available = true;
    reuse_state.preconditioner_ready = true;
    reuse_state.last_build_step = 0;
    reuse_state.build_cfl = reuse_state.current_cfl = 5.0;
    reuse_state.build_dtau_min = reuse_state.current_dtau_min = 0.1;
    reuse_state.build_dtau_max = reuse_state.current_dtau_max = 0.2;
    decision = decide_jacobian_rebuild(reuse, reuse_state);
    if (decision.rebuild) {
        std::cerr << "fresh Jacobian was rebuilt unnecessarily\n";
        return 1;
    }
    reuse_state.newton_iteration = 2;
    reuse_state.previous_residual = 1.0;
    reuse_state.current_residual = 0.9;
    decision = decide_jacobian_rebuild(reuse, reuse_state);
    if (!decision.rebuild || decision.reason != JacobianRebuildReason::newton_stagnation) {
        std::cerr << "Newton stagnation did not request a rebuild\n";
        return 1;
    }

    InexactNewtonPolicy forcing{ 1e-4, 2e-2, 1e-2, 0.05 };
    if (std::abs(forcing.linear_tolerance(0.1, false) - 0.005) > 1e-15) {
        std::cerr << "inexact-Newton forcing tolerance is incorrect\n";
        return 1;
    }

    const BacktrackingConfig line_search{ 12, 1.0 / 4096.0, 0.5, 1e-4, 0.25 };
    const double base = 1.0;
    const auto result = backtracking_line_search(
        base,
        base * base,
        line_search,
        [](double& state, double step) { state += step * (-2.0); },
        [](double state) { return std::isfinite(state); },
        [](double state) { return state * state; });
    if (!result.accepted || result.trials != 2 || std::abs(result.step - 0.5) > 1e-15
        || std::abs(result.state) > 1e-15) {
        std::cerr << "backtracking line search did not accept the expected half step\n";
        return 1;
    }

    return 0;
}
