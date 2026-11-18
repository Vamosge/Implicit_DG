#pragma once

#include <string>
#include <limits>

namespace dg {

enum class RunMode {
    solve,
    benchmark,
    profile,
    check,
    residual_check,
    jacobian_check
};

struct SolverOptions {
    int requested_steps = 0;
    // Dimensionless, physically scaled steady spatial-residual tolerances.
    // These are deliberately separate from the inner Newton/GMRES tolerances.
    double steady_ns_tolerance = 1e-6;
    double steady_sa_tolerance = 1e-6;
    double cfl_initial = 5.0;
    bool cfl_initial_overridden = false;
    double cfl_maximum = 100000.0;
    // Multiplies the local pseudo-time step only in the SA equation. A value
    // below one damps the stiff turbulence update without slowing the four
    // Navier--Stokes equations.
    double sa_cfl_scale = 1.0;
    // Multiplies the BR2 lifting correction only in the SA conservative
    // variable. One preserves the baseline discretization.
    double sa_br2_factor = 1.0;
    bool sa_q2_reset_on_restart = false;
    // Diagnostic continuation: hold density, momentum, and energy exactly at
    // the restart state while solving only the SA transport equation.
    bool freeze_ns = false;
    bool residual_aware_cfl = false;
    // SA-neg is the implemented model. Finite negative working-variable
    // states therefore use its negative branch by default; the legacy floor
    // projection is available only through an explicit CLI request.
    bool limit_negative_sa = false;
    bool allow_legacy_checkpoint = false;
    std::string restart_file;
    std::string checkpoint_file = "checkpoint_latest.bin";
    int checkpoint_every = 100;
    // NaN selects the compiled validation-case default.
    double mach_number = std::numeric_limits<double>::quiet_NaN();
    double reynolds_number = std::numeric_limits<double>::quiet_NaN();
    double angle_of_attack_degrees = std::numeric_limits<double>::quiet_NaN();
};

// requested_steps limits either normal solve or profile mode when positive.
// Zero selects the compiled default for a normal solve. The legacy CFL policy
// remains the default; SA-neg finite-only handling is the default limiter
// policy. The older post-step SA floor remains explicitly selectable.
int run_solver(RunMode mode = RunMode::solve,
    const SolverOptions& options = {});

} // namespace dg
