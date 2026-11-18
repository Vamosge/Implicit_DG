#include "dg/solver.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv)
{
    dg::RunMode mode = dg::RunMode::solve;
    int profile_steps = 0;
    int max_steps = 0;
    bool benchmark_requested = false;
    bool profile_requested = false;
    bool max_steps_requested = false;
    bool check_requested = false;
    bool residual_check_requested = false;
    bool jacobian_check_requested = false;
    dg::SolverOptions options;
    auto parse_positive_integer = [](std::string_view option,
        std::string_view value, int& destination) {
        const char* begin = value.data();
        const char* end = begin + value.size();
        const auto parsed = std::from_chars(begin, end, destination);
        if (parsed.ec != std::errc{} || parsed.ptr != end || destination <= 0) {
            std::cerr << "Invalid " << option << " value: " << value << "\n";
            return false;
        }
        return true;
    };
    auto parse_positive_double = [](std::string_view option,
        std::string_view value, double& destination) {
        std::string text(value);
        char* end = nullptr;
        destination = std::strtod(text.c_str(), &end);
        if (end != text.c_str() + text.size() || !std::isfinite(destination)
            || destination <= 0.0) {
            std::cerr << "Invalid " << option << " value: " << value << "\n";
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--benchmark") {
            mode = dg::RunMode::benchmark;
            benchmark_requested = true;
        }
        else if (argument == "--check-case") {
            mode = dg::RunMode::check;
            check_requested = true;
        }
        else if (argument == "--residual-check") {
            mode = dg::RunMode::residual_check;
            residual_check_requested = true;
        }
        else if (argument == "--jacobian-check") {
            mode = dg::RunMode::jacobian_check;
            jacobian_check_requested = true;
        }
        else if (argument == "--profile-steps") {
            if (i + 1 >= argc) {
                std::cerr << "--profile-steps requires a positive integer\n";
                return 2;
            }
            const std::string_view value(argv[++i]);
            if (!parse_positive_integer("--profile-steps", value, profile_steps)) return 2;
            mode = dg::RunMode::profile;
            profile_requested = true;
        }
        else if (argument == "--max-steps") {
            if (i + 1 >= argc) {
                std::cerr << "--max-steps requires a positive integer\n";
                return 2;
            }
            const std::string_view value(argv[++i]);
            if (!parse_positive_integer("--max-steps", value, max_steps)) return 2;
            max_steps_requested = true;
        }
        else if (argument == "--ns-tolerance") {
            if (i + 1 >= argc || !parse_positive_double("--ns-tolerance",
                    argv[++i], options.steady_ns_tolerance)) return 2;
        }
        else if (argument == "--sa-tolerance") {
            if (i + 1 >= argc || !parse_positive_double("--sa-tolerance",
                    argv[++i], options.steady_sa_tolerance)) return 2;
        }
        else if (argument == "--cfl-max") {
            if (i + 1 >= argc) {
                std::cerr << "--cfl-max requires a positive number\n";
                return 2;
            }
            if (!parse_positive_double("--cfl-max", argv[++i],
                    options.cfl_maximum)) return 2;
        }
        else if (argument == "--cfl-initial") {
            if (i + 1 >= argc) {
                std::cerr << "--cfl-initial requires a positive number\n";
                return 2;
            }
            if (!parse_positive_double("--cfl-initial", argv[++i],
                    options.cfl_initial)) return 2;
            options.cfl_initial_overridden = true;
        }
        else if (argument == "--sa-cfl-scale") {
            if (i + 1 >= argc) {
                std::cerr << "--sa-cfl-scale requires a positive number\n";
                return 2;
            }
            if (!parse_positive_double("--sa-cfl-scale", argv[++i],
                    options.sa_cfl_scale)) return 2;
        }
        else if (argument == "--sa-br2-factor") {
            if (i + 1 >= argc) {
                std::cerr << "--sa-br2-factor requires a positive number\n";
                return 2;
            }
            if (!parse_positive_double("--sa-br2-factor", argv[++i],
                    options.sa_br2_factor)) return 2;
        }
        else if (argument == "--sa-q2-reset-on-restart") {
            options.sa_q2_reset_on_restart = true;
        }
        else if (argument == "--freeze-ns") {
            options.freeze_ns = true;
        }
        else if (argument == "--mach") {
            if (i + 1 >= argc || !parse_positive_double("--mach", argv[++i],
                    options.mach_number)) return 2;
        }
        else if (argument == "--reynolds") {
            if (i + 1 >= argc || !parse_positive_double("--reynolds", argv[++i],
                    options.reynolds_number)) return 2;
        }
        else if (argument == "--alpha-deg") {
            if (i + 1 >= argc) {
                std::cerr << "--alpha-deg requires a finite number\n";
                return 2;
            }
            std::string text(argv[++i]);
            char* end = nullptr;
            options.angle_of_attack_degrees = std::strtod(text.c_str(), &end);
            if (end != text.c_str() + text.size()
                || !std::isfinite(options.angle_of_attack_degrees)) {
                std::cerr << "Invalid --alpha-deg value: " << text << "\n";
                return 2;
            }
        }
        else if (argument == "--cfl-policy") {
            if (i + 1 >= argc) {
                std::cerr << "--cfl-policy requires legacy or residual-aware\n";
                return 2;
            }
            const std::string_view value(argv[++i]);
            if (value == "legacy") options.residual_aware_cfl = false;
            else if (value == "residual-aware") options.residual_aware_cfl = true;
            else {
                std::cerr << "Invalid --cfl-policy value: " << value << "\n";
                return 2;
            }
        }
        else if (argument == "--sa-limiter") {
            if (i + 1 >= argc) {
                std::cerr << "--sa-limiter requires floor or finite-only\n";
                return 2;
            }
            const std::string_view value(argv[++i]);
            if (value == "floor") options.limit_negative_sa = true;
            else if (value == "finite-only") options.limit_negative_sa = false;
            else {
                std::cerr << "Invalid --sa-limiter value: " << value << "\n";
                return 2;
            }
        }
        else if (argument == "--restart") {
            if (i + 1 >= argc) {
                std::cerr << "--restart requires a checkpoint or solution file\n";
                return 2;
            }
            options.restart_file = argv[++i];
        }
        else if (argument == "--allow-legacy-checkpoint") {
            options.allow_legacy_checkpoint = true;
        }
        else if (argument == "--checkpoint-file") {
            if (i + 1 >= argc) {
                std::cerr << "--checkpoint-file requires a path\n";
                return 2;
            }
            options.checkpoint_file = argv[++i];
        }
        else if (argument == "--checkpoint-every") {
            if (i + 1 >= argc) {
                std::cerr << "--checkpoint-every requires a positive integer\n";
                return 2;
            }
            if (!parse_positive_integer("--checkpoint-every", argv[++i],
                    options.checkpoint_every)) return 2;
        }
        else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: gsx_dg [--check-case | --residual-check | --jacobian-check | --benchmark | --profile-steps N | --max-steps N]\n"
                      << "              [--cfl-max VALUE] [--cfl-policy legacy|residual-aware]\n"
                      << "              [--ns-tolerance VALUE] [--sa-tolerance VALUE]\n"
                      << "              [--sa-cfl-scale VALUE] [--sa-br2-factor VALUE]\n"
                      << "              [--sa-q2-reset-on-restart]\n"
                      << "              [--freeze-ns]\n"
                      << "              [--sa-limiter floor|finite-only]\n"
                      << "              [--mach VALUE] [--reynolds VALUE] [--alpha-deg VALUE]\n"
                      << "              [--restart FILE] [--allow-legacy-checkpoint]\n"
                      << "              [--checkpoint-every N]\n"
                      << "  --check-case  Validate the compiled case, mesh, geometry, and AD coloring, then exit.\n"
                      << "  --residual-check  Evaluate and summarize one initial DG RANS-SA residual, then exit.\n"
                      << "  --jacobian-check  Compare the complete AD backward-Euler Jacobian with centered finite differences, optionally at --restart FILE.\n"
                      << "  --benchmark  Measure five warm residuals and one "
                         "AD Jacobian/preconditioner build, then exit.\n"
                      << "  --profile-steps N  Run at most N real implicit "
                         "steps, report phase timings, and suppress output files.\n"
                      << "  --max-steps N  Run at most N normal implicit steps "
                         "and write all regular solution and history files.\n"
                      << "  --ns-tolerance VALUE  Scaled steady NS residual limit (default: 1e-6).\n"
                      << "  --sa-tolerance VALUE  Scaled steady SA residual limit (default: 1e-6).\n"
                      << "  --cfl-max VALUE  Set the pseudo-time CFL ceiling.\n"
                      << "  --cfl-initial VALUE  Explicitly override the starting CFL, including a checkpoint's stored CFL.\n"
                      << "  --cfl-policy residual-aware  Hold or shrink CFL when SA moves away from its best residual.\n"
                      << "  --sa-cfl-scale VALUE  Multiply the pseudo-time step only for the SA equation (default: 1).\n"
                      << "  --sa-br2-factor VALUE  Multiply only the SA BR2 lifting correction (default: 1).\n"
                      << "  --sa-q2-reset-on-restart  Diagnostic: remove degree-two SA modes once after loading a restart while preserving curved-cell averages.\n"
                      << "  --freeze-ns  Diagnostic: hold all Navier--Stokes unknowns at the restart state and solve only SA.\n"
                      << "  --sa-limiter finite-only  Default: let finite negative SA-neg states evolve without an artificial floor.\n"
                      << "  --sa-limiter floor  Legacy diagnostic: project sufficiently negative SA states after a step.\n"
                      << "  --mach VALUE  Override the validation-case freestream Mach number.\n"
                      << "  --reynolds VALUE  Override the validation-case Reynolds number.\n"
                      << "  --alpha-deg VALUE  Override angle of attack in degrees.\n"
                      << "  --restart FILE  Continue from checkpoint_latest.bin or a solution_tend.dat snapshot.\n"
                      << "  --allow-legacy-checkpoint  Permit an unverifiable version-1 binary checkpoint.\n"
                      << "  --checkpoint-file FILE  Set the rolling binary checkpoint path.\n"
                      << "  --checkpoint-every N  Write the rolling checkpoint every N accepted steps.\n";
            return 0;
        }
        else {
            std::cerr << "Unknown option: " << argument << "\n"
                      << "Use --help for supported options.\n";
            return 2;
        }
    }
    const int exclusive_modes = static_cast<int>(benchmark_requested)
        + static_cast<int>(profile_requested)
        + static_cast<int>(max_steps_requested)
        + static_cast<int>(check_requested)
        + static_cast<int>(residual_check_requested)
        + static_cast<int>(jacobian_check_requested);
    if (exclusive_modes > 1) {
        std::cerr << "--check-case, --residual-check, --jacobian-check, --benchmark, --profile-steps, and --max-steps are mutually exclusive\n";
        return 2;
    }
    if ((check_requested || residual_check_requested)
        && !options.restart_file.empty()) {
        std::cerr << "case and residual checks do not load restart files\n";
        return 2;
    }
    if (options.sa_q2_reset_on_restart && options.restart_file.empty()) {
        std::cerr << "--sa-q2-reset-on-restart requires --restart FILE\n";
        return 2;
    }
    if (options.freeze_ns && options.restart_file.empty()) {
        std::cerr << "--freeze-ns requires --restart FILE\n";
        return 2;
    }
    options.requested_steps = profile_requested ? profile_steps : max_steps;
    try {
        return dg::run_solver(mode, options);
    }
    catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << "\n";
        return 1;
    }
}
