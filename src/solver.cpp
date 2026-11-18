// 2D compressible DG RANS-SA solver.
// Q1/Q2/Q3 nodal quadrilateral DG, HLLC convection, BR2 viscosity, and implicit BE/Newton/GMRES marching.
// Core data flow: solver.K -> computeRHS() -> solver.rhs -> Newton/GMRES -> solver.K.

#include "dg/case_config.hpp"
#include "dg/ad_jacobian.hpp"
#include "dg/ad_physics.hpp"
#include "dg/block_ad.hpp"
#include "dg/boundary_conditions.hpp"
#include "dg/boundary_physics.hpp"
#include "dg/cartesian_topology.hpp"
#include "dg/collectives.hpp"
#include "dg/cgns_reader.hpp"
#include "dg/gmsh_reader.hpp"
#include "dg/diagnostics.hpp"
#include "dg/field.hpp"
#include "dg/linear_solver.hpp"
#include "dg/mapped_geometry.hpp"
#include "dg/modal_sensor.hpp"
#include "dg/nonlinear_solver.hpp"
#include "dg/physics.hpp"
#include "dg/reference_element.hpp"
#include "dg/residual_workspace.hpp"
#include "dg/solver.hpp"
#include "dg/steady_driver.hpp"
#include "dg/stencil_coloring.hpp"
#include "dg/viscous_sa.hpp"
#include "dg/wall_geometry.hpp"

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <limits>
#include <functional>
#include <string>
#include <utility>
#include <cstdlib>
#include <cstdint>
#include <cstdio>

#ifndef DG_HAS_OPENMP
#define DG_HAS_OPENMP 0
#endif

#if DG_HAS_OPENMP
#include <omp.h>
#endif


// Compile-time configuration.
#ifndef CGNS_GRID_FILE
#if defined(DG_CASE_BUMP_CHANNEL)
#define CGNS_GRID_FILE "bump_struct_89x41_vol.cgns"
#else
#define CGNS_GRID_FILE "grid_quad_35x25_vol.cgns"
#endif
#endif
#ifndef GMSH_GRID_FILE
#define GMSH_GRID_FILE "naca0012_familyII_225x65.msh"
#endif
#ifndef CGNS_GEOMETRY_GRID_FILE
#define CGNS_GEOMETRY_GRID_FILE "n0012_449-129.cgns"
#endif
#if defined(DG_CASE_NACA0012) && !defined(DG_NACA_OFFICIAL_CGNS)
constexpr const char* COMPILED_MESH_FILE = GMSH_GRID_FILE;
#else
constexpr const char* COMPILED_MESH_FILE = CGNS_GRID_FILE;
#endif



// Solver, physical, CFL, and turbulence-model parameters.
#ifndef NX_CELLS
#if defined(DG_CASE_BUMP_CHANNEL)
#define NX_CELLS 88
#else
#define NX_CELLS 34
#endif
#endif
#ifndef NY_CELLS
#if defined(DG_CASE_BUMP_CHANNEL)
#define NY_CELLS 40
#else
#define NY_CELLS 24
#endif
#endif
constexpr int Nx = NX_CELLS;
constexpr int Ny = NY_CELLS;
#ifndef DG_ORDER
#define DG_ORDER 2
#endif
#ifndef PRECONDITIONER_CHOICE
#define PRECONDITIONER_CHOICE 2
#endif
#ifndef JACOBIAN_MATVEC_CHOICE
#define JACOBIAN_MATVEC_CHOICE 1
#endif
#if DG_ORDER < 1 || DG_ORDER > 3
#error "DG_ORDER must be 1, 2, or 3 for the current Q1/Q2/Q3 tensor-product basis implementation."
#endif
#if PRECONDITIONER_CHOICE < 0 || PRECONDITIONER_CHOICE > 2
#error "PRECONDITIONER_CHOICE must be 0:none, 1:LU-SGS, or 2:block ILU(0)."
#endif
#if JACOBIAN_MATVEC_CHOICE < 0 || JACOBIAN_MATVEC_CHOICE > 2
#error "JACOBIAN_MATVEC_CHOICE must be 0:finite-difference A*v, 1:AD block Jacobian A*v, or 2:multi-secant-corrected AD block Jacobian A*v."
#endif
constexpr int NumEq = 5;
constexpr int PolyOrder = DG_ORDER;
constexpr int Nodes1D = PolyOrder + 1;
constexpr int dimPk = Nodes1D * Nodes1D;
constexpr int GeomOrder = 4;
constexpr int GeomNodes1D = GeomOrder + 1;
constexpr int GeomDimPk = GeomNodes1D * GeomNodes1D;
constexpr int Quad1D = (PolyOrder == 2 ? 5 : Nodes1D);
constexpr int NumQuad = Quad1D * Quad1D;
constexpr int FaceQuad = Quad1D;
static_assert(Quad1D >= 2 && Quad1D <= 5, "gauss_points_1d supports only 2 to 5 quadrature points.");
constexpr int NumCells = Nx * Ny;
constexpr int BlockSize = dimPk * NumEq;
constexpr int NumDofs = NumCells * BlockSize;
const dg::CartesianTopology g_topology(Nx, Ny, dg::case_config::wake_pairs);
const dg::SerialCollectives g_collectives;

// Diagnostics counters for positivity fixes and limiter activations.
static long long g_rho_posfix = 0;
static long long g_p_posfix = 0;
static long long g_nut_posfix = 0;
static long long g_limiter_ns_hits = 0;
static long long g_limiter_sa_hits = 0;

// Enabled only by --profile-steps. The top-level residual calls are serial;
// their internal cell loops may use OpenMP and finish before the accumulator
// is updated, so no atomic operations are required here.
static bool g_profile_rhs_timing_enabled = false;
static double g_profile_rhs_seconds = 0.0;
static long long g_profile_rhs_calls = 0;


// Case bounds are used for validation; physical coordinates come from CGNS.
const double xa = dg::case_config::x_min, xb = dg::case_config::x_max;
const double ya = dg::case_config::y_min, yb = dg::case_config::y_max;


// Freestream, nonlinear-solver, and linear-solver controls.
constexpr double gamma_v = 1.4;
constexpr double gamma1 = gamma_v - 1.0;
double M_inf = dg::case_config::mach_number;
double Re = dg::case_config::reynolds_number;
const double rho_inf = 1.0;
const double p_inf = 1.0 / gamma_v;
const double a_inf = std::sqrt(gamma_v * p_inf / rho_inf);
double u_inf = M_inf * a_inf;
double v_inf = 0.0;
double inflow_angle = 0.0;
const double T_inf = p_inf / rho_inf;
double T0_inf = T_inf * (1.0 + 0.5 * gamma1 * M_inf * M_inf);
double p0_inf = p_inf * std::pow(T0_inf / T_inf, gamma_v / gamma1);
dg::BoundaryConditionSet g_boundary_conditions =
    dg::make_channel_boundary_conditions(p_inf, p0_inf, T0_inf, inflow_angle,
        dg::case_config::upper_boundary_is_symmetry
            ? dg::BoundaryType::SlipWall
            : dg::BoundaryType::Farfield);
const double Pr = 0.72;
const int STEADY_MAX_STEPS = 1000;
const bool STOP_ON_RESIDUAL_CONVERGENCE = true;
#if defined(DG_CASE_NACA0012) || defined(DG_CASE_FLAT_PLATE)
// Validation cases must be steady in their reported engineering outputs as
// well as in the algebraic residual.  For the flat plate this guard monitors
// the three Cf stations and the wall-mean Cf; for NACA it additionally
// monitors Cd, Cl, and Cm.
const bool REQUIRE_PHYSICAL_CONVERGENCE = true;
#else
// The bump/channel case does not yet have a case-specific physical-output
// convergence metric, so retain its established residual-only policy.
const bool REQUIRE_PHYSICAL_CONVERGENCE = false;
#endif
const double CFL_INIT = 5;
const double CFL_SHRINK = 0.5;
const double CFL_MIN = 1.0e-3;
const int CFL_ADAPT_STEP = 500;
const double CFL_RECOVERY_SPAN = 2.0;
const double FCFL_LIMIT = 0.1;
const int CFL_RESIDUAL_TREND_WINDOW_STEPS = 12;
const double CFL_NS_GROW_BEST_FACTOR = 1.05;
const double CFL_SA_GROW_BEST_FACTOR = 1.05;
const double CFL_SA_GROW_STEP_RATIO = 1.02;
const double CFL_NS_GROW_WINDOW_RATIO = 1.0;
const double CFL_SA_GROW_WINDOW_RATIO = 1.0;
const double CFL_SA_REPEAT_SHRINK_FACTOR = 1.5;
const double CFL_SA_EMERGENCY_SHRINK_FACTOR = 2.0;
const int CFL_SA_SHRINK_COOLDOWN_STEPS = 50;
const double CFL_SA_LIMITER_SEVERE_THETA = 0.95;
const int NEWTON_MAX_ITERS = 8;
const double NEWTON_RES_TOL = 1e-6;
const double NEWTON_RES_REDUCTION_TOL = 1e-1;
const double NOOP_RELATIVE_UPDATE_TOL =
    64.0 * std::numeric_limits<double>::epsilon();
const int GMRES_RESTART = 30;
const int GMRES_MAX_ITERS = 120;
const double GMRES_TOL = 1e-4;
const double GMRES_INEXACT_TOL_MAX = 2e-2;
const double GMRES_UPDATE_TOL = 1e-8;
const double GMRES_STRICT_TOL_SLACK = 2.0;
const int LINE_SEARCH_MAX_ITERS = 12;
const double LINE_SEARCH_MIN_ALPHA = 1.0 / 4096.0;
const int JAC_REBUILD_EVERY = 500;
const double JAC_REBUILD_CFL_REL_TOL = 1.0;
const double JAC_REBUILD_DTAU_REL_TOL = 1.0;
const double JAC_REBUILD_STATE_REL_TOL = 1e-2;
const double NEWTON_JAC_STAGNATION_RATIO = 0.80;
const int NEWTON_LINESEARCH_REBUILD_RETRIES = 1;
const double SA_SOURCE_DIAG_FACTOR = 0.3;
const double SA_SOURCE_DIAG_MAX = 1e12;
const double ILU0_DIAG_SHIFT = 1e-8;
const double FD_MATVEC_EPS = 1e-7;
const int SECANT_HISTORY_SIZE = 5;
const double SECANT_CORRECTION_DAMPING = 0.5;
const double SECANT_GRAM_REG = 1e-12;
const int CONSOLE_STEP_EVERY = 50;
const int DIAGNOSTIC_STEP_EVERY = 50;
const double SA_DIAGNOSTIC_NEAR_WALL_DISTANCE = 0.05;
const double PHYSICAL_CONVERGENCE_RELATIVE_TOL = 1e-3;
const int PHYSICAL_CONVERGENCE_HOLD_CHECKS = 2;
const int PHYSICAL_CONVERGENCE_WINDOW_INTERVALS = 10;
constexpr std::array<double, 3> CF_MONITOR_X = {
    0.6321975, 0.75, 0.8678025
};
double mu_ref = rho_inf * M_inf * a_inf / Re;
const double BR2_ETA = 3.0;
double g_sa_br2_factor = 1.0;
const double Pr_t = 0.9;
const double SA_LIMITER_AVG_FLOOR_FACTOR = 0.9;
const double SA_LIMITER_POINT_HARD_FLOOR_FACTOR = 0.9;
const double SA_LIMITER_POINT_REL_TOL = 0.0;

// Fixed robust implicit route: damped Newton/GMRES with
// line search, trial-state checks, Jacobian retry, solution stabilization, and
// FCFL adaptive CFL growth.
constexpr const char* ROBUST_SOLVER_GROUP = "LS_trialcheck_JacRetry_stabilization_FCFL";
constexpr const char* ROBUST_CFL_MODE_NAME = "FCFL";
constexpr const char* JACOBIAN_REUSE_MODE_NAME = "dynamic_rebuild";

enum class PreconditionerMode {
    NONE,
    LU_SGS,
    BLOCK_ILU0
};

enum class JacobianMatvecMode {
    FINITE_DIFFERENCE,
    AD_BLOCK,
    MULTI_SECANT_CORRECTED_AD
};

const PreconditionerMode PRECONDITIONER_MODE = static_cast<PreconditionerMode>(PRECONDITIONER_CHOICE);
const JacobianMatvecMode JACOBIAN_MATVEC_MODE = static_cast<JacobianMatvecMode>(JACOBIAN_MATVEC_CHOICE);

static const char* preconditioner_mode_name()
{
    switch (PRECONDITIONER_MODE) {
    case PreconditionerMode::NONE: return "GMRES_without_preconditioner";
    case PreconditionerMode::LU_SGS: return "GMRES_plus_LU_SGS";
    case PreconditionerMode::BLOCK_ILU0: return "GMRES_plus_block_ILU0";
    }
    return "UNKNOWN";
}

static const char* jacobian_matvec_mode_name()
{
    switch (JACOBIAN_MATVEC_MODE) {
    case JacobianMatvecMode::FINITE_DIFFERENCE: return "finite_difference_matrix_vector_product";
    case JacobianMatvecMode::AD_BLOCK: return "AD_block_Jacobian_matrix_vector_product";
    case JacobianMatvecMode::MULTI_SECANT_CORRECTED_AD: return "multi_secant_corrected_AD_block_Jacobian_matrix_vector_product";
    }
    return "UNKNOWN";
}

// Spalart-Allmaras model constants.
const double SA_SIGMA = 2.0 / 3.0;
const double SA_CB1 = 0.1355;
const double SA_CB2 = 0.622;
const double SA_KAPPA = 0.41;
const double SA_CW2 = 0.3;
const double SA_CW3 = 2.0;
const double SA_CV1 = 7.1;
const double SA_CT3 = 1.2;
const double SA_CT4 = 0.5;
const double SA_CW1 = SA_CB1 / (SA_KAPPA * SA_KAPPA) + (1.0 + SA_CB2) / SA_SIGMA;
const double SA_CN1 = 16.0;
const double SUTHERLAND_S = 110.4 / 300.0;

const dg::ViscousSAParameters g_viscous_sa_parameters = {
    gamma_v,
    p_inf / rho_inf,
    mu_ref,
    Pr,
    Pr_t,
    SUTHERLAND_S,
    SA_SIGMA,
    SA_CB1,
    SA_CB2,
    SA_KAPPA,
    SA_CW1,
    SA_CW2,
    SA_CW3,
    SA_CV1,
    SA_CT3,
    SA_CT4,
    SA_CN1
};


// Numerical floors and mesh-control parameters.
const double SA_D_FLOOR = 1e-6;
// Keep the wall BR2 jump unscaled by default. The dominant Cf error in this
// case is the wall-normal mesh distribution, not the wall lifting strength.
const double BR2_WALL_BOUNDARY_TRACE_FACTOR = 1.0;
#ifndef SA_WALL_BR2_SCALE_VALUE
#define SA_WALL_BR2_SCALE_VALUE 1.0
#endif
const double SA_WALL_BR2_SCALE = SA_WALL_BR2_SCALE_VALUE;
#ifndef NS_WALL_BR2_SCALE_VALUE
#define NS_WALL_BR2_SCALE_VALUE 1.0
#endif
const double NS_WALL_BR2_SCALE = NS_WALL_BR2_SCALE_VALUE;
// NASA/TMBWG VERIF/2DZP reference input uses turbinf = 3.0.  The general
// SA model permits 3--5, but use the case-specific value for reproducibility.
const double SA_NUT_INF_OVER_NU_INF = 3.0;
double SA_INIT_NUT = SA_NUT_INF_OVER_NU_INF * (mu_ref / rho_inf);
const double UTAU_EST = 1.66666666666667e-2;
static double PLATE_X0 = dg::case_config::viscous_wall_x_min;
static double PLATE_X1 = dg::case_config::viscous_wall_x_max;
// Point indices delimiting the wall on logical j=0.  Wall cells occupy the
// half-open interval [begin, end).  Mesh readers populate these values.
static int g_wall_i_begin = -1;
static int g_wall_i_end = -1;
const double PLATE_XMID = dg::case_config::wall_reference_x;
const double GEOM_MIN_J_RELATIVE_TO_BILINEAR = 0.2;
const double GEOM_WALL_CLIP_TOL = 1e-6;

static dg::WallGeometry g_wall_geometry;

static void configure_flow_conditions(const dg::SolverOptions& options)
{
    M_inf = std::isfinite(options.mach_number)
        ? options.mach_number : dg::case_config::mach_number;
    Re = std::isfinite(options.reynolds_number)
        ? options.reynolds_number : dg::case_config::reynolds_number;
    const double alpha_degrees = std::isfinite(options.angle_of_attack_degrees)
        ? options.angle_of_attack_degrees
        : dg::case_config::angle_of_attack_degrees;
    constexpr double pi = 3.141592653589793238462643383279502884;
    inflow_angle = alpha_degrees * pi / 180.0;
    u_inf = M_inf * a_inf * std::cos(inflow_angle);
    v_inf = M_inf * a_inf * std::sin(inflow_angle);
    T0_inf = T_inf * (1.0 + 0.5 * gamma1 * M_inf * M_inf);
    p0_inf = p_inf * std::pow(T0_inf / T_inf, gamma_v / gamma1);
    mu_ref = rho_inf * M_inf * a_inf / Re;
    SA_INIT_NUT = SA_NUT_INF_OVER_NU_INF * (mu_ref / rho_inf);

#if defined(DG_CASE_NACA0012)
    g_boundary_conditions = dg::BoundaryConditionSet{}; // farfield on all outer faces
#else
    g_boundary_conditions = dg::make_channel_boundary_conditions(
        p_inf, p0_inf, T0_inf, inflow_angle,
        dg::case_config::upper_boundary_is_symmetry
            ? dg::BoundaryType::SlipWall : dg::BoundaryType::Farfield);
#endif
}

static inline double flat_plate_wall_y(double x)
{
    return g_wall_geometry.height(x);
}

static inline bool is_solid_wall_x(double x)
{
    return x >= PLATE_X0 - 1e-12 && x <= PLATE_X1 + 1e-12;
}

static inline void gauss_points_1d(double* x, double* w)
{
    dg::gauss_legendre_1d<Quad1D>(x, w);
}

static inline void reference_node_coords(int k, double& r, double& s)
{
    dg::reference_node_coords_2d<PolyOrder>(k, r, s);
}

static inline void reference_geometry_nodes_1d(double* x)
{
    dg::reference_nodes_1d<GeomOrder>(x);
}

static inline void reference_geometry_node_coords(int k, double& r, double& s)
{
    dg::reference_node_coords_2d<GeomOrder>(k, r, s);
}

template <class T>
using Array4D = dg::Field4D<T>;


// Mesh, basis functions, solution arrays, projected gradients, and boundary diagnostics.
struct SolverVariable {
    double gl_p[FaceQuad] = { 0 };
    double gl_w[FaceQuad] = { 0 };

    double phi_vol[NumQuad][dimPk] = { 0 };
    double dphi_dr[NumQuad][dimPk] = { 0 };
    double dphi_ds[NumQuad][dimPk] = { 0 };
    double phi_face[4][FaceQuad][dimPk] = { 0 };
    double inv_mass[dimPk][dimPk] = { 0 };
    std::vector<double> inv_mass_cell;
    double mean_wts[dimPk] = { 0 };

    double x_edges[Nx + 1] = { 0 };
    double h_x[Nx] = { 0 };
    double x_centers[Nx] = { 0 };
    double y_edges[Ny + 1] = { 0 };
    double h_y[Ny] = { 0 };
    double y_centers[Ny] = { 0 };
    std::vector<double> grid_x;
    std::vector<double> grid_y;
    Array4D<double> geom;
    dg::MappedGeometryCache metric_cache;

    Array4D<double> K;       // Accepted/current DG solution.
    Array4D<double> rhs;     // Spatial semi-discrete RHS assembled by computeRHS().
    Array4D<double> gradUX;  // Projected x derivative of the conservative state.
    Array4D<double> gradUY;  // Projected y derivative of the conservative state.
};

static inline size_t grid_vertex_index(int i, int j)
{
    return (size_t)j * (size_t)(Nx + 1) + (size_t)i;
}

static inline double grid_vertex_x(const SolverVariable& S, int i, int j)
{
    return S.grid_x[grid_vertex_index(i, j)];
}

static inline double grid_vertex_y(const SolverVariable& S, int i, int j)
{
    return S.grid_y[grid_vertex_index(i, j)];
}

static inline void bilinear_grid_coords(const SolverVariable& S, int i, int j, double r, double s, double& x, double& y)
{
    const double a = 0.5 * (r + 1.0);
    const double b = 0.5 * (s + 1.0);
    const double w00 = (1.0 - a) * (1.0 - b);
    const double w10 = a * (1.0 - b);
    const double w01 = (1.0 - a) * b;
    const double w11 = a * b;
    x = w00 * grid_vertex_x(S, i, j) + w10 * grid_vertex_x(S, i + 1, j)
        + w01 * grid_vertex_x(S, i, j + 1) + w11 * grid_vertex_x(S, i + 1, j + 1);
    y = w00 * grid_vertex_y(S, i, j) + w10 * grid_vertex_y(S, i + 1, j)
        + w01 * grid_vertex_y(S, i, j + 1) + w11 * grid_vertex_y(S, i + 1, j + 1);
}

static inline void geometry_basis_1d(double x, double* l, double* dl)
{
    double nodes[GeomNodes1D] = {};
    reference_geometry_nodes_1d(nodes);
    dg::lagrange_1d<GeomNodes1D>(x, nodes, l, dl);
}

static inline double geometry_dof_value_from_data(const std::vector<double>& geom_data,
    int i, int j, int k, int comp)
{
    const size_t idx = ((((size_t)i * (size_t)Ny + (size_t)j) * (size_t)GeomDimPk + (size_t)k) * 2u + (size_t)comp);
    return geom_data[idx];
}

static inline double geometry_jacobian_from_data(const std::vector<double>& geom_data,
    int i, int j, double r, double s)
{
    double lr[GeomNodes1D] = {}, ls[GeomNodes1D] = {};
    double dlr[GeomNodes1D] = {}, dls[GeomNodes1D] = {};
    geometry_basis_1d(r, lr, dlr);
    geometry_basis_1d(s, ls, dls);
    double xr = 0.0, xs = 0.0, yr = 0.0, ys = 0.0;
    for (int b = 0; b < GeomNodes1D; ++b) {
        for (int a = 0; a < GeomNodes1D; ++a) {
            const int k = b * GeomNodes1D + a;
            const double dphir = dlr[a] * ls[b];
            const double dphis = lr[a] * dls[b];
            const double xk = geometry_dof_value_from_data(geom_data, i, j, k, 0);
            const double yk = geometry_dof_value_from_data(geom_data, i, j, k, 1);
            xr += xk * dphir;
            xs += xk * dphis;
            yr += yk * dphir;
            ys += yk * dphis;
        }
    }
    return xr * ys - xs * yr;
}

static inline std::vector<double> geometry_jacobian_sample_points()
{
    std::vector<double> pts;
    pts.reserve((size_t)Quad1D + (size_t)GeomNodes1D + 2u);
    pts.push_back(-1.0);
    pts.push_back(1.0);
    double gp[Quad1D] = {}, gw[Quad1D] = {};
    gauss_points_1d(gp, gw);
    for (int q = 0; q < Quad1D; ++q) pts.push_back(gp[q]);
    double nodes[GeomNodes1D] = {};
    reference_geometry_nodes_1d(nodes);
    for (int n = 0; n < GeomNodes1D; ++n) pts.push_back(nodes[n]);
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end(), [](double a, double b) {
        return std::abs(a - b) < 1e-14;
        }), pts.end());
    return pts;
}

static inline void apply_case_wall_correction(SolverVariable& S)
{
    if (!dg::case_config::use_analytic_wall_correction) return;

    constexpr double blend_height = 0.05;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < GeomDimPk; ++k) {
                double rnode = 0.0, snode = 0.0;
                reference_geometry_node_coords(k, rnode, snode);
                const double y_node = S.geom(i, j, k, 1);
                double x_wall = 0.0, y_wall_bilinear = 0.0;
                bilinear_grid_coords(S, i, 0, rnode, -1.0,
                    x_wall, y_wall_bilinear);
                const double wall_distance_from_grid =
                    std::max(0.0, y_node - y_wall_bilinear);
                const double weight =
                    std::max(0.0, 1.0 - wall_distance_from_grid / blend_height);
                if (weight <= 0.0) continue;
                S.geom(i, j, k, 1) += weight
                    * (dg::case_config::wall_height(x_wall) - y_wall_bilinear);
            }
        }
    }
}

static inline bool corrected_geometry_has_healthy_jacobian(const SolverVariable& S,
    const std::vector<double>& bilinear_geom,
    double& minJ,
    double& minJRatio)
{
    minJ = std::numeric_limits<double>::infinity();
    minJRatio = std::numeric_limits<double>::infinity();
    const std::vector<double> pts = geometry_jacobian_sample_points();
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (double r : pts) {
                for (double s : pts) {
                    const double J0 = geometry_jacobian_from_data(bilinear_geom, i, j, r, s);
                    const double J = geometry_jacobian_from_data(S.geom.data, i, j, r, s);
                    minJ = std::min(minJ, J);
                    if (!std::isfinite(J0) || J0 <= 0.0 || !std::isfinite(J) || J <= 0.0) return false;
                    const double ratio = J / J0;
                    minJRatio = std::min(minJRatio, ratio);
                    if (ratio < GEOM_MIN_J_RELATIVE_TO_BILINEAR) return false;
                }
            }
        }
    }
    return true;
}

static inline void cgns_cell_spacings(const SolverVariable& S, int i, int j, double& dx, double& dy)
{
    const double x00 = grid_vertex_x(S, i, j);
    const double y00 = grid_vertex_y(S, i, j);
    const double x10 = grid_vertex_x(S, i + 1, j);
    const double y10 = grid_vertex_y(S, i + 1, j);
    const double x01 = grid_vertex_x(S, i, j + 1);
    const double y01 = grid_vertex_y(S, i, j + 1);
    const double x11 = grid_vertex_x(S, i + 1, j + 1);
    const double y11 = grid_vertex_y(S, i + 1, j + 1);
    const double e_bottom = std::hypot(x10 - x00, y10 - y00);
    const double e_top = std::hypot(x11 - x01, y11 - y01);
    const double e_left = std::hypot(x01 - x00, y01 - y00);
    const double e_right = std::hypot(x11 - x10, y11 - y10);
    dx = std::max(0.5 * (e_bottom + e_top), 1e-30);
    dy = std::max(0.5 * (e_left + e_right), 1e-30);
}

static void build_geometry_dofs_from_grid(SolverVariable& S)
{
    S.geom.reset(Nx, Ny, GeomDimPk, 2, 0.0);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < GeomDimPk; ++k) {
                double rnode = 0.0, snode = 0.0;
                reference_geometry_node_coords(k, rnode, snode);
                double x = 0.0, y = 0.0;
                bilinear_grid_coords(S, i, j, rnode, snode, x, y);
                S.geom(i, j, k, 0) = x;
                S.geom(i, j, k, 1) = y;
            }
        }
    }

    const std::vector<double> bilinear_geom = S.geom.data;
    apply_case_wall_correction(S);
    double minJ = 0.0, minJRatio = 0.0;
    if (!corrected_geometry_has_healthy_jacobian(S, bilinear_geom, minJ, minJRatio)) {
        std::cerr << "[WARN] case geometry failed relative Jacobian check; "
            << "reverting to CGNS bilinear geometry. minJ=" << minJ
            << " minJRatio=" << minJRatio
            << " requiredRatio=" << GEOM_MIN_J_RELATIVE_TO_BILINEAR << "\n";
        S.geom.data = bilinear_geom;
    }
    else {
        std::cout << "[GEOM] "
            << (dg::case_config::use_analytic_wall_correction
                ? "CGNS plus analytic wall correction"
                : "CGNS bilinear geometry")
            << " minJ=" << minJ
            << " minJRatio=" << minJRatio << "\n";
    }
}

static void build_geometry_dofs_from_nested_grid(SolverVariable& S,
    const dg::StructuredGrid2D& fine)
{
    const int fine_ni = GeomOrder * Nx + 1;
    const int fine_nj = GeomOrder * Ny + 1;
    if (fine.ni != fine_ni || fine.nj != fine_nj) {
        std::cerr << "[FATAL] nested geometry grid has " << fine.ni << "x" << fine.nj
            << " points; expected " << fine_ni << "x" << fine_nj << "\n";
        std::exit(1);
    }

    double max_shared_point_error = 0.0;
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            const size_t coarse_index = grid_vertex_index(i, j);
            const size_t fine_index = static_cast<size_t>(GeomOrder * j) * fine.ni
                + static_cast<size_t>(GeomOrder * i);
            max_shared_point_error = std::max(max_shared_point_error,
                std::hypot(S.grid_x[coarse_index] - fine.x[fine_index],
                    S.grid_y[coarse_index] - fine.y[fine_index]));
        }
    }
    if (!std::isfinite(max_shared_point_error) || max_shared_point_error > 1e-9) {
        std::cerr << "[FATAL] topology and geometry CGNS grids are not nested; max shared-point error="
            << max_shared_point_error << std::endl;
        std::exit(1);
    }

    S.geom.reset(Nx, Ny, GeomDimPk, 2, 0.0);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int b = 0; b < GeomNodes1D; ++b) {
                for (int a = 0; a < GeomNodes1D; ++a) {
                    const int k = b * GeomNodes1D + a;
                    const int fine_i = GeomOrder * i + a;
                    const int fine_j = GeomOrder * j + b;
                    const size_t fine_index = static_cast<size_t>(fine_j) * fine.ni + fine_i;
                    S.geom(i, j, k, 0) = fine.x[fine_index];
                    S.geom(i, j, k, 1) = fine.y[fine_index];
                }
            }
        }
    }

    double min_jacobian = std::numeric_limits<double>::infinity();
    double max_jacobian = 0.0;
    const std::vector<double> points = geometry_jacobian_sample_points();
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (double r : points) {
                for (double s : points) {
                    const double jacobian = geometry_jacobian_from_data(S.geom.data, i, j, r, s);
                    min_jacobian = std::min(min_jacobian, jacobian);
                    max_jacobian = std::max(max_jacobian, jacobian);
                    if (!std::isfinite(jacobian) || jacobian <= 0.0) {
                        std::cerr << "[FATAL] non-positive nested Q4 geometry Jacobian at cell ("
                            << i << "," << j << "), (r,s)=(" << r << "," << s
                            << "): J=" << jacobian << std::endl;
                        std::exit(1);
                    }
                }
            }
        }
    }
    std::cout << "[GEOM] nested CGNS Q4 geometry sharedPointError="
        << max_shared_point_error << " minJ=" << min_jacobian
        << " maxJ=" << max_jacobian << "\n";
}

static std::vector<unsigned char> read_binary_file_or_die(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[FATAL] cannot open CGNS grid file: " << path << "\n";
        std::exit(1);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes((size_t)n);
    if (n > 0) in.read(reinterpret_cast<char*>(bytes.data()), n);
    if (!in && n > 0) {
        std::cerr << "[FATAL] failed while reading CGNS grid file: " << path << "\n";
        std::exit(1);
    }
    return bytes;
}

static size_t find_bytes(const std::vector<unsigned char>& bytes, const char* needle, size_t start)
{
    const size_t n = std::strlen(needle);
    if (n == 0 || bytes.size() < n || start > bytes.size() - n) return std::string::npos;
    for (size_t p = start; p <= bytes.size() - n; ++p) {
        if (std::memcmp(bytes.data() + p, needle, n) == 0) return p;
    }
    return std::string::npos;
}

static double read_le_double(const std::vector<unsigned char>& bytes, size_t off)
{
    double v = 0.0;
    if (off + sizeof(double) > bytes.size()) return std::numeric_limits<double>::quiet_NaN();
    std::memcpy(&v, bytes.data() + off, sizeof(double));
    return v;
}

static std::vector<size_t> find_adf_data_starts(const std::vector<unsigned char>& bytes)
{
    std::vector<size_t> starts;
    for (const char* marker : { "snTEDaTa", "TaiLDaTa" }) {
        size_t pos = 0;
        while (true) {
            pos = find_bytes(bytes, marker, pos);
            if (pos == std::string::npos) break;
            if (pos + 20 < bytes.size()) starts.push_back(pos + 20);
            ++pos;
        }
    }
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    return starts;
}

struct CgnsCoordArray {
    size_t start = std::string::npos;
    int nk = 1;
    bool i_major = true;
};

static size_t cgns_packed_coord_offset(int i, int j, int ni, int nj, int nk, bool i_major)
{
    const size_t point = i_major
        ? ((size_t)i * (size_t)nj + (size_t)j)
        : ((size_t)j * (size_t)ni + (size_t)i);
    return point * (size_t)nk * sizeof(double);
}

static bool cgns_repeated_spanwise_values_match(const std::vector<unsigned char>& bytes,
    size_t start, int i, int j, int ni, int nj, int nk, bool i_major)
{
    const size_t base = start + cgns_packed_coord_offset(i, j, ni, nj, nk, i_major);
    const double v0 = read_le_double(bytes, base);
    if (!std::isfinite(v0)) return false;
    for (int k = 1; k < nk; ++k) {
        const double vk = read_le_double(bytes, base + (size_t)k * sizeof(double));
        if (!std::isfinite(vk) || std::abs(v0 - vk) > 1e-10) return false;
    }
    return true;
}

static double read_cgns_coord_value(const std::vector<unsigned char>& bytes,
    size_t start, int i, int j, int ni, int nj, int nk, bool i_major)
{
    const size_t base = start + cgns_packed_coord_offset(i, j, ni, nj, nk, i_major);
    return read_le_double(bytes, base);
}

static bool looks_like_cgns_x_array(const std::vector<unsigned char>& bytes,
    size_t start, int nk, bool i_major)
{
    const int ni = Nx + 1;
    const int nj = Ny + 1;
    if (start + (size_t)ni * nj * nk * sizeof(double) > bytes.size()) return false;
    for (int j : {0, nj / 2, nj - 1}) {
        double last = -1e300;
        for (int i = 0; i < ni; ++i) {
            if (!cgns_repeated_spanwise_values_match(bytes, start, i, j, ni, nj, nk, i_major)) return false;
            const double x = read_cgns_coord_value(bytes, start, i, j, ni, nj, nk, i_major);
            if (x < last - 1e-10) return false;
            last = x;
        }
    }
    const double xmin = read_cgns_coord_value(bytes, start, 0, 0, ni, nj, nk, i_major);
    const double xmax = read_cgns_coord_value(bytes, start, ni - 1, 0, ni, nj, nk, i_major);
    return std::abs(xmin - xa) < 1e-8 && std::abs(xmax - xb) < 1e-8;
}

static bool looks_like_cgns_vertical_array(const std::vector<unsigned char>& bytes,
    size_t start, int nk, bool i_major)
{
    const int ni = Nx + 1;
    const int nj = Ny + 1;
    if (start + (size_t)ni * nj * nk * sizeof(double) > bytes.size()) return false;
    for (int i = 0; i < ni; ++i) {
        double last = -1e-12;
        for (int j = 0; j < nj; ++j) {
            if (!cgns_repeated_spanwise_values_match(bytes, start, i, j, ni, nj, nk, i_major)) return false;
            const double y = read_cgns_coord_value(bytes, start, i, j, ni, nj, nk, i_major);
            if (y < -1e-10 || y > yb + 1e-8 || y < last - 1e-10) return false;
            last = y;
        }
        const double ytop = read_cgns_coord_value(bytes, start, i, nj - 1, ni, nj, nk, i_major);
        if (std::abs(ytop - yb) > 1e-8) return false;
    }
    return true;
}

static bool find_cgns_coord_array(const std::vector<unsigned char>& bytes, size_t start,
    bool vertical, CgnsCoordArray& out)
{
    for (int nk : {1, 2}) {
        for (bool i_major : {true, false}) {
            const bool ok = vertical
                ? looks_like_cgns_vertical_array(bytes, start, nk, i_major)
                : looks_like_cgns_x_array(bytes, start, nk, i_major);
            if (ok) {
                out.start = start;
                out.nk = nk;
                out.i_major = i_major;
                return true;
            }
        }
    }
    return false;
}

static void load_structured_adf_cgns_grid(SolverVariable& S, const std::string& path)
{
#if defined(DG_USE_CGNS)
#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
    const dg::StructuredGrid2D grid = dg::read_structured_cgns_2d(
        path, Nx + 1, Ny + 1);
    const dg::StructuredGrid2D geometry_grid = dg::read_structured_cgns_2d(
        CGNS_GEOMETRY_GRID_FILE, GeomOrder * Nx + 1, GeomOrder * Ny + 1);
#elif defined(DG_CASE_NACA0012)
    const dg::StructuredGrid2D grid = dg::read_logically_structured_gmsh_2d(
        path, Nx + 1, Ny + 1, 2,
#if defined(DG_NACA_SOURCE_NI) && defined(DG_NACA_SOURCE_NJ)
        DG_NACA_SOURCE_NI, DG_NACA_SOURCE_NJ
#else
        Nx + 1, Ny + 1
#endif
    );
#else
    const dg::StructuredGrid2D grid = dg::read_structured_cgns_2d(path, Nx + 1, Ny + 1);
#endif
    const int ni = Nx + 1;
    const int nj = Ny + 1;
    if (grid.ni != ni || grid.nj != nj) {
        std::cerr << "[FATAL] CGNS grid has " << grid.ni << "x" << grid.nj
            << " points, but this executable expects " << ni << "x" << nj
            << ". CGNS filenames describe points; NX_CELLS/NY_CELLS describe cells.\n";
        std::exit(1);
    }
    S.grid_x = grid.x;
    S.grid_y = grid.y;
    g_wall_i_begin = grid.wall_i_begin;
    g_wall_i_end = grid.wall_i_end;
#if defined(DG_CASE_NACA0012)
    if (g_wall_i_begin < 0 || g_wall_i_end <= g_wall_i_begin
        || g_wall_i_end > Nx) {
        throw std::runtime_error(
            "NACA mesh does not provide a valid viscous-wall point range");
    }
#endif

    std::vector<double> wall_profile_x(ni);
    std::vector<double> wall_profile_y(ni);
    for (int i = 0; i < ni; ++i) {
        wall_profile_x[i] = grid_vertex_x(S, i, 0);
        wall_profile_y[i] = grid_vertex_y(S, i, 0);
    }
#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
    std::vector<double> airfoil_x;
    std::vector<double> airfoil_y;
    airfoil_x.reserve(GeomOrder * (g_wall_i_end - g_wall_i_begin) + 1);
    airfoil_y.reserve(airfoil_x.capacity());
    for (int fine_i = GeomOrder * g_wall_i_begin;
        fine_i <= GeomOrder * g_wall_i_end; ++fine_i) {
        const size_t index = static_cast<size_t>(fine_i);
        airfoil_x.push_back(geometry_grid.x[index]);
        airfoil_y.push_back(geometry_grid.y[index]);
    }
    g_wall_geometry.set_polyline(std::move(airfoil_x), std::move(airfoil_y));
    PLATE_X0 = dg::case_config::viscous_wall_x_min;
    PLATE_X1 = dg::case_config::viscous_wall_x_max;
#else
    if (grid.wall_i_begin >= 0 && grid.wall_i_end >= grid.wall_i_begin) {
        const int first = std::max(0, std::min(grid.wall_i_begin, ni - 1));
        const int last = std::max(0, std::min(grid.wall_i_end, ni - 1));
#if defined(DG_CASE_NACA0012)
        std::vector<double> airfoil_x(wall_profile_x.begin() + first,
            wall_profile_x.begin() + last + 1);
        std::vector<double> airfoil_y(wall_profile_y.begin() + first,
            wall_profile_y.begin() + last + 1);
        g_wall_geometry.set_polyline(std::move(airfoil_x), std::move(airfoil_y));
        PLATE_X0 = dg::case_config::viscous_wall_x_min;
        PLATE_X1 = dg::case_config::viscous_wall_x_max;
#else
        g_wall_geometry.set_profile(std::move(wall_profile_x), std::move(wall_profile_y),
            static_cast<std::size_t>(first), static_cast<std::size_t>(last));
        PLATE_X0 = g_wall_geometry.x_begin();
        PLATE_X1 = g_wall_geometry.x_end();

        if (dg::case_config::use_analytic_wall_correction) {
            constexpr int analytic_segments = 4096;
            std::vector<double> analytic_x(analytic_segments + 1);
            std::vector<double> analytic_y(analytic_segments + 1);
            for (int n = 0; n <= analytic_segments; ++n) {
                const double t = static_cast<double>(n) / analytic_segments;
                const double x = PLATE_X0 + t * (PLATE_X1 - PLATE_X0);
                analytic_x[n] = x;
                analytic_y[n] = dg::case_config::wall_height(x);
            }
            g_wall_geometry.set_profile(std::move(analytic_x), std::move(analytic_y),
                0, analytic_segments);
        }
#endif
    }
    else {
        const auto closest_wall_index = [&](double target_x) {
            int best = 0;
            double best_distance = std::abs(wall_profile_x[0] - target_x);
            for (int i = 1; i < ni; ++i) {
                const double distance = std::abs(wall_profile_x[i] - target_x);
                if (distance < best_distance) {
                    best = i;
                    best_distance = distance;
                }
            }
            return best;
        };
        const int first = closest_wall_index(dg::case_config::viscous_wall_x_min);
        const int last = closest_wall_index(dg::case_config::viscous_wall_x_max);
        g_wall_i_begin = std::min(first, last);
        g_wall_i_end = std::max(first, last);
        g_wall_geometry.set_profile(std::move(wall_profile_x), std::move(wall_profile_y),
            static_cast<std::size_t>(g_wall_i_begin),
            static_cast<std::size_t>(g_wall_i_end));
        PLATE_X0 = g_wall_geometry.x_begin();
        PLATE_X1 = g_wall_geometry.x_end();
        std::cout << "[CGNS] no viscous-wall BC metadata; using case wall range\n";
    }
#endif

    std::cout << "[MESH] "
#if defined(DG_CASE_NACA0012)
#if defined(DG_NACA_OFFICIAL_CGNS)
        << "CGNS logical C-grid reader with nested Q4 geometry: points="
#else
        << "Gmsh logical C-grid reader: points="
#endif
#else
        << "CGNS library reader: points="
#endif
        << grid.ni << "x" << grid.nj
        << " boundaries=" << grid.boundaries.size()
        << " viscous_wall_x=[" << PLATE_X0 << ", " << PLATE_X1 << "]\n";
    for (const dg::BoundaryPatch& patch : grid.boundaries) {
        std::cout << "[CGNS] boundary name=\"" << patch.name << "\" kind="
            << static_cast<int>(patch.kind) << " range=("
            << patch.begin[0] << "," << patch.begin[1] << "," << patch.begin[2]
            << ")..(" << patch.end[0] << "," << patch.end[1] << ","
            << patch.end[2] << ")\n";
    }
#else
    const std::vector<unsigned char> bytes = read_binary_file_or_die(path);
    const int ni = Nx + 1;
    const int nj = Ny + 1;
    CgnsCoordArray x_array;
    CgnsCoordArray y_array;
    for (const size_t start : find_adf_data_starts(bytes)) {
        if (x_array.start == std::string::npos) find_cgns_coord_array(bytes, start, false, x_array);
        if (y_array.start == std::string::npos) find_cgns_coord_array(bytes, start, true, y_array);
    }
    if (x_array.start == std::string::npos || y_array.start == std::string::npos) {
        std::cerr << "[FATAL] could not locate CoordinateX/vertical coordinate arrays in ADF CGNS grid: "
            << path << "\n";
        std::exit(1);
    }

    S.grid_x.assign((size_t)ni * nj, 0.0);
    S.grid_y.assign((size_t)ni * nj, 0.0);
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            S.grid_x[grid_vertex_index(i, j)] = read_cgns_coord_value(
                bytes, x_array.start, i, j, ni, nj, x_array.nk, x_array.i_major);
            S.grid_y[grid_vertex_index(i, j)] = read_cgns_coord_value(
                bytes, y_array.start, i, j, ni, nj, y_array.nk, y_array.i_major);
        }
    }

    std::cout << "[CGNS] CoordinateX layout="
        << (x_array.i_major ? "i-major" : "j-major") << " nk=" << x_array.nk
        << " vertical layout=" << (y_array.i_major ? "i-major" : "j-major")
        << " nk=" << y_array.nk << "\n";
    if (dg::case_config::use_analytic_wall_correction) {
        constexpr int analytic_segments = 4096;
        std::vector<double> analytic_x(analytic_segments + 1);
        std::vector<double> analytic_y(analytic_segments + 1);
        for (int n = 0; n <= analytic_segments; ++n) {
            const double t = static_cast<double>(n) / analytic_segments;
            const double x = PLATE_X0 + t * (PLATE_X1 - PLATE_X0);
            analytic_x[n] = x;
            analytic_y[n] = dg::case_config::wall_height(x);
        }
        g_wall_geometry.set_profile(std::move(analytic_x), std::move(analytic_y),
            0, analytic_segments);
    }
    else {
        g_wall_geometry.set_flat_wall(PLATE_X0, PLATE_X1);
    }
#endif

    for (int i = 0; i <= Nx; ++i) S.x_edges[i] = grid_vertex_x(S, i, 0);
    for (int i = 0; i < Nx; ++i) {
        S.h_x[i] = 0.5 * (S.x_edges[i + 1] - S.x_edges[i]);
        S.x_centers[i] = 0.5 * (S.x_edges[i + 1] + S.x_edges[i]);
    }
    for (int j = 0; j <= Ny; ++j) S.y_edges[j] = grid_vertex_y(S, 0, j);
    for (int j = 0; j < Ny; ++j) {
        S.h_y[j] = 0.5 * (S.y_edges[j + 1] - S.y_edges[j]);
        S.y_centers[j] = 0.5 * (S.y_edges[j + 1] + S.y_edges[j]);
    }

#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
    build_geometry_dofs_from_nested_grid(S, geometry_grid);
#else
    build_geometry_dofs_from_grid(S);
#endif
}

static inline double cell_inv_mass_entry(const SolverVariable& S, int i, int j, int k, int m)
{
    if (!S.inv_mass_cell.empty()) {
        const size_t cell = (size_t)i * (size_t)Ny + (size_t)j;
        const size_t idx = ((cell * dimPk + (size_t)k) * dimPk + (size_t)m);
        return S.inv_mass_cell[idx];
    }
    (void)i;
    (void)j;
    return S.inv_mass[k][m];
}

static inline void eval_tensor_basis(double r, double s, double* phi, double* dphi_dr_out = nullptr, double* dphi_ds_out = nullptr)
{
    dg::tensor_lagrange_2d<PolyOrder>(r, s, phi, dphi_dr_out, dphi_ds_out);
}


// Dense inverse for the element mass matrix.
void invertMatrixDimPk(const double* A, double* inv)
{
    double mat[dimPk][2 * dimPk];
    for (int i = 0; i < dimPk; i++) {
        for (int j = 0; j < dimPk; j++) mat[i][j] = A[i * dimPk + j];
        for (int j = dimPk; j < 2 * dimPk; j++) mat[i][j] = (j - dimPk == i ? 1.0 : 0.0);
    }
    for (int i = 0; i < dimPk; i++) {
        int pivot = i;
        for (int j = i + 1; j < dimPk; j++) if (std::abs(mat[j][i]) > std::abs(mat[pivot][i])) pivot = j;
        for (int k = 0; k < 2 * dimPk; k++) std::swap(mat[i][k], mat[pivot][k]);
        double p = mat[i][i]; if (std::abs(p) < 1e-18) p = 1e-18;
        for (int k = i; k < 2 * dimPk; k++) mat[i][k] /= p;
        for (int k = 0; k < dimPk; k++) if (k != i) {
            double f = mat[k][i]; for (int j = i; j < 2 * dimPk; j++) mat[k][j] -= f * mat[i][j];
        }
    }
    for (int i = 0; i < dimPk; i++) for (int j = 0; j < dimPk; j++) inv[i * dimPk + j] = mat[i][j + dimPk];
}

void GetInitial(double x, double y, double* U)
{
    const double rho = rho_inf;
    const double u = u_inf;
    const double v = v_inf;
    const double p = p_inf;
    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = p / gamma1 + 0.5 * rho * (u * u + v * v);
    U[4] = rho * SA_INIT_NUT;
}

// Reconstruction, diagnostics, and verification-output utilities.
inline void cell_average(const SolverVariable& S, const Array4D<double>& K, int i, int j, double* Uavg);
void computeGradients(SolverVariable& S, const Array4D<double>& Kin);
void computeRHS(SolverVariable& S, const Array4D<double>& Kin);
dg::SALimiterReport apply_limiters(SolverVariable& S, Array4D<double>& Ktar,
    bool limit_negative_sa);
inline void face_point_coords(const SolverVariable& S, int i, int j, int f, int g, double& x, double& y);
inline void physical_coords(const SolverVariable& S, int i, int j, double r, double s, double& x, double& y);
inline double cell_jacobian_mean(const SolverVariable& S, int i, int j);
inline void contravariant_metrics(const SolverVariable& S, int i, int j, double r, double s,
    double& xr, double& xs, double& yr, double& ys, double& J);
inline void face_metric(const SolverVariable& S, int i, int j, int f, int g,
    double& nx, double& ny, double& ds);
inline bool is_plate_wall_face(int f, double x);
inline bool is_symmetry_slip_face(int f, double x);
inline bool is_zero_viscous_flux_face(int f, double x);
inline bool boundary_viscous_trace_2dzp(const double* UL, int f, double x, double y, double nx, double ny, double* UB);
void computeFaceLifting(const SolverVariable& S, const double* jump, int f, int i_idx, int j_idx, double* rX, double* rY, double scale);
inline void Viscous_Flux(const double* U, const double* dUx, const double* dUy, double* Fv, double* Gv);
inline double turbulent_mu_from_U(const double* U);
inline double laminar_mu_from_cons(const double* U);
inline double sa_eddy_mu_from_rho_nut(double rho, double nut, double mu_lam);
static inline void add_constant_to_cell_component(Array4D<double>& K, int i, int j, int e, double delta)
{
    if (delta == 0.0) return;
    for (int k = 0; k < dimPk; ++k) K(i, j, k, e) += delta;
}
static inline void set_cell_constant_state(Array4D<double>& K, int i, int j, const double* U)
{
    for (int k = 0; k < dimPk; ++k) {
        for (int e = 0; e < NumEq; ++e) K(i, j, k, e) = U[e];
    }
}
static inline void enforce_cellavg_positivity(SolverVariable& S, Array4D<double>& K,
    const double rho_floor = 1e-10,
    const double p_floor = 1e-8)
{
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            double Uavg[NumEq];
            cell_average(S, K, i, j, Uavg);
            double rho = Uavg[0];
            double ru = Uavg[1];
            double rv = Uavg[2];
            double E = Uavg[3];
            double q = Uavg[4];

            bool avg_finite = true;
            for (int e = 0; e < NumEq; ++e) avg_finite = avg_finite && std::isfinite(Uavg[e]);
            if (!avg_finite) {
                double Usafe[NumEq];
                double xc, yc;
                physical_coords(S, i, j, 0.0, 0.0, xc, yc);
                GetInitial(xc, yc, Usafe);
                set_cell_constant_state(K, i, j, Usafe);
                ++g_rho_posfix;
                ++g_p_posfix;
                ++g_nut_posfix;
                continue;
            }


            if (!std::isfinite(rho) || rho < rho_floor)
            {
                ++g_rho_posfix;
                const double rho_cur = (std::isfinite(rho) ? rho : rho_floor);
                const double ru_cur = (std::isfinite(ru) ? ru : 0.0);
                const double rv_cur = (std::isfinite(rv) ? rv : 0.0);
                const double E_cur = (std::isfinite(E) ? E : 0.0);
                const double q_cur = (std::isfinite(q) ? q : 0.0);
                const double rho_ref = std::max(rho_cur, rho_floor);
                const double u = ru_cur / rho_ref;
                const double v = rv_cur / rho_ref;
                const double nut = q_cur / rho_ref;
                const double kinetic_old = 0.5 * rho_ref * (u * u + v * v);
                const double p_old = std::max(gamma1 * (E_cur - kinetic_old), p_floor);
                const double rho_target = rho_floor;
                const double ru_target = rho_target * u;
                const double rv_target = rho_target * v;
                const double q_target = rho_target * nut;
                const double kinetic_new = 0.5 * rho_target * (u * u + v * v);
                const double E_target = p_old / gamma1 + kinetic_new;
                add_constant_to_cell_component(K, i, j, 0, rho_target - rho_cur);
                add_constant_to_cell_component(K, i, j, 1, ru_target - ru_cur);
                add_constant_to_cell_component(K, i, j, 2, rv_target - rv_cur);
                add_constant_to_cell_component(K, i, j, 3, E_target - E_cur);
                add_constant_to_cell_component(K, i, j, 4, q_target - q_cur);
                cell_average(S, K, i, j, Uavg);
                rho = Uavg[0]; ru = Uavg[1]; rv = Uavg[2]; E = Uavg[3]; q = Uavg[4];
            }

            const double rho_safe = std::max(rho, rho_floor);
            const double u = ru / rho_safe;
            const double v = rv / rho_safe;
            double p = gamma1 * (E - 0.5 * rho_safe * (u * u + v * v));

            if (!std::isfinite(p) || p < p_floor)
            {
                ++g_p_posfix;
                const double E_target = p_floor / gamma1 + 0.5 * rho_safe * (u * u + v * v);
                const double E_cur = (std::isfinite(E) ? E : E_target);
                add_constant_to_cell_component(K, i, j, 3, E_target - E_cur);
                cell_average(S, K, i, j, Uavg);
                rho = Uavg[0]; ru = Uavg[1]; rv = Uavg[2]; E = Uavg[3]; q = Uavg[4];
            }

            const double rho_safe2 = std::max(rho, rho_floor);
            const double nut = q / rho_safe2;
            if (!std::isfinite(nut))
            {
                ++g_nut_posfix;
                const double q_cur = (std::isfinite(q) ? q : 0.0);
                const double q_target = 0.0;
                add_constant_to_cell_component(K, i, j, 4, q_target - q_cur);
            }
        }
    }
}

static inline dg::SALimiterReport apply_solution_stabilization(
    SolverVariable& S, Array4D<double>& K, bool limit_negative_sa)
{
    enforce_cellavg_positivity(S, K);
    dg::SALimiterReport report = apply_limiters(S, K, limit_negative_sa);
    enforce_cellavg_positivity(S, K);
    return report;
}




inline void cell_average(const SolverVariable& S, const Array4D<double>& K, int i, int j, double* Uavg)
{
    for (int e = 0; e < NumEq; e++) Uavg[e] = 0.0;
    double wsum = 0.0;
    for (int q = 0; q < NumQuad; q++) {
        double uq[NumEq] = { 0 };
        for (int k = 0; k < dimPk; k++) {
            const double ph = S.phi_vol[q][k];
            for (int e = 0; e < NumEq; e++) uq[e] += K(i, j, k, e) * ph;
        }
        const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
        const double Jq = S.metric_cache.volume(i, j, q).jacobian;
        for (int e = 0; e < NumEq; e++) Uavg[e] += w * Jq * uq[e];
        wsum += w * Jq;
    }
    const double inv = 1.0 / (wsum + 1e-30);
    for (int e = 0; e < NumEq; e++) Uavg[e] *= inv;
}

struct SAQ2ResetReport {
    int cells = 0;
    double highest_order_energy_before = 0.0;
    double highest_order_energy_after = 0.0;
    double change_l2 = 0.0;
    double change_max_abs = 0.0;
    double cell_average_error_max_abs = 0.0;
    double cell_average_error_max_rel = 0.0;
};

static SAQ2ResetReport reset_sa_q2_modes_preserving_average(
    const SolverVariable& S, Array4D<double>& K)
{
    SAQ2ResetReport report;
    if constexpr (PolyOrder != 2) return report;
    double change_square = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            std::array<double, 9> original{};
            for (int k = 0; k < 9; ++k) original[k] = K(i, j, k, 4);
            report.highest_order_energy_before +=
                dg::q2_modal_energy(original.data()).highest_order_energy;

            double average_before[NumEq] = {};
            cell_average(S, K, i, j, average_before);
            const std::array<double, 9> filtered =
                dg::q2_remove_degree_two_modes(original.data());
            for (int k = 0; k < 9; ++k) K(i, j, k, 4) = filtered[k];

            double average_filtered[NumEq] = {};
            cell_average(S, K, i, j, average_filtered);
            const double correction = average_before[4] - average_filtered[4];
            for (int k = 0; k < 9; ++k) K(i, j, k, 4) += correction;

            double average_after[NumEq] = {};
            cell_average(S, K, i, j, average_after);
            const double average_error = average_after[4] - average_before[4];
            report.cell_average_error_max_abs = std::max(
                report.cell_average_error_max_abs, std::abs(average_error));
            report.cell_average_error_max_rel = std::max(
                report.cell_average_error_max_rel, std::abs(average_error)
                    / std::max(std::abs(average_before[4]), 1e-300));
            std::array<double, 9> final_values{};
            for (int k = 0; k < 9; ++k) {
                final_values[k] = K(i, j, k, 4);
                const double change = final_values[k] - original[k];
                change_square += change * change;
                report.change_max_abs = std::max(report.change_max_abs,
                    std::abs(change));
            }
            report.highest_order_energy_after +=
                dg::q2_modal_energy(final_values.data()).highest_order_energy;
            ++report.cells;
        }
    }
    report.change_l2 = std::sqrt(change_square
        / std::max(1, report.cells * 9));
    return report;
}

static inline void eval_phi_at(const SolverVariable& S, double r, double s, double* phi)
{
    (void)S;
    eval_tensor_basis(r, s, phi);
}

static double cgns_wall_y_at_x(double x);

static void write_solution_snapshot(const SolverVariable& solver, const Array4D<double>& K,
    const std::string& fname, bool write_tecplot_connectivity = true)
{
    const int solution_nodes = Nx * Ny * NumQuad;
    const int solution_elements = Nx * Ny * (Quad1D - 1) * (Quad1D - 1);
    dg::io::TecplotQuadrilateralWriter output;
    output.open(fname, write_tecplot_connectivity, solution_nodes, solution_elements);
    int clipped_below_wall_points = 0;
    int raw_below_wall_points = 0;
    double max_wall_penetration = 0.0;
    for (int i = 0; i < Nx; i++) for (int j = 0; j < Ny; j++) {
        for (int q = 0; q < NumQuad; q++) {
            const double r = solver.gl_p[q / Quad1D];
            const double s = solver.gl_p[q % Quad1D];
            double px = 0.0, py = 0.0;
            physical_coords(solver, i, j, r, s, px, py);
#if !defined(DG_CASE_NACA0012)
            const double ywall = cgns_wall_y_at_x(px);
            if (py < ywall) {
                ++raw_below_wall_points;
                max_wall_penetration = std::max(max_wall_penetration, ywall - py);
                if (ywall - py > GEOM_WALL_CLIP_TOL) {
                    ++clipped_below_wall_points;
                }
                py = ywall;
            }
#endif
            double un[NumEq] = { 0 };
            for (int k = 0; k < dimPk; k++)
                for (int e = 0; e < NumEq; e++)
                    un[e] += K(i, j, k, e) * solver.phi_vol[q][k];
            const double rho = std::max(un[0], 1e-30);
            const double u = un[1] / rho;
            const double v = un[2] / rho;
            const double p = gamma1 * (un[3] - 0.5 * (un[1] * un[1] + un[2] * un[2]) / rho);
            const double nut = un[4] / rho;
            const double mu_t = turbulent_mu_from_U(un);
            output.write_point({
                px,
                py,
                rho,
                u,
                v,
                p,
                nut,
                mu_t,
                mu_t / std::max(mu_ref, 1e-30) });
        }
    }
    if (raw_below_wall_points > 0) {
        const char* level = (clipped_below_wall_points > 0) ? "[WARN]" : "[INFO]";
        std::cerr << level << " write_solution_snapshot found " << raw_below_wall_points
            << " points below the wall profile in " << fname
            << " (strong=" << clipped_below_wall_points
            << ", tol=" << GEOM_WALL_CLIP_TOL
            << ", max_penetration=" << max_wall_penetration << ").\n";
    }
    if (!write_tecplot_connectivity) return;
    auto snapshot_node_id = [](int i, int j, int a, int b) {
        const int q = a * Quad1D + b;
        return ((i * Ny + j) * NumQuad + q) + 1;
        };
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        for (int a = 0; a < Quad1D - 1; ++a) {
            for (int b = 0; b < Quad1D - 1; ++b) {
                output.write_quadrilateral(
                    snapshot_node_id(i, j, a, b),
                    snapshot_node_id(i, j, a + 1, b),
                    snapshot_node_id(i, j, a + 1, b + 1),
                    snapshot_node_id(i, j, a, b + 1));
            }
        }
    }
}

struct ContinuationState {
    bool available = false;
    double initial_sa_residual = std::numeric_limits<double>::quiet_NaN();
    double previous_max_residual = std::numeric_limits<double>::quiet_NaN();
    double previous_ns_residual = std::numeric_limits<double>::quiet_NaN();
    double previous_sa_residual = std::numeric_limits<double>::quiet_NaN();
    double best_max_residual = std::numeric_limits<double>::quiet_NaN();
    double best_ns_residual = std::numeric_limits<double>::quiet_NaN();
    double best_sa_residual = std::numeric_limits<double>::quiet_NaN();
    double sa_shrink_reference = std::numeric_limits<double>::quiet_NaN();
    long long last_sa_shrink_total_step = 0;
    int residual_convergence_hold = 0;
    int physical_convergence_hold = 0;
    double last_physical_relative_change =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<dg::steady::ResidualGrowthGuard::ResidualPair>
        residual_growth_history;
    std::vector<dg::steady::PhysicalConvergenceSample>
        physical_convergence_history;
};

struct RestartState {
    long long completed_steps = 0;
    double pseudo_time = 0.0;
    double cfl = CFL_INIT;
    bool binary_checkpoint = false;
    std::uint32_t checkpoint_version = 0;
    ContinuationState continuation;
};

static constexpr char CHECKPOINT_MAGIC[8] = {
    'G', 'S', 'X', 'D', 'G', 'C', 'P', '1'
};

static void checkpoint_hash_bytes(std::uint64_t& hash,
    const void* data, std::size_t byte_count)
{
    constexpr std::uint64_t fnv_prime = 1099511628211ull;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t byte = 0; byte < byte_count; ++byte) {
        hash ^= bytes[byte];
        hash *= fnv_prime;
    }
}

template <class T>
static void checkpoint_hash_value(std::uint64_t& hash, const T& value)
{
    checkpoint_hash_bytes(hash, &value, sizeof(value));
}

static std::uint64_t compute_checkpoint_signature(const SolverVariable& solver)
{
    constexpr std::uint64_t fnv_offset = 14695981039346656037ull;
    std::uint64_t hash = fnv_offset;
    checkpoint_hash_bytes(hash, dg::case_config::checkpoint_id,
        std::strlen(dg::case_config::checkpoint_id));
    const std::array<int, 8> discretization = {
        Nx, Ny, PolyOrder, GeomOrder, dimPk, NumEq,
        dg::case_config::wake_pairs, Quad1D
    };
    checkpoint_hash_bytes(hash, discretization.data(),
        discretization.size() * sizeof(discretization[0]));
    checkpoint_hash_value(hash, M_inf);
    checkpoint_hash_value(hash, Re);
    checkpoint_hash_value(hash, inflow_angle);
    checkpoint_hash_value(hash, SA_INIT_NUT);
    checkpoint_hash_bytes(hash, solver.grid_x.data(),
        solver.grid_x.size() * sizeof(double));
    checkpoint_hash_bytes(hash, solver.grid_y.data(),
        solver.grid_y.size() * sizeof(double));
    checkpoint_hash_bytes(hash, solver.geom.data.data(),
        solver.geom.data.size() * sizeof(double));
    return hash;
}

template <class T>
static void checkpoint_write_value(std::ofstream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
static void checkpoint_read_value(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
}

static std::string sibling_file_path(const std::string& path,
    const std::string& filename)
{
    const std::size_t separator = path.find_last_of("/\\");
    return separator == std::string::npos
        ? filename : path.substr(0, separator + 1) + filename;
}

static bool recover_v2_continuation_summary(const std::string& checkpoint_path,
    RestartState& state)
{
    const std::string summary_path = sibling_file_path(checkpoint_path,
        "run_summary.dat");
    std::ifstream summary(summary_path);
    if (!summary) return false;

    ContinuationState recovered;
    std::string key, line;
    while (summary >> key) {
        if (!key.empty() && key[0] == '#') {
            std::getline(summary, line);
        }
        else if (key == "best_max_l2scaled") {
            summary >> recovered.best_max_residual;
        }
        else if (key == "best_max_ns_l2scaled") {
            summary >> recovered.best_ns_residual;
        }
        else if (key == "best_sa_l2scaled") {
            summary >> recovered.best_sa_residual;
        }
        else if (key == "last_max_l2scaled") {
            summary >> recovered.previous_max_residual;
        }
        else if (key == "last_max_ns_l2scaled") {
            summary >> recovered.previous_ns_residual;
        }
        else if (key == "last_sa_l2scaled") {
            summary >> recovered.previous_sa_residual;
        }
        else if (key == "physical_convergence_hold") {
            summary >> recovered.physical_convergence_hold;
        }
        else if (key == "last_physical_relative_change") {
            summary >> recovered.last_physical_relative_change;
        }
        else {
            std::getline(summary, line);
        }
    }
    auto valid = [](double value) {
        return std::isfinite(value) && value > 0.0;
        };
    if (!valid(recovered.best_max_residual)
        || !valid(recovered.best_ns_residual)
        || !valid(recovered.best_sa_residual)
        || !valid(recovered.previous_max_residual)
        || !valid(recovered.previous_ns_residual)
        || !valid(recovered.previous_sa_residual)) {
        return false;
    }
    recovered.sa_shrink_reference = recovered.best_sa_residual;
    recovered.last_sa_shrink_total_step = state.completed_steps
        - CFL_SA_SHRINK_COOLDOWN_STEPS;
    recovered.available = true;
    state.continuation = recovered;
    std::cout << "[RESTART] recovered version-2 adaptive state from \""
        << summary_path << "\"" << std::endl;
    return true;
}

static void write_binary_checkpoint(const std::string& path,
    const Array4D<double>& K, long long completed_steps,
    double pseudo_time, double cfl, std::uint64_t checkpoint_signature,
    const ContinuationState& continuation)
{
    const std::string temporary_path = path + ".tmp";
    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(
        "Could not open checkpoint for writing: " + temporary_path);
    output.write(CHECKPOINT_MAGIC, sizeof(CHECKPOINT_MAGIC));
    const std::uint32_t version = 4;
    const std::uint32_t nx = Nx;
    const std::uint32_t ny = Ny;
    const std::uint32_t basis = dimPk;
    const std::uint32_t equations = NumEq;
    const std::uint64_t value_count = static_cast<std::uint64_t>(K.data.size());
    checkpoint_write_value(output, version);
    checkpoint_write_value(output, nx);
    checkpoint_write_value(output, ny);
    checkpoint_write_value(output, basis);
    checkpoint_write_value(output, equations);
    checkpoint_write_value(output, checkpoint_signature);
    checkpoint_write_value(output, completed_steps);
    checkpoint_write_value(output, pseudo_time);
    checkpoint_write_value(output, cfl);
    checkpoint_write_value(output, value_count);
    checkpoint_write_value(output, continuation.initial_sa_residual);
    checkpoint_write_value(output, continuation.previous_max_residual);
    checkpoint_write_value(output, continuation.previous_ns_residual);
    checkpoint_write_value(output, continuation.previous_sa_residual);
    checkpoint_write_value(output, continuation.best_max_residual);
    checkpoint_write_value(output, continuation.best_ns_residual);
    checkpoint_write_value(output, continuation.best_sa_residual);
    checkpoint_write_value(output, continuation.sa_shrink_reference);
    checkpoint_write_value(output, continuation.last_sa_shrink_total_step);
    const std::int32_t residual_hold = continuation.residual_convergence_hold;
    const std::int32_t physical_hold = continuation.physical_convergence_hold;
    checkpoint_write_value(output, residual_hold);
    checkpoint_write_value(output, physical_hold);
    checkpoint_write_value(output, continuation.last_physical_relative_change);
    const std::uint32_t history_size = static_cast<std::uint32_t>(
        continuation.residual_growth_history.size());
    checkpoint_write_value(output, history_size);
    for (const auto& residuals : continuation.residual_growth_history) {
        checkpoint_write_value(output, residuals.first);
        checkpoint_write_value(output, residuals.second);
    }
    const std::uint32_t physical_history_size =
        static_cast<std::uint32_t>(
            continuation.physical_convergence_history.size());
    checkpoint_write_value(output, physical_history_size);
    for (const auto& sample : continuation.physical_convergence_history) {
        for (double value : sample) checkpoint_write_value(output, value);
    }
    output.write(reinterpret_cast<const char*>(K.data.data()),
        static_cast<std::streamsize>(value_count * sizeof(double)));
    output.close();
    if (!output) throw std::runtime_error(
        "Checkpoint write failed: " + temporary_path);
    std::remove(path.c_str());
    if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("Could not install checkpoint: " + path);
    }
}

static RestartState read_binary_checkpoint(const std::string& path,
    Array4D<double>& K, std::uint64_t expected_signature,
    bool allow_legacy_checkpoint)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(
        "Could not open restart checkpoint: " + path);
    char magic[sizeof(CHECKPOINT_MAGIC)] = {};
    input.read(magic, sizeof(magic));
    if (std::memcmp(magic, CHECKPOINT_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid checkpoint signature: " + path);
    }
    std::uint32_t version = 0, nx = 0, ny = 0, basis = 0, equations = 0;
    std::uint64_t value_count = 0;
    RestartState state;
    checkpoint_read_value(input, version);
    state.checkpoint_version = version;
    checkpoint_read_value(input, nx);
    checkpoint_read_value(input, ny);
    checkpoint_read_value(input, basis);
    checkpoint_read_value(input, equations);
    std::uint64_t stored_signature = 0;
    if (version >= 2) {
        checkpoint_read_value(input, stored_signature);
    }
    checkpoint_read_value(input, state.completed_steps);
    checkpoint_read_value(input, state.pseudo_time);
    checkpoint_read_value(input, state.cfl);
    checkpoint_read_value(input, value_count);
    if (version >= 3) {
        ContinuationState& continuation = state.continuation;
        checkpoint_read_value(input, continuation.initial_sa_residual);
        checkpoint_read_value(input, continuation.previous_max_residual);
        checkpoint_read_value(input, continuation.previous_ns_residual);
        checkpoint_read_value(input, continuation.previous_sa_residual);
        checkpoint_read_value(input, continuation.best_max_residual);
        checkpoint_read_value(input, continuation.best_ns_residual);
        checkpoint_read_value(input, continuation.best_sa_residual);
        checkpoint_read_value(input, continuation.sa_shrink_reference);
        checkpoint_read_value(input, continuation.last_sa_shrink_total_step);
        std::int32_t residual_hold = 0, physical_hold = 0;
        checkpoint_read_value(input, residual_hold);
        checkpoint_read_value(input, physical_hold);
        continuation.residual_convergence_hold = std::max(residual_hold, 0);
        continuation.physical_convergence_hold = std::max(physical_hold, 0);
        checkpoint_read_value(input,
            continuation.last_physical_relative_change);
        std::uint32_t history_size = 0;
        checkpoint_read_value(input, history_size);
        const std::uint32_t maximum_history_size = static_cast<std::uint32_t>(
            std::max(CFL_RESIDUAL_TREND_WINDOW_STEPS, 1) + 1);
        if (history_size > maximum_history_size) {
            throw std::runtime_error(
                "Checkpoint residual trend history is invalid: " + path);
        }
        continuation.residual_growth_history.resize(history_size);
        for (auto& residuals : continuation.residual_growth_history) {
            checkpoint_read_value(input, residuals.first);
            checkpoint_read_value(input, residuals.second);
        }
        continuation.available = true;
    }
    if (version >= 4) {
        ContinuationState& continuation = state.continuation;
        std::uint32_t physical_history_size = 0;
        checkpoint_read_value(input, physical_history_size);
        const std::uint32_t maximum_physical_history_size =
            static_cast<std::uint32_t>(
                std::max(PHYSICAL_CONVERGENCE_WINDOW_INTERVALS, 1) + 1);
        if (physical_history_size > maximum_physical_history_size) {
            throw std::runtime_error(
                "Checkpoint physical convergence history is invalid: "
                + path);
        }
        continuation.physical_convergence_history.resize(
            physical_history_size);
        for (auto& sample : continuation.physical_convergence_history) {
            for (double& value : sample) checkpoint_read_value(input, value);
        }
    }
    if ((version != 1 && version != 2 && version != 3 && version != 4)
        || nx != Nx || ny != Ny || basis != dimPk
        || equations != NumEq || value_count != K.data.size()) {
        throw std::runtime_error(
            "Checkpoint dimensions do not match this executable: " + path);
    }
    if (version == 1 && !allow_legacy_checkpoint) {
        throw std::runtime_error(
            "Legacy version-1 checkpoint has no case/mesh/geometry signature: "
            + path + ". Start a fresh run or pass --allow-legacy-checkpoint "
            "only when compatibility is known.");
    }
    if (version >= 2 && stored_signature != expected_signature) {
        throw std::runtime_error(
            "Checkpoint case/mesh/geometry signature does not match this run: "
            + path);
    }
    if (version == 1) {
        std::cerr << "[WARN] accepting unverifiable legacy version-1 checkpoint \""
            << path << "\" by explicit request.\n";
    }
    input.read(reinterpret_cast<char*>(K.data.data()),
        static_cast<std::streamsize>(value_count * sizeof(double)));
    if (!input) throw std::runtime_error(
        "Checkpoint data are incomplete: " + path);
    state.binary_checkpoint = true;
    if (version == 2 && !state.continuation.available) {
        recover_v2_continuation_summary(path, state);
    }
    return state;
}

static RestartState read_tecplot_solution_restart(const SolverVariable& solver,
    const std::string& path, Array4D<double>& K, double initial_cfl)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error(
        "Could not open Tecplot restart solution: " + path);
    std::string line;
    if (!std::getline(input, line) || line.find("Variables=") == std::string::npos
        || !std::getline(input, line)
        || line.find("SolutionGaussPoints") == std::string::npos) {
        throw std::runtime_error(
            "Unsupported Tecplot restart header: " + path);
    }
    double maximum_coordinate_error = 0.0;
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        double rhs[dimPk][NumEq] = {};
        for (int q = 0; q < NumQuad; ++q) {
            double x = 0.0, y = 0.0, rho = 0.0, u = 0.0, v = 0.0;
            double p = 0.0, nut = 0.0, mu_t = 0.0, mu_ratio = 0.0;
            if (!(input >> x >> y >> rho >> u >> v >> p >> nut
                    >> mu_t >> mu_ratio)) {
                throw std::runtime_error(
                    "Tecplot restart ended before all solution points: " + path);
            }
            (void)mu_t;
            (void)mu_ratio;
            double expected_x = 0.0, expected_y = 0.0;
            physical_coords(solver, i, j,
                solver.gl_p[q / Quad1D], solver.gl_p[q % Quad1D],
                expected_x, expected_y);
            maximum_coordinate_error = std::max(maximum_coordinate_error,
                std::hypot(x - expected_x, y - expected_y));
            const double state[NumEq] = {
                rho,
                rho * u,
                rho * v,
                p / gamma1 + 0.5 * rho * (u * u + v * v),
                rho * nut
            };
            const double weight = solver.gl_w[q / Quad1D]
                * solver.gl_w[q % Quad1D]
                * solver.metric_cache.volume(i, j, q).jacobian;
            for (int k = 0; k < dimPk; ++k) {
                for (int e = 0; e < NumEq; ++e) {
                    rhs[k][e] += weight * solver.phi_vol[q][k] * state[e];
                }
            }
        }
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            double coefficient = 0.0;
            for (int m = 0; m < dimPk; ++m) {
                coefficient += cell_inv_mass_entry(solver, i, j, k, m)
                    * rhs[m][e];
            }
            K(i, j, k, e) = coefficient;
        }
    }
    std::cout << "[RESTART] imported Tecplot Gauss-point solution from \""
        << path << "\" max_coordinate_error=" << maximum_coordinate_error
        << std::endl;
    RestartState state;
    state.cfl = initial_cfl;
    const std::string summary_path = sibling_file_path(path, "run_summary.dat");
    std::ifstream summary(summary_path);
    if (summary) {
        std::string key;
        long long fallback_steps = 0;
        bool have_total_steps = false;
        while (summary >> key) {
            if (!key.empty() && key[0] == '#') {
                std::getline(summary, line);
            }
            else if (key == "total_completed_steps") {
                summary >> state.completed_steps;
                have_total_steps = true;
            }
            else if (key == "last_loop_step") {
                summary >> fallback_steps;
            }
            else if (key == "pseudo_time_final") {
                summary >> state.pseudo_time;
            }
            else if (key == "cfl_final") {
                summary >> state.cfl;
            }
            else {
                std::getline(summary, line);
            }
        }
        if (!have_total_steps) state.completed_steps = fallback_steps;
        std::cout << "[RESTART] loaded metadata from \"" << summary_path
            << "\"" << std::endl;
    }
    return state;
}

static RestartState load_restart(const SolverVariable& solver,
    const std::string& path, Array4D<double>& K, double initial_cfl,
    std::uint64_t expected_signature, bool allow_legacy_checkpoint)
{
    std::ifstream probe(path, std::ios::binary);
    if (!probe) throw std::runtime_error("Could not open restart file: " + path);
    char magic[sizeof(CHECKPOINT_MAGIC)] = {};
    probe.read(magic, sizeof(magic));
    const bool binary = probe.gcount() == sizeof(magic)
        && std::memcmp(magic, CHECKPOINT_MAGIC, sizeof(magic)) == 0;
    probe.close();
    return binary
        ? read_binary_checkpoint(path, K, expected_signature,
            allow_legacy_checkpoint)
        : read_tecplot_solution_restart(solver, path, K, initial_cfl);
}

static double cgns_wall_y_at_x(double x)
{
    if (!is_solid_wall_x(x)) return 0.0;
    return flat_plate_wall_y(x);
}

inline void eval_state_at(const SolverVariable& S, const Array4D<double>& K, int i, int j, double r, double s, double* U)
{
    double phi[dimPk];
    eval_phi_at(S, r, s, phi);
    for (int e = 0; e < NumEq; ++e) U[e] = 0.0;
    for (int k = 0; k < dimPk; ++k) {
        const double ph = phi[k];
        for (int e = 0; e < NumEq; ++e) U[e] += K(i, j, k, e) * ph;
    }
}

inline void eval_grad_at(const SolverVariable& S, int i, int j, double r, double s, double* dUx, double* dUy)
{
    double phi[dimPk];
    eval_phi_at(S, r, s, phi);
    for (int e = 0; e < NumEq; ++e) { dUx[e] = 0.0; dUy[e] = 0.0; }
    for (int k = 0; k < dimPk; ++k) {
        const double ph = phi[k];
        for (int e = 0; e < NumEq; ++e) {
            dUx[e] += S.gradUX(i, j, k, e) * ph;
            dUy[e] += S.gradUY(i, j, k, e) * ph;
        }
    }
}

inline double turbulent_mu_from_U(const double* U)
{
    const double r = std::max(U[0], 1e-12);
    const double nut = U[4] / r;
    return sa_eddy_mu_from_rho_nut(r, nut, laminar_mu_from_cons(U));
}

inline double wall_distance(double x, double y);



struct PlateWallFaceMetrics {
    double x = 0.0;
    double y = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double ds_metric = 0.0;
    double tau_w = 0.0;
    double tau_w_raw = 0.0;
    double rho_w = rho_inf;
    double p_w = p_inf;
    double mu_lam_w = mu_ref;
    double u_t_interior = 0.0;
    double u_n_interior = 0.0;
    double nut_interior = 0.0;
    double mut_interior = 0.0;
    double mu_sa_wall = mu_ref / SA_SIGMA;
    double sa_flux_n_raw = 0.0;
    double sa_flux_n_br2 = 0.0;
    double sa_flux_n_lift = 0.0;
    double du_t_dn_raw = 0.0;
    double du_t_dn_br2 = 0.0;
    double du_t_dn_lift = 0.0;
    double dnut_dn_raw = 0.0;
    double dnut_dn_br2 = 0.0;
    double dnut_dn_lift = 0.0;
    double du_t_dn_direct = 0.0;
    double dnut_dn_direct = 0.0;
    double wall_distance_value = 0.0;
    bool valid = false;
};

inline double wall_shear_on_wall_from_flux(const double* Fv, const double* Gv,
    double nx, double ny)
{
    const double tx = -ny;
    const double ty = nx;
    const double traction_x = Fv[1] * nx + Gv[1] * ny;
    const double traction_y = Fv[2] * nx + Gv[2] * ny;
    return -(traction_x * tx + traction_y * ty);
}

inline void face_metric_at_reference(const SolverVariable& S, int i, int j, int f,
    double face_ref, double& nx, double& ny, double& ds)
{
    double r = 0.0, s = 0.0;
    if (f == 0) { r = face_ref; s = -1.0; }
    else if (f == 1) { r = 1.0; s = face_ref; }
    else if (f == 2) { r = face_ref; s = 1.0; }
    else { r = -1.0; s = face_ref; }

    double xr = 0.0, xs = 0.0, yr = 0.0, ys = 0.0, J = 0.0;
    contravariant_metrics(S, i, j, r, s, xr, xs, yr, ys, J);
    double nx_ds = 0.0, ny_ds = 0.0;
    if (f == 0) { nx_ds = yr; ny_ds = -xr; ds = std::sqrt(xr * xr + yr * yr); }
    else if (f == 2) { nx_ds = -yr; ny_ds = xr; ds = std::sqrt(xr * xr + yr * yr); }
    else if (f == 1) { nx_ds = ys; ny_ds = -xs; ds = std::sqrt(xs * xs + ys * ys); }
    else { nx_ds = -ys; ny_ds = xs; ds = std::sqrt(xs * xs + ys * ys); }
    const double inv = 1.0 / std::max(ds, 1e-30);
    nx = nx_ds * inv;
    ny = ny_ds * inv;
}

inline double plate_coverage_length_in_cell(const SolverVariable& S, int i)
{
    const double xL = std::max(S.x_edges[i], PLATE_X0);
    const double xR = std::min(S.x_edges[i + 1], PLATE_X1);
    return std::max(xR - xL, 0.0);
}

inline double plate_segment_center_in_cell(const SolverVariable& S, int i)
{
    const double xL = std::max(S.x_edges[i], PLATE_X0);
    const double xR = std::min(S.x_edges[i + 1], PLATE_X1);
    return 0.5 * (xL + xR);
}

inline PlateWallFaceMetrics plate_wall_point_metrics(const SolverVariable& S,
    const Array4D<double>& K,
    int i,
    double r_face)
{
    PlateWallFaceMetrics metrics;
    const int j = 0;
    const int f = 0;
    double xFace = 0.0, yFace = 0.0;
    physical_coords(S, i, j, r_face, -1.0, xFace, yFace);
#if defined(DG_CASE_NACA0012)
    // Use topology instead of an x test at the sharp trailing edge, where the
    // wall and wake-cut coordinates meet.  The range comes from mesh metadata.
    if (i < g_wall_i_begin || i >= g_wall_i_end) return metrics;
#else
    if (!is_plate_wall_face(f, xFace)) return metrics;
#endif
    double nx = 0.0, ny = 0.0, ds_metric = 0.0;
    face_metric_at_reference(S, i, j, f, r_face, nx, ny, ds_metric);
    const double tx = -ny;
    const double ty = nx;

    double phi_pt[dimPk] = { 0.0 };
    double dphir_pt[dimPk] = { 0.0 };
    double dphis_pt[dimPk] = { 0.0 };
    eval_tensor_basis(r_face, -1.0, phi_pt, dphir_pt, dphis_pt);

    double uL[NumEq] = { 0.0 }, uWall[NumEq] = { 0.0 };
    double duxL[NumEq] = { 0.0 }, duyL[NumEq] = { 0.0 };
    double dUr_direct[NumEq] = { 0.0 }, dUs_direct[NumEq] = { 0.0 };
    for (int k = 0; k < dimPk; ++k) {
        const double ph = phi_pt[k];
        for (int e = 0; e < NumEq; ++e) {
            uL[e] += K(i, j, k, e) * ph;
            duxL[e] += S.gradUX(i, j, k, e) * ph;
            duyL[e] += S.gradUY(i, j, k, e) * ph;
            dUr_direct[e] += K(i, j, k, e) * dphir_pt[k];
            dUs_direct[e] += K(i, j, k, e) * dphis_pt[k];
        }
    }
    if (!boundary_viscous_trace_2dzp(uL, f, xFace, yFace, nx, ny, uWall)) return metrics;

    metrics.x = xFace;
    metrics.y = yFace;
    metrics.nx = nx;
    metrics.ny = ny;
    metrics.tx = tx;
    metrics.ty = ty;
    metrics.ds_metric = ds_metric;
    metrics.wall_distance_value = wall_distance(metrics.x, metrics.y);
    {
        const double rhoL = std::max(uL[0], 1e-12);
        const double u = uL[1] / rhoL;
        const double v = uL[2] / rhoL;
        const double nut = uL[4] / rhoL;
        metrics.u_t_interior = u * tx + v * ty;
        metrics.u_n_interior = u * nx + v * ny;
        metrics.nut_interior = nut;
        metrics.mut_interior = turbulent_mu_from_U(uL);
    }

    auto primitive_wall_derivatives = [&](const double* dUx, const double* dUy,
        double& du_t_dn, double& dnut_dn) {
        const double rho = std::max(uWall[0], 1e-12);
        const double uw = uWall[1] / rho;
        const double vw = uWall[2] / rho;
        const double nutw = uWall[4] / rho;
        const double du_dx = (dUx[1] - uw * dUx[0]) / rho;
        const double du_dy = (dUy[1] - uw * dUy[0]) / rho;
        const double dv_dx = (dUx[2] - vw * dUx[0]) / rho;
        const double dv_dy = (dUy[2] - vw * dUy[0]) / rho;
        const double dnut_dx = (dUx[4] - nutw * dUx[0]) / rho;
        const double dnut_dy = (dUy[4] - nutw * dUy[0]) / rho;
        const double dut_dx = tx * du_dx + ty * dv_dx;
        const double dut_dy = tx * du_dy + ty * dv_dy;
        du_t_dn = dut_dx * nx + dut_dy * ny;
        dnut_dn = dnut_dx * nx + dnut_dy * ny;
        };

    double Fv[NumEq] = { 0.0 }, Gv[NumEq] = { 0.0 };
    Viscous_Flux(uWall, duxL, duyL, Fv, Gv);
    metrics.tau_w_raw = wall_shear_on_wall_from_flux(Fv, Gv, nx, ny);
    metrics.sa_flux_n_raw = Fv[4] * nx + Gv[4] * ny;
    primitive_wall_derivatives(duxL, duyL, metrics.du_t_dn_raw, metrics.dnut_dn_raw);

    double duxDirect[NumEq] = { 0.0 }, duyDirect[NumEq] = { 0.0 };
    {
        double xr = 0.0, xs = 0.0, yr = 0.0, ys = 0.0, J = 0.0;
        contravariant_metrics(S, i, j, r_face, -1.0, xr, xs, yr, ys, J);
        const double invJ = 1.0 / std::max(J, 1e-30);
        for (int e = 0; e < NumEq; ++e) {
            duxDirect[e] = (dUr_direct[e] * ys - dUs_direct[e] * yr) * invJ;
            duyDirect[e] = (-dUr_direct[e] * xs + dUs_direct[e] * xr) * invJ;
        }
    }
    primitive_wall_derivatives(duxDirect, duyDirect, metrics.du_t_dn_direct, metrics.dnut_dn_direct);

    double jump[FaceQuad][NumEq] = { 0.0 };
    for (int g = 0; g < FaceQuad; ++g) {
        double uFace[NumEq] = { 0.0 }, uFaceWall[NumEq] = { 0.0 };
        for (int k = 0; k < dimPk; ++k) {
            const double ph = S.phi_face[f][g][k];
            for (int e = 0; e < NumEq; ++e) uFace[e] += K(i, j, k, e) * ph;
        }
        double xg, yg;
        face_point_coords(S, i, j, f, g, xg, yg);
        double nxg = 0.0, nyg = 0.0, dsg = 0.0;
        face_metric(S, i, j, f, g, nxg, nyg, dsg);
        boundary_viscous_trace_2dzp(uFace, f, xg, yg, nxg, nyg, uFaceWall);
        for (int e = 0; e < NumEq; ++e) jump[g][e] = uFace[e] - uFaceWall[e];
        for (int e = 1; e <= 3; ++e) jump[g][e] *= BR2_WALL_BOUNDARY_TRACE_FACTOR * NS_WALL_BR2_SCALE;
        jump[g][4] *= BR2_WALL_BOUNDARY_TRACE_FACTOR * SA_WALL_BR2_SCALE
            * g_sa_br2_factor;
    }

    double rX[dimPk][NumEq] = { 0.0 }, rY[dimPk][NumEq] = { 0.0 };
    computeFaceLifting(S, &jump[0][0], f, i, j, &rX[0][0], &rY[0][0], BR2_ETA);
    double duxBR2[NumEq] = { 0.0 }, duyBR2[NumEq] = { 0.0 };
    for (int e = 0; e < NumEq; ++e) {
        duxBR2[e] = duxL[e];
        duyBR2[e] = duyL[e];
    }
    for (int k = 0; k < dimPk; ++k) {
        const double ph = phi_pt[k];
        for (int e = 0; e < NumEq; ++e) {
            duxBR2[e] -= rX[k][e] * ph;
            duyBR2[e] -= rY[k][e] * ph;
        }
    }

    Viscous_Flux(uWall, duxBR2, duyBR2, Fv, Gv);
    metrics.tau_w = wall_shear_on_wall_from_flux(Fv, Gv, nx, ny);
    metrics.sa_flux_n_br2 = Fv[4] * nx + Gv[4] * ny;
    metrics.sa_flux_n_lift = metrics.sa_flux_n_raw - metrics.sa_flux_n_br2;
    primitive_wall_derivatives(duxBR2, duyBR2, metrics.du_t_dn_br2, metrics.dnut_dn_br2);
    metrics.du_t_dn_lift = metrics.du_t_dn_raw - metrics.du_t_dn_br2;
    metrics.dnut_dn_lift = metrics.dnut_dn_raw - metrics.dnut_dn_br2;
    metrics.rho_w = std::max(uWall[0], 1e-12);
    const double uw = uWall[1] / metrics.rho_w;
    const double vw = uWall[2] / metrics.rho_w;
    metrics.p_w = std::max(gamma1 * (uWall[3] - 0.5 * metrics.rho_w * (uw * uw + vw * vw)), 1e-12);
    metrics.mu_lam_w = laminar_mu_from_cons(uWall);
    metrics.mu_sa_wall = metrics.mu_lam_w / SA_SIGMA;
    metrics.valid = true;
    return metrics;
}

inline PlateWallFaceMetrics plate_wall_face_metrics(const SolverVariable& S, const Array4D<double>& K, int i)
{
    PlateWallFaceMetrics metrics;
    const double xL = std::max(S.x_edges[i], PLATE_X0);
    const double xR = std::min(S.x_edges[i + 1], PLATE_X1);
    if (xR <= xL) return metrics;

    const double rL = std::max(-1.0, std::min(1.0, (xL - S.x_centers[i]) / std::max(S.h_x[i], 1e-30)));
    const double rR = std::max(-1.0, std::min(1.0, (xR - S.x_centers[i]) / std::max(S.h_x[i], 1e-30)));
    if (!(rR > rL)) return metrics;





    double tau_sum = 0.0;
    double tau_raw_sum = 0.0;
    double rho_sum = 0.0;
    double p_sum = 0.0;
    double mu_lam_sum = 0.0;
    double wsum = 0.0;
    const double sub_jac = 0.5 * (rR - rL);
    for (int g = 0; g < FaceQuad; ++g) {
        const double r = 0.5 * (rL + rR) + sub_jac * S.gl_p[g];
        const PlateWallFaceMetrics point = plate_wall_point_metrics(S, K, i, r);
        if (!point.valid) return metrics;
        const double tau_xy = point.tau_w;
        const double w = S.gl_w[g] * sub_jac * S.h_x[i];
        tau_sum += w * tau_xy;
        tau_raw_sum += w * point.tau_w_raw;
        rho_sum += w * point.rho_w;
        p_sum += w * point.p_w;
        mu_lam_sum += w * point.mu_lam_w;
        wsum += w;
    }

    if (wsum > 0.0) {
        metrics.tau_w = tau_sum / wsum;
        metrics.tau_w_raw = tau_raw_sum / wsum;
        metrics.rho_w = rho_sum / wsum;
        metrics.p_w = p_sum / wsum;
        metrics.mu_lam_w = mu_lam_sum / wsum;
        metrics.valid = true;
    }
    return metrics;
}


inline void boundary_layer_metrics(const std::vector<double>& y, const std::vector<double>& uu,
    double& delta99, double& dstar, double& theta, double& H12)
{
    delta99 = y.back();
    for (size_t k = 1; k < y.size(); ++k) {
        if (uu[k] >= 0.99) {
            const double y0 = y[k - 1], y1 = y[k];
            const double u0 = uu[k - 1], u1 = uu[k];
            const double t = (0.99 - u0) / std::max(u1 - u0, 1e-12);
            delta99 = y0 + t * (y1 - y0);
            break;
        }
    }
    dstar = 0.0; theta = 0.0;
    for (size_t k = 1; k < y.size(); ++k) {
        const double dy = y[k] - y[k - 1];
        const double f1a = 1.0 - uu[k - 1], f1b = 1.0 - uu[k];
        const double f2a = uu[k - 1] * (1.0 - uu[k - 1]), f2b = uu[k] * (1.0 - uu[k]);
        dstar += 0.5 * dy * (f1a + f1b);
        theta += 0.5 * dy * (f2a + f2b);
    }
    H12 = dstar / std::max(theta, 1e-30);
}

#if defined(DG_CASE_NACA0012)
struct NacaForceCoefficients {
    double cd = 0.0;
    double cl = 0.0;
    double cm_c4 = 0.0;
    double cd_pressure = 0.0;
    double cl_pressure = 0.0;
    double cd_viscous = 0.0;
    double cl_viscous = 0.0;
    double cm_pressure = 0.0;
    double cm_viscous = 0.0;
};

static NacaForceCoefficients compute_naca_force_coefficients(
    const SolverVariable& S, const Array4D<double>& K)
{
    NacaForceCoefficients coefficients;
    const double qinf = 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    double pressure_force_x = 0.0, pressure_force_y = 0.0;
    double viscous_force_x = 0.0, viscous_force_y = 0.0;
    double pressure_moment = 0.0, viscous_moment = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int g = 0; g < FaceQuad; ++g) {
            const PlateWallFaceMetrics wall =
                plate_wall_point_metrics(S, K, i, S.gl_p[g]);
            if (!wall.valid) continue;
            const double weight = S.gl_w[g] * wall.ds_metric;
            const double pressure_gauge = wall.p_w - p_inf;
            const double pressure_dfx = pressure_gauge * wall.nx * weight;
            const double pressure_dfy = pressure_gauge * wall.ny * weight;
            const double viscous_dfx = wall.tau_w * wall.tx * weight;
            const double viscous_dfy = wall.tau_w * wall.ty * weight;
            pressure_force_x += pressure_dfx;
            pressure_force_y += pressure_dfy;
            viscous_force_x += viscous_dfx;
            viscous_force_y += viscous_dfy;
            pressure_moment += (wall.x - 0.25) * pressure_dfy
                - wall.y * pressure_dfx;
            viscous_moment += (wall.x - 0.25) * viscous_dfy
                - wall.y * viscous_dfx;
        }
    }
    const double scale = 1.0 / std::max(qinf, 1e-30);
    const double ca = std::cos(inflow_angle);
    const double sa = std::sin(inflow_angle);
    coefficients.cd_pressure =
        (pressure_force_x * ca + pressure_force_y * sa) * scale;
    coefficients.cl_pressure =
        (-pressure_force_x * sa + pressure_force_y * ca) * scale;
    coefficients.cd_viscous =
        (viscous_force_x * ca + viscous_force_y * sa) * scale;
    coefficients.cl_viscous =
        (-viscous_force_x * sa + viscous_force_y * ca) * scale;
    coefficients.cm_pressure = pressure_moment * scale;
    coefficients.cm_viscous = viscous_moment * scale;
    coefficients.cd = coefficients.cd_pressure + coefficients.cd_viscous;
    coefficients.cl = coefficients.cl_pressure + coefficients.cl_viscous;
    coefficients.cm_c4 = coefficients.cm_pressure + coefficients.cm_viscous;
    return coefficients;
}

static double aerodynamic_output_relative_change(
    const NacaForceCoefficients& current,
    const NacaForceCoefficients& previous)
{
    const std::array<double, 3> current_values = {
        current.cd, current.cl, current.cm_c4
    };
    const std::array<double, 3> previous_values = {
        previous.cd, previous.cl, previous.cm_c4
    };
    double difference_squared = 0.0;
    double current_squared = 0.0;
    for (std::size_t component = 0; component < current_values.size(); ++component) {
        if (!std::isfinite(current_values[component])
            || !std::isfinite(previous_values[component])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double difference = current_values[component]
            - previous_values[component];
        difference_squared += difference * difference;
        current_squared += current_values[component] * current_values[component];
    }
    if (current_squared <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(difference_squared / current_squared);
}
#endif


static void write_verification_outputs(SolverVariable& S, const Array4D<double>& K) {
    computeRHS(S, K);
    const double qinf = 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf);
#if defined(DG_CASE_NACA0012)
    struct SurfaceSample {
        int cell = 0;
        int face_gauss = 0;
        double x = 0.0;
        double y = 0.0;
        double cp = 0.0;
        double cf = 0.0;
        double nx = 0.0;
        double ny = 0.0;
    };
    std::array<std::vector<SurfaceSample>, 2> surface_samples;
    std::ofstream surface("naca0012_surface.dat");
    surface << std::setprecision(16)
        << "# surface_id: 0=lower, 1=upper; each surface is ordered leading_edge_to_trailing_edge\n"
        << "# surface_id surface_point cell face_gauss x y Cp Cf nx ny\n";
    double pressure_force_x = 0.0, pressure_force_y = 0.0;
    double viscous_force_x = 0.0, viscous_force_y = 0.0;
    double pressure_moment = 0.0, viscous_moment = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int g = 0; g < FaceQuad; ++g) {
            const PlateWallFaceMetrics wall =
                plate_wall_point_metrics(S, K, i, S.gl_p[g]);
            if (!wall.valid) continue;
            const double weight = S.gl_w[g] * wall.ds_metric;
            const double cp = (wall.p_w - p_inf) / std::max(qinf, 1e-30);
            double oriented_tau = wall.tau_w;
            if (wall.tx * u_inf + wall.ty * v_inf < 0.0) oriented_tau = -oriented_tau;
            const double cf = oriented_tau / std::max(qinf, 1e-30);
            const int leading_edge_index =
                g_wall_i_begin + (g_wall_i_end - g_wall_i_begin) / 2;
            const int surface_id = i < leading_edge_index ? 0 : 1;
            surface_samples[static_cast<size_t>(surface_id)].push_back({
                i, g, wall.x, wall.y, cp, cf, wall.nx, wall.ny });
            // Gauge pressure is mathematically equivalent on a closed body
            // and avoids amplifying tiny discrete surface-closure errors by
            // the much larger freestream static pressure.
            const double pressure_gauge = wall.p_w - p_inf;
            const double pressure_dfx = pressure_gauge * wall.nx * weight;
            const double pressure_dfy = pressure_gauge * wall.ny * weight;
            const double viscous_dfx = wall.tau_w * wall.tx * weight;
            const double viscous_dfy = wall.tau_w * wall.ty * weight;
            pressure_force_x += pressure_dfx;
            pressure_force_y += pressure_dfy;
            viscous_force_x += viscous_dfx;
            viscous_force_y += viscous_dfy;
            pressure_moment += (wall.x - 0.25) * pressure_dfy
                - wall.y * pressure_dfx;
            viscous_moment += (wall.x - 0.25) * viscous_dfy
                - wall.y * viscous_dfx;
        }
    }
    for (int surface_id = 0; surface_id < 2; ++surface_id) {
        std::vector<SurfaceSample>& samples = surface_samples[static_cast<size_t>(surface_id)];
        std::sort(samples.begin(), samples.end(), [](const SurfaceSample& a,
            const SurfaceSample& b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.cell != b.cell) return a.cell < b.cell;
            return a.face_gauss < b.face_gauss;
        });
        for (size_t point = 0; point < samples.size(); ++point) {
            const SurfaceSample& sample = samples[point];
            surface << surface_id << ' ' << point << ' ' << sample.cell << ' '
                << sample.face_gauss << ' ' << sample.x << ' ' << sample.y << ' '
                << sample.cp << ' ' << sample.cf << ' ' << sample.nx << ' '
                << sample.ny << '\n';
        }
        if (surface_id == 0) surface << '\n';
    }
    const double ca = std::cos(inflow_angle);
    const double sa = std::sin(inflow_angle);
    const double force_scale = 1.0 / std::max(qinf, 1e-30);
    const double cd_pressure = (pressure_force_x * ca + pressure_force_y * sa) * force_scale;
    const double cl_pressure = (-pressure_force_x * sa + pressure_force_y * ca) * force_scale;
    const double cd_viscous = (viscous_force_x * ca + viscous_force_y * sa) * force_scale;
    const double cl_viscous = (-viscous_force_x * sa + viscous_force_y * ca) * force_scale;
    const double cd = cd_pressure + cd_viscous;
    const double cl = cl_pressure + cl_viscous;
    const double cm_pressure = pressure_moment * force_scale;
    const double cm_viscous = viscous_moment * force_scale;
    const double force_x = pressure_force_x + viscous_force_x;
    const double force_y = pressure_force_y + viscous_force_y;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::ofstream forces("naca0012_forces.dat");
    forces << std::setprecision(16)
        << "# coefficients use chord=1 and q_inf=0.5*rho_inf*|V_inf|^2\n"
        << "# grid_family " << dg::case_config::grid_family << '\n'
        << "# reference_comparison "
        << dg::case_config::reference_comparison << '\n'
        << "# farfield_point_vortex_correction "
        << dg::case_config::farfield_correction << '\n'
        << "# moment_convention " << dg::case_config::moment_convention << '\n'
        << "alpha_degrees " << inflow_angle * 180.0 / pi << '\n'
        << "force_x " << force_x << '\n'
        << "force_y " << force_y << '\n'
        << "Cd " << cd << '\n'
        << "Cl " << cl << '\n'
        << "Cm_c4 " << cm_pressure + cm_viscous << '\n'
        << "Cd_pressure " << cd_pressure << '\n'
        << "Cl_pressure " << cl_pressure << '\n'
        << "Cd_viscous " << cd_viscous << '\n'
        << "Cl_viscous " << cl_viscous << '\n'
        << "Cm_pressure " << cm_pressure << '\n'
        << "Cm_viscous " << cm_viscous << '\n';
    std::cout << "[NACA0012] wrote naca0012_surface.dat and naca0012_forces.dat"
        << "  Cd=" << cd << " Cl=" << cl << '\n';
    return;
#endif
    const double cf_den = qinf;
    auto locate_x = [&](double xt, int& i, double& r) {
        i = 0;
        for (int ii = 0; ii < Nx; ++ii) {
            if (xt >= S.x_edges[ii] - 1e-12 && xt <= S.x_edges[ii + 1] + 1e-12) {
                i = ii;
                break;
            }
        }
        const double xc = S.x_centers[i];
        r = std::max(-1.0, std::min(1.0, (xt - xc) / std::max(S.h_x[i], 1e-30)));
        };

    std::ofstream fcf_grid(std::string("cf_") + dg::case_config::short_name + ".dat");
    std::ofstream fwall_diag("wall_flux_diagnostics.dat");
    fcf_grid << "# x y_wall Cf tau_w\n";
    fcf_grid << "# Cf = tau_w / (0.5*rho_inf*u_inf^2), tau_w is fluid-on-wall shear in the +wall-tangent direction\n";
    fcf_grid << "# wall-face Gauss points only\n";
    fwall_diag << "# x y_wall nx ny tx ty ds_metric Cf tau_w tau_w_raw "
        << "rho_w p_w mu_lam_w u_t_interior u_n_interior nut_interior "
        << "mut_interior mut_over_muinf_interior du_t_dn_direct du_t_dn_raw "
        << "du_t_dn_br2 du_t_dn_lift dnut_dn_direct dnut_dn_raw dnut_dn_br2 "
        << "dnut_dn_lift mu_sa_wall sa_flux_n_raw sa_flux_n_br2 sa_flux_n_lift "
        << "wall_distance\n";
    for (int i = 0; i < Nx; ++i) {
        const double xL = std::max(S.x_edges[i], PLATE_X0);
        const double xR = std::min(S.x_edges[i + 1], PLATE_X1);
        if (xR <= xL) continue;
        for (int g = 0; g < FaceQuad; ++g) {
            const double r = S.gl_p[g];
            const double x = S.x_centers[i] + r * S.h_x[i];
            if (x < xL - 1e-12 || x > xR + 1e-12) continue;
            const PlateWallFaceMetrics wall = plate_wall_point_metrics(S, K, i, r);
            if (!wall.valid) continue;
            const double cf = wall.tau_w / std::max(cf_den, 1e-30);
            fcf_grid << std::setprecision(16) << wall.x << " " << wall.y << " " << cf << " " << wall.tau_w << "\n";
            fwall_diag << std::setprecision(16)
                << wall.x << " " << wall.y << " "
                << wall.nx << " " << wall.ny << " "
                << wall.tx << " " << wall.ty << " "
                << wall.ds_metric << " "
                << cf << " " << wall.tau_w << " " << wall.tau_w_raw << " "
                << wall.rho_w << " " << wall.p_w << " " << wall.mu_lam_w << " "
                << wall.u_t_interior << " " << wall.u_n_interior << " "
                << wall.nut_interior << " " << wall.mut_interior << " "
                << (wall.mut_interior / std::max(mu_ref, 1e-30)) << " "
                << wall.du_t_dn_direct << " " << wall.du_t_dn_raw << " "
                << wall.du_t_dn_br2 << " " << wall.du_t_dn_lift << " "
                << wall.dnut_dn_direct << " " << wall.dnut_dn_raw << " "
                << wall.dnut_dn_br2 << " " << wall.dnut_dn_lift << " "
                << wall.mu_sa_wall << " "
                << wall.sa_flux_n_raw << " " << wall.sa_flux_n_br2 << " "
                << wall.sa_flux_n_lift << " "
                << wall.wall_distance_value << "\n";
        }
    }
    fcf_grid.close();
    fwall_diag.close();

    auto write_profile_at_x = [&](double xt,
        std::ofstream* fmut_profile,
        std::ofstream* fu_profile) {
            double utau = std::numeric_limits<double>::quiet_NaN();
            double nu_wall = mu_ref / rho_inf;
            if (fu_profile) {
                int iw = 0;
                double rw = 0.0;
                locate_x(xt, iw, rw);
                const PlateWallFaceMetrics wall = plate_wall_point_metrics(S, K, iw, rw);
                if (wall.valid) {
                    utau = std::sqrt(std::max(std::abs(wall.tau_w) / std::max(wall.rho_w, 1e-30), 1e-30));
                    nu_wall = wall.mu_lam_w / std::max(wall.rho_w, 1e-30);
                }
                else {
                    utau = std::sqrt(std::max(std::abs(qinf * 0.003) / rho_inf, 1e-30));
                }
            }
            std::vector<std::pair<int, double>> traces;
            for (int ii = 0; ii < Nx; ++ii) {
                if (xt < S.x_edges[ii] - 1e-12 || xt > S.x_edges[ii + 1] + 1e-12) continue;
                const double rr = std::max(-1.0, std::min(1.0,
                    (xt - S.x_centers[ii]) / std::max(S.h_x[ii], 1e-30)));
                traces.emplace_back(ii, rr);
            }
            if (traces.empty()) {
                int i = 0;
                double r = 0.0;
                locate_x(xt, i, r);
                traces.emplace_back(i, r);
            }
            for (int j = 0; j < Ny; ++j) {
                for (int gy = 0; gy < Quad1D; ++gy) {
                    const double s = S.gl_p[gy];
                    double x_phys = 0.0, y_phys = 0.0;
                    double mut_left = std::numeric_limits<double>::quiet_NaN();
                    double mut_right = std::numeric_limits<double>::quiet_NaN();
                    double mut_sum = 0.0;
                    double u_sum = 0.0;
                    for (const auto& tr : traces) {
                        double Ut[NumEq];
                        eval_state_at(S, K, tr.first, j, tr.second, s, Ut);
                        double xtmp = 0.0, ytmp = 0.0;
                        physical_coords(S, tr.first, j, tr.second, s, xtmp, ytmp);
                        x_phys += xtmp;
                        y_phys += ytmp;
                        const double rho_t = std::max(Ut[0], 1e-12);
                        const double mut_t = turbulent_mu_from_U(Ut);
                        const double u_t = Ut[1] / rho_t;
                        if (!std::isfinite(mut_left)) mut_left = mut_t;
                        mut_right = mut_t;
                        mut_sum += mut_t;
                        u_sum += u_t;
                    }
                    const double inv_traces = 1.0 / std::max((double)traces.size(), 1.0);
                    x_phys *= inv_traces;
                    y_phys *= inv_traces;
                    if (traces.size() == 1) mut_right = mut_left;
                    const double mut_average = mut_sum * inv_traces;
                    const double u_average = u_sum * inv_traces;
                    const double y = std::max(y_phys - cgns_wall_y_at_x(xt), 0.0);
                    if (fmut_profile) {
                        (*fmut_profile) << std::setprecision(16)
                            << x_phys << " " << y_phys << " " << y << " "
                            << mut_left / std::max(mu_ref, 1e-30) << " "
                            << mut_right / std::max(mu_ref, 1e-30) << " "
                            << mut_average / std::max(mu_ref, 1e-30) << " "
                            << traces.size() << "\n";
                    }
                    if (fu_profile) {
                        const double yplus = y * utau / std::max(nu_wall, 1e-30);
                        const double uplus = u_average / std::max(utau, 1e-30);
                        (*fu_profile) << std::setprecision(16)
                            << xt << " " << x_phys << " " << y_phys << " " << y << " "
                            << yplus << " " << uplus << " " << u_average << " "
                            << utau << " " << nu_wall << "\n";
                    }
                }
            }
        };

    std::ofstream fmut_profile_097("mut_0.97.dat");
    fmut_profile_097 << "# x y y_minus_ywall mut_left_over_muinf mut_right_over_muinf mut_average_over_muinf trace_count\n";
    fmut_profile_097 << "# If trace_count=1, left/right/average are the same single-cell trace; if trace_count=2, left/right are the two DG traces at the x interface.\n";
    write_profile_at_x(0.97, &fmut_profile_097, nullptr);
    fmut_profile_097.close();

    std::ofstream fuplus_yplus(std::string("uplus_yplus_")
        + dg::case_config::short_name + ".dat");
    fuplus_yplus << "# x_target x y y_minus_ywall yplus uplus u utau nu_wall\n";
    write_profile_at_x(0.97008, nullptr, &fuplus_yplus);
    write_profile_at_x(1.90334, nullptr, &fuplus_yplus);
    fuplus_yplus.close();

}


using PrimitiveState = dg::PrimitiveState;

inline PrimitiveState freestream_primitive()
{
    PrimitiveState V;
    V.rho = rho_inf;
    V.u = u_inf;
    V.v = v_inf;
    V.p = p_inf;
    V.nut = SA_INIT_NUT;
    return V;
}

inline dg::boundary_physics::GasProperties boundary_gas_properties()
{
    dg::boundary_physics::GasProperties gas;
    gas.gamma = gamma_v;
    gas.gamma_minus_one = gamma1;
    gas.freestream_mach = M_inf;
    gas.freestream = freestream_primitive();
    return gas;
}

inline PrimitiveState cons_to_primitive(const double* U)
{
    return dg::conservative_to_primitive(U, gamma_v);
}

inline void primitive_to_cons(const PrimitiveState& V, double* U)
{
    dg::primitive_to_conservative(V, gamma_v, U);
}

inline void velocity_to_normal_tangent(double u, double v, double nx, double ny, double& un, double& ut)
{
    dg::velocity_to_normal_tangent(u, v, nx, ny, un, ut);
}

inline void normal_tangent_to_velocity(double un, double ut, double nx, double ny, double& u, double& v)
{
    dg::normal_tangent_to_velocity(un, ut, nx, ny, u, v);
}

inline PrimitiveState characteristic_farfield_boundary(
    const PrimitiveState& Vi,
    const PrimitiveState& Vext,
    double nx,
    double ny)
{
    return dg::boundary_physics::characteristic_farfield_boundary(
        Vi, Vext, nx, ny, boundary_gas_properties());
}

inline PrimitiveState characteristic_pressure_outlet_boundary(
    const PrimitiveState& Vi,
    double p_back,
    double nx,
    double ny)
{
    return dg::boundary_physics::characteristic_pressure_outlet_boundary(
        Vi, p_back, nx, ny, boundary_gas_properties());
}

inline PrimitiveState characteristic_total_inflow_boundary(
    const PrimitiveState& Vi,
    double p0,
    double T0,
    double alpha,
    double nx,
    double ny)
{
    return dg::boundary_physics::characteristic_total_inflow_boundary(
        Vi, p0, T0, alpha, nx, ny, boundary_gas_properties());
}

inline void face_point_coords(const SolverVariable& S, int i, int j, int f, int g, double& x, double& y)
{
    const dg::FaceQuadratureMetric& metric = S.metric_cache.face(i, j, f, g);
    x = metric.x;
    y = metric.y;
}

inline void physical_coords(const SolverVariable& S, int i, int j, double r, double s, double& x, double& y)
{
    double lr[GeomNodes1D] = {}, ls[GeomNodes1D] = {};
    double dlr[GeomNodes1D] = {}, dls[GeomNodes1D] = {};
    geometry_basis_1d(r, lr, dlr);
    geometry_basis_1d(s, ls, dls);
    (void)dlr;
    (void)dls;
    x = 0.0;
    y = 0.0;
    for (int b = 0; b < GeomNodes1D; ++b) {
        for (int a = 0; a < GeomNodes1D; ++a) {
            const int k = b * GeomNodes1D + a;
            const double ph = lr[a] * ls[b];
            x += S.geom(i, j, k, 0) * ph;
            y += S.geom(i, j, k, 1) * ph;
        }
    }
}

inline void contravariant_metrics(const SolverVariable& S, int i, int j, double r, double s,
    double& xr, double& xs, double& yr, double& ys, double& J)
{
    double lr[GeomNodes1D] = {}, ls[GeomNodes1D] = {};
    double dlr[GeomNodes1D] = {}, dls[GeomNodes1D] = {};
    geometry_basis_1d(r, lr, dlr);
    geometry_basis_1d(s, ls, dls);
    xr = 0.0;
    xs = 0.0;
    yr = 0.0;
    ys = 0.0;
    for (int b = 0; b < GeomNodes1D; ++b) {
        for (int a = 0; a < GeomNodes1D; ++a) {
            const int k = b * GeomNodes1D + a;
            const double dphir = dlr[a] * ls[b];
            const double dphis = lr[a] * dls[b];
            xr += S.geom(i, j, k, 0) * dphir;
            xs += S.geom(i, j, k, 0) * dphis;
            yr += S.geom(i, j, k, 1) * dphir;
            ys += S.geom(i, j, k, 1) * dphis;
        }
    }
    J = std::max(xr * ys - xs * yr, 1e-30);
}

static void build_mapped_geometry_cache(SolverVariable& S)
{
    S.metric_cache.reset(Nx, Ny, NumQuad, 4, FaceQuad);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int q = 0; q < NumQuad; ++q) {
                const double r = S.gl_p[q / Quad1D];
                const double s = S.gl_p[q % Quad1D];
                dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
                physical_coords(S, i, j, r, s, metric.x, metric.y);
                contravariant_metrics(S, i, j, r, s,
                    metric.xr, metric.xs, metric.yr, metric.ys,
                    metric.jacobian);
            }

            for (int f = 0; f < 4; ++f) {
                for (int g = 0; g < FaceQuad; ++g) {
                    double r = 0.0;
                    double s = 0.0;
                    if (f == 0) { r = S.gl_p[g]; s = -1.0; }
                    else if (f == 1) { r = 1.0; s = S.gl_p[g]; }
                    else if (f == 2) { r = S.gl_p[g]; s = 1.0; }
                    else { r = -1.0; s = S.gl_p[g]; }

                    dg::FaceQuadratureMetric& metric =
                        S.metric_cache.face(i, j, f, g);
                    physical_coords(S, i, j, r, s, metric.x, metric.y);
                    double xr = 0.0, xs = 0.0, yr = 0.0, ys = 0.0, J = 0.0;
                    contravariant_metrics(S, i, j, r, s,
                        xr, xs, yr, ys, J);
                    double nx_ds = 0.0;
                    double ny_ds = 0.0;
                    if (f == 0) {
                        nx_ds = yr; ny_ds = -xr;
                        metric.measure = std::hypot(xr, yr);
                    }
                    else if (f == 2) {
                        nx_ds = -yr; ny_ds = xr;
                        metric.measure = std::hypot(xr, yr);
                    }
                    else if (f == 1) {
                        nx_ds = ys; ny_ds = -xs;
                        metric.measure = std::hypot(xs, ys);
                    }
                    else {
                        nx_ds = -ys; ny_ds = xs;
                        metric.measure = std::hypot(xs, ys);
                    }
                    const double inverse_measure =
                        1.0 / std::max(metric.measure, 1e-30);
                    metric.nx = nx_ds * inverse_measure;
                    metric.ny = ny_ds * inverse_measure;
                }
            }
        }
    }

    std::cout << "[GEOM] cached volume/face metrics memory_MB="
        << (static_cast<double>(S.metric_cache.memory_bytes()) /
            (1024.0 * 1024.0)) << "\n";
}

inline double cell_jacobian_mean(const SolverVariable& S, int i, int j)
{
    double sum = 0.0;
    double wsum = 0.0;
    for (int q = 0; q < NumQuad; ++q) {
        const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
        sum += w * S.metric_cache.volume(i, j, q).jacobian;
        wsum += w;
    }
    return sum / std::max(wsum, 1e-30);
}

static void build_cell_mass_matrices(SolverVariable& S)
{
    S.inv_mass_cell.assign((size_t)NumCells * dimPk * dimPk, 0.0);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            double M[dimPk * dimPk] = { 0.0 };
            for (int q = 0; q < NumQuad; ++q) {
                const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
                const double Jq = S.metric_cache.volume(i, j, q).jacobian;
                for (int a = 0; a < dimPk; ++a) {
                    for (int b = 0; b < dimPk; ++b) {
                        M[a * dimPk + b] += w * S.phi_vol[q][a] * S.phi_vol[q][b] * Jq;
                    }
                }
            }
            double inv[dimPk * dimPk] = { 0.0 };
            invertMatrixDimPk(M, inv);
            const size_t cell = (size_t)i * (size_t)Ny + (size_t)j;
            for (int a = 0; a < dimPk; ++a) {
                for (int b = 0; b < dimPk; ++b) {
                    S.inv_mass_cell[(cell * dimPk + (size_t)a) * dimPk + (size_t)b] = inv[a * dimPk + b];
                }
            }
        }
    }
}

inline void face_metric(const SolverVariable& S, int i, int j, int f, int g,
    double& nx, double& ny, double& ds)
{
    const dg::FaceQuadratureMetric& metric = S.metric_cache.face(i, j, f, g);
    nx = metric.nx;
    ny = metric.ny;
    ds = metric.measure;
}

// Characteristic/open-boundary states; weak viscous wall trace with no-slip and nu_tilde = 0.
inline const dg::BoundaryCondition& boundary_condition(int f, double x)
{
    const bool on_solid_wall = (f == 0) && is_solid_wall_x(x);
    return g_boundary_conditions.resolve(f, on_solid_wall);
}

inline void boundary_state_2dzp(const double* UL, int f, double x, double y, double nx, double ny, double* UR) {
    (void)y;
    const PrimitiveState Vi = cons_to_primitive(UL);
    const dg::BoundaryCondition& condition = boundary_condition(f, x);
    if (condition.type == dg::BoundaryType::TotalInflow) {
        const PrimitiveState Vb = characteristic_total_inflow_boundary(Vi,
            condition.total_pressure, condition.total_temperature,
            condition.flow_angle, nx, ny);
        primitive_to_cons(Vb, UR);
    }
    else if (condition.type == dg::BoundaryType::PressureOutlet) {
        const PrimitiveState Vb = characteristic_pressure_outlet_boundary(Vi,
            condition.static_pressure, nx, ny);
        primitive_to_cons(Vb, UR);
    }
    else if (condition.type == dg::BoundaryType::Farfield) {
        const PrimitiveState Vb = characteristic_farfield_boundary(Vi, freestream_primitive(), nx, ny);
        primitive_to_cons(Vb, UR);
    }
    else if (dg::is_wall(condition.type)) {
        double un = 0.0, ut = 0.0;
        velocity_to_normal_tangent(Vi.u, Vi.v, nx, ny, un, ut);
        PrimitiveState Vr = Vi;
        normal_tangent_to_velocity(-un, ut, nx, ny, Vr.u, Vr.v);
        primitive_to_cons(Vr, UR);
    }
}

inline bool is_plate_wall_face(int f, double x)
{
    return dg::has_strict_viscous_trace(boundary_condition(f, x).type);
}

inline bool is_symmetry_slip_face(int f, double x)
{
    return boundary_condition(f, x).type == dg::BoundaryType::SlipWall;
}

inline bool is_zero_viscous_flux_face(int f, double x)
{
    return dg::has_zero_viscous_flux(boundary_condition(f, x).type);
}

inline void wall_viscous_dirichlet_state(const double* UL, double* UB)
{



    const double rho = std::max(UL[0], 1e-12);
    const double u = UL[1] / rho;
    const double v = UL[2] / rho;
    const double p = std::max(gamma1 * (UL[3] - 0.5 * rho * (u * u + v * v)), 1e-12);
    const double T = p / rho;
    const double rho_w = rho;
    const double p_w = std::max(rho_w * T, 1e-12);
    UB[0] = rho_w;
    UB[1] = 0.0;
    UB[2] = 0.0;
    UB[3] = p_w / gamma1;
    UB[4] = 0.0;
}

inline bool boundary_viscous_trace_2dzp(const double* UL, int f, double x, double y, double nx, double ny, double* UB)
{
    if (is_plate_wall_face(f, x)) {
        (void)y;
        wall_viscous_dirichlet_state(UL, UB);
        return true;
    }
    if (is_symmetry_slip_face(f, x)) {
        (void)y;



        dg::reflect_normal(UL, nx, ny, UB);
        return false;
    }
    (void)y;


    for (int e = 0; e < NumEq; ++e) UB[e] = UL[e];
    return false;
}


// Read the NASA/TMR CGNS grid, precompute DG basis data, and initialize freestream.
void init_solver(SolverVariable& S) {
    S.K.reset(Nx, Ny, dimPk, NumEq); S.rhs.reset(Nx, Ny, dimPk, NumEq);
    S.gradUX.reset(Nx, Ny, dimPk, NumEq); S.gradUY.reset(Nx, Ny, dimPk, NumEq);

    load_structured_adf_cgns_grid(S,
#if defined(DG_CASE_NACA0012) && !defined(DG_NACA_OFFICIAL_CGNS)
        GMSH_GRID_FILE
#else
        CGNS_GRID_FILE
#endif
    );
    (void)wall_distance(PLATE_XMID, flat_plate_wall_y(PLATE_XMID) + SA_D_FLOOR);
    const double nu_inf = mu_ref / rho_inf;
    const int i_mid = (int)(std::min_element(S.x_edges, S.x_edges + Nx + 1,
        [](double a, double b) { return std::abs(a - PLATE_XMID) < std::abs(b - PLATE_XMID); }) - S.x_edges);
    const double dy1_plate = std::abs(grid_vertex_y(S, std::max(0, std::min(Nx, i_mid)), 1)
        - grid_vertex_y(S, std::max(0, std::min(Nx, i_mid)), 0));
    const double dy1_far = std::abs(grid_vertex_y(S, 0, 1) - grid_vertex_y(S, 0, 0));
    const double yplus_plate = 0.5 * dy1_plate * UTAU_EST / std::max(nu_inf, 1e-30);
    const double yplus_far = 0.5 * dy1_far * UTAU_EST / std::max(nu_inf, 1e-30);
    double dx_min = 1e300, dx_max = 0.0;
    for (int i = 0; i < Nx; ++i) {
        const double dxi = S.x_edges[i + 1] - S.x_edges[i];
        dx_min = std::min(dx_min, dxi);
        dx_max = std::max(dx_max, dxi);
    }
    int i_le = 0;
    double xbest = std::abs(S.x_edges[0] - PLATE_X0);
    for (int i = 1; i <= Nx; ++i) {
        const double d = std::abs(S.x_edges[i] - PLATE_X0);
        if (d < xbest) { xbest = d; i_le = i; }
    }
    std::cout << "Mesh info: file=\"" << COMPILED_MESH_FILE << "\""
        << "  dy1_plate=" << dy1_plate
        << "  dy1_far=" << dy1_far
        << "  yplus_plate(first-center, est)=" << yplus_plate
        << "  yplus_far(first-center, est)=" << yplus_far
        << "  dx_min=" << dx_min
        << "  dx_max=" << dx_max
        << "  plate_start_edge_index=" << i_le << std::endl;

    gauss_points_1d(S.gl_p, S.gl_w);
    for (int q = 0; q < NumQuad; q++) {
        double r = S.gl_p[q / Quad1D], s = S.gl_p[q % Quad1D];
        eval_tensor_basis(r, s, S.phi_vol[q], S.dphi_dr[q], S.dphi_ds[q]);
    }
    for (int f = 0; f < 4; f++) for (int g = 0; g < FaceQuad; g++) {
        double r, s; if (f == 0) { r = S.gl_p[g]; s = -1.0; }
        else if (f == 1) { r = 1.0; s = S.gl_p[g]; }
        else if (f == 2) { r = S.gl_p[g]; s = 1.0; }
        else { r = -1.0; s = S.gl_p[g]; }
        eval_tensor_basis(r, s, S.phi_face[f][g]);
    }
    build_mapped_geometry_cache(S);
    double M[dimPk * dimPk] = { 0 };
    for (int q = 0; q < NumQuad; q++) {
        double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
        for (int i = 0; i < dimPk; i++) for (int j = 0; j < dimPk; j++) M[i * dimPk + j] += w * S.phi_vol[q][i] * S.phi_vol[q][j];
    }
    invertMatrixDimPk(M, &S.inv_mass[0][0]);
    for (int i = 0; i < dimPk; i++) { double rowsum = 0; for (int j = 0; j < dimPk; j++) rowsum += M[i * dimPk + j]; S.mean_wts[i] = rowsum / 4.0; }
    build_cell_mass_matrices(S);

    for (int i = 0; i < Nx; i++) for (int j = 0; j < Ny; j++) {
        for (int k = 0; k < dimPk; k++) {
            double rnode = 0.0, snode = 0.0;
            reference_node_coords(k, rnode, snode);
            double px = 0.0, py = 0.0;
            physical_coords(S, i, j, rnode, snode, px, py);
            GetInitial(px, py, &S.K(i, j, k, 0));
        }
    }
}




// Viscous and Spalart-Allmaras flux/source routines. Euler HLLC is in
// dg/physics.hpp so the stateless kernel can be tested independently.
inline double laminar_mu_from_cons(const double* U)
{
    return dg::laminar_viscosity<double, dg::DoubleMath>(
        U, g_viscous_sa_parameters);
}

inline double sa_eddy_mu_from_rho_nut(double rho, double nut, double mu_lam)
{
    return dg::sa_eddy_viscosity<double, dg::DoubleMath>(
        rho, nut, mu_lam, g_viscous_sa_parameters);
}


inline double wall_distance(double x, double y) {
    return g_wall_geometry.distance(x, y, SA_D_FLOOR);
}

static void write_boundary_junction_diagnostic(const SolverVariable& S,
    const std::string& path)
{
    dg::io::TextOutputFile output;
    output.open(path);
    output << "# bottom-boundary classification audit\n"
        << "# wall_cell_index_range " << g_wall_i_begin << " "
        << g_wall_i_end << " (half-open)\n"
        << "# cell_i face_gauss x y index_wall coordinate_wall match "
        << "wall_distance\n";

    int samples = 0;
    int mismatches = 0;
    for (int i = 0; i < Nx; ++i) {
        const bool index_wall = g_wall_i_begin >= 0
            && i >= g_wall_i_begin && i < g_wall_i_end;
        for (int g = 0; g < FaceQuad; ++g) {
            double x = 0.0, y = 0.0;
            face_point_coords(S, i, 0, 0, g, x, y);
            const bool coordinate_wall = is_plate_wall_face(0, x);
            const bool match = g_wall_i_begin < 0
                || coordinate_wall == index_wall;
            output << i << " " << g << " " << std::setprecision(16)
                << x << " " << y << " " << (index_wall ? 1 : 0)
                << " " << (coordinate_wall ? 1 : 0)
                << " " << (match ? 1 : 0)
                << " " << wall_distance(x, y) << "\n";
            ++samples;
            if (!match) ++mismatches;
        }
    }
    output.close();
    std::cout << "[BOUNDARY AUDIT] bottom_face_samples=" << samples
        << " classification_mismatches=" << mismatches
        << " wall_cell_range=[" << g_wall_i_begin << ","
        << g_wall_i_end << ")"
        << (g_wall_i_begin < 0 ? " metadata_unavailable" : "")
        << std::endl;
}

inline double sa_destruction_distance(double d_raw) {
    return std::max(d_raw, 1e-12);
}


inline void Viscous_Flux(const double* U, const double* dUx, const double* dUy, double* Fv, double* Gv) {
    dg::viscous_flux<double, dg::DoubleMath>(
        U, dUx, dUy, g_viscous_sa_parameters, Fv, Gv);
}


static inline double sa_source_ramp(double x)
{
    (void)x;
    return 1.0;
}

inline dg::SASourceTerms<double> SA_Source_Terms(const double* U,
    const double* dUx, const double* dUy, double x, double y) {
    const double ramp = sa_source_ramp(x);
    if (ramp <= 0.0) return {};
    const double d = wall_distance(x, y);
    const double d_destroy = sa_destruction_distance(d);
    dg::SASourceTerms<double> terms =
        dg::sa_source_terms<double, dg::DoubleMath>(U, dUx, dUy,
            d, d_destroy, g_viscous_sa_parameters);
    const double total = ramp * terms.total;
    terms.production *= ramp;
    terms.destruction *= ramp;
    terms.cross_diffusion *= ramp;
    // Preserve the exact arithmetic used by the pre-diagnostic residual path.
    // The separately scaled fields are observational components only.
    terms.total = total;
    return terms;
}

inline double SA_Source(const double* U, const double* dUx, const double* dUy,
    double x, double y) {
    return SA_Source_Terms(U, dUx, dUy, x, y).total;
}

static inline double sa_source_diag_stiffness_point(const double* U, const double* dUx, const double* dUy,
    double x, double y)
{
    (void)y;
    const double ramp = sa_source_ramp(x);
    if (ramp <= 0.0) return 0.0;
    const double r = std::max(U[0], 1e-12);
    const double inv_r = 1.0 / r;
    const double u = U[1] * inv_r;
    const double v = U[2] * inv_r;
    const double ury = (dUy[1] - u * dUy[0]) * inv_r;
    const double vrx = (dUx[2] - v * dUx[0]) * inv_r;
    const double vort = std::abs(vrx - ury);
    const double production_rate = SA_CB1 * vort;
    const double stiff_rate = production_rate;
    const double lambda = ramp * SA_SOURCE_DIAG_FACTOR * stiff_rate;
    if (!std::isfinite(lambda)) return SA_SOURCE_DIAG_MAX;
    return std::min(std::max(lambda, 0.0), SA_SOURCE_DIAG_MAX);
}

static inline double cell_sa_source_diag_stiffness(const SolverVariable& S,
    const Array4D<double>& K,
    int i,
    int j)
{
    double lambda = 0.0;
    for (int q = 0; q < NumQuad; ++q) {
        double Uq[NumEq] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double dUxq[NumEq] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double dUyq[NumEq] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        for (int k = 0; k < dimPk; ++k) {
            const double ph = S.phi_vol[q][k];
            for (int e = 0; e < NumEq; ++e) {
                Uq[e] += K(i, j, k, e) * ph;
                dUxq[e] += S.gradUX(i, j, k, e) * ph;
                dUyq[e] += S.gradUY(i, j, k, e) * ph;
            }
        }
        double x = 0.0, y = 0.0;
        const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
        x = metric.x;
        y = metric.y;
        lambda = std::max(lambda, sa_source_diag_stiffness_point(Uq, dUxq, dUyq, x, y));
    }
    return lambda;
}


// BR2 projected gradients and face-lifting operators.
void computeFaceLifting(const SolverVariable& S, const double* jump, int f, int i_idx, int j_idx, double* rX, double* rY, double scale) {
    double bX[dimPk][NumEq] = { 0 };
    double bY[dimPk][NumEq] = { 0 };
    for (int g = 0; g < FaceQuad; g++) {
        double nx = 0.0, ny = 0.0, ds = 0.0;
        face_metric(S, i_idx, j_idx, f, g, nx, ny, ds);
        for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
            const double face_rhs = S.gl_w[g] * jump[g * NumEq + e] * S.phi_face[f][g][k] * ds;
            bX[k][e] += face_rhs * nx;
            bY[k][e] += face_rhs * ny;
        }
    }
    for (int e = 0; e < NumEq; e++) for (int k = 0; k < dimPk; k++) {
        double accX = 0.0, accY = 0.0;
        for (int m = 0; m < dimPk; m++) {
            const double Minv = cell_inv_mass_entry(S, i_idx, j_idx, k, m);
            accX += Minv * bX[m][e];
            accY += Minv * bY[m][e];
        }
        rX[k * NumEq + e] = accX * scale;
        rY[k * NumEq + e] = accY * scale;
    }
}


void computeGradients(SolverVariable& S, const Array4D<double>& Kin) {
    std::fill(S.gradUX.data.begin(), S.gradUX.data.end(), 0.0);
    std::fill(S.gradUY.data.begin(), S.gradUY.data.end(), 0.0);
#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int cell = 0; cell < NumCells; ++cell) {
        const int i = cell / Ny;
        const int j = cell % Ny;
        for (int q = 0; q < NumQuad; q++) {
            double uq[NumEq] = { 0 }; for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) uq[e] += Kin(i, j, k, e) * S.phi_vol[q][k];
            double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
            const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
            const double xr = metric.xr;
            const double xs = metric.xs;
            const double yr = metric.yr;
            const double ys = metric.ys;
            for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                const double grad_phi_x_J = S.dphi_dr[q][k] * ys - S.dphi_ds[q][k] * yr;
                const double grad_phi_y_J = -S.dphi_dr[q][k] * xs + S.dphi_ds[q][k] * xr;
                S.gradUX(i, j, k, e) -= w * uq[e] * grad_phi_x_J;
                S.gradUY(i, j, k, e) -= w * uq[e] * grad_phi_y_J;
            }
        }
        for (int f = 0; f < 4; f++) {
            const auto neighbor = g_topology.face_neighbor(i, j, f);
            for (int g = 0; g < FaceQuad; g++) {
                double nx = 0.0, ny = 0.0, ds = 0.0;
                face_metric(S, i, j, f, g, nx, ny, ds);
                double uL[NumEq] = { 0 }, uR[NumEq] = { 0 };
                bool strict_wall_viscous = false;
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) uL[e] += Kin(i, j, k, e) * S.phi_face[f][g][k];
                const int ni = neighbor.i;
                const int nj = neighbor.j;
                const bool isInteriorFace = neighbor.interior;
                if (isInteriorFace) {
                    const int nf = neighbor.face;
                    const int ng = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
                    for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++)
                        uR[e] += Kin(ni, nj, k, e) * S.phi_face[nf][ng][k];
                }
                else {
                    double xFace, yFace;
                    face_point_coords(S, i, j, f, g, xFace, yFace);
                    strict_wall_viscous = boundary_viscous_trace_2dzp(uL, f, xFace, yFace, nx, ny, uR);
                }
                double u_avg[NumEq];
                for (int e = 0; e < NumEq; e++) {
                    if (!isInteriorFace && strict_wall_viscous) {
                        u_avg[e] = uR[e];
                    }
                    else {
                        u_avg[e] = 0.5 * (uL[e] + uR[e]);
                    }
                }
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                    S.gradUX(i, j, k, e) += S.gl_w[g] * u_avg[e] * nx * S.phi_face[f][g][k] * ds;
                    S.gradUY(i, j, k, e) += S.gl_w[g] * u_avg[e] * ny * S.phi_face[f][g][k] * ds;
                }
            }
        }
        for (int e = 0; e < NumEq; e++) {
            double bx[dimPk], by[dimPk]; for (int k = 0; k < dimPk; k++) { bx[k] = S.gradUX(i, j, k, e); by[k] = S.gradUY(i, j, k, e); }
            for (int k = 0; k < dimPk; k++) {
                double ax = 0.0, ay = 0.0;
                for (int m = 0; m < dimPk; m++) {
                    const double Minv = cell_inv_mass_entry(S, i, j, k, m);
                    ax += Minv * bx[m];
                    ay += Minv * by[m];
                }
                S.gradUX(i, j, k, e) = ax;
                S.gradUY(i, j, k, e) = ay;
            }
        }
    }
}
struct ResidualStats {
    std::array<double, NumEq> l2 = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::array<double, NumEq> linf = { 0.0, 0.0, 0.0, 0.0, 0.0 };
};

static inline std::array<double, NumEq> build_residual_scales(bool use_sa_bl_scale) {
    const double Lref = std::max(PLATE_X1 - PLATE_X0, 1.0);
    const double uref = std::max(u_inf, a_inf);
    const double Einf = p_inf / gamma1 + 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    const double mass_scale = rho_inf * uref / Lref;
    const double mom_scale = (rho_inf * uref * uref + p_inf) / Lref;
    const double energy_scale = ((Einf + p_inf) * uref) / Lref;
    const double Cf_ref = 0.074 * std::pow(std::max(Re, 1.0), -0.2);
    const double utau_ref = std::sqrt(std::max(Cf_ref * 0.5, 0.0)) * u_inf;
    const double delta_ref = 0.37 * Lref * std::pow(std::max(Re, 1.0), -0.2);
    const double nut_bl_ref = SA_KAPPA * utau_ref * delta_ref;
    const double nut_scale = use_sa_bl_scale ? std::max(nut_bl_ref, SA_INIT_NUT) : SA_INIT_NUT;
    const double q_scale = std::max(rho_inf * nut_scale * uref / Lref, 1e-20);
    return { mass_scale, mom_scale, mom_scale, energy_scale, q_scale };
}

static inline std::array<double, NumEq> build_residual_scales_bl() {
    return build_residual_scales(true);
}

static inline std::array<double, NumEq> build_residual_scales_init() {
    return build_residual_scales(false);
}

static inline ResidualStats compute_array_residual_stats(const SolverVariable& S, const Array4D<double>& R) {
    ResidualStats rs;
    double total_area = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const double Jbar = cell_jacobian_mean(S, i, j);
            total_area += 4.0 * Jbar;
            for (int k = 0; k < dimPk; ++k) {
                for (int e = 0; e < NumEq; ++e) {
                    const double rk = R(i, j, k, e);
                    if (!std::isfinite(rk)) {
                        rs.l2[e] = std::numeric_limits<double>::infinity();
                        rs.linf[e] = std::numeric_limits<double>::infinity();
                    }
                    else {
                        rs.linf[e] = std::max(rs.linf[e], std::abs(rk));
                    }
                }
            }
            for (int q = 0; q < NumQuad; ++q) {
                const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
                double rq[NumEq] = { 0.0 };
                for (int k = 0; k < dimPk; ++k) {
                    const double ph = S.phi_vol[q][k];
                    for (int e = 0; e < NumEq; ++e) rq[e] += R(i, j, k, e) * ph;
                }
                for (int e = 0; e < NumEq; ++e) {
                    if (!std::isfinite(rq[e])) {
                        rs.l2[e] = std::numeric_limits<double>::infinity();
                        rs.linf[e] = std::numeric_limits<double>::infinity();
                    }
                    else if (std::isfinite(rs.l2[e])) {
                        const double Jq = S.metric_cache.volume(i, j, q).jacobian;
                        rs.l2[e] += w * rq[e] * rq[e] * Jq;
                        rs.linf[e] = std::max(rs.linf[e], std::abs(rq[e]));
                    }
                }
            }
        }
    }
    total_area = g_collectives.sum(total_area);
    const double inv_area = 1.0 / std::max(total_area, 1e-300);
    for (int e = 0; e < NumEq; ++e) {
        rs.l2[e] = g_collectives.sum(rs.l2[e]);
        rs.linf[e] = g_collectives.maximum(rs.linf[e]);
        if (std::isfinite(rs.l2[e])) rs.l2[e] = std::sqrt(rs.l2[e] * inv_area);
    }
    return rs;
}

static inline ResidualStats compute_rhs_residual_stats(const SolverVariable& S) {
    return compute_array_residual_stats(S, S.rhs);
}

struct WallOutputMonitor {
    std::array<double, 3> cf = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    };
    double mean_cf = std::numeric_limits<double>::quiet_NaN();
};

static WallOutputMonitor compute_wall_output_monitor(const SolverVariable& S,
    const Array4D<double>& K)
{
    WallOutputMonitor monitor;
    const double qinf = 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    for (std::size_t sample = 0; sample < CF_MONITOR_X.size(); ++sample) {
        const double target_x = CF_MONITOR_X[sample];
#if defined(DG_CASE_NACA0012)
        double cf_sum = 0.0;
        int cf_count = 0;
        const int leading_edge_index =
            g_wall_i_begin + (g_wall_i_end - g_wall_i_begin) / 2;
        for (int surface = 0; surface < 2; ++surface) {
            const int begin = surface == 0
                ? g_wall_i_begin : leading_edge_index;
            const int end = surface == 0
                ? leading_edge_index : g_wall_i_end;
            int best_i = begin;
            double best_r = 0.0;
            double best_distance = std::numeric_limits<double>::infinity();
            for (int i = begin; i < end; ++i) {
                for (int g = 0; g < FaceQuad; ++g) {
                    double x = 0.0, y = 0.0;
                    physical_coords(S, i, 0, S.gl_p[g], -1.0, x, y);
                    const double distance = std::abs(x - target_x);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_i = i;
                        best_r = S.gl_p[g];
                    }
                }
            }
            const PlateWallFaceMetrics wall =
                plate_wall_point_metrics(S, K, best_i, best_r);
            if (wall.valid) {
                double oriented_tau = wall.tau_w;
                if (wall.tx * u_inf + wall.ty * v_inf < 0.0) {
                    oriented_tau = -oriented_tau;
                }
                cf_sum += oriented_tau / std::max(qinf, 1e-30);
                ++cf_count;
            }
        }
        if (cf_count > 0) monitor.cf[sample] = cf_sum / cf_count;
#else
        int cell_i = 0;
        double best_distance = std::numeric_limits<double>::infinity();
        for (int i = 0; i < Nx; ++i) {
            const double distance = target_x < S.x_edges[i]
                ? S.x_edges[i] - target_x
                : (target_x > S.x_edges[i + 1]
                    ? target_x - S.x_edges[i + 1] : 0.0);
            if (distance < best_distance) {
                best_distance = distance;
                cell_i = i;
            }
        }
        const double r = std::max(-1.0, std::min(1.0,
            (target_x - S.x_centers[cell_i]) /
                std::max(S.h_x[cell_i], 1e-30)));
        const PlateWallFaceMetrics wall =
            plate_wall_point_metrics(S, K, cell_i, r);
        if (wall.valid) {
            monitor.cf[sample] = wall.tau_w / std::max(qinf, 1e-30);
        }
#endif
    }

    double shear_integral = 0.0;
    double wall_measure = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int g = 0; g < FaceQuad; ++g) {
            const PlateWallFaceMetrics wall =
                plate_wall_point_metrics(S, K, i, S.gl_p[g]);
            if (!wall.valid) continue;
            const double weight = S.gl_w[g] * wall.ds_metric;
            double oriented_tau = wall.tau_w;
#if defined(DG_CASE_NACA0012)
            if (wall.tx * u_inf + wall.ty * v_inf < 0.0) {
                oriented_tau = -oriented_tau;
            }
#endif
            shear_integral += weight * oriented_tau;
            wall_measure += weight;
        }
    }
    if (wall_measure > 0.0) {
        monitor.mean_cf = shear_integral /
            (std::max(qinf, 1e-30) * wall_measure);
    }
    return monitor;
}

static double wall_output_relative_change(const WallOutputMonitor& current,
    const WallOutputMonitor& previous)
{
    double difference_squared = 0.0;
    double current_squared = 0.0;
    for (std::size_t sample = 0; sample < current.cf.size(); ++sample) {
        if (!std::isfinite(current.cf[sample])
            || !std::isfinite(previous.cf[sample])) continue;
        const double difference = current.cf[sample] - previous.cf[sample];
        difference_squared += difference * difference;
        current_squared += current.cf[sample] * current.cf[sample];
    }
    if (std::isfinite(current.mean_cf) && std::isfinite(previous.mean_cf)) {
        const double difference = current.mean_cf - previous.mean_cf;
        difference_squared += difference * difference;
        current_squared += current.mean_cf * current.mean_cf;
    }
    if (current_squared <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(difference_squared / current_squared);
}

static dg::io::SAResidualDiagnosticRecord compute_sa_residual_diagnostic(
    const SolverVariable& S,
    const Array4D<double>& K,
    int local_step,
    long long total_step,
    double pseudo_time,
    double solver_cpu_seconds,
    double cfl,
    double global_l2_scaled,
    double sa_residual_scale)
{
    dg::io::SAResidualDiagnosticRecord record;
    record.local_step = local_step;
    record.total_step = total_step;
    record.pseudo_time = pseudo_time;
    record.solver_cpu_seconds = solver_cpu_seconds;
    record.cfl = cfl;
    record.global_l2_scaled = global_l2_scaled;

    double upstream_square = 0.0, upstream_area = 0.0;
    double wall_square = 0.0, wall_area = 0.0;
    double downstream_square = 0.0, downstream_area = 0.0;
    double airfoil_band_square = 0.0, airfoil_band_area = 0.0;
    double wake_band_square = 0.0, wake_band_area = 0.0;
    double wake_pair_common_square = 0.0;
    double wake_pair_antisymmetric_square = 0.0;
    double wake_pair_measure = 0.0;
    double near_square = 0.0, near_area = 0.0;
    double outer_square = 0.0, outer_area = 0.0;
    double wall_begin_outside_square = 0.0, wall_begin_outside_area = 0.0;
    double wall_begin_inside_square = 0.0, wall_begin_inside_area = 0.0;
    double wall_end_inside_square = 0.0, wall_end_inside_area = 0.0;
    double wall_end_outside_square = 0.0, wall_end_outside_area = 0.0;
    double wall_core_band_square = 0.0, wall_core_band_area = 0.0;
    double maximum_cell_l2 = -1.0;
    const double scale = std::max(sa_residual_scale, 1e-30);
    constexpr int junction_band_rows = 4;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            double cell_square = 0.0;
            double cell_area = 0.0;
            for (int q = 0; q < NumQuad; ++q) {
                double residual_sa = 0.0;
                double state[NumEq] = {};
                for (int k = 0; k < dimPk; ++k) {
                    const double basis = S.phi_vol[q][k];
                    residual_sa += S.rhs(i, j, k, 4) * basis;
                    for (int e = 0; e < NumEq; ++e) {
                        state[e] += K(i, j, k, e) * basis;
                    }
                }
                const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
                const double weight = S.gl_w[q / Quad1D]
                    * S.gl_w[q % Quad1D] * metric.jacobian;
                const double square = residual_sa * residual_sa * weight;
                cell_square += square;
                cell_area += weight;

                const double distance = wall_distance(metric.x, metric.y);
                if (metric.x < PLATE_X0) {
                    upstream_square += square;
                    upstream_area += weight;
                }
                else if (metric.x <= PLATE_X1) {
                    wall_square += square;
                    wall_area += weight;
                }
                else {
                    downstream_square += square;
                    downstream_area += weight;
                }
                if (j < junction_band_rows && g_wall_i_begin >= 0
                    && g_wall_i_end > g_wall_i_begin) {
                    if (i == g_wall_i_begin - 1) {
                        wall_begin_outside_square += square;
                        wall_begin_outside_area += weight;
                    }
                    else if (i == g_wall_i_begin) {
                        wall_begin_inside_square += square;
                        wall_begin_inside_area += weight;
                    }
                    else if (i == g_wall_i_end - 1) {
                        wall_end_inside_square += square;
                        wall_end_inside_area += weight;
                    }
                    else if (i == g_wall_i_end) {
                        wall_end_outside_square += square;
                        wall_end_outside_area += weight;
                    }
                    else if (i > g_wall_i_begin && i < g_wall_i_end - 1) {
                        wall_core_band_square += square;
                        wall_core_band_area += weight;
                    }
                }
#if defined(DG_CASE_NACA0012)
                constexpr int wake_cells_per_side =
                    dg::case_config::wake_pairs;
                constexpr int airfoil_cell_begin = wake_cells_per_side;
                constexpr int airfoil_cell_end = Nx - wake_cells_per_side;
                if (j <= 3 && i >= airfoil_cell_begin
                    && i < airfoil_cell_end) {
                    airfoil_band_square += square;
                    airfoil_band_area += weight;
                }
                if (j <= 3 && (i < airfoil_cell_begin
                    || i >= airfoil_cell_end)) {
                    wake_band_square += square;
                    wake_band_area += weight;
                }
#endif
                if (distance <= SA_DIAGNOSTIC_NEAR_WALL_DISTANCE) {
                    near_square += square;
                    near_area += weight;
                }
                else {
                    outer_square += square;
                    outer_area += weight;
                }

                const double magnitude = std::abs(residual_sa);
                if (magnitude > record.maximum_abs) {
                    const double rho = std::max(state[0], 1e-12);
                    const double nu_tilde = state[4] / rho;
                    const double mu_lam = laminar_mu_from_cons(state);
                    const double nu_lam = mu_lam / rho;
                    record.maximum_abs = magnitude;
                    record.maximum_scaled = magnitude / scale;
                    record.maximum_i = i;
                    record.maximum_j = j;
                    record.maximum_quadrature = q;
                    record.maximum_x = metric.x;
                    record.maximum_y = metric.y;
                    record.maximum_wall_distance = distance;
                    record.maximum_rho = rho;
                    record.maximum_nu_tilde = nu_tilde;
                    record.maximum_chi = nu_tilde / std::max(nu_lam, 1e-30);
                    record.maximum_mut_over_mu_inf =
                        turbulent_mu_from_U(state) / std::max(mu_ref, 1e-30);
                }
            }
            const double cell_l2_scaled = std::sqrt(cell_square /
                std::max(cell_area, 1e-300)) / scale;
            if (cell_l2_scaled > maximum_cell_l2) {
                maximum_cell_l2 = cell_l2_scaled;
                record.maximum_cell_i = i;
                record.maximum_cell_j = j;
                record.maximum_cell_l2_scaled = cell_l2_scaled;
            }
        }
    }

#if defined(DG_CASE_NACA0012)
    // Resolve the SA residual on the two coincident sides of the C-grid cut
    // into a common component and an antisymmetric seam component.  Iterate
    // only one side of every pair to avoid double counting.
    for (int i = 0; i < dg::case_config::wake_pairs; ++i) {
        const auto neighbor = g_topology.face_neighbor(i, 0, 0);
        if (!neighbor.interior) continue;
        for (int g = 0; g < FaceQuad; ++g) {
            const int neighbor_g = neighbor.reverse_quadrature
                ? FaceQuad - 1 - g : g;
            double residual_left = 0.0;
            double residual_right = 0.0;
            for (int k = 0; k < dimPk; ++k) {
                residual_left += S.rhs(i, 0, k, 4)
                    * S.phi_face[0][g][k];
                residual_right += S.rhs(neighbor.i, neighbor.j, k, 4)
                    * S.phi_face[neighbor.face][neighbor_g][k];
            }
            double nx = 0.0, ny = 0.0, measure = 0.0;
            face_metric(S, i, 0, 0, g, nx, ny, measure);
            const double weight = S.gl_w[g] * measure;
            const double common = 0.5 * (residual_left + residual_right);
            const double antisymmetric = 0.5
                * (residual_left - residual_right);
            wake_pair_common_square += common * common * weight;
            wake_pair_antisymmetric_square +=
                antisymmetric * antisymmetric * weight;
            wake_pair_measure += weight;
        }
    }
#endif

    auto scaled_region_l2 = [scale](double square, double area) {
        if (area <= 0.0) return 0.0;
        return std::sqrt(square / area) / scale;
    };
    record.upstream_l2_scaled =
        scaled_region_l2(upstream_square, upstream_area);
    record.wall_patch_l2_scaled = scaled_region_l2(wall_square, wall_area);
    record.downstream_l2_scaled =
        scaled_region_l2(downstream_square, downstream_area);
    record.airfoil_band_l2_scaled =
        scaled_region_l2(airfoil_band_square, airfoil_band_area);
    record.wake_band_l2_scaled =
        scaled_region_l2(wake_band_square, wake_band_area);
    record.wake_pair_common_l2_scaled =
        scaled_region_l2(wake_pair_common_square, wake_pair_measure);
    record.wake_pair_antisymmetric_l2_scaled =
        scaled_region_l2(wake_pair_antisymmetric_square,
            wake_pair_measure);
    record.near_wall_l2_scaled = scaled_region_l2(near_square, near_area);
    record.outer_l2_scaled = scaled_region_l2(outer_square, outer_area);
    record.wall_begin_outside_l2_scaled = scaled_region_l2(
        wall_begin_outside_square, wall_begin_outside_area);
    record.wall_begin_inside_l2_scaled = scaled_region_l2(
        wall_begin_inside_square, wall_begin_inside_area);
    record.wall_end_inside_l2_scaled = scaled_region_l2(
        wall_end_inside_square, wall_end_inside_area);
    record.wall_end_outside_l2_scaled = scaled_region_l2(
        wall_end_outside_square, wall_end_outside_area);
    record.wall_core_band_l2_scaled = scaled_region_l2(
        wall_core_band_square, wall_core_band_area);
    return record;
}

static inline double compute_density_update_l2(const SolverVariable& S,
    const Array4D<double>& Knew,
    const Array4D<double>& Kold)
{
    double sum = 0.0;
    double total_area = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const double Jbar = cell_jacobian_mean(S, i, j);
            total_area += 4.0 * Jbar;
            for (int q = 0; q < NumQuad; ++q) {
                const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
                double drho = 0.0;
                for (int k = 0; k < dimPk; ++k) {
                    drho += (Knew(i, j, k, 0) - Kold(i, j, k, 0)) * S.phi_vol[q][k];
                }
                if (!std::isfinite(drho)) return std::numeric_limits<double>::infinity();
                const double Jq = S.metric_cache.volume(i, j, q).jacobian;
                sum += w * drho * drho * Jq;
            }
        }
    }
    sum = g_collectives.sum(sum);
    total_area = g_collectives.sum(total_area);
    return std::sqrt(sum / std::max(total_area, 1e-300));
}

static inline double max_scaled_l2(const ResidualStats& rs, const std::array<double, NumEq>& scales) {
    double m = 0.0;
    for (int e = 0; e < NumEq; ++e) m = std::max(m, rs.l2[e] / std::max(scales[e], 1e-30));
    return m;
}

static inline int cell_id(int i, int j) { return g_topology.cell_id(i, j); }
static inline int cell_i_from_id(int c) { return g_topology.cell_i(c); }
static inline int cell_j_from_id(int c) { return g_topology.cell_j(c); }

static inline int local_dof(int k, int e) { return k * NumEq + e; }
static inline void dof_from_local(int p, int& k, int& e) { k = p / NumEq; e = p % NumEq; }

static inline double solution_scale(int e) {
    const double Einf = p_inf / gamma1 + 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    if (e == 0) return std::max(rho_inf, 1e-30);
    if (e == 1 || e == 2) return std::max(rho_inf * std::max(u_inf, a_inf), 1e-30);
    if (e == 3) return std::max(Einf, 1e-30);
    return std::max(rho_inf * SA_INIT_NUT, 1e-30);
}

static inline std::array<double, NumEq> build_solution_scales() {
    return { solution_scale(0), solution_scale(1), solution_scale(2), solution_scale(3), solution_scale(4) };
}

static inline void array_to_vector_inplace(const Array4D<double>& K, std::vector<double>& v) {
    if ((int)v.size() != NumDofs) v.resize(NumDofs);
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        const int base = cell_id(i, j) * BlockSize;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            v[base + local_dof(k, e)] = K(i, j, k, e);
        }
    }
}

static inline double dot_vec(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return g_collectives.sum(s);
}

static inline double norm_vec(const std::vector<double>& a) {
    return std::sqrt(std::max(dot_vec(a, a), 0.0));
}

static inline void axpy_vec(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
}

static bool solve_small_dense_system(std::vector<double> A,
    std::vector<double> b,
    std::vector<double>& x)
{
    const int n = (int)b.size();
    if (n <= 0 || (int)A.size() != n * n) return false;
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double pivot_abs = std::abs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            const double cand = std::abs(A[i * n + k]);
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = i;
            }
        }
        if (!std::isfinite(pivot_abs) || pivot_abs < 1e-300) return false;
        if (pivot != k) {
            for (int j = k; j < n; ++j) std::swap(A[k * n + j], A[pivot * n + j]);
            std::swap(b[k], b[pivot]);
        }
        const double diag = A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            const double factor = A[i * n + k] / diag;
            A[i * n + k] = 0.0;
            for (int j = k + 1; j < n; ++j) A[i * n + j] -= factor * A[k * n + j];
            b[i] -= factor * b[k];
        }
    }

    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double rhs = b[i];
        for (int j = i + 1; j < n; ++j) rhs -= A[i * n + j] * x[j];
        const double diag = A[i * n + i];
        if (!std::isfinite(diag) || std::abs(diag) < 1e-300) return false;
        x[i] = rhs / diag;
        if (!std::isfinite(x[i])) return false;
    }
    return true;
}

static inline double scaled_l2_from_array(const SolverVariable& S,
    const Array4D<double>& R,
    const std::array<double, NumEq>& scales)
{
    const ResidualStats rs = compute_array_residual_stats(S, R);
    return max_scaled_l2(rs, scales);
}

static inline double relative_array_change(const Array4D<double>& a, const Array4D<double>& b)
{
    double diff2 = 0.0;
    double base2 = 0.0;
    const size_t n = a.data.size();
    for (size_t idx = 0; idx < n; ++idx) {
        const double av = a.data[idx];
        const double bv = b.data[idx];
        const double d = av - bv;
        diff2 += d * d;
        base2 += bv * bv;
    }
    diff2 = g_collectives.sum(diff2);
    base2 = g_collectives.sum(base2);
    return std::sqrt(diff2) / std::max(std::sqrt(base2), 1e-300);
}

static inline std::vector<int> neighbor_cells_self_first(int c) {
    return g_topology.manhattan_neighbors(c, 2);
}

static dg::StencilColoring g_ad_coloring;

static void ensure_topology_safe_ad_coloring()
{
    if (g_ad_coloring.cell_color.size() == static_cast<std::size_t>(NumCells)) return;
    std::vector<std::vector<int>> row_stencils(static_cast<std::size_t>(NumCells));
    for (int row = 0; row < NumCells; ++row) {
        row_stencils[static_cast<std::size_t>(row)] = neighbor_cells_self_first(row);
    }

    if constexpr (dg::case_config::wake_pairs == 0) {
        g_ad_coloring.cell_color.resize(static_cast<std::size_t>(NumCells));
        g_ad_coloring.color_count = dg::radius_two_color_count;
        for (int cell = 0; cell < NumCells; ++cell) {
            g_ad_coloring.cell_color[static_cast<std::size_t>(cell)] =
                dg::radius_two_color(cell_i_from_id(cell), cell_j_from_id(cell));
        }
    }
    else {
        g_ad_coloring = dg::greedy_stencil_coloring(row_stencils);
    }
    dg::audit_stencil_coloring(row_stencils, g_ad_coloring);
    if (g_ad_coloring.color_count <= 0 || g_ad_coloring.collision_pairs != 0) {
        std::cerr << "[FATAL] invalid AD cell coloring: colors="
            << g_ad_coloring.color_count
            << " collision_rows=" << g_ad_coloring.collision_rows
            << " collision_pairs=" << g_ad_coloring.collision_pairs << '\n';
        std::exit(1);
    }
    std::cout << "[AD COLOR] "
        << (dg::case_config::wake_pairs == 0
            ? "Cartesian radius-two" : "topology-safe greedy two-ring")
        << " colors=" << g_ad_coloring.color_count
        << " collision_rows=" << g_ad_coloring.collision_rows
        << " collision_pairs=" << g_ad_coloring.collision_pairs << '\n';
}

static inline int ilu0_ad_color(int c) {
    ensure_topology_safe_ad_coloring();
    return g_ad_coloring.cell_color[static_cast<std::size_t>(c)];
}



static inline bool cellavg_state_valid(const SolverVariable& S, const Array4D<double>& K,
    double* rho_min_out = nullptr, double* p_min_out = nullptr, double* nut_min_out = nullptr)
{
    auto eval_state_metrics = [](const double* U, double& rho, double& p, double& nut) {
        rho = U[0];
        const double rho_safe = std::max(rho, 1e-12);
        const double u = U[1] / rho_safe;
        const double v = U[2] / rho_safe;
        p = gamma1 * (U[3] - 0.5 * rho_safe * (u * u + v * v));
        nut = U[4] / rho_safe;
        };
    auto update_validity = [&](const double* U, double& rho_min, double& p_min, double& nut_min, bool& ok) {
        double rho = 0.0, p = 0.0, nut = 0.0;
        eval_state_metrics(U, rho, p, nut);
        rho_min = std::min(rho_min, rho);
        p_min = std::min(p_min, p);
        nut_min = std::min(nut_min, nut);
        ok = ok && std::isfinite(rho) && std::isfinite(p) && std::isfinite(nut)
            && rho > 0.0 && p > 0.0;
        };
    auto eval_with_phi = [&](int i, int j, const double* phi, double* U) {
        for (int e = 0; e < NumEq; ++e) U[e] = 0.0;
        for (int k = 0; k < dimPk; ++k) {
            const double ph = phi[k];
            for (int e = 0; e < NumEq; ++e) U[e] += K(i, j, k, e) * ph;
        }
        };

    double rho_min = 1e300, p_min = 1e300, nut_min = 1e300;
    bool ok = true;
    for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        double Uavg[NumEq];
        cell_average(S, K, i, j, Uavg);
        update_validity(Uavg, rho_min, p_min, nut_min, ok);

        for (int k = 0; k < dimPk; ++k) {
            double Uk[NumEq];
            for (int e = 0; e < NumEq; ++e) Uk[e] = K(i, j, k, e);
            update_validity(Uk, rho_min, p_min, nut_min, ok);
        }

        for (int q = 0; q < NumQuad; ++q) {
            double Uq[NumEq];
            eval_with_phi(i, j, S.phi_vol[q], Uq);
            update_validity(Uq, rho_min, p_min, nut_min, ok);
        }

        for (int f = 0; f < 4; ++f) {
            for (int g = 0; g < FaceQuad; ++g) {
                double Uf[NumEq];
                eval_with_phi(i, j, S.phi_face[f][g], Uf);
                update_validity(Uf, rho_min, p_min, nut_min, ok);
            }
        }
    }
    if (rho_min_out) *rho_min_out = rho_min;
    if (p_min_out) *p_min_out = p_min;
    if (nut_min_out) *nut_min_out = nut_min;
    return ok;
}

static void compute_local_dtau(const SolverVariable& S,
    const Array4D<double>& K,
    double cfl,
    std::vector<double>& dtau,
    double* dtau_min_out = nullptr,
    double* dtau_max_out = nullptr)
{
    dtau.assign(NumCells, 0.0);
    double dtau_min = 1e300;
    double dtau_max = 0.0;
    constexpr double dg_order_factor = 2.0 * PolyOrder + 1.0;
    constexpr double visc_factor = 4.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            double Uavg[NumEq];
            cell_average(S, K, i, j, Uavg);
            const double rho = std::max(Uavg[0], 1e-12);
            const double u = Uavg[1] / rho;
            const double v = Uavg[2] / rho;
            const double p = std::max(gamma1 * (Uavg[3] - 0.5 * rho * (u * u + v * v)), 1e-12);
            const double a = std::sqrt(gamma_v * p / rho);
            const double nut = std::max(Uavg[4] / rho, 0.0);
            const double mu_lam = laminar_mu_from_cons(Uavg);
            const double nu_eff = mu_lam / rho + nut;
            double dx = 0.0, dy = 0.0;
            cgns_cell_spacings(S, i, j, dx, dy);
            const double conv = (std::abs(u) + a) / dx + (std::abs(v) + a) / dy;
            const double visc = visc_factor * nu_eff * (1.0 / (dx * dx) + 1.0 / (dy * dy));
            const double denom = dg_order_factor * std::max(conv + visc, 1e-30);
            const double dti = cfl / denom;
            dtau[cell_id(i, j)] = dti;
            dtau_min = std::min(dtau_min, dti);
            dtau_max = std::max(dtau_max, dti);
        }
    }
    if (dtau_min_out) *dtau_min_out = dtau_min;
    if (dtau_max_out) *dtau_max_out = dtau_max;
}

static inline void compute_be_residual_local(SolverVariable& S,
    const Array4D<double>& Kstar,
    const Array4D<double>& Kn,
    const std::vector<double>& dtau,
    double sa_cfl_scale,
    Array4D<double>& F)
{
    computeRHS(S, Kstar);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const double dti = dtau[cell_id(i, j)];
            for (int k = 0; k < dimPk; ++k) {
                for (int e = 0; e < NumEq; ++e) {
                    const double equation_dt = e == 4
                        ? sa_cfl_scale * dti : dti;
                    F(i, j, k, e) = Kstar(i, j, k, e)
                        - Kn(i, j, k, e) - equation_dt * S.rhs(i, j, k, e);
                }
            }
        }
    }
}

static inline void restrict_residual_to_sa(Array4D<double>& residual)
{
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < 4; ++e) {
            residual(i, j, k, e) = 0.0;
        }
    }
}

static inline void restrict_vector_to_sa(std::vector<double>& values)
{
    for (int c = 0; c < NumCells; ++c) {
        const int base = c * BlockSize;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < 4; ++e) {
            values[base + local_dof(k, e)] = 0.0;
        }
    }
}

static inline void restore_ns_state(Array4D<double>& state,
    const Array4D<double>& frozen_reference)
{
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < 4; ++e) {
            state(i, j, k, e) = frozen_reference(i, j, k, e);
        }
    }
}

static inline void vector_add_to_array(const Array4D<double>& base,
    const std::vector<double>& direction,
    double alpha,
    Array4D<double>& out)
{
    out = base;
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        const int base_id = cell_id(i, j) * BlockSize;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            out(i, j, k, e) += alpha * direction[base_id + local_dof(k, e)];
        }
    }
}

static inline void finite_difference_be_matvec(SolverVariable& S,
    const Array4D<double>& Kbase,
    const Array4D<double>& Kn,
    const std::vector<double>& dtau,
    double sa_cfl_scale,
    const Array4D<double>& Fbase,
    const std::vector<double>& v,
    std::vector<double>& out)
{
    const double vnorm = std::max(norm_vec(v), 1e-300);
    double knorm2 = 0.0;
    for (double val : Kbase.data) knorm2 += val * val;
    const double knorm = std::sqrt(std::max(knorm2, 0.0));
    const double eps = FD_MATVEC_EPS * std::max(1.0, knorm) / vnorm;

    Array4D<double> Kpert(Nx, Ny, dimPk, NumEq);
    Array4D<double> Fpert(Nx, Ny, dimPk, NumEq);
    vector_add_to_array(Kbase, v, eps, Kpert);
    compute_be_residual_local(S, Kpert, Kn, dtau, sa_cfl_scale, Fpert);

    out.assign(NumDofs, 0.0);
    for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
        const int base_id = cell_id(i, j) * BlockSize;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            out[base_id + local_dof(k, e)] = (Fpert(i, j, k, e) - Fbase(i, j, k, e)) / eps;
        }
    }
}

using BlockEntry = dg::linear::BlockEntry<BlockSize>;
using DenseBlock = dg::linear::DenseBlock<BlockSize>;

static inline double& block_at(DenseBlock& block, int row, int col) {
    return dg::linear::block_at<BlockSize>(block, row, col);
}

static inline BlockEntry* find_block(std::vector<BlockEntry>& row, int col) {
    return dg::linear::find_block<BlockSize>(row, col);
}

static inline void zero_block(DenseBlock& block) {
    dg::linear::zero_block<BlockSize>(block);
}

// One derivative slot per unknown in a cell block. For the default Q2 case,
// BlockSize = 9 basis functions * 5 equations = 45. This is unrelated to the
// 13 spatial colors used to seed nonoverlapping cell stencils.
using AD45 = dg::BlockAD<BlockSize>;
using dg::ad_abs;
using dg::ad_exp;
using dg::ad_max;
using dg::ad_min;
using dg::ad_pow;
using dg::ad_sqrt;

static inline AD45 make_seed(double value, int p)
{
    return dg::make_seed<BlockSize>(value, static_cast<std::size_t>(p));
}

using PrimitiveStateAD = dg::adphysics::PrimitiveState<BlockSize>;

static inline dg::adphysics::GasProperties ad_gas_properties()
{
    dg::adphysics::GasProperties gas;
    gas.gamma = gamma_v;
    gas.gamma_minus_one = gamma1;
    gas.freestream_density = rho_inf;
    gas.freestream_u = u_inf;
    gas.freestream_v = v_inf;
    gas.freestream_pressure = p_inf;
    gas.freestream_nu_tilde = SA_INIT_NUT;
    gas.freestream_mach = M_inf;
    return gas;
}

static inline PrimitiveStateAD freestream_primitive_ad()
{
    return dg::adphysics::freestream_primitive<BlockSize>(ad_gas_properties());
}

static inline PrimitiveStateAD cons_to_primitive_ad(const AD45* U)
{
    return dg::adphysics::conservative_to_primitive<BlockSize>(U, ad_gas_properties());
}

static inline void primitive_to_cons_ad(const PrimitiveStateAD& V, AD45* U)
{
    dg::adphysics::primitive_to_conservative<BlockSize>(V, U, ad_gas_properties());
}

static inline void velocity_to_normal_tangent_ad(const AD45& u, const AD45& v, double nx, double ny, AD45& un, AD45& ut)
{
    dg::adphysics::velocity_to_normal_tangent<BlockSize>(u, v, nx, ny, un, ut);
}

static inline void normal_tangent_to_velocity_ad(const AD45& un, const AD45& ut, double nx, double ny, AD45& u, AD45& v)
{
    dg::adphysics::normal_tangent_to_velocity<BlockSize>(un, ut, nx, ny, u, v);
}

static inline PrimitiveStateAD characteristic_farfield_boundary_ad(
    const PrimitiveStateAD& Vi,
    const PrimitiveStateAD& Vext,
    double nx,
    double ny)
{
    return dg::adphysics::characteristic_farfield_boundary<BlockSize>(
        Vi, Vext, nx, ny, ad_gas_properties());
}

static inline PrimitiveStateAD characteristic_pressure_outlet_boundary_ad(
    const PrimitiveStateAD& Vi,
    double p_back,
    double nx,
    double ny)
{
    return dg::adphysics::characteristic_pressure_outlet_boundary<BlockSize>(
        Vi, p_back, nx, ny, ad_gas_properties());
}

static inline PrimitiveStateAD characteristic_total_inflow_boundary_ad(
    const PrimitiveStateAD& Vi,
    double p0,
    double T0,
    double alpha,
    double nx,
    double ny)
{
    return dg::adphysics::characteristic_total_inflow_boundary<BlockSize>(
        Vi, p0, T0, alpha, nx, ny, ad_gas_properties());
}

static void compute_be_residual_ad(SolverVariable& S,
    const Array4D<AD45>& Kstar,
    const Array4D<double>& Kn,
    const std::vector<double>& dtau,
    double sa_cfl_scale,
    Array4D<AD45>& F,
    Array4D<AD45>& rhs_ad,
    Array4D<AD45>& gradUX,
    Array4D<AD45>& gradUY);

using ADJacobianWorkspace = dg::ad::JacobianWorkspace<Array4D<AD45>>;

struct BlockSparseMatrix {
    std::vector<std::vector<BlockEntry>> rows;
    ADJacobianWorkspace ad_workspace;

    void init_pattern() {
        ensure_topology_safe_ad_coloring();
        rows.assign(NumCells, {});
        for (int c = 0; c < NumCells; ++c) {
            const std::vector<int> nb = neighbor_cells_self_first(c);
            rows[c].reserve(nb.size());
            for (int col : nb) {
                BlockEntry e;
                e.col = col;
                zero_block(e.a);
                rows[c].push_back(e);
            }
            std::sort(rows[c].begin(), rows[c].end(), [](const BlockEntry& a, const BlockEntry& b) { return a.col < b.col; });
        }
    }

    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        dg::linear::block_sparse_matvec<BlockSize>(rows, x, y);
    }

    void matvec_with_current_dtau(const std::vector<double>& x,
        std::vector<double>& y,
        const std::vector<double>& dtau_current,
        const std::vector<double>& dtau_build) const
    {
        matvec(x, y);
        if ((int)dtau_current.size() != NumCells || (int)dtau_build.size() != NumCells) return;
        for (int c = 0; c < NumCells; ++c) {
            const double base_dt = dtau_build[c];
            const double cur_dt = dtau_current[c];
            if (!(base_dt > 0.0) || !std::isfinite(base_dt) || !std::isfinite(cur_dt)) continue;
            const double scale = cur_dt / base_dt;
            const int base = c * BlockSize;
            for (int p = 0; p < BlockSize; ++p) {
                y[base + p] = x[base + p] + scale * (y[base + p] - x[base + p]);
            }
        }
    }

    bool assemble_ad(SolverVariable& S,
        const Array4D<double>& Kbase,
        const Array4D<double>& Kn,
        const std::vector<double>& dtau,
        double sa_cfl_scale,
        const Array4D<double>& Fbase,
        bool include_sa_stabilization = true,
        bool sa_only = false)
    {
        (void)Fbase;
        if ((int)rows.size() != NumCells) init_pattern();

        constexpr std::size_t field_size =
            static_cast<std::size_t>(Nx) * Ny * dimPk * NumEq;
        ad_workspace.ensure_allocated([](Array4D<AD45>& field) {
            if (field.data.size() != field_size) {
                field.reset(Nx, Ny, dimPk, NumEq);
            }
        });
        Array4D<AD45>& Kad = ad_workspace.state;
        Array4D<AD45>& Fad = ad_workspace.residual;
        const bool assembled = dg::ad::assemble_colored_block_jacobian<BlockSize>(
            rows,
            NumCells,
            g_ad_coloring.color_count,
            [](int cell) { return ilu0_ad_color(cell); },
            [&](int cell, bool seeded) {
                const int i = cell / Ny;
                const int j = cell % Ny;
                for (int k = 0; k < dimPk; ++k) {
                    for (int e = 0; e < NumEq; ++e) {
                        const int p = local_dof(k, e);
                        Kad(i, j, k, e) = seeded ? make_seed(Kbase(i, j, k, e), p) : AD45(Kbase(i, j, k, e));
                    }
                }
            },
            [&]() {
                compute_be_residual_ad(S, Kad, Kn, dtau, sa_cfl_scale, Fad,
                    ad_workspace.rhs, ad_workspace.gradient_x, ad_workspace.gradient_y);
                return true;
            },
            [](int column_cell) {
                return neighbor_cells_self_first(column_cell);
            },
            [&](int row_cell, int, DenseBlock& block) {
                const int ri = cell_i_from_id(row_cell);
                const int rj = cell_j_from_id(row_cell);
                for (int q = 0; q < BlockSize; ++q) {
                    int qk, qe;
                    dof_from_local(q, qk, qe);
                    for (int p = 0; p < BlockSize; ++p) {
                        block_at(block, q, p) = Fad(ri, rj, qk, qe).d[p];
                    }
                }
            });
        if (!assembled) return false;
        if (include_sa_stabilization && (int)dtau.size() == NumCells) {
            for (int c = 0; c < NumCells; ++c) {
                BlockEntry* diag = find_block(rows[c], c);
                if (!diag) continue;
                const int i = cell_i_from_id(c);
                const int j = cell_j_from_id(c);
                const double lambda_sa = cell_sa_source_diag_stiffness(S, Kbase, i, j);
                const double add = sa_cfl_scale
                    * std::max(dtau[c], 0.0) * lambda_sa;
                if (!std::isfinite(add) || add <= 0.0) continue;
                for (int k = 0; k < dimPk; ++k) {
                    const int p = local_dof(k, 4);
                    block_at(diag->a, p, p) += add;
                }
            }
        }
        if (sa_only
            && !dg::linear::constrain_to_equation<BlockSize, NumEq>(rows, 4)) {
            return false;
        }
        return true;
    }

    std::uint64_t bitwise_fingerprint() const
    {
        constexpr std::uint64_t fnv_offset = 1469598103934665603ull;
        constexpr std::uint64_t fnv_prime = 1099511628211ull;
        std::uint64_t fingerprint = fnv_offset;
        for (const auto& row : rows) {
            for (const BlockEntry& entry : row) {
                for (double value : entry.a) {
                    std::uint64_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    for (int byte = 0; byte < 8; ++byte) {
                        fingerprint ^= (bits >> (8 * byte)) & 0xffu;
                        fingerprint *= fnv_prime;
                    }
                }
            }
        }
        return fingerprint;
    }
};

static int run_jacobian_directional_check(SolverVariable& S,
    const Array4D<double>& Kbase, double cfl, double sa_cfl_scale)
{
    constexpr double perturbation = 1.0e-7;
    // Centered finite differences lose relative accuracy in very weak
    // cross-equation couplings. 5e-4 still detects a material Jacobian error
    // while admitting the measured FD cancellation floor on this operator.
    constexpr double relative_tolerance = 5.0e-4;
    constexpr double scaled_absolute_tolerance = 5.0e-8;
    constexpr std::array<const char*, NumEq> equation_names = {
        "rho", "rho_u", "rho_v", "rho_E", "rho_nu_tilde"
    };
    const std::array<double, NumEq> reference_scales = {
        rho_inf,
        std::max(std::abs(rho_inf * u_inf), 0.1),
        std::max(std::abs(rho_inf * v_inf), 0.1),
        p_inf / gamma1 + 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf),
        std::max(rho_inf * SA_INIT_NUT, 1.0e-12)
    };

    std::vector<double> dtau;
    double dtau_min = 0.0;
    double dtau_max = 0.0;
    compute_local_dtau(S, Kbase, cfl, dtau, &dtau_min, &dtau_max);
    Array4D<double> Fbase(Nx, Ny, dimPk, NumEq);
    compute_be_residual_local(S, Kbase, Kbase, dtau, sa_cfl_scale, Fbase);

    BlockSparseMatrix jacobian;
    const auto assembly_start = std::chrono::steady_clock::now();
    // Compare only the derivative of the actual BE residual. The production
    // solver's positive SA diagonal is a stabilization term, not part of F.
    if (!jacobian.assemble_ad(S, Kbase, Kbase, dtau, sa_cfl_scale,
            Fbase, false)) {
        std::cerr << "[JACOBIAN CHECK] status=failed reason=assembly_failed\n";
        return 1;
    }
    const double assembly_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - assembly_start).count();

    enum class DirectionKind { mixed, ns_only, sa_only };
    const std::array<std::pair<const char*, DirectionKind>, 3> directions = {{
        {"mixed", DirectionKind::mixed},
        {"ns_only", DirectionKind::ns_only},
        {"sa_only", DirectionKind::sa_only}
    }};
    bool passed = true;
    double worst_relative_error = 0.0;
    std::cout << std::scientific << std::setprecision(12)
        << "[JACOBIAN CHECK] cfl=" << cfl
        << " sa_cfl_scale=" << sa_cfl_scale
        << " epsilon=" << perturbation
        << " dtau_min=" << dtau_min
        << " dtau_max=" << dtau_max
        << " colors=" << g_ad_coloring.color_count
        << " assembly_seconds=" << assembly_seconds << "\n";

    for (const auto& named_direction : directions) {
        const char* name = named_direction.first;
        const DirectionKind kind = named_direction.second;
        std::vector<double> direction(NumDofs, 0.0);
        std::uint64_t random_state = 0x9e3779b97f4a7c15ull;
        for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
            const int base = cell_id(i, j) * BlockSize;
            for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
                random_state ^= random_state >> 12;
                random_state ^= random_state << 25;
                random_state ^= random_state >> 27;
                const std::uint64_t bits = random_state
                    * 2685821657736338717ull;
                const double random_value = 2.0 * static_cast<double>(
                    bits >> 11) / 9007199254740992.0 - 1.0;
                const bool active = kind == DirectionKind::mixed
                    || (kind == DirectionKind::ns_only && e < 4)
                    || (kind == DirectionKind::sa_only && e == 4);
                if (!active) continue;
                const double scale = std::max(
                    std::abs(Kbase(i, j, k, e)), reference_scales[e]);
                direction[base + local_dof(k, e)] = random_value * scale;
            }
        }

        std::vector<double> ad_product;
        jacobian.matvec(direction, ad_product);
        Array4D<double> Kplus(Nx, Ny, dimPk, NumEq);
        Array4D<double> Kminus(Nx, Ny, dimPk, NumEq);
        Array4D<double> Fplus(Nx, Ny, dimPk, NumEq);
        Array4D<double> Fminus(Nx, Ny, dimPk, NumEq);
        vector_add_to_array(Kbase, direction, perturbation, Kplus);
        vector_add_to_array(Kbase, direction, -perturbation, Kminus);
        compute_be_residual_local(S, Kplus, Kbase, dtau,
            sa_cfl_scale, Fplus);
        compute_be_residual_local(S, Kminus, Kbase, dtau,
            sa_cfl_scale, Fminus);

        std::array<double, NumEq> ad_square{};
        std::array<double, NumEq> fd_square{};
        std::array<double, NumEq> error_square{};
        std::array<double, NumEq> error_max{};
        for (int i = 0; i < Nx; ++i) for (int j = 0; j < Ny; ++j) {
            const int base = cell_id(i, j) * BlockSize;
            for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
                const int index = base + local_dof(k, e);
                const double fd = (Fplus(i, j, k, e)
                    - Fminus(i, j, k, e)) / (2.0 * perturbation);
                const double error = ad_product[index] - fd;
                ad_square[e] += ad_product[index] * ad_product[index];
                fd_square[e] += fd * fd;
                error_square[e] += error * error;
                error_max[e] = std::max(error_max[e], std::abs(error));
            }
        }
        double total_ad_square = 0.0;
        double total_fd_square = 0.0;
        double total_error_square = 0.0;
        double direction_worst = 0.0;
        double direction_worst_significant = 0.0;
        constexpr double equation_dofs = static_cast<double>(
            Nx * Ny * dimPk);
        for (int e = 0; e < NumEq; ++e) {
            total_ad_square += ad_square[e];
            total_fd_square += fd_square[e];
            total_error_square += error_square[e];
            const double relative = std::sqrt(error_square[e])
                / std::max({std::sqrt(ad_square[e]),
                    std::sqrt(fd_square[e]), 1.0e-300});
            const double ad_l2 = std::sqrt(ad_square[e]);
            const double fd_l2 = std::sqrt(fd_square[e]);
            const double error_l2 = std::sqrt(error_square[e]);
            const double scaled_absolute = error_l2
                / (std::sqrt(equation_dofs) * reference_scales[e]);
            const bool equation_passed = relative <= relative_tolerance
                || scaled_absolute <= scaled_absolute_tolerance;
            direction_worst = std::max(direction_worst, relative);
            if (!equation_passed) {
                direction_worst_significant = std::max(
                    direction_worst_significant, relative);
            }
            std::cout << "[JACOBIAN CHECK] direction=" << name
                << " equation=" << equation_names[e]
                << " relative_l2=" << relative
                << " scaled_absolute_l2=" << scaled_absolute
                << " ad_l2=" << ad_l2
                << " fd_l2=" << fd_l2
                << " error_l2=" << error_l2
                << " max_abs=" << error_max[e] << "\n";
        }
        const double total_relative = std::sqrt(total_error_square)
            / std::max({std::sqrt(total_ad_square),
                std::sqrt(total_fd_square), 1.0e-300});
        worst_relative_error = std::max(worst_relative_error,
            direction_worst);
        passed = passed && direction_worst_significant == 0.0;
        std::cout << "[JACOBIAN CHECK] direction=" << name
            << " total_relative_l2=" << total_relative
            << " worst_equation_relative_l2=" << direction_worst
            << " worst_significant_relative_l2="
            << direction_worst_significant << "\n";
    }
    std::cout << "[JACOBIAN CHECK] status="
        << (passed ? "ok" : "failed")
        << " tolerance=" << relative_tolerance
        << " scaled_absolute_tolerance=" << scaled_absolute_tolerance
        << " worst_relative_l2=" << worst_relative_error << "\n";
    return passed ? 0 : 1;
}

using BlockILU0Preconditioner = dg::linear::BlockILU0Preconditioner<BlockSize>;
using IdentityPreconditioner = dg::linear::IdentityPreconditioner;
using BlockLUSGSPreconditioner = dg::linear::BlockLUSGSPreconditioner<BlockSize>;
using GMRESResult = dg::linear::GMRESResult;
using DoubleResidualWorkspace =
    dg::CellResidualWorkspace<double, dimPk, FaceQuad, NumEq>;

enum SATermComponent {
    SA_CONVECTION = 0,
    SA_VISCOSITY_VOLUME = 1,
    SA_VISCOSITY_INTERIOR_CONSISTENT = 2,
    SA_VISCOSITY_INTERIOR_LIFTING = 3,
    SA_VISCOSITY_WALL_FACE = 4,
    SA_VISCOSITY_OTHER_BOUNDARY = 5,
    SA_PRODUCTION = 6,
    SA_DESTRUCTION = 7,
    SA_CROSS_DIFFUSION = 8,
    SA_TERM_COUNT = 9
};
static_assert(SA_TERM_COUNT == dg::io::sa_term_balance_component_count,
    "SA term diagnostic component count mismatch");

static const char* sa_term_component_name(int component)
{
    constexpr std::array<const char*, SA_TERM_COUNT> names = {{
        "convection", "viscosity_volume", "viscosity_interior_consistent",
        "viscosity_interior_lifting", "viscosity_wall_face",
        "viscosity_other_boundary", "production", "destruction",
        "cross_diffusion"
    }};
    return component >= 0 && component < SA_TERM_COUNT
        ? names[static_cast<std::size_t>(component)] : "unavailable";
}

struct SATermCellAccumulator {
    std::array<std::array<double, dimPk>, SA_TERM_COUNT> value{};
    void clear() {
        for (auto& component : value) component.fill(0.0);
    }
};

static inline void accumulate_cell_volume_residual(const SolverVariable& S,
    const Array4D<double>& Kin, int i, int j, DoubleResidualWorkspace& work,
    SATermCellAccumulator* sa_terms = nullptr)
{
    for (int q = 0; q < NumQuad; ++q) {
        work.clear_volume_point();
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            work.state[e] += Kin(i, j, k, e) * S.phi_vol[q][k];
            work.gradient_x[e] += S.gradUX(i, j, k, e) * S.phi_vol[q][k];
            work.gradient_y[e] += S.gradUY(i, j, k, e) * S.phi_vol[q][k];
        }
        const double rho = std::max(work.state[0], 1e-12);
        const double u = work.state[1] / rho;
        const double v = work.state[2] / rho;
        const double p = std::max(gamma1 *
            (work.state[3] - 0.5 * rho * (u * u + v * v)), 1e-12);
        work.convective_x[0] = work.state[1];
        work.convective_x[1] = rho * u * u + p;
        work.convective_x[2] = rho * u * v;
        work.convective_x[3] = u * (work.state[3] + p);
        work.convective_x[4] = work.state[4] * u;
        work.convective_y[0] = work.state[2];
        work.convective_y[1] = rho * u * v;
        work.convective_y[2] = rho * v * v + p;
        work.convective_y[3] = v * (work.state[3] + p);
        work.convective_y[4] = work.state[4] * v;
        Viscous_Flux(work.state.data(), work.gradient_x.data(), work.gradient_y.data(),
            work.viscous_x.data(), work.viscous_y.data());

        const double quadrature_weight =
            S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
        const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
        const dg::SASourceTerms<double> source_terms = SA_Source_Terms(
            work.state.data(), work.gradient_x.data(), work.gradient_y.data(),
            metric.x, metric.y);
        const double source_sa = source_terms.total;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            const double grad_phi_x_jacobian =
                S.dphi_dr[q][k] * metric.ys - S.dphi_ds[q][k] * metric.yr;
            const double grad_phi_y_jacobian =
                -S.dphi_dr[q][k] * metric.xs + S.dphi_ds[q][k] * metric.xr;
            const double convection = quadrature_weight *
                (work.convective_x[e] * grad_phi_x_jacobian
                    + work.convective_y[e] * grad_phi_y_jacobian);
            const double viscosity = quadrature_weight *
                (-work.viscous_x[e] * grad_phi_x_jacobian
                    + -work.viscous_y[e] * grad_phi_y_jacobian);
            const double source = e == 4
                ? quadrature_weight * source_sa * S.phi_vol[q][k] * metric.jacobian
                : 0.0;
            work.cell(k, e) += convection + viscosity + source;
            if (sa_terms != nullptr && e == 4) {
                const double source_projection = quadrature_weight
                    * S.phi_vol[q][k] * metric.jacobian;
                sa_terms->value[SA_CONVECTION][k] += convection;
                sa_terms->value[SA_VISCOSITY_VOLUME][k] += viscosity;
                sa_terms->value[SA_PRODUCTION][k] +=
                    source_projection * source_terms.production;
                sa_terms->value[SA_DESTRUCTION][k] +=
                    source_projection * source_terms.destruction;
                sa_terms->value[SA_CROSS_DIFFUSION][k] +=
                    source_projection * source_terms.cross_diffusion;
            }
        }
    }
}

static inline void accumulate_cell_face_residual(const SolverVariable& S,
    const Array4D<double>& Kin, int i, int j, int face,
    DoubleResidualWorkspace& work, SATermCellAccumulator* sa_terms = nullptr)
{
    work.clear_face_data();
    const auto neighbor = g_topology.face_neighbor(i, j, face);
    const int neighbor_i = neighbor.i;
    const int neighbor_j = neighbor.j;
    const bool interior = neighbor.interior;
    const int neighbor_face = neighbor.face;

    for (int g = 0; g < FaceQuad; ++g) {
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            work.face(work.face_left, g, e) +=
                Kin(i, j, k, e) * S.phi_face[face][g][k];
        }
        if (interior) {
            const int neighbor_g = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
            for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
                work.face(work.face_right_convective, g, e) +=
                    Kin(neighbor_i, neighbor_j, k, e)
                    * S.phi_face[neighbor_face][neighbor_g][k];
            }
            for (int e = 0; e < NumEq; ++e) {
                work.face(work.face_right_viscous, g, e) =
                    work.face(work.face_right_convective, g, e);
            }
        }
        else {
            double x_face = 0.0, y_face = 0.0;
            face_point_coords(S, i, j, face, g, x_face, y_face);
            double nx = 0.0, ny = 0.0, measure = 0.0;
            face_metric(S, i, j, face, g, nx, ny, measure);
            double* left = &work.face_left[g * NumEq];
            boundary_state_2dzp(left, face, x_face, y_face, nx, ny,
                &work.face_right_convective[g * NumEq]);
            work.strict_wall_viscous[g] = boundary_viscous_trace_2dzp(left,
                face, x_face, y_face, nx, ny,
                &work.face_right_viscous[g * NumEq]);
            work.zero_viscous_flux[g] = is_zero_viscous_flux_face(face, x_face);
        }

        for (int e = 0; e < NumEq; ++e) {
            work.face(work.face_jump, g, e) =
                work.face(work.face_left, g, e)
                - work.face(work.face_right_viscous, g, e);
        }
        if (!interior && work.strict_wall_viscous[g]) {
            for (int e = 1; e <= 3; ++e) {
                work.face(work.face_jump, g, e) *=
                    BR2_WALL_BOUNDARY_TRACE_FACTOR * NS_WALL_BR2_SCALE;
            }
            work.face(work.face_jump, g, 4) *=
                BR2_WALL_BOUNDARY_TRACE_FACTOR * SA_WALL_BR2_SCALE;
        }
        work.face(work.face_jump, g, 4) *= g_sa_br2_factor;
    }

    computeFaceLifting(S, work.face_jump.data(), face, i, j,
        work.lifting_x.data(), work.lifting_y.data(), BR2_ETA);
    if (interior) {
        for (int g = 0; g < FaceQuad; ++g) {
            const int neighbor_g = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
            for (int e = 0; e < NumEq; ++e) {
                work.face(work.face_auxiliary, neighbor_g, e) =
                    -work.face(work.face_jump, g, e);
            }
        }
        computeFaceLifting(S, work.face_auxiliary.data(), neighbor_face,
            neighbor_i, neighbor_j, work.adjacent_lifting_x.data(),
            work.adjacent_lifting_y.data(), BR2_ETA);
    }

    for (int g = 0; g < FaceQuad; ++g) {
        work.clear_face_point();
        std::array<double, NumEq> base_left_gradient_x;
        std::array<double, NumEq> base_left_gradient_y;
        std::array<double, NumEq> base_right_gradient_x;
        std::array<double, NumEq> base_right_gradient_y;
        if (sa_terms != nullptr && interior) {
            base_left_gradient_x.fill(0.0);
            base_left_gradient_y.fill(0.0);
            base_right_gradient_x.fill(0.0);
            base_right_gradient_y.fill(0.0);
        }
        double nx = 0.0, ny = 0.0, measure = 0.0;
        face_metric(S, i, j, face, g, nx, ny, measure);
        for (int e = 0; e < NumEq; ++e) {
            work.state[e] = work.face(work.face_left, g, e);
            work.right_convective[e] =
                work.face(work.face_right_convective, g, e);
            work.right_viscous[e] = work.face(work.face_right_viscous, g, e);
        }
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            if (sa_terms != nullptr && interior) {
                base_left_gradient_x[e] +=
                    S.gradUX(i, j, k, e) * S.phi_face[face][g][k];
                base_left_gradient_y[e] +=
                    S.gradUY(i, j, k, e) * S.phi_face[face][g][k];
            }
            work.gradient_x[e] +=
                (S.gradUX(i, j, k, e) - work.lifting(work.lifting_x, k, e))
                * S.phi_face[face][g][k];
            work.gradient_y[e] +=
                (S.gradUY(i, j, k, e) - work.lifting(work.lifting_y, k, e))
                * S.phi_face[face][g][k];
        }
        if (interior) {
            const int neighbor_g = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
            for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
                if (sa_terms != nullptr) {
                    base_right_gradient_x[e] +=
                        S.gradUX(neighbor_i, neighbor_j, k, e)
                        * S.phi_face[neighbor_face][neighbor_g][k];
                    base_right_gradient_y[e] +=
                        S.gradUY(neighbor_i, neighbor_j, k, e)
                        * S.phi_face[neighbor_face][neighbor_g][k];
                }
                work.right_gradient_x[e] +=
                    (S.gradUX(neighbor_i, neighbor_j, k, e)
                        - work.lifting(work.adjacent_lifting_x, k, e))
                    * S.phi_face[neighbor_face][neighbor_g][k];
                work.right_gradient_y[e] +=
                    (S.gradUY(neighbor_i, neighbor_j, k, e)
                        - work.lifting(work.adjacent_lifting_y, k, e))
                    * S.phi_face[neighbor_face][neighbor_g][k];
            }
        }
        else {
            work.right_gradient_x = work.gradient_x;
            work.right_gradient_y = work.gradient_y;
        }

        dg::hllc_euler_flux(work.state.data(), work.right_convective.data(),
            nx, ny, gamma_v, work.numerical_flux.data());
        const double rho_left = std::max(work.state[0], 1e-12);
        const double rho_right = std::max(work.right_convective[0], 1e-12);
        const double mass_flux = work.numerical_flux[0];
        work.numerical_flux[4] = mass_flux * (mass_flux >= 0.0
            ? work.state[4] / rho_left
            : work.right_convective[4] / rho_right);

        for (int e = 0; e < NumEq; ++e) {
            if (!interior && work.strict_wall_viscous[g]) {
                work.averaged_state[e] = work.right_viscous[e];
                work.averaged_gradient_x[e] = work.gradient_x[e];
                work.averaged_gradient_y[e] = work.gradient_y[e];
            }
            else {
                work.averaged_state[e] =
                    0.5 * (work.state[e] + work.right_viscous[e]);
                work.averaged_gradient_x[e] =
                    0.5 * (work.gradient_x[e] + work.right_gradient_x[e]);
                work.averaged_gradient_y[e] =
                    0.5 * (work.gradient_y[e] + work.right_gradient_y[e]);
            }
        }
        double consistent_sa_viscous_normal = 0.0;
        if (sa_terms != nullptr && interior) {
            std::array<double, NumEq> base_averaged_gradient_x{};
            std::array<double, NumEq> base_averaged_gradient_y{};
            std::array<double, NumEq> consistent_viscous_x{};
            std::array<double, NumEq> consistent_viscous_y{};
            for (int e = 0; e < NumEq; ++e) {
                base_averaged_gradient_x[e] = 0.5
                    * (base_left_gradient_x[e] + base_right_gradient_x[e]);
                base_averaged_gradient_y[e] = 0.5
                    * (base_left_gradient_y[e] + base_right_gradient_y[e]);
            }
            Viscous_Flux(work.averaged_state.data(),
                base_averaged_gradient_x.data(), base_averaged_gradient_y.data(),
                consistent_viscous_x.data(), consistent_viscous_y.data());
            consistent_sa_viscous_normal = consistent_viscous_x[4] * nx
                + consistent_viscous_y[4] * ny;
        }
        Viscous_Flux(work.averaged_state.data(), work.averaged_gradient_x.data(),
            work.averaged_gradient_y.data(), work.viscous_x.data(),
            work.viscous_y.data());
        for (int e = 0; e < NumEq; ++e) {
            work.viscous_normal[e] =
                work.viscous_x[e] * nx + work.viscous_y[e] * ny;
        }
        if (!interior && work.zero_viscous_flux[g]) {
            work.viscous_normal.fill(0.0);
        }
        if (!interior && work.strict_wall_viscous[g]) {
            work.viscous_normal[3] = 0.0;
        }
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            const double convection = -S.gl_w[g] * work.numerical_flux[e]
                * S.phi_face[face][g][k] * measure;
            const double viscosity = S.gl_w[g] * work.viscous_normal[e]
                * S.phi_face[face][g][k] * measure;
            work.cell(k, e) += convection + viscosity;
            if (sa_terms != nullptr && e == 4) {
                sa_terms->value[SA_CONVECTION][k] += convection;
                if (interior) {
                    const double consistent_viscosity = S.gl_w[g]
                        * consistent_sa_viscous_normal
                        * S.phi_face[face][g][k] * measure;
                    sa_terms->value[SA_VISCOSITY_INTERIOR_CONSISTENT][k] +=
                        consistent_viscosity;
                    sa_terms->value[SA_VISCOSITY_INTERIOR_LIFTING][k] +=
                        viscosity - consistent_viscosity;
                }
                else {
                    const int viscosity_component = work.strict_wall_viscous[g]
                        ? SA_VISCOSITY_WALL_FACE
                        : SA_VISCOSITY_OTHER_BOUNDARY;
                    sa_terms->value[viscosity_component][k] += viscosity;
                }
            }
        }
    }
}

static inline void store_mass_inverted_cell_rhs(SolverVariable& S,
    int i, int j, DoubleResidualWorkspace& work)
{
    for (int e = 0; e < NumEq; ++e) {
        for (int k = 0; k < dimPk; ++k) work.mass_rhs[k] = work.cell(k, e);
        for (int k = 0; k < dimPk; ++k) {
            double value = 0.0;
            for (int m = 0; m < dimPk; ++m) {
                value += cell_inv_mass_entry(S, i, j, k, m) * work.mass_rhs[m];
            }
            S.rhs(i, j, k, e) = value;
        }
    }
}

static inline void assemble_cell_rhs(SolverVariable& S,
    const Array4D<double>& Kin, int i, int j, DoubleResidualWorkspace& work)
{
    work.clear_cell();
    accumulate_cell_volume_residual(S, Kin, i, j, work);
    for (int face = 0; face < 4; ++face) {
        accumulate_cell_face_residual(S, Kin, i, j, face, work);
    }
    store_mass_inverted_cell_rhs(S, i, j, work);
}

// DG semi-discrete RHS assembly. Each iteration owns one output cell and one
// reusable workspace, which is the thread-safety boundary for later OpenMP.
void computeRHS(SolverVariable& S, const Array4D<double>& Kin) {
    const auto rhs_start = g_profile_rhs_timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    computeGradients(S, Kin);
#if DG_HAS_OPENMP
#pragma omp parallel
    {
        DoubleResidualWorkspace work;
#pragma omp for schedule(static)
        for (int cell = 0; cell < NumCells; ++cell) {
            const int i = cell / Ny;
            const int j = cell % Ny;
            assemble_cell_rhs(S, Kin, i, j, work);
        }
    }
#else
    DoubleResidualWorkspace work;
    for (int cell = 0; cell < NumCells; ++cell) {
        const int i = cell / Ny;
        const int j = cell % Ny;
        assemble_cell_rhs(S, Kin, i, j, work);
    }
#endif
    if (g_profile_rhs_timing_enabled) {
        g_profile_rhs_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rhs_start).count();
        ++g_profile_rhs_calls;
    }
}

static void assemble_sa_term_field(SolverVariable& S,
    const Array4D<double>& Kin, Array4D<double>& term_field,
    Array4D<double>& interior_face_field)
{
    term_field.reset(Nx, Ny, dimPk, SA_TERM_COUNT);
    interior_face_field.reset(Nx, Ny, dimPk, 8);
    computeGradients(S, Kin);
#if DG_HAS_OPENMP
#pragma omp parallel
    {
        DoubleResidualWorkspace work;
        SATermCellAccumulator terms;
        SATermCellAccumulator face_terms;
#pragma omp for schedule(static)
        for (int cell = 0; cell < NumCells; ++cell) {
            const int i = cell / Ny;
            const int j = cell % Ny;
            work.clear_cell();
            terms.clear();
            accumulate_cell_volume_residual(S, Kin, i, j, work, &terms);
            for (int face = 0; face < 4; ++face) {
                face_terms.clear();
                accumulate_cell_face_residual(S, Kin, i, j, face, work,
                    &face_terms);
                for (int component = 0; component < SA_TERM_COUNT; ++component) {
                    for (int k = 0; k < dimPk; ++k) {
                        terms.value[component][k] +=
                            face_terms.value[component][k];
                    }
                }
                for (int k = 0; k < dimPk; ++k) {
                    double consistent = 0.0;
                    double lifting = 0.0;
                    for (int m = 0; m < dimPk; ++m) {
                        const double inverse_mass =
                            cell_inv_mass_entry(S, i, j, k, m);
                        consistent += inverse_mass
                            * face_terms.value[SA_VISCOSITY_INTERIOR_CONSISTENT][m];
                        lifting += inverse_mass
                            * face_terms.value[SA_VISCOSITY_INTERIOR_LIFTING][m];
                    }
                    interior_face_field(i, j, k, 2 * face) = consistent;
                    interior_face_field(i, j, k, 2 * face + 1) = lifting;
                }
            }
            for (int component = 0; component < SA_TERM_COUNT; ++component) {
                for (int k = 0; k < dimPk; ++k) {
                    double value = 0.0;
                    for (int m = 0; m < dimPk; ++m) {
                        value += cell_inv_mass_entry(S, i, j, k, m)
                            * terms.value[component][m];
                    }
                    term_field(i, j, k, component) = value;
                }
            }
        }
    }
#else
    DoubleResidualWorkspace work;
    SATermCellAccumulator terms;
    SATermCellAccumulator face_terms;
    for (int cell = 0; cell < NumCells; ++cell) {
        const int i = cell / Ny;
        const int j = cell % Ny;
        work.clear_cell();
        terms.clear();
        accumulate_cell_volume_residual(S, Kin, i, j, work, &terms);
        for (int face = 0; face < 4; ++face) {
            face_terms.clear();
            accumulate_cell_face_residual(S, Kin, i, j, face, work,
                &face_terms);
            for (int component = 0; component < SA_TERM_COUNT; ++component) {
                for (int k = 0; k < dimPk; ++k) {
                    terms.value[component][k] += face_terms.value[component][k];
                }
            }
            for (int k = 0; k < dimPk; ++k) {
                double consistent = 0.0;
                double lifting = 0.0;
                for (int m = 0; m < dimPk; ++m) {
                    const double inverse_mass = cell_inv_mass_entry(S, i, j, k, m);
                    consistent += inverse_mass
                        * face_terms.value[SA_VISCOSITY_INTERIOR_CONSISTENT][m];
                    lifting += inverse_mass
                        * face_terms.value[SA_VISCOSITY_INTERIOR_LIFTING][m];
                }
                interior_face_field(i, j, k, 2 * face) = consistent;
                interior_face_field(i, j, k, 2 * face + 1) = lifting;
            }
        }
        for (int component = 0; component < SA_TERM_COUNT; ++component) {
            for (int k = 0; k < dimPk; ++k) {
                double value = 0.0;
                for (int m = 0; m < dimPk; ++m) {
                    value += cell_inv_mass_entry(S, i, j, k, m)
                        * terms.value[component][m];
                }
                term_field(i, j, k, component) = value;
            }
        }
    }
#endif
}

static void populate_sa_dominant_face_jump(const SolverVariable& S,
    const Array4D<double>& K, dg::io::SATermBalanceRecord& record)
{
    const int i = record.hotspot_i;
    const int j = record.hotspot_j;
    const int face = record.hotspot_dominant_interior_face;
    if (i < 0 || j < 0 || face < 0) return;
    const auto neighbor = g_topology.face_neighbor(i, j, face);
    if (!neighbor.interior) return;

    double largest_jump = -1.0;
    for (int g = 0; g < FaceQuad; ++g) {
        const int neighbor_g = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
        double left[NumEq] = {};
        double right[NumEq] = {};
        double left_gradient_x[NumEq] = {};
        double left_gradient_y[NumEq] = {};
        double right_gradient_x[NumEq] = {};
        double right_gradient_y[NumEq] = {};
        for (int k = 0; k < dimPk; ++k) {
            const double left_basis = S.phi_face[face][g][k];
            const double right_basis =
                S.phi_face[neighbor.face][neighbor_g][k];
            for (int e = 0; e < NumEq; ++e) {
                left[e] += K(i, j, k, e) * left_basis;
                right[e] += K(neighbor.i, neighbor.j, k, e) * right_basis;
                left_gradient_x[e] += S.gradUX(i, j, k, e) * left_basis;
                left_gradient_y[e] += S.gradUY(i, j, k, e) * left_basis;
                right_gradient_x[e] +=
                    S.gradUX(neighbor.i, neighbor.j, k, e) * right_basis;
                right_gradient_y[e] +=
                    S.gradUY(neighbor.i, neighbor.j, k, e) * right_basis;
            }
        }
        const double left_rho = std::max(left[0], 1e-12);
        const double right_rho = std::max(right[0], 1e-12);
        const double left_nut = left[4] / left_rho;
        const double right_nut = right[4] / right_rho;
        const double nut_jump = left_nut - right_nut;
        double nx = 0.0, ny = 0.0, measure = 0.0;
        face_metric(S, i, j, face, g, nx, ny, measure);
        const double left_nut_x =
            (left_gradient_x[4] - left_nut * left_gradient_x[0]) / left_rho;
        const double left_nut_y =
            (left_gradient_y[4] - left_nut * left_gradient_y[0]) / left_rho;
        const double right_nut_x =
            (right_gradient_x[4] - right_nut * right_gradient_x[0]) / right_rho;
        const double right_nut_y =
            (right_gradient_y[4] - right_nut * right_gradient_y[0]) / right_rho;
        const double normal_gradient_jump =
            (left_nut_x - right_nut_x) * nx
            + (left_nut_y - right_nut_y) * ny;
        if (std::abs(nut_jump) > largest_jump) {
            largest_jump = std::abs(nut_jump);
            record.hotspot_dominant_face_quadrature = g;
            face_point_coords(S, i, j, face, g,
                record.hotspot_dominant_face_x,
                record.hotspot_dominant_face_y);
            record.hotspot_dominant_face_qsa_jump = left[4] - right[4];
            record.hotspot_dominant_face_nu_tilde_jump = nut_jump;
            record.hotspot_dominant_face_normal_gradient_jump =
                normal_gradient_jump;
        }
    }
}

static void populate_wake_interface_audit(const SolverVariable& S,
    const Array4D<double>& K, dg::io::SATermBalanceRecord& record)
{
    if constexpr (dg::case_config::wake_pairs == 0) return;

    auto relative_cancellation = [](double left, double right) {
        const double denominator = std::abs(left) + std::abs(right);
        if (denominator <= 1e-300) return 0.0;
        return std::abs(left + right) / denominator;
    };
    auto integrated_component = [](const SATermCellAccumulator& terms,
        int component) {
        double value = 0.0;
        // The nodal Lagrange basis is a partition of unity, so summing all
        // modal/nodal test-function contributions recovers the face integral
        // for the constant test function.
        for (int k = 0; k < dimPk; ++k) {
            value += terms.value[component][k];
        }
        return value;
    };

    DoubleResidualWorkspace left_work;
    DoubleResidualWorkspace right_work;
    SATermCellAccumulator left_terms;
    SATermCellAccumulator right_terms;
    for (int i = 0; i < dg::case_config::wake_pairs; ++i) {
        const auto neighbor = g_topology.face_neighbor(i, 0, 0);
        if (!neighbor.interior) continue;

        left_work.clear_cell();
        right_work.clear_cell();
        left_terms.clear();
        right_terms.clear();
        accumulate_cell_face_residual(S, K, i, 0, 0,
            left_work, &left_terms);
        accumulate_cell_face_residual(S, K, neighbor.i, neighbor.j,
            neighbor.face, right_work, &right_terms);

        const double left_convection = integrated_component(left_terms,
            SA_CONVECTION);
        const double right_convection = integrated_component(right_terms,
            SA_CONVECTION);
        const double left_consistent = integrated_component(left_terms,
            SA_VISCOSITY_INTERIOR_CONSISTENT);
        const double right_consistent = integrated_component(right_terms,
            SA_VISCOSITY_INTERIOR_CONSISTENT);
        const double left_lifting = integrated_component(left_terms,
            SA_VISCOSITY_INTERIOR_LIFTING);
        const double right_lifting = integrated_component(right_terms,
            SA_VISCOSITY_INTERIOR_LIFTING);
        const double left_viscosity = left_consistent + left_lifting;
        const double right_viscosity = right_consistent + right_lifting;
        const double left_total = left_convection + left_viscosity;
        const double right_total = right_convection + right_viscosity;

        record.wake_interface_convection_cancel_rel_max = std::max(
            record.wake_interface_convection_cancel_rel_max,
            relative_cancellation(left_convection, right_convection));
        record.wake_interface_consistent_cancel_rel_max = std::max(
            record.wake_interface_consistent_cancel_rel_max,
            relative_cancellation(left_consistent, right_consistent));
        record.wake_interface_lifting_cancel_rel_max = std::max(
            record.wake_interface_lifting_cancel_rel_max,
            relative_cancellation(left_lifting, right_lifting));
        record.wake_interface_viscosity_cancel_rel_max = std::max(
            record.wake_interface_viscosity_cancel_rel_max,
            relative_cancellation(left_viscosity, right_viscosity));
        record.wake_interface_total_cancel_rel_max = std::max(
            record.wake_interface_total_cancel_rel_max,
            relative_cancellation(left_total, right_total));

        for (int g = 0; g < FaceQuad; ++g) {
            const int neighbor_g = neighbor.reverse_quadrature
                ? FaceQuad - 1 - g : g;
            double left_rho = 0.0;
            double right_rho = 0.0;
            double left_qsa = 0.0;
            double right_qsa = 0.0;
            for (int k = 0; k < dimPk; ++k) {
                const double left_basis = S.phi_face[0][g][k];
                const double right_basis =
                    S.phi_face[neighbor.face][neighbor_g][k];
                left_rho += K(i, 0, k, 0) * left_basis;
                right_rho += K(neighbor.i, neighbor.j, k, 0) * right_basis;
                left_qsa += K(i, 0, k, 4) * left_basis;
                right_qsa += K(neighbor.i, neighbor.j, k, 4) * right_basis;
            }
            const double left_nu_tilde = left_qsa
                / std::max(left_rho, 1e-12);
            const double right_nu_tilde = right_qsa
                / std::max(right_rho, 1e-12);
            record.wake_interface_nu_tilde_jump_abs_max = std::max(
                record.wake_interface_nu_tilde_jump_abs_max,
                std::abs(left_nu_tilde - right_nu_tilde));
        }
    }
}

static dg::io::SATermBalanceRecord compute_sa_term_balance(
    SolverVariable& S,
    const Array4D<double>& K,
    const dg::io::SAResidualDiagnosticRecord& hotspot,
    int local_step,
    long long total_step,
    double pseudo_time,
    double solver_cpu_seconds,
    double cfl,
    double sa_residual_scale)
{
    Array4D<double> term_field;
    Array4D<double> interior_face_field;
    assemble_sa_term_field(S, K, term_field, interior_face_field);

    dg::io::SATermBalanceRecord record;
    record.local_step = local_step;
    record.total_step = total_step;
    record.pseudo_time = pseudo_time;
    record.solver_cpu_seconds = solver_cpu_seconds;
    record.cfl = cfl;
    record.hotspot_i = hotspot.maximum_i;
    record.hotspot_j = hotspot.maximum_j;
    record.hotspot_quadrature = hotspot.maximum_quadrature;
    record.hotspot_x = hotspot.maximum_x;
    record.hotspot_y = hotspot.maximum_y;
    record.hotspot_wall_distance = hotspot.maximum_wall_distance;
    record.hotspot_rho = hotspot.maximum_rho;
    record.hotspot_nu_tilde = hotspot.maximum_nu_tilde;
    record.hotspot_chi = hotspot.maximum_chi;
    record.hotspot_touches_wall = record.hotspot_j == 0
        && record.hotspot_i >= g_wall_i_begin
        && record.hotspot_i < g_wall_i_end;
#if defined(DG_CASE_NACA0012)
    record.hotspot_touches_wake_interface = record.hotspot_j == 0
        && (record.hotspot_i < dg::case_config::wake_pairs
            || record.hotspot_i >= Nx - dg::case_config::wake_pairs);
#endif
    if (record.hotspot_i >= 0 && record.hotspot_j >= 0) {
        for (int face = 0; face < 4; ++face) {
            if (!g_topology.face_neighbor(record.hotspot_i,
                    record.hotspot_j, face).interior) {
                record.hotspot_touches_open_boundary =
                    !record.hotspot_touches_wall;
            }
        }
    }
    populate_wake_interface_audit(S, K, record);

    if (record.hotspot_i >= 0 && record.hotspot_j >= 0
        && record.hotspot_quadrature >= 0) {
        double dominant_score = -1.0;
        for (int face = 0; face < 4; ++face) {
            if (!g_topology.face_neighbor(record.hotspot_i,
                    record.hotspot_j, face).interior) continue;
            double consistent = 0.0;
            double lifting = 0.0;
            for (int k = 0; k < dimPk; ++k) {
                const double basis = S.phi_vol[record.hotspot_quadrature][k];
                consistent += interior_face_field(record.hotspot_i,
                    record.hotspot_j, k, 2 * face) * basis;
                lifting += interior_face_field(record.hotspot_i,
                    record.hotspot_j, k, 2 * face + 1) * basis;
            }
            const double score = std::abs(consistent) + std::abs(lifting);
            if (score > dominant_score) {
                dominant_score = score;
                record.hotspot_dominant_interior_face = face;
                record.hotspot_dominant_face_consistent = consistent;
                record.hotspot_dominant_face_lifting = lifting;
                record.hotspot_dominant_face_total = consistent + lifting;
            }
        }
        populate_sa_dominant_face_jump(S, K, record);
        if constexpr (PolyOrder == 2) {
            double hotspot_nodal[9] = {};
            for (int k = 0; k < 9; ++k) {
                hotspot_nodal[k] = K(record.hotspot_i,
                    record.hotspot_j, k, 4);
            }
            const dg::Q2ModalEnergy hotspot_modal =
                dg::q2_modal_energy(hotspot_nodal);
            record.hotspot_sa_q2_fraction =
                hotspot_modal.highest_order_fraction;
            record.hotspot_sa_q2_rms_over_mean =
                hotspot_modal.highest_order_rms_over_mean;

            if (record.hotspot_dominant_interior_face >= 0) {
                const auto neighbor = g_topology.face_neighbor(
                    record.hotspot_i, record.hotspot_j,
                    record.hotspot_dominant_interior_face);
                if (neighbor.interior) {
                    record.dominant_neighbor_i = neighbor.i;
                    record.dominant_neighbor_j = neighbor.j;
                    double neighbor_nodal[9] = {};
                    for (int k = 0; k < 9; ++k) {
                        neighbor_nodal[k] = K(neighbor.i, neighbor.j, k, 4);
                    }
                    const dg::Q2ModalEnergy neighbor_modal =
                        dg::q2_modal_energy(neighbor_nodal);
                    record.dominant_neighbor_sa_q2_fraction =
                        neighbor_modal.highest_order_fraction;
                    record.dominant_neighbor_sa_q2_rms_over_mean =
                        neighbor_modal.highest_order_rms_over_mean;
                }
            }
        }
    }

    const double scale = std::max(sa_residual_scale, 1e-30);
    std::array<double, SA_TERM_COUNT> global_square{};
    std::array<double, SA_TERM_COUNT> airfoil_square{};
    std::array<double, SA_TERM_COUNT> wake_square{};
    double global_viscosity_square = 0.0;
    double airfoil_viscosity_square = 0.0;
    double wake_viscosity_square = 0.0;
    double global_error_square = 0.0;
    double global_area = 0.0;
    double airfoil_area = 0.0;
    double wake_area = 0.0;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int q = 0; q < NumQuad; ++q) {
                std::array<double, SA_TERM_COUNT> point_terms{};
                double total_rhs = 0.0;
                for (int k = 0; k < dimPk; ++k) {
                    const double basis = S.phi_vol[q][k];
                    total_rhs += S.rhs(i, j, k, 4) * basis;
                    for (int component = 0; component < SA_TERM_COUNT; ++component) {
                        point_terms[component] +=
                            term_field(i, j, k, component) * basis;
                    }
                }
                const double viscosity_total = point_terms[SA_VISCOSITY_VOLUME]
                    + point_terms[SA_VISCOSITY_INTERIOR_CONSISTENT]
                    + point_terms[SA_VISCOSITY_INTERIOR_LIFTING]
                    + point_terms[SA_VISCOSITY_WALL_FACE]
                    + point_terms[SA_VISCOSITY_OTHER_BOUNDARY];
                const double reconstruction = point_terms[SA_CONVECTION]
                    + viscosity_total + point_terms[SA_PRODUCTION]
                    - point_terms[SA_DESTRUCTION]
                    + point_terms[SA_CROSS_DIFFUSION];
                const double error = reconstruction - total_rhs;
                const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
                const double weight = S.gl_w[q / Quad1D]
                    * S.gl_w[q % Quad1D] * metric.jacobian;
                global_area += weight;
                global_error_square += error * error * weight;
                global_viscosity_square += viscosity_total * viscosity_total * weight;
                for (int component = 0; component < SA_TERM_COUNT; ++component) {
                    global_square[component] += point_terms[component]
                        * point_terms[component] * weight;
                }
#if defined(DG_CASE_NACA0012)
                constexpr int wake_cells_per_side = dg::case_config::wake_pairs;
                constexpr int airfoil_cell_begin = wake_cells_per_side;
                constexpr int airfoil_cell_end = Nx - wake_cells_per_side;
                if (j <= 3 && i >= airfoil_cell_begin && i < airfoil_cell_end) {
                    airfoil_area += weight;
                    airfoil_viscosity_square +=
                        viscosity_total * viscosity_total * weight;
                    for (int component = 0; component < SA_TERM_COUNT; ++component) {
                        airfoil_square[component] += point_terms[component]
                            * point_terms[component] * weight;
                    }
                }
                if (j <= 3 && (i < airfoil_cell_begin || i >= airfoil_cell_end)) {
                    wake_area += weight;
                    wake_viscosity_square +=
                        viscosity_total * viscosity_total * weight;
                    for (int component = 0; component < SA_TERM_COUNT; ++component) {
                        wake_square[component] += point_terms[component]
                            * point_terms[component] * weight;
                    }
                }
#endif
                if (i == record.hotspot_i && j == record.hotspot_j
                    && q == record.hotspot_quadrature) {
                    record.hotspot_terms = point_terms;
                    record.hotspot_viscosity_total = viscosity_total;
                    record.hotspot_total_rhs = total_rhs;
                    record.hotspot_reconstruction = reconstruction;
                    record.hotspot_reconstruction_error = error;

                    double state[NumEq] = {};
                    double gradient_x[NumEq] = {};
                    double gradient_y[NumEq] = {};
                    for (int k = 0; k < dimPk; ++k) {
                        const double basis = S.phi_vol[q][k];
                        for (int e = 0; e < NumEq; ++e) {
                            state[e] += K(i, j, k, e) * basis;
                            gradient_x[e] += S.gradUX(i, j, k, e) * basis;
                            gradient_y[e] += S.gradUY(i, j, k, e) * basis;
                        }
                    }
                    const dg::SASourceTerms<double> source = SA_Source_Terms(
                        state, gradient_x, gradient_y, metric.x, metric.y);
                    record.point_source_production = source.production;
                    record.point_source_destruction = source.destruction;
                    record.point_source_cross_diffusion = source.cross_diffusion;
                    record.point_source_net = source.total;

                    const double rho = std::max(state[0], 1e-12);
                    const double u = state[1] / rho;
                    const double v = state[2] / rho;
                    const double du_dy = (gradient_y[1]
                        - u * gradient_y[0]) / rho;
                    const double dv_dx = (gradient_x[2]
                        - v * gradient_x[0]) / rho;
                    record.hotspot_vorticity = std::abs(dv_dx - du_dy);
                    const double mu_lam = laminar_mu_from_cons(state);
                    const double mu_t = sa_eddy_mu_from_rho_nut(rho,
                        state[4] / rho, mu_lam);
                    const double mu_scale = std::max(mu_ref, 1e-30);
                    record.hotspot_mu_lam_over_mu_inf = mu_lam / mu_scale;
                    record.hotspot_mu_t_over_mu_inf = mu_t / mu_scale;
                    record.hotspot_mu_effective_over_mu_inf =
                        (mu_lam + mu_t) / mu_scale;

                    double sum_abs = 0.0;
                    double dominant_abs = -1.0;
                    for (int component = 0; component < SA_TERM_COUNT;
                            ++component) {
                        const double magnitude = std::abs(
                            point_terms[component]);
                        sum_abs += magnitude;
                        if (magnitude > dominant_abs) {
                            dominant_abs = magnitude;
                            record.hotspot_dominant_component = component;
                        }
                    }
                    if (sum_abs > 0.0) {
                        record.hotspot_dominant_abs_fraction =
                            dominant_abs / sum_abs;
                        record.hotspot_cancellation_ratio =
                            std::abs(reconstruction) / sum_abs;
                    }
                }
            }
        }
    }

    for (int component = 0; component < SA_TERM_COUNT; ++component) {
        record.global_l2_scaled[component] =
            std::sqrt(global_square[component] / std::max(global_area, 1e-300)) / scale;
        if (airfoil_area > 0.0) {
            record.airfoil_band_l2_scaled[component] =
                std::sqrt(airfoil_square[component] / airfoil_area) / scale;
        }
        if (wake_area > 0.0) {
            record.wake_band_l2_scaled[component] =
                std::sqrt(wake_square[component] / wake_area) / scale;
        }
    }
    record.global_viscosity_total_l2_scaled =
        std::sqrt(global_viscosity_square / std::max(global_area, 1e-300)) / scale;
    if (airfoil_area > 0.0) {
        record.airfoil_band_viscosity_total_l2_scaled =
            std::sqrt(airfoil_viscosity_square / airfoil_area) / scale;
    }
    if (wake_area > 0.0) {
        record.wake_band_viscosity_total_l2_scaled =
            std::sqrt(wake_viscosity_square / wake_area) / scale;
    }
    record.global_reconstruction_error_l2_scaled =
        std::sqrt(global_error_square / std::max(global_area, 1e-300)) / scale;
    record.global_dominant_component = 0;
    for (int component = 1; component < SA_TERM_COUNT; ++component) {
        if (record.global_l2_scaled[component]
            > record.global_l2_scaled[record.global_dominant_component]) {
            record.global_dominant_component = component;
        }
    }
    return record;
}

static void populate_sa_term_growth(dg::io::SATermBalanceRecord& record,
    const std::array<double, SA_TERM_COUNT>* previous)
{
    if (previous == nullptr) return;
    double fastest_ratio = -1.0;
    for (int component = 0; component < SA_TERM_COUNT; ++component) {
        const double old_value = (*previous)[component];
        if (!std::isfinite(old_value) || old_value <= 1e-300) continue;
        const double ratio = record.global_l2_scaled[component] / old_value;
        if (std::isfinite(ratio) && ratio > fastest_ratio) {
            fastest_ratio = ratio;
            record.global_fastest_growth_component = component;
            record.global_fastest_growth_ratio = ratio;
        }
    }
}

static std::uint64_t rhs_bitwise_fingerprint(const Array4D<double>& rhs)
{
    constexpr std::uint64_t fnv_offset = 1469598103934665603ull;
    constexpr std::uint64_t fnv_prime = 1099511628211ull;
    std::uint64_t fingerprint = fnv_offset;
    for (double value : rhs.data) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 8; ++byte) {
            fingerprint ^= (bits >> (8 * byte)) & 0xffu;
            fingerprint *= fnv_prime;
        }
    }
    return fingerprint;
}

static void boundary_state_2dzp_ad(const AD45* UL, int f, double x, double y, double nx, double ny, AD45* UR) {
    (void)y;
    const PrimitiveStateAD Vi = cons_to_primitive_ad(UL);
    const dg::BoundaryCondition& condition = boundary_condition(f, x);
    if (condition.type == dg::BoundaryType::TotalInflow) {
        const PrimitiveStateAD Vb = characteristic_total_inflow_boundary_ad(Vi,
            condition.total_pressure, condition.total_temperature,
            condition.flow_angle, nx, ny);
        primitive_to_cons_ad(Vb, UR);
    }
    else if (condition.type == dg::BoundaryType::PressureOutlet) {
        const PrimitiveStateAD Vb = characteristic_pressure_outlet_boundary_ad(Vi,
            condition.static_pressure, nx, ny);
        primitive_to_cons_ad(Vb, UR);
    }
    else if (condition.type == dg::BoundaryType::Farfield) {
        const PrimitiveStateAD Vb = characteristic_farfield_boundary_ad(Vi, freestream_primitive_ad(), nx, ny);
        primitive_to_cons_ad(Vb, UR);
    }
    else if (dg::is_wall(condition.type)) {
        AD45 un, ut;
        velocity_to_normal_tangent_ad(Vi.u, Vi.v, nx, ny, un, ut);
        PrimitiveStateAD Vr = Vi;
        normal_tangent_to_velocity_ad(-un, ut, nx, ny, Vr.u, Vr.v);
        primitive_to_cons_ad(Vr, UR);
    }
}

static inline void wall_viscous_dirichlet_state_ad(const AD45* UL, AD45* UB)
{
    const AD45 rho = ad_max(UL[0], 1e-12);
    const AD45 u = UL[1] / rho;
    const AD45 v = UL[2] / rho;
    const AD45 p = ad_max(gamma1 * (UL[3] - 0.5 * rho * (u * u + v * v)), 1e-12);
    const AD45 T = p / rho;
    const AD45 rho_w = rho;
    const AD45 p_w = ad_max(rho_w * T, 1e-12);
    UB[0] = rho_w;
    UB[1] = 0.0;
    UB[2] = 0.0;
    UB[3] = p_w / gamma1;
    UB[4] = 0.0;
}

static inline bool boundary_viscous_trace_2dzp_ad(const AD45* UL, int f, double x, double y, double nx, double ny, AD45* UB)
{
    if (is_plate_wall_face(f, x)) {
        (void)y;
        wall_viscous_dirichlet_state_ad(UL, UB);
        return true;
    }
    if (is_symmetry_slip_face(f, x)) {
        (void)y;
        const PrimitiveStateAD Vi = cons_to_primitive_ad(UL);
        PrimitiveStateAD Vr = Vi;
        AD45 un, ut;
        velocity_to_normal_tangent_ad(Vi.u, Vi.v, nx, ny, un, ut);
        normal_tangent_to_velocity_ad(-un, ut, nx, ny, Vr.u, Vr.v);
        primitive_to_cons_ad(Vr, UB);
        return false;
    }
    (void)y;
    for (int e = 0; e < NumEq; ++e) UB[e] = UL[e];
    return false;
}

static void HLLC_Flux_ad(const AD45* UL, const AD45* UR, double nx, double ny, AD45* flux) {
    auto get_prim = [](const AD45* U, AD45& r, AD45& u, AD45& v, AD45& p) {
        r = ad_max(U[0], 1e-12);
        u = U[1] / r;
        v = U[2] / r;
        p = ad_max(gamma1 * (U[3] - 0.5 * r * (u * u + v * v)), 1e-12);
        };
    AD45 rL, uL, vL, pL, rR, uR, vR, pR;
    get_prim(UL, rL, uL, vL, pL);
    get_prim(UR, rR, uR, vR, pR);
    AD45 unL = uL * nx + vL * ny, unR = uR * nx + vR * ny;
    AD45 aL = ad_sqrt(gamma_v * pL / rL), aR = ad_sqrt(gamma_v * pR / rR);
    AD45 SL = ad_min(unL - aL, unR - aR);
    AD45 SR = ad_max(unL + aL, unR + aR);
    if (SL.v >= 0.0) {
        flux[0] = UL[0] * unL; flux[1] = UL[1] * unL + pL * nx; flux[2] = UL[2] * unL + pL * ny; flux[3] = (UL[3] + pL) * unL;
    }
    else if (SR.v <= 0.0) {
        flux[0] = UR[0] * unR; flux[1] = UR[1] * unR + pR * nx; flux[2] = UR[2] * unR + pR * ny; flux[3] = (UR[3] + pR) * unR;
    }
    else {
        AD45 denom = rL * (SL - unL) - rR * (SR - unR);
        if (std::abs(denom.v) < 1e-10) denom.v = (denom.v >= 0 ? 1e-10 : -1e-10);
        AD45 Sstar = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / denom;
        if (Sstar.v >= 0.0) {
            AD45 dSL = SL - Sstar;
            if (std::abs(dSL.v) < 1e-10) dSL.v = (dSL.v >= 0 ? 1e-10 : -1e-10);
            AD45 f = rL * (SL - unL) / dSL;
            AD45 dSL_un = SL - unL;
            if (std::abs(dSL_un.v) < 1e-10) dSL_un.v = (dSL_un.v >= 0 ? 1e-10 : -1e-10);
            AD45 Us[4] = { f, f * (uL + (Sstar - unL) * nx), f * (vL + (Sstar - unL) * ny), f * (UL[3] / rL + (Sstar - unL) * (Sstar + pL / (rL * dSL_un))) };
            flux[0] = UL[0] * unL + SL * (Us[0] - UL[0]); flux[1] = (UL[1] * unL + pL * nx) + SL * (Us[1] - UL[1]); flux[2] = (UL[2] * unL + pL * ny) + SL * (Us[2] - UL[2]); flux[3] = (UL[3] + pL) * unL + SL * (Us[3] - UL[3]);
        }
        else {
            AD45 dSR = SR - Sstar;
            if (std::abs(dSR.v) < 1e-10) dSR.v = (dSR.v >= 0 ? 1e-10 : -1e-10);
            AD45 f = rR * (SR - unR) / dSR;
            AD45 dSR_un = SR - unR;
            if (std::abs(dSR_un.v) < 1e-10) dSR_un.v = (dSR_un.v >= 0 ? 1e-10 : -1e-10);
            AD45 Us[4] = { f, f * (uR + (Sstar - unR) * nx), f * (vR + (Sstar - unR) * ny), f * (UR[3] / rR + (Sstar - unR) * (Sstar + pR / (rR * dSR_un))) };
            flux[0] = UR[0] * unR + SR * (Us[0] - UR[0]); flux[1] = (UR[1] * unR + pR * nx) + SR * (Us[1] - UR[1]); flux[2] = (UR[2] * unR + pR * ny) + SR * (Us[2] - UR[2]); flux[3] = (UR[3] + pR) * unR + SR * (Us[3] - UR[3]);
        }
    }
}

static inline void Viscous_Flux_ad(
    const AD45* U, const AD45* dUx, const AD45* dUy, AD45* Fv, AD45* Gv) {
    dg::viscous_flux<AD45, dg::BlockADMath<BlockSize>>(
        U, dUx, dUy, g_viscous_sa_parameters, Fv, Gv);
}

static inline AD45 SA_Source_ad(
    const AD45* U, const AD45* dUx, const AD45* dUy, double x, double y) {
    const double ramp = sa_source_ramp(x);
    if (ramp <= 0.0) return AD45(0.0);
    const double d = wall_distance(x, y);
    const double d_destroy = sa_destruction_distance(d);
    const auto terms = dg::sa_source_terms<AD45,
        dg::BlockADMath<BlockSize>>(U, dUx, dUy, d, d_destroy,
            g_viscous_sa_parameters);
    return ramp * terms.total;
}

static void computeFaceLifting_ad(const SolverVariable& S, const AD45* jump, int f, int i_idx, int j_idx, AD45* rX, AD45* rY, double scale) {
    std::array<AD45, dimPk * NumEq> bX = {}, bY = {};
    for (int g = 0; g < FaceQuad; g++) {
        double nx = 0.0, ny = 0.0, ds = 0.0;
        face_metric(S, i_idx, j_idx, f, g, nx, ny, ds);
        for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
            const AD45 face_rhs = S.gl_w[g] * jump[g * NumEq + e] * S.phi_face[f][g][k] * ds;
            bX[k * NumEq + e] += face_rhs * nx;
            bY[k * NumEq + e] += face_rhs * ny;
        }
    }
    for (int e = 0; e < NumEq; e++) for (int k = 0; k < dimPk; k++) {
        AD45 accX = 0.0, accY = 0.0;
        for (int m = 0; m < dimPk; m++) {
            const double Minv = cell_inv_mass_entry(S, i_idx, j_idx, k, m);
            accX += Minv * bX[m * NumEq + e];
            accY += Minv * bY[m * NumEq + e];
        }
        rX[k * NumEq + e] = accX * scale;
        rY[k * NumEq + e] = accY * scale;
    }
}

static void computeGradients_ad(const SolverVariable& S, const Array4D<AD45>& Kin, Array4D<AD45>& gradUX, Array4D<AD45>& gradUY) {
#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int cell = 0; cell < NumCells; ++cell) {
        const int i = cell / Ny;
        const int j = cell % Ny;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            gradUX(i, j, k, e) = AD45{};
            gradUY(i, j, k, e) = AD45{};
        }
        for (int q = 0; q < NumQuad; q++) {
            AD45 uq[NumEq] = {};
            for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                uq[e] += Kin(i, j, k, e) * S.phi_vol[q][k];
            }
            const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
            const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
            const double xr = metric.xr;
            const double xs = metric.xs;
            const double yr = metric.yr;
            const double ys = metric.ys;
            for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                const double grad_phi_x_J = S.dphi_dr[q][k] * ys - S.dphi_ds[q][k] * yr;
                const double grad_phi_y_J = -S.dphi_dr[q][k] * xs + S.dphi_ds[q][k] * xr;
                gradUX(i, j, k, e) = gradUX(i, j, k, e) - w * uq[e] * grad_phi_x_J;
                gradUY(i, j, k, e) = gradUY(i, j, k, e) - w * uq[e] * grad_phi_y_J;
            }
        }
        for (int f = 0; f < 4; f++) {
            const auto neighbor = g_topology.face_neighbor(i, j, f);
            for (int g = 0; g < FaceQuad; g++) {
                double nx = 0.0, ny = 0.0, ds = 0.0;
                face_metric(S, i, j, f, g, nx, ny, ds);
                AD45 uL[NumEq] = {}, uR[NumEq] = {};
                bool strict_wall_viscous = false;
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) uL[e] = uL[e] + Kin(i, j, k, e) * S.phi_face[f][g][k];
                const int ni = neighbor.i;
                const int nj = neighbor.j;
                const bool isInteriorFace = neighbor.interior;
                if (isInteriorFace) {
                    const int nf = neighbor.face;
                    const int ng = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
                    for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) uR[e] = uR[e] + Kin(ni, nj, k, e) * S.phi_face[nf][ng][k];
                }
                else {
                    double xFace, yFace;
                    face_point_coords(S, i, j, f, g, xFace, yFace);
                    strict_wall_viscous = boundary_viscous_trace_2dzp_ad(uL, f, xFace, yFace, nx, ny, uR);
                }
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                    AD45 u_avg = 0.5 * (uL[e] + uR[e]);
                    if (!isInteriorFace && strict_wall_viscous) {
                        u_avg = uR[e];
                    }
                    gradUX(i, j, k, e) = gradUX(i, j, k, e) + S.gl_w[g] * u_avg * nx * S.phi_face[f][g][k] * ds;
                    gradUY(i, j, k, e) = gradUY(i, j, k, e) + S.gl_w[g] * u_avg * ny * S.phi_face[f][g][k] * ds;
                }
            }
        }
        for (int e = 0; e < NumEq; e++) {
            AD45 bx[dimPk], by[dimPk];
            for (int k = 0; k < dimPk; k++) { bx[k] = gradUX(i, j, k, e); by[k] = gradUY(i, j, k, e); }
            for (int k = 0; k < dimPk; k++) {
                AD45 ax = 0.0, ay = 0.0;
                for (int m = 0; m < dimPk; m++) {
                    const double Minv = cell_inv_mass_entry(S, i, j, k, m);
                    ax += Minv * bx[m];
                    ay += Minv * by[m];
                }
                gradUX(i, j, k, e) = ax;
                gradUY(i, j, k, e) = ay;
            }
        }
    }
}

static void computeRHS_ad(SolverVariable& S, const Array4D<AD45>& Kin,
    Array4D<AD45>& rhs, Array4D<AD45>& gradUX, Array4D<AD45>& gradUY)
{
    computeGradients_ad(S, Kin, gradUX, gradUY);
#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int cell = 0; cell < NumCells; ++cell) {
        const int i = cell / Ny;
        const int j = cell % Ny;
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            rhs(i, j, k, e) = AD45{};
        }
        for (int q = 0; q < NumQuad; q++) {
            std::array<AD45, NumEq> uq = {}, dux = {}, duy = {};
            for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                uq[e] += Kin(i, j, k, e) * S.phi_vol[q][k];
                dux[e] += gradUX(i, j, k, e) * S.phi_vol[q][k];
                duy[e] += gradUY(i, j, k, e) * S.phi_vol[q][k];
            }
            AD45 r = ad_max(uq[0], 1e-12), u = uq[1] / r, v = uq[2] / r, p = ad_max(gamma1 * (uq[3] - 0.5 * r * (u * u + v * v)), 1e-12);
            std::array<AD45, NumEq> Fc = {}, Gc = {}, Fv = {}, Gv = {};
            Fc[0] = uq[1]; Fc[1] = r * u * u + p; Fc[2] = r * u * v; Fc[3] = u * (uq[3] + p); Fc[4] = uq[4] * u;
            Gc[0] = uq[2]; Gc[1] = r * u * v; Gc[2] = r * v * v + p; Gc[3] = v * (uq[3] + p); Gc[4] = uq[4] * v;
            Viscous_Flux_ad(uq.data(), dux.data(), duy.data(), Fv.data(), Gv.data());
            const double w = S.gl_w[q / Quad1D] * S.gl_w[q % Quad1D];
            const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
            const double x = metric.x;
            const double y = metric.y;
            const double xr = metric.xr;
            const double xs = metric.xs;
            const double yr = metric.yr;
            const double ys = metric.ys;
            const double J = metric.jacobian;
            AD45 srcSA = SA_Source_ad(uq.data(), dux.data(), duy.data(), x, y);
            for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                const double grad_phi_x_J = S.dphi_dr[q][k] * ys - S.dphi_ds[q][k] * yr;
                const double grad_phi_y_J = -S.dphi_dr[q][k] * xs + S.dphi_ds[q][k] * xr;
                const AD45 termV = w * (-Fv[e] * grad_phi_x_J - Gv[e] * grad_phi_y_J);
                rhs(i, j, k, e) = rhs(i, j, k, e)
                    + w * (Fc[e] * grad_phi_x_J + Gc[e] * grad_phi_y_J)
                    + termV
                    + ((e == 4) ? (w * srcSA * S.phi_vol[q][k] * J) : AD45(0.0));
            }
        }
        for (int f = 0; f < 4; f++) {
            std::array<AD45, FaceQuad * NumEq> uFL = {}, uFR_conv = {},
                uFR_visc = {}, jump = {};
            bool strict_wall_viscous[FaceQuad] = {};
            bool zero_viscous_flux_boundary[FaceQuad] = {};
            const auto neighbor = g_topology.face_neighbor(i, j, f);
            const int ni = neighbor.i;
            const int nj = neighbor.j;
            const bool isInteriorFace = neighbor.interior;
            const int nf = neighbor.face;
            auto face_idx = [](int g, int e) { return g * NumEq + e; };
            for (int g = 0; g < FaceQuad; g++) {
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                    uFL[face_idx(g, e)] += Kin(i, j, k, e) * S.phi_face[f][g][k];
                }
                if (isInteriorFace) {
                    const int ng = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
                    for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                        uFR_conv[face_idx(g, e)] += Kin(ni, nj, k, e) * S.phi_face[nf][ng][k];
                    }
                    for (int e = 0; e < NumEq; ++e) uFR_visc[face_idx(g, e)] = uFR_conv[face_idx(g, e)];
                }
                else {
                    double xFace, yFace;
                    face_point_coords(S, i, j, f, g, xFace, yFace);
                    double nx = 0.0, ny = 0.0, ds = 0.0;
                    face_metric(S, i, j, f, g, nx, ny, ds);
                    boundary_state_2dzp_ad(&uFL[face_idx(g, 0)], f, xFace, yFace, nx, ny, &uFR_conv[face_idx(g, 0)]);
                    strict_wall_viscous[g] = boundary_viscous_trace_2dzp_ad(&uFL[face_idx(g, 0)], f, xFace, yFace, nx, ny, &uFR_visc[face_idx(g, 0)]);
                    zero_viscous_flux_boundary[g] = is_zero_viscous_flux_face(f, xFace);
                }
                for (int e = 0; e < NumEq; e++) jump[face_idx(g, e)] = uFL[face_idx(g, e)] - uFR_visc[face_idx(g, e)];
                if (!isInteriorFace && strict_wall_viscous[g]) {
                    for (int e = 1; e <= 3; ++e) {
                        jump[face_idx(g, e)] = jump[face_idx(g, e)] * BR2_WALL_BOUNDARY_TRACE_FACTOR * NS_WALL_BR2_SCALE;
                    }
                    jump[face_idx(g, 4)] = jump[face_idx(g, 4)] * BR2_WALL_BOUNDARY_TRACE_FACTOR * SA_WALL_BR2_SCALE;
                }
                jump[face_idx(g, 4)] = jump[face_idx(g, 4)]
                    * g_sa_br2_factor;
            }
            std::array<AD45, dimPk * NumEq> rX = {}, rY = {};
            computeFaceLifting_ad(S, jump.data(), f, i, j, rX.data(), rY.data(), BR2_ETA);
            std::array<AD45, dimPk * NumEq> rX_adj = {}, rY_adj = {};
            if (isInteriorFace) {
                std::array<AD45, FaceQuad * NumEq> jmp_adj = {};
                for (int gg = 0; gg < FaceQuad; gg++) {
                    const int neighbor_g = neighbor.reverse_quadrature ? FaceQuad - 1 - gg : gg;
                    for (int ee = 0; ee < NumEq; ee++) {
                        jmp_adj[face_idx(neighbor_g, ee)] = -jump[face_idx(gg, ee)];
                    }
                }
                computeFaceLifting_ad(S, jmp_adj.data(), nf, ni, nj, rX_adj.data(), rY_adj.data(), BR2_ETA);
            }
            for (int g = 0; g < FaceQuad; g++) {
                double nx = 0.0, ny = 0.0, ds = 0.0;
                face_metric(S, i, j, f, g, nx, ny, ds);
                std::array<AD45, NumEq> uL = {}, uR_conv_g = {}, uR_visc_g = {};
                std::array<AD45, NumEq> duxL = {}, duyL = {}, duxR = {}, duyR = {};
                std::array<AD45, NumEq> fluxC = {}, Fv = {}, Gv = {},
                    u_avg = {}, dux_avg = {}, duy_avg = {}, viscous_normal = {};
                for (int e = 0; e < NumEq; ++e) {
                    uL[e] = uFL[face_idx(g, e)];
                    uR_conv_g[e] = uFR_conv[face_idx(g, e)];
                    uR_visc_g[e] = uFR_visc[face_idx(g, e)];
                }
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                    duxL[e] += (gradUX(i, j, k, e) - rX[k * NumEq + e]) * S.phi_face[f][g][k];
                    duyL[e] += (gradUY(i, j, k, e) - rY[k * NumEq + e]) * S.phi_face[f][g][k];
                }
                if (isInteriorFace) {
                    const int neighbor_g = neighbor.reverse_quadrature ? FaceQuad - 1 - g : g;
                    for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                        duxR[e] += (gradUX(ni, nj, k, e) - rX_adj[k * NumEq + e]) * S.phi_face[nf][neighbor_g][k];
                        duyR[e] += (gradUY(ni, nj, k, e) - rY_adj[k * NumEq + e]) * S.phi_face[nf][neighbor_g][k];
                    }
                }
                else {
                    for (int e = 0; e < NumEq; e++) { duxR[e] = duxL[e]; duyR[e] = duyL[e]; }
                }
                AD45 fluxEuler[4];
                HLLC_Flux_ad(uL.data(), uR_conv_g.data(), nx, ny, fluxEuler);
                for (int e = 0; e < 4; e++) fluxC[e] = fluxEuler[e];
                AD45 rhoL = ad_max(uL[0], 1e-12), rhoR = ad_max(uR_conv_g[0], 1e-12);
                AD45 nutL = uL[4] / rhoL, nutR = uR_conv_g[4] / rhoR;
                const AD45 mdot_hllc = fluxEuler[0];
                fluxC[4] = mdot_hllc * (mdot_hllc.v >= 0.0 ? nutL : nutR);
                if (!isInteriorFace && strict_wall_viscous[g]) {
                    for (int e = 0; e < NumEq; e++) {
                        u_avg[e] = uR_visc_g[e];
                        dux_avg[e] = duxL[e];
                        duy_avg[e] = duyL[e];
                    }
                }
                else {
                    for (int e = 0; e < NumEq; e++) {
                        u_avg[e] = 0.5 * (uL[e] + uR_visc_g[e]);
                        dux_avg[e] = 0.5 * (duxL[e] + duxR[e]);
                        duy_avg[e] = 0.5 * (duyL[e] + duyR[e]);
                    }
                }
                Viscous_Flux_ad(u_avg.data(), dux_avg.data(), duy_avg.data(), Fv.data(), Gv.data());
                for (int e = 0; e < NumEq; ++e) viscous_normal[e] = Fv[e] * nx + Gv[e] * ny;
                if (!isInteriorFace && zero_viscous_flux_boundary[g]) {
                    for (int e = 0; e < NumEq; ++e) viscous_normal[e] = 0.0;
                }
                if (!isInteriorFace && strict_wall_viscous[g]) {
                    viscous_normal[3] = 0.0;
                }
                for (int k = 0; k < dimPk; k++) for (int e = 0; e < NumEq; e++) {
                    const AD45 faceV = S.gl_w[g] * (viscous_normal[e]) * S.phi_face[f][g][k] * ds;
                    rhs(i, j, k, e) = rhs(i, j, k, e)
                        - S.gl_w[g] * fluxC[e] * S.phi_face[f][g][k] * ds
                        + faceV;
                }
            }
        }
        for (int e = 0; e < NumEq; e++) {
            std::array<AD45, dimPk> b = {};
            for (int k = 0; k < dimPk; k++) b[k] = rhs(i, j, k, e);
            for (int k = 0; k < dimPk; k++) {
                AD45 acc = 0.0;
                for (int m = 0; m < dimPk; m++) acc = acc + cell_inv_mass_entry(S, i, j, k, m) * b[m];
                rhs(i, j, k, e) = acc;
            }
        }
    }
}

static void compute_be_residual_ad(SolverVariable& S,
    const Array4D<AD45>& Kstar,
    const Array4D<double>& Kn,
    const std::vector<double>& dtau,
    double sa_cfl_scale,
    Array4D<AD45>& F,
    Array4D<AD45>& rhs_ad,
    Array4D<AD45>& gradUX,
    Array4D<AD45>& gradUY)
{
    computeRHS_ad(S, Kstar, rhs_ad, gradUX, gradUY);
#if DG_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int cell = 0; cell < NumCells; ++cell) {
        const int i = cell / Ny;
        const int j = cell % Ny;
        const double dti = dtau[cell];
        for (int k = 0; k < dimPk; ++k) for (int e = 0; e < NumEq; ++e) {
            const double equation_dt = e == 4
                ? sa_cfl_scale * dti : dti;
            F(i, j, k, e) = Kstar(i, j, k, e) - Kn(i, j, k, e)
                - equation_dt * rhs_ad(i, j, k, e);
        }
    }
}

// Bounded positivity limiter for NS variables; SA-neg allows locally negative nu_tilde.
dg::SALimiterReport apply_limiters(SolverVariable& S, Array4D<double>& Ktar,
    bool limit_negative_sa) {
    dg::SALimiterReport report;
    const double rho_floor = 1e-10;
    const double p_floor = 1e-11;

    auto pressure = [](const double* U) {
        const double rho = std::max(U[0], 1e-30);
        const double u = U[1] / rho;
        const double v = U[2] / rho;
        return gamma1 * (U[3] - 0.5 * rho * (u * u + v * v));
        };

    auto eval_with_phi = [&](const Array4D<double>& K, int i, int j, const double* phi, double* U) {
        for (int e = 0; e < NumEq; ++e) U[e] = 0.0;
        for (int k = 0; k < dimPk; ++k) {
            const double ph = phi[k];
            for (int e = 0; e < NumEq; ++e) U[e] += K(i, j, k, e) * ph;
        }
        };

    auto theta_for_pressure = [&](const double* Uavg, const double* Uq) {
        const double p_q = pressure(Uq);
        if (std::isfinite(p_q) && p_q >= p_floor) return 1.0;
        double lo = 0.0, hi = 1.0;
        double Ut[NumEq];
        for (int it = 0; it < 48; ++it) {
            const double mid = 0.5 * (lo + hi);
            for (int e = 0; e < NumEq; ++e) Ut[e] = Uavg[e] + mid * (Uq[e] - Uavg[e]);
            const double p_mid = pressure(Ut);
            if (std::isfinite(p_mid) && p_mid >= p_floor) lo = mid;
            else hi = mid;
        }
        return lo;
        };

    for (int i = 0; i < Nx; i++) for (int j = 0; j < Ny; j++) {
        double Uavg[NumEq];
        cell_average(S, Ktar, i, j, Uavg);

        double theta_ns = 1.0;
        const double rho_avg = Uavg[0];
        const double p_avg = pressure(Uavg);
        bool ns_avg_ok = std::isfinite(rho_avg) && std::isfinite(p_avg)
            && rho_avg > rho_floor && p_avg > p_floor;

        auto visit_state_ns = [&](const double* Uq) {
            for (int e = 0; e < 4; ++e) {
                if (!std::isfinite(Uq[e])) {
                    theta_ns = 0.0;
                    return;
                }
            }
            if (Uq[0] < rho_floor) {
                theta_ns = std::min(theta_ns, (rho_avg - rho_floor) / (rho_avg - Uq[0] + 1e-30));
            }
            theta_ns = std::min(theta_ns, theta_for_pressure(Uavg, Uq));
            };

        double Uq[NumEq];
        if (ns_avg_ok) {
            for (int k = 0; k < dimPk; ++k) {
                for (int e = 0; e < NumEq; ++e) Uq[e] = Ktar(i, j, k, e);
                visit_state_ns(Uq);
            }
            for (int q = 0; q < NumQuad; ++q) {
                eval_with_phi(Ktar, i, j, S.phi_vol[q], Uq);
                visit_state_ns(Uq);
            }
            for (int f = 0; f < 4; ++f) for (int g = 0; g < FaceQuad; ++g) {
                eval_with_phi(Ktar, i, j, S.phi_face[f][g], Uq);
                visit_state_ns(Uq);
            }

            theta_ns = std::max(0.0, std::min(1.0, theta_ns));
            if (theta_ns < 1.0) {
                ++g_limiter_ns_hits;
                for (int k = 0; k < dimPk; ++k) {
                    for (int e = 0; e < 4; ++e) {
                        Ktar(i, j, k, e) = Uavg[e] + theta_ns * (Ktar(i, j, k, e) - Uavg[e]);
                    }
                }
                cell_average(S, Ktar, i, j, Uavg);
            }
        }

        cell_average(S, Ktar, i, j, Uavg);
        const double rho_sa_avg = std::max(Uavg[0], rho_floor);
        const double nu_lam_avg = laminar_mu_from_cons(Uavg) / rho_sa_avg;
        const double nut_floor_avg = -SA_LIMITER_AVG_FLOOR_FACTOR * SA_CN1 * nu_lam_avg;
        const double q_floor_avg = rho_sa_avg * nut_floor_avg;
        bool sa_cell_limited = false;
        if (!std::isfinite(Uavg[4])
            || (limit_negative_sa && Uavg[4] < q_floor_avg)) {
            ++g_nut_posfix;
            ++g_limiter_sa_hits;
            const double q_cur = std::isfinite(Uavg[4]) ? Uavg[4] : 0.0;
            const double correction = q_floor_avg - q_cur;
            add_constant_to_cell_component(Ktar, i, j, 4, correction);
            for (int k = 0; k < dimPk; ++k) report.record_change(correction);
            sa_cell_limited = true;
            ++report.average_corrections;
            if (report.minimum_theta > 0.0) {
                double x = 0.0, y = 0.0;
                physical_coords(S, i, j, 0.0, 0.0, x, y);
                report.minimum_theta = 0.0;
                report.worst_i = i;
                report.worst_j = j;
                report.worst_kind = dg::SASampleKind::cell_average;
                report.worst_local_index = -1;
                report.worst_face = -1;
                report.worst_x = x;
                report.worst_y = y;
                report.worst_rho = rho_sa_avg;
                report.worst_nu_lam = nu_lam_avg;
                report.worst_nu_tilde = q_cur / rho_sa_avg;
                report.worst_chi = report.worst_nu_tilde / std::max(nu_lam_avg, 1e-30);
                report.worst_floor = nut_floor_avg;
            }
            cell_average(S, Ktar, i, j, Uavg);
        }

        double theta_sa = 1.0;
        dg::SASampleKind cell_worst_kind = dg::SASampleKind::none;
        int cell_worst_local_index = -1;
        int cell_worst_face = -1;
        double cell_worst_x = std::numeric_limits<double>::quiet_NaN();
        double cell_worst_y = std::numeric_limits<double>::quiet_NaN();
        double cell_worst_rho = std::numeric_limits<double>::quiet_NaN();
        double cell_worst_nu_lam = std::numeric_limits<double>::quiet_NaN();
        double cell_worst_nut = std::numeric_limits<double>::quiet_NaN();
        double cell_worst_floor = std::numeric_limits<double>::quiet_NaN();
        auto visit_state_sa = [&](const double* Uq, dg::SASampleKind kind,
            int local_index, int face, double x, double y) {
            if (!std::isfinite(Uq[4]) || !std::isfinite(Uq[0]) || Uq[0] <= rho_floor) {
                theta_sa = 0.0;
                cell_worst_kind = kind;
                cell_worst_local_index = local_index;
                cell_worst_face = face;
                cell_worst_x = x;
                cell_worst_y = y;
                return;
            }
            const double rho_q = std::max(Uq[0], rho_floor);
            const double nu_lam_q = laminar_mu_from_cons(Uq) / rho_q;
            const double nut_hard_floor =
                -SA_LIMITER_POINT_HARD_FLOOR_FACTOR * SA_CN1 * nu_lam_q;
            const double q_hard_floor = rho_q * nut_hard_floor;
            const double q_tol = SA_LIMITER_POINT_REL_TOL *
                std::max({ std::abs(Uavg[4]), std::abs(q_hard_floor), rho_q * nu_lam_q, 1e-30 });
            if (limit_negative_sa && Uq[4] < q_hard_floor - q_tol) {
                const double theta_candidate = (Uavg[4] - q_hard_floor) / (Uavg[4] - Uq[4] + 1e-30);
                if (theta_candidate < theta_sa) {
                    theta_sa = theta_candidate;
                    cell_worst_kind = kind;
                    cell_worst_local_index = local_index;
                    cell_worst_face = face;
                    cell_worst_x = x;
                    cell_worst_y = y;
                    cell_worst_rho = rho_q;
                    cell_worst_nu_lam = nu_lam_q;
                    cell_worst_nut = Uq[4] / rho_q;
                    cell_worst_floor = nut_hard_floor;
                }
            }
            };

        for (int k = 0; k < dimPk; ++k) {
            for (int e = 0; e < NumEq; ++e) Uq[e] = Ktar(i, j, k, e);
            double r = 0.0, s = 0.0, x = 0.0, y = 0.0;
            reference_node_coords(k, r, s);
            physical_coords(S, i, j, r, s, x, y);
            visit_state_sa(Uq, dg::SASampleKind::nodal_dof, k, -1, x, y);
        }
        for (int q = 0; q < NumQuad; ++q) {
            eval_with_phi(Ktar, i, j, S.phi_vol[q], Uq);
            const dg::VolumeMetric& metric = S.metric_cache.volume(i, j, q);
            visit_state_sa(Uq, dg::SASampleKind::volume_quadrature, q, -1,
                metric.x, metric.y);
        }
        for (int f = 0; f < 4; ++f) for (int g = 0; g < FaceQuad; ++g) {
            eval_with_phi(Ktar, i, j, S.phi_face[f][g], Uq);
            const dg::FaceQuadratureMetric& metric = S.metric_cache.face(i, j, f, g);
            visit_state_sa(Uq, dg::SASampleKind::face_quadrature, g, f,
                metric.x, metric.y);
        }
        theta_sa = std::max(0.0, std::min(1.0, theta_sa));
        if (theta_sa < 1.0) {
            ++g_limiter_sa_hits;
            ++report.point_corrections;
            sa_cell_limited = true;
            if (theta_sa < report.minimum_theta) {
                report.minimum_theta = theta_sa;
                report.worst_i = i;
                report.worst_j = j;
                report.worst_kind = cell_worst_kind;
                report.worst_local_index = cell_worst_local_index;
                report.worst_face = cell_worst_face;
                report.worst_x = cell_worst_x;
                report.worst_y = cell_worst_y;
                report.worst_rho = cell_worst_rho;
                report.worst_nu_lam = cell_worst_nu_lam;
                report.worst_nu_tilde = cell_worst_nut;
                report.worst_chi = cell_worst_nut /
                    std::max(cell_worst_nu_lam, 1e-30);
                report.worst_floor = cell_worst_floor;
            }
            for (int k = 0; k < dimPk; ++k) {
                const double old_value = Ktar(i, j, k, 4);
                const double new_value = Uavg[4]
                    + theta_sa * (old_value - Uavg[4]);
                report.record_change(new_value - old_value);
                Ktar(i, j, k, 4) = new_value;
            }
        }
        if (sa_cell_limited) ++report.limited_cells;
    }
    return report;
}


// Local pseudo-time BE main loop with damped Newton, AD Jacobian, GMRES, and block ILU(0).
int dg::run_solver(RunMode mode, const SolverOptions& options) {
    if (!std::isfinite(options.steady_ns_tolerance)
        || options.steady_ns_tolerance <= 0.0
        || !std::isfinite(options.steady_sa_tolerance)
        || options.steady_sa_tolerance <= 0.0) {
        std::cerr << "Steady NS and SA tolerances must be finite positive numbers."
            << std::endl;
        return 2;
    }
    if (!std::isfinite(options.sa_cfl_scale)
        || options.sa_cfl_scale <= 0.0) {
        std::cerr << "SA CFL scale must be a finite positive number."
            << std::endl;
        return 2;
    }
    if (!std::isfinite(options.sa_br2_factor)
        || options.sa_br2_factor <= 0.0) {
        std::cerr << "SA BR2 factor must be a finite positive number."
            << std::endl;
        return 2;
    }
    g_sa_br2_factor = options.sa_br2_factor;
    if (options.sa_q2_reset_on_restart && options.restart_file.empty()) {
        std::cerr << "SA Q2 reset requires a restart file." << std::endl;
        return 2;
    }
    if constexpr (PolyOrder != 2) {
        if (options.sa_q2_reset_on_restart) {
            std::cerr << "SA Q2 reset requires a Q2 solver build." << std::endl;
            return 2;
        }
    }
    configure_flow_conditions(options);
    if (options.limit_negative_sa) {
        std::cerr << "[WARN] SA floor limiter is a legacy diagnostic mode. "
            << "It post-processes accepted SA-neg states and may prevent "
            << "nonlinear convergence; use finite-only for validation runs."
            << std::endl;
    }
    const bool benchmark_mode = mode == RunMode::benchmark;
    const bool profile_mode = mode == RunMode::profile;
    const bool check_mode = mode == RunMode::check;
    const bool residual_check_mode = mode == RunMode::residual_check;
    const bool jacobian_check_mode = mode == RunMode::jacobian_check;
    const bool output_enabled = mode == RunMode::solve;
    const int steady_step_limit = options.requested_steps > 0
        ? options.requested_steps
        : STEADY_MAX_STEPS;
    // Phase 1: read the grid, precompute DG data, and create the freestream initial state.
    SolverVariable solver; init_solver(solver);
    ensure_topology_safe_ad_coloring();
    const std::uint64_t checkpoint_signature =
        compute_checkpoint_signature(solver);
    if (check_mode) {
        const auto mesh_x_bounds = std::minmax_element(
            solver.grid_x.begin(), solver.grid_x.end());
        const auto mesh_y_bounds = std::minmax_element(
            solver.grid_y.begin(), solver.grid_y.end());
        const std::size_t initialized_bytes =
            (solver.grid_x.size() + solver.grid_y.size()
                + solver.geom.data.size() + solver.K.data.size()
                + solver.rhs.data.size() + solver.gradUX.data.size()
                + solver.gradUY.data.size() + solver.inv_mass_cell.size())
                * sizeof(double)
            + solver.metric_cache.memory_bytes();
        std::cout << "[CASE CHECK] status=ok\n"
            << "[CASE CHECK] case=" << dg::case_config::name
            << " grid_family=" << dg::case_config::grid_family << "\n"
            << "[CASE CHECK] mesh_file=\"" << COMPILED_MESH_FILE << "\"\n"
            << "[CASE CHECK] cells=" << Nx << "x" << Ny
            << " points=" << (Nx + 1) << "x" << (Ny + 1)
            << " equations=" << NumEq
            << " basis_dofs_per_cell=" << dimPk << "\n"
            << "[CASE CHECK] mesh_x_bounds=" << *mesh_x_bounds.first
            << " " << *mesh_x_bounds.second
            << " mesh_y_bounds=" << *mesh_y_bounds.first
            << " " << *mesh_y_bounds.second << "\n"
            << "[CASE CHECK] viscous_wall_x_range=" << PLATE_X0
            << " " << PLATE_X1
            << " wall_point_index_range=" << g_wall_i_begin
            << " " << g_wall_i_end << "\n"
            << "[CASE CHECK] mach=" << M_inf << " reynolds=" << Re
            << " alpha_degrees="
            << inflow_angle * 180.0 / 3.14159265358979323846 << "\n"
            << "[CASE CHECK] ad_colors=" << g_ad_coloring.color_count
            << " collision_rows=" << g_ad_coloring.collision_rows
            << " collision_pairs=" << g_ad_coloring.collision_pairs << "\n"
            << "[CASE CHECK] initialized_memory_MB="
            << static_cast<double>(initialized_bytes) / (1024.0 * 1024.0)
            << " checkpoint_signature=0x" << std::hex
            << checkpoint_signature << std::dec << "\n";
#if DG_HAS_OPENMP
        std::cout << "[CASE CHECK] parallel=openmp max_threads="
            << omp_get_max_threads() << "\n";
#else
        std::cout << "[CASE CHECK] parallel=serial\n";
#endif
        return 0;
    }
    if (residual_check_mode) {
        const auto residual_start = std::chrono::steady_clock::now();
        computeRHS(solver, solver.K);
        const double residual_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - residual_start).count();
        const ResidualStats residual = compute_rhs_residual_stats(solver);
        const std::array<double, NumEq> scales = build_residual_scales_init();
        constexpr std::array<const char*, NumEq> equation_names = {
            "rho", "rho_u", "rho_v", "rho_E", "rho_nu_tilde"
        };
        bool finite_residual = true;
        double max_l2_scaled = 0.0;
        double max_linf_scaled = 0.0;
        for (int equation = 0; equation < NumEq; ++equation) {
            finite_residual = finite_residual
                && std::isfinite(residual.l2[equation])
                && std::isfinite(residual.linf[equation]);
            max_l2_scaled = std::max(max_l2_scaled,
                residual.l2[equation] / std::max(scales[equation], 1e-30));
            max_linf_scaled = std::max(max_linf_scaled,
                residual.linf[equation] / std::max(scales[equation], 1e-30));
        }
        double rho_min = 0.0;
        double p_min = 0.0;
        double nut_min = 0.0;
        const bool valid_state = cellavg_state_valid(solver, solver.K,
            &rho_min, &p_min, &nut_min);
        const WallOutputMonitor wall =
            compute_wall_output_monitor(solver, solver.K);
        const bool ok = finite_residual && valid_state;
        std::cout << std::scientific << std::setprecision(12);
        std::cout << "[RESIDUAL CHECK] status=" << (ok ? "ok" : "failed")
            << " max_l2_scaled=" << max_l2_scaled
            << " max_linf_scaled=" << max_linf_scaled
            << " rhs_fingerprint=0x" << std::hex
            << rhs_bitwise_fingerprint(solver.rhs) << std::dec << "\n";
        for (int equation = 0; equation < NumEq; ++equation) {
            std::cout << "[RESIDUAL CHECK] equation="
                << equation_names[equation]
                << " l2=" << residual.l2[equation]
                << " linf=" << residual.linf[equation]
                << " scale=" << scales[equation]
                << " l2_scaled="
                << residual.l2[equation] / std::max(scales[equation], 1e-30)
                << " linf_scaled="
                << residual.linf[equation] / std::max(scales[equation], 1e-30)
                << "\n";
        }
        std::cout << "[RESIDUAL CHECK] state_valid="
            << (valid_state ? 1 : 0)
            << " rho_min=" << rho_min
            << " p_min=" << p_min
            << " nu_tilde_min=" << nut_min << "\n";
        std::cout << "[RESIDUAL CHECK] wall_cf=" << wall.cf[0]
            << " " << wall.cf[1] << " " << wall.cf[2]
            << " mean_cf=" << wall.mean_cf << "\n";
#if defined(DG_CASE_NACA0012)
        const NacaForceCoefficients aerodynamic =
            compute_naca_force_coefficients(solver, solver.K);
        std::cout << "[RESIDUAL CHECK] cd=" << aerodynamic.cd
            << " cl=" << aerodynamic.cl
            << " cm_c4=" << aerodynamic.cm_c4
            << " cd_pressure=" << aerodynamic.cd_pressure
            << " cd_viscous=" << aerodynamic.cd_viscous << "\n";
#endif
        std::cout << "[RESIDUAL CHECK] residual_seconds="
            << residual_seconds << "\n";
        return ok ? 0 : 1;
    }
    RestartState restart_state;
    SAQ2ResetReport sa_q2_reset_report;
    restart_state.cfl = options.cfl_initial;
    if (!options.restart_file.empty()) {
        restart_state = load_restart(solver, options.restart_file,
            solver.K, options.cfl_initial, checkpoint_signature,
            options.allow_legacy_checkpoint);
        const double stored_restart_cfl = restart_state.cfl;
        if (options.cfl_initial_overridden) {
            restart_state.cfl = options.cfl_initial;
        }
        std::cout << "[RESTART] source=\"" << options.restart_file
            << "\" kind="
            << (restart_state.binary_checkpoint
                ? "binary_checkpoint" : "tecplot_projection")
            << " completed_steps=" << restart_state.completed_steps
            << " pseudo_time=" << restart_state.pseudo_time
            << " cfl=" << restart_state.cfl;
        if (restart_state.binary_checkpoint) {
            std::cout << " checkpoint_version="
                << restart_state.checkpoint_version
                << " adaptive_state="
                << (restart_state.continuation.available
                    ? "restored" : "reset");
        }
        if (options.cfl_initial_overridden) {
            std::cout << " stored_cfl=" << stored_restart_cfl
                << " explicit_cfl_override=1";
        }
        std::cout << std::endl;
    }
    if (options.sa_q2_reset_on_restart) {
        sa_q2_reset_report = reset_sa_q2_modes_preserving_average(
            solver, solver.K);
        // The transformed field is a new diagnostic initial condition, not an
        // exact continuation of the old nonlinear trajectory.
        restart_state.continuation = {};
        std::cout << std::scientific << std::setprecision(12)
            << "[SA Q2 RESET] cells=" << sa_q2_reset_report.cells
            << " highest_order_energy_before="
            << sa_q2_reset_report.highest_order_energy_before
            << " highest_order_energy_after="
            << sa_q2_reset_report.highest_order_energy_after
            << " change_l2=" << sa_q2_reset_report.change_l2
            << " change_max_abs=" << sa_q2_reset_report.change_max_abs
            << " cell_average_error_max_abs="
            << sa_q2_reset_report.cell_average_error_max_abs
            << " cell_average_error_max_rel="
            << sa_q2_reset_report.cell_average_error_max_rel << "\n";
    }
    if (jacobian_check_mode) {
        const double check_cfl = std::max(CFL_MIN, restart_state.cfl);
        return run_jacobian_directional_check(solver, solver.K,
            check_cfl, options.sa_cfl_scale);
    }
    const double cfl_start = std::max(CFL_MIN, restart_state.cfl);
    const double cfl_max = std::max(cfl_start, options.cfl_maximum);
    const double cfl_growth_reference = std::min(cfl_start, CFL_INIT);
    const double cfl_adapt_grow = dg::steady::cfl_accepted_growth_factor(
        cfl_growth_reference, cfl_max, CFL_ADAPT_STEP, CFL_RECOVERY_SPAN);
    const long long restart_step_offset = restart_state.completed_steps;
    double t = restart_state.pseudo_time;
    int step = 0;
    auto start_time = std::chrono::steady_clock::now();
    const std::clock_t cpu_start = std::clock();
    auto cpu_seconds = [&]() {
        return (double)(std::clock() - cpu_start) / (double)CLOCKS_PER_SEC;
        };
    std::cout << "Starting DG RANS-SA Simulation (" << dg::case_config::name << "): Grid=" << Nx << "x" << Ny
        << ", M_inf=" << M_inf << ", Re_L=" << Re << std::endl;
    std::cout << "Grid family: " << dg::case_config::grid_family << std::endl;
    std::cout << "Robustness group: " << ROBUST_SOLVER_GROUP
        << "  CFL mode: " << ROBUST_CFL_MODE_NAME << std::endl;
    std::cout << "SA pseudo-time CFL scale: "
        << options.sa_cfl_scale << std::endl;
    std::cout << "SA BR2 lifting factor: "
        << options.sa_br2_factor << std::endl;
    if (options.freeze_ns) {
        std::cout << "[MODE] SA-only diagnostic: Navier--Stokes degrees of freedom "
            "are frozen at the restart state." << std::endl;
    }
#if DG_HAS_OPENMP
    std::cout << "[PARALLEL] OpenMP enabled max_threads="
        << omp_get_max_threads() << " schedule=static" << std::endl;
#else
    std::cout << "[PARALLEL] serial build (OpenMP unavailable or disabled)"
        << std::endl;
#endif
    if (output_enabled) {
        dg::io::TextOutputFile fcase("case_conditions.dat");
        double dx_min = 1e300, dx_max = 0.0;
        for (int i = 0; i < Nx; ++i) {
            const double dxi = solver.x_edges[i + 1] - solver.x_edges[i];
            dx_min = std::min(dx_min, dxi);
            dx_max = std::max(dx_max, dxi);
        }
        int i_x0 = 0;
        double x0_dist = std::abs(solver.x_edges[0] - PLATE_X0);
        for (int i = 1; i <= Nx; ++i) {
            const double dist = std::abs(solver.x_edges[i] - PLATE_X0);
            if (dist < x0_dist) { x0_dist = dist; i_x0 = i; }
        }
        const double dx_left_near_x0 = (i_x0 > 0 && i_x0 <= Nx) ? solver.x_edges[i_x0] - solver.x_edges[i_x0 - 1] : std::numeric_limits<double>::quiet_NaN();
        const double dx_left_far = (i_x0 > 0) ? solver.x_edges[1] - solver.x_edges[0] : std::numeric_limits<double>::quiet_NaN();
        const double dx_right_near_x0 = (i_x0 >= 0 && i_x0 < Nx) ? solver.x_edges[i_x0 + 1] - solver.x_edges[i_x0] : std::numeric_limits<double>::quiet_NaN();
        const double dx_right_far = (i_x0 < Nx) ? solver.x_edges[Nx] - solver.x_edges[Nx - 1] : std::numeric_limits<double>::quiet_NaN();
        const double nu_inf = mu_ref / rho_inf;
        const int i_mid = (int)(std::min_element(solver.x_edges, solver.x_edges + Nx + 1,
            [](double a, double b) { return std::abs(a - PLATE_XMID) < std::abs(b - PLATE_XMID); }) - solver.x_edges);
        const double dy1_plate = std::abs(grid_vertex_y(solver, std::max(0, std::min(Nx, i_mid)), 1)
            - grid_vertex_y(solver, std::max(0, std::min(Nx, i_mid)), 0));
        const double dy1_far = std::abs(grid_vertex_y(solver, 0, 1) - grid_vertex_y(solver, 0, 0));
        const double yplus_plate = (0.5 * dy1_plate) * UTAU_EST / std::max(nu_inf, 1e-30);
        const double yplus_far = (0.5 * dy1_far) * UTAU_EST / std::max(nu_inf, 1e-30);
        const auto mesh_x_bounds = std::minmax_element(
            solver.grid_x.begin(), solver.grid_x.end());
        const auto mesh_y_bounds = std::minmax_element(
            solver.grid_y.begin(), solver.grid_y.end());
        fcase << "# DG RANS-SA runtime conditions\n";
        fcase << "case_name " << dg::case_config::name << "\n";
        fcase << "study_focus " << dg::case_config::study_focus << "\n";
        fcase << "grid_family " << dg::case_config::grid_family << "\n";
#if defined(DG_CASE_NACA0012)
        fcase << "reference_comparison "
            << dg::case_config::reference_comparison << "\n";
        fcase << "farfield_point_vortex_correction "
            << dg::case_config::farfield_correction << "\n";
        fcase << "moment_convention "
            << dg::case_config::moment_convention << "\n";
        fcase << "viscous_wall_point_index_range "
            << g_wall_i_begin << " " << g_wall_i_end << "\n";
        fcase << "viscous_wall_cell_index_range "
            << g_wall_i_begin << " " << (g_wall_i_end - 1) << "\n";
#endif
        fcase << "Nx_cells " << Nx << "\n";
        fcase << "Ny_cells " << Ny << "\n";
        fcase << "reference_grid_points " << (Nx + 1) << " " << (Ny + 1) << "\n";
        fcase << "ad_jacobian_color_count " << g_ad_coloring.color_count << "\n";
        fcase << "ad_jacobian_color_collision_rows "
            << g_ad_coloring.collision_rows << "\n";
        fcase << "ad_jacobian_color_collision_pairs "
            << g_ad_coloring.collision_pairs << "\n";
        fcase << "mesh_source "
#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
            << "CGNS_nested_logical_C_grid\n";
#else
            << (dg::case_config::gmsh_logical_c_grid
                ? "Gmsh_logical_C_grid" : "CGNS_library") << "\n";
#endif
        fcase << "mesh_file " << COMPILED_MESH_FILE << "\n";
        fcase << "x_domain " << xa << " " << xb << "\n";
        fcase << "y_domain " << ya << " " << yb << "\n";
        fcase << "viscous_wall_x_range " << PLATE_X0 << " " << PLATE_X1 << "\n";
        fcase << "mesh_x_bounds " << *mesh_x_bounds.first << " "
            << *mesh_x_bounds.second << "\n";
        fcase << "mesh_y_bounds " << *mesh_y_bounds.first << " "
            << *mesh_y_bounds.second << "\n";
        fcase << "cgns_coordinate_x CoordinateX\n";
        fcase << "cgns_coordinate_y_solver CoordinateY_or_CoordinateZ_detected_by_range\n";
        fcase << "cgns_spanwise_coordinate_ignored CoordinateY\n";
        fcase << "estimated_yplus_wall_first_center " << yplus_plate << "\n";
        fcase << "estimated_yplus_far_first_center " << yplus_far << "\n";
        fcase << "mesh_dx_left_near_wall_start " << dx_left_near_x0 << "\n";
        fcase << "mesh_dx_left_far " << dx_left_far << "\n";
        fcase << "mesh_dx_right_near_wall_start " << dx_right_near_x0 << "\n";
        fcase << "mesh_dx_right_far " << dx_right_far << "\n";
        fcase << "viscous_wall_start_edge_index " << i_x0 << "\n";
        fcase << "dx_min " << dx_min << "\n";
        fcase << "dx_max " << dx_max << "\n";
        fcase << "time_integrator LocalPseudoTransientBackwardEuler\n";
        fcase << "steady_mode local_pseudo_time\n";
        fcase << "equation_solve_mode "
            << (options.freeze_ns ? "sa_only_frozen_ns" : "coupled_rans_sa") << "\n";
        fcase << "nonlinear_solver damped_Newton\n";
        fcase << "linear_solver GMRES\n";
        fcase << "use_visc 1\n";
        fcase << "use_conduction 1\n";
        fcase << "use_limiter 1\n";
        fcase << "dg_order Q" << PolyOrder << "\n";
        fcase << "basis_nodes_1d " << Nodes1D << "\n";
        fcase << "volume_quadrature_points_1d " << Quad1D << "\n";
        fcase << "face_quadrature_points_1d " << FaceQuad << "\n";
        fcase << "quadrature_mode " << (Quad1D > Nodes1D ? "over_integration" : "standard_p_plus_1") << "\n";
        fcase << "basis_dofs_per_cell " << dimPk << "\n";
        fcase << "jacobian_matvec_mode " << jacobian_matvec_mode_name() << "\n";
        fcase << "jacobian_reuse_mode " << JACOBIAN_REUSE_MODE_NAME << "\n";
        fcase << "secant_history_size " << SECANT_HISTORY_SIZE << "\n";
        fcase << "secant_correction_damping " << SECANT_CORRECTION_DAMPING << "\n";
        fcase << "secant_gram_regularization " << SECANT_GRAM_REG << "\n";
        fcase << "preconditioner " << preconditioner_mode_name() << "\n";
        fcase << "steady_max_steps " << STEADY_MAX_STEPS << "\n";
        fcase << "effective_step_limit " << steady_step_limit << "\n";
        fcase << "console_step_every " << CONSOLE_STEP_EVERY << "\n";
        fcase << "diagnostic_step_every " << DIAGNOSTIC_STEP_EVERY << "\n";
        fcase << "sa_term_balance_every " << DIAGNOSTIC_STEP_EVERY << "\n";
        fcase << "sa_term_balance_reconstruction convection+viscosity_volume+viscosity_interior_consistent+viscosity_interior_lifting+viscosity_wall_face+viscosity_other_boundary+production-destruction+cross_diffusion\n";
        fcase << "sa_diagnostic_near_wall_distance "
            << SA_DIAGNOSTIC_NEAR_WALL_DISTANCE << "\n";
        fcase << "cf_monitor_x " << CF_MONITOR_X[0] << " "
            << CF_MONITOR_X[1] << " " << CF_MONITOR_X[2] << "\n";
        fcase << "stop_on_residual_convergence " << (STOP_ON_RESIDUAL_CONVERGENCE ? 1 : 0) << "\n";
        fcase << "requires_physical_convergence "
            << ((REQUIRE_PHYSICAL_CONVERGENCE && !options.freeze_ns) ? 1 : 0) << "\n";
        fcase << "cfl_init " << cfl_start << "\n";
        fcase << "cfl_initial_override_requested "
            << (options.cfl_initial_overridden ? 1 : 0) << "\n";
        fcase << "cfl_max " << cfl_max << "\n";
        fcase << "sa_cfl_scale " << options.sa_cfl_scale << "\n";
        fcase << "sa_br2_factor " << options.sa_br2_factor << "\n";
        fcase << "sa_q2_reset_on_restart "
            << (options.sa_q2_reset_on_restart ? 1 : 0) << "\n";
        fcase << "freeze_ns " << (options.freeze_ns ? 1 : 0) << "\n";
        fcase << "sa_q2_reset_change_l2 "
            << sa_q2_reset_report.change_l2 << "\n";
        fcase << "sa_q2_reset_cell_average_error_max_abs "
            << sa_q2_reset_report.cell_average_error_max_abs << "\n";
        fcase << "cfl_shrink " << CFL_SHRINK << "\n";
        fcase << "cfl_adapt_step " << CFL_ADAPT_STEP << "\n";
        fcase << "cfl_recovery_span " << CFL_RECOVERY_SPAN << "\n";
        fcase << "fcfl_limit " << FCFL_LIMIT << "\n";
        fcase << "cfl_adapt_grow " << cfl_adapt_grow << "\n";
        fcase << "cfl_policy "
            << (options.residual_aware_cfl ? "residual_aware" : "legacy") << "\n";
        fcase << "cfl_residual_trend_window_steps "
            << CFL_RESIDUAL_TREND_WINDOW_STEPS << "\n";
        fcase << "cfl_ns_grow_best_factor " << CFL_NS_GROW_BEST_FACTOR << "\n";
        fcase << "cfl_sa_grow_best_factor " << CFL_SA_GROW_BEST_FACTOR << "\n";
        fcase << "cfl_sa_grow_step_ratio " << CFL_SA_GROW_STEP_RATIO << "\n";
        fcase << "cfl_ns_grow_window_ratio " << CFL_NS_GROW_WINDOW_RATIO << "\n";
        fcase << "cfl_sa_grow_window_ratio " << CFL_SA_GROW_WINDOW_RATIO << "\n";
        fcase << "cfl_sa_repeat_shrink_factor " << CFL_SA_REPEAT_SHRINK_FACTOR << "\n";
        fcase << "cfl_sa_emergency_shrink_factor "
            << CFL_SA_EMERGENCY_SHRINK_FACTOR << "\n";
        fcase << "cfl_sa_shrink_cooldown_steps " << CFL_SA_SHRINK_COOLDOWN_STEPS << "\n";
        fcase << "cfl_sa_limiter_severe_theta " << CFL_SA_LIMITER_SEVERE_THETA << "\n";
        fcase << "sa_limiter_mode "
            << (options.limit_negative_sa ? "floor" : "finite_only") << "\n";
        fcase << "restart_file "
            << (options.restart_file.empty() ? "none" : options.restart_file) << "\n";
        fcase << "restart_kind "
            << (options.restart_file.empty() ? "none"
                : (restart_state.binary_checkpoint
                    ? "binary_checkpoint" : "tecplot_projection"))
            << "\n";
        fcase << "restart_completed_steps " << restart_step_offset << "\n";
        fcase << "checkpoint_file " << options.checkpoint_file << "\n";
        fcase << "checkpoint_every " << options.checkpoint_every << "\n";
        fcase << "checkpoint_format_version 4\n";
        fcase << "restart_adaptive_state "
            << (restart_state.continuation.available
                ? "restored" : "reset") << "\n";
        fcase << "checkpoint_signature 0x" << std::hex
            << checkpoint_signature << std::dec << "\n";
        fcase << "allow_legacy_checkpoint "
            << (options.allow_legacy_checkpoint ? 1 : 0) << "\n";
        fcase << "newton_max_iters " << NEWTON_MAX_ITERS << "\n";
        fcase << "newton_res_tol " << NEWTON_RES_TOL << "\n";
        fcase << "newton_res_reduction_tol " << NEWTON_RES_REDUCTION_TOL << "\n";
        fcase << "noop_relative_update_tol " << NOOP_RELATIVE_UPDATE_TOL << "\n";
        fcase << "res_tol_ns " << options.steady_ns_tolerance << "\n";
        fcase << "res_tol_sa " << options.steady_sa_tolerance << "\n";
        fcase << "res_tol_sa_relative_gate disabled\n";
        fcase << "gmres_restart " << GMRES_RESTART << "\n";
        fcase << "gmres_max_iters " << GMRES_MAX_ITERS << "\n";
        fcase << "gmres_tol " << GMRES_TOL << "\n";
        fcase << "gmres_inexact_tol_max " << GMRES_INEXACT_TOL_MAX << "\n";
        fcase << "gmres_update_tol " << GMRES_UPDATE_TOL << "\n";
        fcase << "gmres_strict_tol_slack " << GMRES_STRICT_TOL_SLACK << "\n";
        fcase << "jac_rebuild_every " << JAC_REBUILD_EVERY << "\n";
        fcase << "jac_rebuild_cfl_rel_tol " << JAC_REBUILD_CFL_REL_TOL << "\n";
        fcase << "jac_rebuild_dtau_rel_tol " << JAC_REBUILD_DTAU_REL_TOL << "\n";
        fcase << "jac_rebuild_state_rel_tol " << JAC_REBUILD_STATE_REL_TOL << "\n";
        fcase << "newton_jac_stagnation_ratio " << NEWTON_JAC_STAGNATION_RATIO << "\n";
        fcase << "newton_linesearch_rebuild_retries " << NEWTON_LINESEARCH_REBUILD_RETRIES << "\n";
        fcase << "robust_cfl_mode " << ROBUST_CFL_MODE_NAME << "\n";
        fcase << "robust_solver_group " << ROBUST_SOLVER_GROUP << "\n";
        fcase << "br2_eta " << BR2_ETA << "\n";
        fcase << "curved_cell_mass_matrix physical_J_weighted\n";
        fcase << "geometry_storage mesh_vertex_backed_nodal_geom_dofs\n";
        fcase << "geometry_order Q" << GeomOrder << "\n";
        fcase << "geometry_nodes_1d " << GeomNodes1D << "\n";
        fcase << "geometry_dofs_per_cell " << GeomDimPk << "\n";
        fcase << "geometry_mapping "
#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
            << "nested_CGNS_Q4_points_to_DG_Q"
#else
            << (dg::case_config::use_analytic_wall_correction
                ? "mesh_vertex_plus_analytic_bump_correction_to_DG_Q"
                : "mesh_vertex_bilinear_to_DG_Q")
#endif
            << GeomOrder << "\n";
        fcase << "physical_convergence_relative_tolerance "
            << PHYSICAL_CONVERGENCE_RELATIVE_TOL << "\n";
        fcase << "physical_convergence_window_intervals "
            << PHYSICAL_CONVERGENCE_WINDOW_INTERVALS << "\n";
        fcase << "physical_convergence_window_steps "
            << PHYSICAL_CONVERGENCE_WINDOW_INTERVALS
                * DIAGNOSTIC_STEP_EVERY << "\n";
        fcase << "physical_convergence_hold_checks "
            << PHYSICAL_CONVERGENCE_HOLD_CHECKS << "\n";
        fcase << "wall_face_geometry mesh_DG_geometry_basis_derivatives\n";
        fcase << "metric_terms mesh_DG_geometry_basis_derivatives\n";
        fcase << "wall_distance "
            << (dg::case_config::use_analytic_wall_correction
                ? "analytic_bump_polyline_distance"
                : "cgns_bottom_boundary_polyline_distance")
            << "\n";
        fcase << "geometry_min_j_relative_to_bilinear " << GEOM_MIN_J_RELATIVE_TO_BILINEAR << "\n";
        fcase << "br2_lifting_normal local_face_gauss_normal\n";
        fcase << "cell_average physical_J_weighted\n";
        fcase << "sa_limiter_avg_floor_factor " << SA_LIMITER_AVG_FLOOR_FACTOR << "\n";
        fcase << "sa_limiter_point_hard_floor_factor " << SA_LIMITER_POINT_HARD_FLOOR_FACTOR << "\n";
        fcase << "sa_limiter_point_relative_tolerance " << SA_LIMITER_POINT_REL_TOL << "\n";
        fcase << "sa_init_nut_over_nu_inf " << (SA_INIT_NUT / std::max(mu_ref / rho_inf, 1e-30)) << "\n";
        fcase << "mu_inf " << mu_ref << "\n";
        fcase << "sa_init_mut_over_mu_inf " << (sa_eddy_mu_from_rho_nut(rho_inf, SA_INIT_NUT, mu_ref) / std::max(mu_ref, 1e-30)) << "\n";
        fcase << "sa_source_diag_factor " << SA_SOURCE_DIAG_FACTOR << "\n";
        fcase << "sa_source_diag_max " << SA_SOURCE_DIAG_MAX << "\n";
        fcase << "Re_L " << Re << "\n";
        fcase << "Ma_inf " << M_inf << "\n";
        fcase << "alpha_degrees " << inflow_angle * 180.0 / 3.14159265358979323846 << "\n";
        fcase << "reference_temperature_kelvin "
            << dg::case_config::reference_temperature_kelvin << "\n";
        fcase << "coefficient_denominator 0.5*rho_inf*V_inf^2\n";
        fcase << "cf_tau_sign fluid_on_wall_positive_along_wall_tangent\n";
        fcase << "inflow_total_pressure " << p0_inf << "\n";
        fcase << "inflow_total_temperature " << T0_inf << "\n";
        fcase << "inflow_angle " << inflow_angle << "\n";
        fcase << "BC_left characteristic total-pressure/total-temperature inflow boundary\n";
        fcase << "BC_right characteristic static-pressure outlet boundary\n";
        fcase << "BC_top characteristic farfield boundary\n";
        fcase << "BC_bottom symmetry/slip outside CGNS viscous-wall patch; adiabatic no-slip with nu_tilde=0 on patch\n";
    }


    // Phase 2: create initial solution and convergence/diagnostic output streams.
    if (output_enabled) write_solution_snapshot(solver, solver.K, "solution_t0.dat");

    dg::io::ResidualHistoryWriter residual_history;
    dg::io::NonlinearHistoryWriter nonlinear_history;
    dg::io::EventHistoryWriter event_history;
    dg::io::SALimiterHistoryWriter sa_limiter_history;
    dg::io::SAResidualDiagnosticWriter sa_residual_diagnostics;
    dg::io::SATermBalanceWriter sa_term_balance;
    std::array<double, SA_TERM_COUNT> previous_sa_term_l2{};
    bool have_previous_sa_term_l2 = false;
    auto finalize_sa_term_trend = [&](dg::io::SATermBalanceRecord& record) {
        populate_sa_term_growth(record, have_previous_sa_term_l2
            ? &previous_sa_term_l2 : nullptr);
        previous_sa_term_l2 = record.global_l2_scaled;
        have_previous_sa_term_l2 = true;
    };
    dg::io::PhysicalConvergenceWriter physical_convergence;
    dg::io::TextOutputFile aerodynamic_convergence;
    if (output_enabled) {
        residual_history.open("residual_history.dat");
        nonlinear_history.open("nonlinear_residual_history.dat");
        event_history.open("event_history.dat", dg::case_config::name,
            ROBUST_SOLVER_GROUP, ROBUST_CFL_MODE_NAME);
        sa_limiter_history.open("sa_limiter_history.dat");
        sa_residual_diagnostics.open("sa_residual_diagnostics.dat",
            SA_DIAGNOSTIC_NEAR_WALL_DISTANCE, g_wall_i_begin,
            g_wall_i_end, 4);
        write_boundary_junction_diagnostic(solver,
            "boundary_junction_diagnostics.dat");
        sa_term_balance.open("sa_term_balance.dat");
        physical_convergence.open("physical_convergence.dat",
            CF_MONITOR_X[0], CF_MONITOR_X[1], CF_MONITOR_X[2]);
#if defined(DG_CASE_NACA0012)
        aerodynamic_convergence.open("aerodynamic_convergence.dat");
        aerodynamic_convergence
            << "# local_step total_step pseudo_time solver_cpu_seconds cfl "
            << "Cd Cl Cm_c4 Cd_pressure Cl_pressure Cd_viscous Cl_viscous "
            << "Cm_pressure Cm_viscous\n";
#endif
    }
    std::cout << "[INIT] computing initial RHS/residual..." << std::endl;
    const auto initial_rhs_start = std::chrono::steady_clock::now();
    computeRHS(solver, solver.K);
    const double initial_rhs_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initial_rhs_start).count();
    std::cout << "[PERF] initial_rhs_s=" << initial_rhs_seconds << std::endl;
    const ResidualStats res_ref = compute_rhs_residual_stats(solver);
    const std::array<double, NumEq> residual_scales = build_residual_scales_init();
    const std::array<double, NumEq> residual_scales_bl = build_residual_scales_bl();
    const std::array<double, NumEq> solution_scales = build_solution_scales();
    double initial_max_l2scaled = 0.0;
    double initial_max_ns_l2scaled = 0.0;
    double initial_sa_l2scaled = 0.0;
    double initial_sa_l2scaled_bl = 0.0;
    double sa_convergence_reference = std::numeric_limits<double>::quiet_NaN();
    WallOutputMonitor previous_wall_output;
    bool have_previous_wall_output = false;
#if defined(DG_CASE_NACA0012)
    NacaForceCoefficients previous_aerodynamic_output;
    bool have_previous_aerodynamic_output = false;
#endif
    dg::steady::PhysicalConvergenceConfig physical_guard_config;
    physical_guard_config.window_intervals =
        PHYSICAL_CONVERGENCE_WINDOW_INTERVALS;
#if defined(DG_CASE_NACA0012)
    physical_guard_config.active_components = 7;
#else
    physical_guard_config.active_components = 4;
#endif
    dg::steady::PhysicalConvergenceGuard physical_convergence_guard(
        physical_guard_config);
    if (restart_state.checkpoint_version >= 4) {
        physical_convergence_guard.restore_history(
            restart_state.continuation.physical_convergence_history);
    }
    auto make_physical_sample = [](const WallOutputMonitor& wall
#if defined(DG_CASE_NACA0012)
        , const NacaForceCoefficients& aerodynamic
#endif
        ) {
        dg::steady::PhysicalConvergenceSample sample = {
            wall.cf[0], wall.cf[1], wall.cf[2], wall.mean_cf,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()
        };
#if defined(DG_CASE_NACA0012)
        sample[4] = aerodynamic.cd;
        sample[5] = aerodynamic.cl;
        sample[6] = aerodynamic.cm_c4;
#endif
        return sample;
        };
    {
        std::array<double, NumEq> l2scaled = { 0,0,0,0,0 };
        std::array<double, NumEq> linfscaled = { 0,0,0,0,0 };
        double max_l2scaled = 0.0, max_linfscaled = 0.0;
        double max_ns_l2scaled = 0.0;
        for (int e = 0; e < NumEq; ++e) {
            l2scaled[e] = res_ref.l2[e] / std::max(residual_scales[e], 1e-30);
            linfscaled[e] = res_ref.linf[e] / std::max(residual_scales[e], 1e-30);
            max_l2scaled = std::max(max_l2scaled, l2scaled[e]);
            max_linfscaled = std::max(max_linfscaled, linfscaled[e]);
            if (e < 4) max_ns_l2scaled = std::max(max_ns_l2scaled, l2scaled[e]);
        }
        initial_max_l2scaled = max_l2scaled;
        initial_max_ns_l2scaled = max_ns_l2scaled;
        initial_sa_l2scaled = l2scaled[4];
        initial_sa_l2scaled_bl = res_ref.l2[4] / std::max(residual_scales_bl[4], 1e-30);
        sa_convergence_reference = restart_state.continuation.available
                && std::isfinite(
                    restart_state.continuation.initial_sa_residual)
                && restart_state.continuation.initial_sa_residual > 0.0
            ? restart_state.continuation.initial_sa_residual
            : std::max(initial_sa_l2scaled, 1e-300);
        if (output_enabled) residual_history.write({
            0,
            t,
            cpu_seconds(),
            0.0,
            cfl_start,
            0.0,
            0.0,
            0.0,
            -300.0,
            std::vector<double>(res_ref.l2.begin(), res_ref.l2.end()),
            std::vector<double>(l2scaled.begin(), l2scaled.end()),
            std::vector<double>(res_ref.linf.begin(), res_ref.linf.end()),
            std::vector<double>(linfscaled.begin(), linfscaled.end()),
            max_l2scaled,
            max_linfscaled,
            initial_sa_l2scaled,
            initial_sa_l2scaled_bl,
            initial_sa_l2scaled
                / std::max(sa_convergence_reference, 1e-300) });
        std::cout << "[INIT] initial maxResL2scaled=" << std::scientific << max_l2scaled
            << " maxNSL2scaled=" << max_ns_l2scaled
            << " saL2scaledInit=" << initial_sa_l2scaled
            << " saL2scaledBL=" << initial_sa_l2scaled_bl
            << " maxResLinfscaled=" << max_linfscaled << std::endl;
        if (output_enabled) {
            const long long total_step = restart_step_offset;
            const dg::io::SAResidualDiagnosticRecord sa_diagnostic =
                compute_sa_residual_diagnostic(solver, solver.K, 0, total_step,
                    t, cpu_seconds(), cfl_start, initial_sa_l2scaled,
                    residual_scales[4]);
            sa_residual_diagnostics.write(sa_diagnostic);
            dg::io::SATermBalanceRecord sa_balance =
                compute_sa_term_balance(solver, solver.K, sa_diagnostic, 0,
                    total_step, t, cpu_seconds(), cfl_start,
                    residual_scales[4]);
            finalize_sa_term_trend(sa_balance);
            sa_term_balance.write(sa_balance);
            std::cout << "[SA-BALANCE] total_step=" << total_step
                << " hotspot=(" << sa_balance.hotspot_i << ","
                << sa_balance.hotspot_j << ") conv="
                << sa_balance.hotspot_terms[SA_CONVECTION] << " visc_volume="
                << sa_balance.hotspot_terms[SA_VISCOSITY_VOLUME]
                << " visc_interior_consistent="
                << sa_balance.hotspot_terms[SA_VISCOSITY_INTERIOR_CONSISTENT]
                << " visc_interior_lifting="
                << sa_balance.hotspot_terms[SA_VISCOSITY_INTERIOR_LIFTING]
                << " visc_wall_face="
                << sa_balance.hotspot_terms[SA_VISCOSITY_WALL_FACE]
                << " visc_other_boundary="
                << sa_balance.hotspot_terms[SA_VISCOSITY_OTHER_BOUNDARY]
                << " visc_total=" << sa_balance.hotspot_viscosity_total << " prod="
                << sa_balance.hotspot_terms[SA_PRODUCTION] << " dest="
                << sa_balance.hotspot_terms[SA_DESTRUCTION] << " cross="
                << sa_balance.hotspot_terms[SA_CROSS_DIFFUSION] << " rhs="
                << sa_balance.hotspot_total_rhs << " dominant_face="
                << sa_balance.hotspot_dominant_interior_face
                << " face_consistent="
                << sa_balance.hotspot_dominant_face_consistent
                << " face_lifting=" << sa_balance.hotspot_dominant_face_lifting
                << " nut_jump=" << sa_balance.hotspot_dominant_face_nu_tilde_jump
                << " dominant_term=" << sa_term_component_name(
                    sa_balance.hotspot_dominant_component)
                << " dominant_fraction="
                << sa_balance.hotspot_dominant_abs_fraction
                << " cancellation_ratio="
                << sa_balance.hotspot_cancellation_ratio
                << " vorticity=" << sa_balance.hotspot_vorticity
                << " mu_eff_over_mu_inf="
                << sa_balance.hotspot_mu_effective_over_mu_inf
                << " region_wall=" << sa_balance.hotspot_touches_wall
                << " region_wake="
                << sa_balance.hotspot_touches_wake_interface
                << " region_open="
                << sa_balance.hotspot_touches_open_boundary
                << " fastest_growth_term=" << sa_term_component_name(
                    sa_balance.global_fastest_growth_component)
                << " fastest_growth_ratio="
                << sa_balance.global_fastest_growth_ratio
                << " reconstruction_error="
                << sa_balance.hotspot_reconstruction_error << std::endl;
            previous_wall_output = compute_wall_output_monitor(solver, solver.K);
            have_previous_wall_output = true;
#if defined(DG_CASE_NACA0012)
            previous_aerodynamic_output =
                compute_naca_force_coefficients(solver, solver.K);
            have_previous_aerodynamic_output = true;
#endif
            if (physical_convergence_guard.history().empty()) {
                physical_convergence_guard.observe(make_physical_sample(
                    previous_wall_output
#if defined(DG_CASE_NACA0012)
                    , previous_aerodynamic_output
#endif
                    ));
            }
            dg::io::PhysicalConvergenceRecord physical_record;
            physical_record.local_step = 0;
            physical_record.total_step = total_step;
            physical_record.pseudo_time = t;
            physical_record.solver_cpu_seconds = cpu_seconds();
            physical_record.cfl = cfl_start;
            physical_record.cf_x1 = previous_wall_output.cf[0];
            physical_record.cf_x2 = previous_wall_output.cf[1];
            physical_record.cf_x3 = previous_wall_output.cf[2];
            physical_record.cf_wall_mean = previous_wall_output.mean_cf;
            physical_record.cf_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.aerodynamic_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.convergence_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.convergence_window_intervals =
                PHYSICAL_CONVERGENCE_WINDOW_INTERVALS;
            physical_record.cf_x1_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.cf_x2_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.cf_x3_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.cf_mean_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.cd_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.cl_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
            physical_record.cm_window_relative_change =
                std::numeric_limits<double>::quiet_NaN();
#if defined(DG_CASE_NACA0012)
            physical_record.cd = previous_aerodynamic_output.cd;
            physical_record.cl = previous_aerodynamic_output.cl;
            physical_record.cm_c4 = previous_aerodynamic_output.cm_c4;
#else
            physical_record.cd = std::numeric_limits<double>::quiet_NaN();
            physical_record.cl = std::numeric_limits<double>::quiet_NaN();
            physical_record.cm_c4 = std::numeric_limits<double>::quiet_NaN();
#endif
            physical_convergence.write(physical_record);
#if defined(DG_CASE_NACA0012)
            aerodynamic_convergence << 0 << " " << total_step << " "
                << std::setprecision(16) << t << " " << cpu_seconds() << " "
                << cfl_start << " " << previous_aerodynamic_output.cd << " "
                << previous_aerodynamic_output.cl << " "
                << previous_aerodynamic_output.cm_c4 << " "
                << previous_aerodynamic_output.cd_pressure << " "
                << previous_aerodynamic_output.cl_pressure << " "
                << previous_aerodynamic_output.cd_viscous << " "
                << previous_aerodynamic_output.cl_viscous << " "
                << previous_aerodynamic_output.cm_pressure << " "
                << previous_aerodynamic_output.cm_viscous << "\n";
            aerodynamic_convergence.flush();
#endif
        }
    }

    if (benchmark_mode) {
        constexpr int residual_repetitions = 5;
        std::cout << "[BENCH] rhs_fingerprint=0x" << std::hex
            << rhs_bitwise_fingerprint(solver.rhs) << std::dec << std::endl;
        std::array<double, residual_repetitions> samples = {};
        for (int repetition = 0; repetition < residual_repetitions; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            computeRHS(solver, solver.K);
            samples[repetition] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
        std::array<double, residual_repetitions> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double mean = 0.0;
        for (double sample : samples) mean += sample;
        mean /= residual_repetitions;
        std::cout << "[BENCH] residual_repetitions=" << residual_repetitions
            << " min_s=" << sorted.front()
            << " median_s=" << sorted[residual_repetitions / 2]
            << " mean_s=" << mean << std::endl;
    }

    const int RES_CHECK_EVERY = 1;
    const int RES_MIN_STEP = 50;
    const int RES_HOLD_CHECKS = 5;
    int res_hold = restart_state.continuation.available
        ? restart_state.continuation.residual_convergence_hold : 0;
    bool converged_marked = false;
    int converged_step = -1;
    double converged_time = -1.0;
    int physical_convergence_hold = restart_state.checkpoint_version >= 4
        ? restart_state.continuation.physical_convergence_hold : 0;
    bool physical_convergence_ready =
        !REQUIRE_PHYSICAL_CONVERGENCE
        || physical_convergence_hold >= PHYSICAL_CONVERGENCE_HOLD_CHECKS;
    double last_physical_relative_change =
        restart_state.checkpoint_version >= 4
        ? restart_state.continuation.last_physical_relative_change
        : std::numeric_limits<double>::quiet_NaN();

    // Phase 3: allocate nonlinear/linear solver state reused by all steady steps.
    BlockSparseMatrix jacobian;
    IdentityPreconditioner identity_preconditioner;
    BlockLUSGSPreconditioner lusgs_preconditioner;
    BlockILU0Preconditioner ilu0_preconditioner;
    Array4D<double> K_n = solver.K;
    Array4D<double> F(Nx, Ny, dimPk, NumEq);
    Array4D<double> Ftrial(Nx, Ny, dimPk, NumEq);
    std::vector<double> Kbase_vec;
    std::vector<double> Fbase_vec;
    std::vector<double> bvec;
    std::vector<double> delta_vec;
    std::vector<double> dtau;
    std::vector<std::vector<double>> secant_s_hist;
    std::vector<std::vector<double>> secant_e_hist;
    std::vector<double> secant_gram_hist;
    dg::steady::CflController cfl_controller({
        cfl_start, CFL_MIN, cfl_max, CFL_SHRINK, cfl_adapt_grow, FCFL_LIMIT });
    double& cfl_current = cfl_controller.current_reference();
    auto restored_positive = [](double restored, double fallback) {
        return std::isfinite(restored) && restored > 0.0
            ? restored : fallback;
        };
    const ContinuationState& restored = restart_state.continuation;
    double previous_max_l2scaled = restored_positive(
        restored.previous_max_residual,
        std::max(initial_max_l2scaled, 1e-300));
    double previous_max_ns_l2scaled = restored_positive(
        restored.previous_ns_residual,
        std::max(initial_max_ns_l2scaled, 1e-300));
    double previous_sa_l2scaled = restored_positive(
        restored.previous_sa_residual,
        std::max(initial_sa_l2scaled, 1e-300));
    double best_max_l2scaled = std::min(restored_positive(
        restored.best_max_residual, previous_max_l2scaled),
        std::max(initial_max_l2scaled, 1e-300));
    double best_max_ns_l2scaled = std::min(restored_positive(
        restored.best_ns_residual, previous_max_ns_l2scaled),
        std::max(initial_max_ns_l2scaled, 1e-300));
    double best_sa_l2scaled = std::min(restored_positive(
        restored.best_sa_residual, previous_sa_l2scaled),
        std::max(initial_sa_l2scaled, 1e-300));
    double last_rho_update_l2 = 0.0;
    double last_lg_rho_update_l2 = -300.0;
    int jacobian_last_build_step = -1000000;
    double jacobian_build_cfl = -1.0;
    double jacobian_build_dtau_min = -1.0;
    double jacobian_build_dtau_max = -1.0;
    std::vector<double> jacobian_build_dtau;
    Array4D<double> K_jacobian_base = solver.K;
    long long limiter_ns_last_print = g_limiter_ns_hits;
    long long limiter_sa_last_print = g_limiter_sa_hits;
    double sa_shrink_reference = std::min(restored_positive(
        restored.sa_shrink_reference, best_sa_l2scaled),
        std::max(initial_sa_l2scaled, 1e-300));
    long long last_sa_residual_shrink_total_step = restored.available
        ? restored.last_sa_shrink_total_step
        : restart_step_offset - CFL_SA_SHRINK_COOLDOWN_STEPS;
    dg::steady::ResidualGrowthGuard residual_growth_guard({
        CFL_RESIDUAL_TREND_WINDOW_STEPS,
        CFL_NS_GROW_BEST_FACTOR,
        CFL_SA_GROW_BEST_FACTOR,
        CFL_NS_GROW_WINDOW_RATIO,
        CFL_SA_GROW_WINDOW_RATIO });
    if (restored.available) {
        residual_growth_guard.restore_history(
            restored.residual_growth_history);
    }

    auto current_continuation_state = [&]() {
        ContinuationState state;
        state.available = true;
        state.initial_sa_residual = sa_convergence_reference;
        state.previous_max_residual = previous_max_l2scaled;
        state.previous_ns_residual = previous_max_ns_l2scaled;
        state.previous_sa_residual = previous_sa_l2scaled;
        state.best_max_residual = best_max_l2scaled;
        state.best_ns_residual = best_max_ns_l2scaled;
        state.best_sa_residual = best_sa_l2scaled;
        state.sa_shrink_reference = sa_shrink_reference;
        state.last_sa_shrink_total_step =
            last_sa_residual_shrink_total_step;
        state.residual_convergence_hold = res_hold;
        state.physical_convergence_hold = physical_convergence_hold;
        state.last_physical_relative_change =
            last_physical_relative_change;
        state.residual_growth_history = residual_growth_guard.history();
        state.physical_convergence_history =
            physical_convergence_guard.history();
        return state;
        };

    auto active_preconditioner_ready = [&]() {
        switch (PRECONDITIONER_MODE) {
        case PreconditionerMode::NONE: return identity_preconditioner.ready();
        case PreconditionerMode::LU_SGS: return lusgs_preconditioner.ready();
        case PreconditionerMode::BLOCK_ILU0: return ilu0_preconditioner.ready();
        }
        return false;
        };
    auto set_active_preconditioner_not_ready = [&]() {
        switch (PRECONDITIONER_MODE) {
        case PreconditionerMode::NONE:
            identity_preconditioner.invalidate();
            break;
        case PreconditionerMode::LU_SGS:
            lusgs_preconditioner.invalidate();
            break;
        case PreconditionerMode::BLOCK_ILU0:
            ilu0_preconditioner.invalidate();
            break;
        }
        };
    auto build_active_preconditioner = [&]() {
        switch (PRECONDITIONER_MODE) {
        case PreconditionerMode::NONE:
            return identity_preconditioner.build_from_matrix(jacobian, ILU0_DIAG_SHIFT);
        case PreconditionerMode::LU_SGS:
            return lusgs_preconditioner.build_from_matrix(jacobian, ILU0_DIAG_SHIFT);
        case PreconditionerMode::BLOCK_ILU0:
            return ilu0_preconditioner.build_from_matrix(jacobian, ILU0_DIAG_SHIFT);
        }
        return false;
        };
    auto apply_active_preconditioner = [&](const std::vector<double>& in, std::vector<double>& out) {
        switch (PRECONDITIONER_MODE) {
        case PreconditionerMode::NONE:
            identity_preconditioner.apply(in, out);
            break;
        case PreconditionerMode::LU_SGS:
            lusgs_preconditioner.apply(in, out);
            break;
        case PreconditionerMode::BLOCK_ILU0:
            ilu0_preconditioner.apply(in, out);
            break;
        }
        };

    const double run_nan = std::numeric_limits<double>::quiet_NaN();
    std::string run_outcome = "running";
    std::string run_failure_reason = "none";
    int run_failure_step = -1;
    double run_failure_time = run_nan;
    double run_failure_cfl = run_nan;
    double run_failure_dtau_min = run_nan;
    double run_failure_dtau_max = run_nan;
    double run_failure_max_l2scaled = previous_max_l2scaled;
    double run_failure_max_ns_l2scaled = previous_max_ns_l2scaled;
    double run_failure_sa_l2scaled = previous_sa_l2scaled;
    double run_failure_rho_min = run_nan;
    double run_failure_p_min = run_nan;
    double run_failure_nut_min = run_nan;
    dg::steady::RunStatistics run_statistics;
    auto& run_total_newton_iters = run_statistics.total_newton_iterations;
    auto& run_total_gmres_iters = run_statistics.total_gmres_iterations;
    auto& run_rejected_steps = run_statistics.rejected_steps;
    auto& run_line_search_failures = run_statistics.line_search_failures;
    auto& run_gmres_failures = run_statistics.gmres_failures;
    auto& run_newton_failures = run_statistics.newton_failures;
    auto& run_invalid_raw_failures = run_statistics.invalid_raw_failures;
    auto& run_invalid_post_failures = run_statistics.invalid_post_failures;
    auto& run_cfl_grows = run_statistics.cfl_grows;
    auto& run_cfl_shrinks = run_statistics.cfl_shrinks;
    long long accepted_steady_steps = 0;
    dg::steady::ProfileStatistics profile_statistics;
    auto& profile_dtau_seconds = profile_statistics.dtau_seconds;
    auto& profile_jacobian_seconds = profile_statistics.jacobian_seconds;
    auto& profile_preconditioner_seconds = profile_statistics.preconditioner_build_seconds;
    auto& profile_gmres_seconds = profile_statistics.gmres_seconds;
    auto& profile_gmres_matvec_seconds = profile_statistics.gmres_matvec_seconds;
    auto& profile_gmres_preconditioner_apply_seconds =
        profile_statistics.gmres_preconditioner_apply_seconds;
    auto& profile_line_search_seconds = profile_statistics.line_search_seconds;
    auto& profile_stabilization_seconds = profile_statistics.stabilization_seconds;
    auto& profile_jacobian_builds = profile_statistics.jacobian_builds;
    auto& profile_gmres_calls = profile_statistics.gmres_calls;
    auto& profile_gmres_matvec_calls = profile_statistics.gmres_matvec_calls;
    auto& profile_gmres_preconditioner_apply_calls =
        profile_statistics.gmres_preconditioner_apply_calls;
    auto& profile_line_search_trials = profile_statistics.line_search_trials;
    auto& profile_attempted_steps = profile_statistics.attempted_steps;
    auto& profile_accepted_steps = profile_statistics.accepted_steps;

    auto log_event = [&](int event_step,
        double event_time,
        const std::string& event_type,
        double old_cfl,
        double new_cfl,
        double event_dtau_min,
        double event_dtau_max,
        double event_max_l2,
        double event_max_ns,
        double event_sa_l2,
        double rho_min,
        double p_min,
        double nut_min,
        const std::string& detail) {
            event_history.write({
                event_step,
                event_time,
                cpu_seconds(),
                event_type,
                old_cfl,
                new_cfl,
                event_dtau_min,
                event_dtau_max,
                event_max_l2,
                event_max_ns,
                event_sa_l2,
                rho_min,
                p_min,
                nut_min,
                detail });
        };
    auto count_nonlinear_failure = [&](const std::string& reason) {
        run_statistics.record_nonlinear_failure(reason);
        };
    auto mark_run_failure = [&](const std::string& reason,
        int failure_step,
        double failure_time,
        double failure_cfl,
        double failure_dtau_min,
        double failure_dtau_max,
        double failure_max_l2,
        double failure_max_ns,
        double failure_sa_l2,
        double rho_min,
        double p_min,
        double nut_min) {
            if (run_outcome == "failed") return;
            run_outcome = "failed";
            run_failure_reason = reason;
            run_failure_step = failure_step;
            run_failure_time = failure_time;
            run_failure_cfl = failure_cfl;
            run_failure_dtau_min = failure_dtau_min;
            run_failure_dtau_max = failure_dtau_max;
            run_failure_max_l2scaled = failure_max_l2;
            run_failure_max_ns_l2scaled = failure_max_ns;
            run_failure_sa_l2scaled = failure_sa_l2;
            run_failure_rho_min = rho_min;
            run_failure_p_min = p_min;
            run_failure_nut_min = nut_min;
        };
    auto write_run_summary = [&](double solver_wall_seconds,
        double solver_cpu_seconds,
        double total_wall_seconds,
        double total_cpu_seconds) {
        dg::io::TextOutputFile fsum("run_summary.dat");
        int i_x0 = 0;
        double x0_dist = std::abs(solver.x_edges[0] - PLATE_X0);
        for (int i = 1; i <= Nx; ++i) {
            const double dist = std::abs(solver.x_edges[i] - PLATE_X0);
            if (dist < x0_dist) { x0_dist = dist; i_x0 = i; }
        }
        const double dx_left_near_x0 = (i_x0 > 0 && i_x0 <= Nx) ? solver.x_edges[i_x0] - solver.x_edges[i_x0 - 1] : std::numeric_limits<double>::quiet_NaN();
        const double dx_right_near_x0 = (i_x0 >= 0 && i_x0 < Nx) ? solver.x_edges[i_x0 + 1] - solver.x_edges[i_x0] : std::numeric_limits<double>::quiet_NaN();
        const double nu_inf = mu_ref / rho_inf;
        const int i_mid = (int)(std::min_element(solver.x_edges, solver.x_edges + Nx + 1,
            [](double a, double b) { return std::abs(a - PLATE_XMID) < std::abs(b - PLATE_XMID); }) - solver.x_edges);
        const double dy1_plate = std::abs(grid_vertex_y(solver, std::max(0, std::min(Nx, i_mid)), 1)
            - grid_vertex_y(solver, std::max(0, std::min(Nx, i_mid)), 0));
        const double dy1_far = std::abs(grid_vertex_y(solver, 0, 1) - grid_vertex_y(solver, 0, 0));
        const double yplus_plate = (0.5 * dy1_plate) * UTAU_EST / std::max(nu_inf, 1e-30);
        const double yplus_far = (0.5 * dy1_far) * UTAU_EST / std::max(nu_inf, 1e-30);
        const auto summary_mesh_x_bounds = std::minmax_element(
            solver.grid_x.begin(), solver.grid_x.end());
        const auto summary_mesh_y_bounds = std::minmax_element(
            solver.grid_y.begin(), solver.grid_y.end());
        fsum << std::setprecision(16);
        fsum << "# DG RANS-SA robustness run summary\n";
        fsum << "study_focus " << dg::case_config::study_focus << "\n";
        fsum << "case_name " << dg::case_config::name << "\n";
        fsum << "grid_family " << dg::case_config::grid_family << "\n";
#if defined(DG_CASE_NACA0012)
        fsum << "reference_comparison "
            << dg::case_config::reference_comparison << "\n";
        fsum << "farfield_point_vortex_correction "
            << dg::case_config::farfield_correction << "\n";
        fsum << "moment_convention "
            << dg::case_config::moment_convention << "\n";
        fsum << "viscous_wall_point_index_range "
            << g_wall_i_begin << " " << g_wall_i_end << "\n";
        fsum << "viscous_wall_cell_index_range "
            << g_wall_i_begin << " " << (g_wall_i_end - 1) << "\n";
#endif
        fsum << "grid_cells " << Nx << " " << Ny << "\n";
        fsum << "reference_grid_points " << (Nx + 1) << " " << (Ny + 1) << "\n";
        fsum << "ad_jacobian_color_count " << g_ad_coloring.color_count << "\n";
        fsum << "ad_jacobian_color_collision_rows "
            << g_ad_coloring.collision_rows << "\n";
        fsum << "ad_jacobian_color_collision_pairs "
            << g_ad_coloring.collision_pairs << "\n";
        fsum << "mesh_source "
#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
            << "CGNS_nested_logical_C_grid\n";
#else
            << (dg::case_config::gmsh_logical_c_grid
                ? "Gmsh_logical_C_grid" : "CGNS_library") << "\n";
#endif
        fsum << "mesh_file " << COMPILED_MESH_FILE << "\n";
        fsum << "viscous_wall_x_range " << PLATE_X0 << " " << PLATE_X1 << "\n";
        fsum << "mesh_x_bounds " << *summary_mesh_x_bounds.first << " "
            << *summary_mesh_x_bounds.second << "\n";
        fsum << "mesh_y_bounds " << *summary_mesh_y_bounds.first << " "
            << *summary_mesh_y_bounds.second << "\n";
        fsum << "cgns_coordinate_x CoordinateX\n";
        fsum << "cgns_coordinate_y_solver CoordinateY_or_CoordinateZ_detected_by_range\n";
        fsum << "cgns_spanwise_coordinate_ignored remaining_constant_coordinate\n";
        fsum << "estimated_yplus_wall_first_center " << yplus_plate << "\n";
        fsum << "estimated_yplus_far_first_center " << yplus_far << "\n";
        fsum << "utau_est_for_yplus " << UTAU_EST << "\n";
        fsum << "mesh_dx_left_near_wall_start " << dx_left_near_x0 << "\n";
        fsum << "mesh_dx_right_near_wall_start " << dx_right_near_x0 << "\n";
        fsum << "Ma_inf " << M_inf << "\n";
        fsum << "Re_L " << Re << "\n";
        fsum << "alpha_degrees " << inflow_angle * 180.0 / 3.14159265358979323846 << "\n";
        fsum << "reference_temperature_kelvin "
            << dg::case_config::reference_temperature_kelvin << "\n";
        fsum << "coefficient_denominator 0.5*rho_inf*V_inf^2\n";
        fsum << "cf_tau_sign fluid_on_wall_positive_along_wall_tangent\n";
        fsum << "time_integrator LocalPseudoTransientBackwardEuler\n";
        fsum << "equation_solve_mode "
            << (options.freeze_ns ? "sa_only_frozen_ns" : "coupled_rans_sa") << "\n";
        fsum << "nonlinear_solver Newton_GMRES\n";
        fsum << "use_visc 1\n";
        fsum << "use_conduction 1\n";
        fsum << "use_limiter 1\n";
        fsum << "dg_order Q" << PolyOrder << "\n";
        fsum << "basis_nodes_1d " << Nodes1D << "\n";
        fsum << "volume_quadrature_points_1d " << Quad1D << "\n";
        fsum << "face_quadrature_points_1d " << FaceQuad << "\n";
        fsum << "quadrature_mode " << (Quad1D > Nodes1D ? "over_integration" : "standard_p_plus_1") << "\n";
        fsum << "basis_dofs_per_cell " << dimPk << "\n";
        fsum << "mu_inf " << mu_ref << "\n";
        fsum << "sa_init_nut_over_nu_inf " << (SA_INIT_NUT / std::max(mu_ref / rho_inf, 1e-30)) << "\n";
        fsum << "sa_init_mut_over_mu_inf " << (sa_eddy_mu_from_rho_nut(rho_inf, SA_INIT_NUT, mu_ref) / std::max(mu_ref, 1e-30)) << "\n";
        fsum << "jacobian_matvec_mode " << jacobian_matvec_mode_name() << "\n";
        fsum << "jacobian_reuse_mode " << JACOBIAN_REUSE_MODE_NAME << "\n";
        fsum << "secant_history_size " << SECANT_HISTORY_SIZE << "\n";
        fsum << "secant_correction_damping " << SECANT_CORRECTION_DAMPING << "\n";
        fsum << "secant_gram_regularization " << SECANT_GRAM_REG << "\n";
        fsum << "preconditioner " << preconditioner_mode_name() << "\n";
        fsum << "stop_on_residual_convergence " << (STOP_ON_RESIDUAL_CONVERGENCE ? 1 : 0) << "\n";
        fsum << "requires_physical_convergence "
            << ((REQUIRE_PHYSICAL_CONVERGENCE && !options.freeze_ns) ? 1 : 0) << "\n";
        fsum << "compiled_default_max_steps " << STEADY_MAX_STEPS << "\n";
        fsum << "effective_step_limit " << steady_step_limit << "\n";
        fsum << "console_step_every " << CONSOLE_STEP_EVERY << "\n";
        fsum << "diagnostic_step_every " << DIAGNOSTIC_STEP_EVERY << "\n";
        fsum << "sa_term_balance_every " << DIAGNOSTIC_STEP_EVERY << "\n";
        fsum << "sa_term_balance_reconstruction convection+viscosity_volume+viscosity_interior_consistent+viscosity_interior_lifting+viscosity_wall_face+viscosity_other_boundary+production-destruction+cross_diffusion\n";
        fsum << "sa_diagnostic_near_wall_distance "
            << SA_DIAGNOSTIC_NEAR_WALL_DISTANCE << "\n";
        fsum << "cf_monitor_x " << CF_MONITOR_X[0] << " "
            << CF_MONITOR_X[1] << " " << CF_MONITOR_X[2] << "\n";
        fsum << "robust_solver_group " << ROBUST_SOLVER_GROUP << "\n";
        fsum << "robust_cfl_mode " << ROBUST_CFL_MODE_NAME << "\n";
        fsum << "cfl_init " << cfl_start << "\n";
        fsum << "cfl_initial_override_requested "
            << (options.cfl_initial_overridden ? 1 : 0) << "\n";
        fsum << "cfl_adapt_step " << CFL_ADAPT_STEP << "\n";
        fsum << "cfl_recovery_span " << CFL_RECOVERY_SPAN << "\n";
        fsum << "fcfl_limit " << FCFL_LIMIT << "\n";
        fsum << "cfl_max " << cfl_max << "\n";
        fsum << "sa_cfl_scale " << options.sa_cfl_scale << "\n";
        fsum << "sa_br2_factor " << options.sa_br2_factor << "\n";
        fsum << "sa_q2_reset_on_restart "
            << (options.sa_q2_reset_on_restart ? 1 : 0) << "\n";
        fsum << "freeze_ns " << (options.freeze_ns ? 1 : 0) << "\n";
        fsum << "sa_q2_reset_change_l2 "
            << sa_q2_reset_report.change_l2 << "\n";
        fsum << "sa_q2_reset_cell_average_error_max_abs "
            << sa_q2_reset_report.cell_average_error_max_abs << "\n";
        fsum << "cfl_adapt_grow " << cfl_adapt_grow << "\n";
        fsum << "cfl_policy "
            << (options.residual_aware_cfl ? "residual_aware" : "legacy") << "\n";
        fsum << "cfl_residual_trend_window_steps "
            << CFL_RESIDUAL_TREND_WINDOW_STEPS << "\n";
        fsum << "cfl_ns_grow_best_factor " << CFL_NS_GROW_BEST_FACTOR << "\n";
        fsum << "cfl_sa_grow_best_factor " << CFL_SA_GROW_BEST_FACTOR << "\n";
        fsum << "cfl_sa_grow_step_ratio " << CFL_SA_GROW_STEP_RATIO << "\n";
        fsum << "cfl_ns_grow_window_ratio " << CFL_NS_GROW_WINDOW_RATIO << "\n";
        fsum << "cfl_sa_grow_window_ratio " << CFL_SA_GROW_WINDOW_RATIO << "\n";
        fsum << "cfl_sa_repeat_shrink_factor " << CFL_SA_REPEAT_SHRINK_FACTOR << "\n";
        fsum << "cfl_sa_emergency_shrink_factor "
            << CFL_SA_EMERGENCY_SHRINK_FACTOR << "\n";
        fsum << "cfl_sa_shrink_cooldown_steps " << CFL_SA_SHRINK_COOLDOWN_STEPS << "\n";
        fsum << "cfl_sa_limiter_severe_theta " << CFL_SA_LIMITER_SEVERE_THETA << "\n";
        fsum << "sa_limiter_mode "
            << (options.limit_negative_sa ? "floor" : "finite_only") << "\n";
        fsum << "restart_file "
            << (options.restart_file.empty() ? "none" : options.restart_file) << "\n";
        fsum << "restart_kind "
            << (options.restart_file.empty() ? "none"
                : (restart_state.binary_checkpoint
                    ? "binary_checkpoint" : "tecplot_projection"))
            << "\n";
        fsum << "restart_completed_steps " << restart_step_offset << "\n";
        fsum << "checkpoint_file " << options.checkpoint_file << "\n";
        fsum << "checkpoint_every " << options.checkpoint_every << "\n";
        fsum << "checkpoint_format_version 4\n";
        fsum << "restart_adaptive_state "
            << (restart_state.continuation.available
                ? "restored" : "reset") << "\n";
        fsum << "checkpoint_signature 0x" << std::hex
            << checkpoint_signature << std::dec << "\n";
        fsum << "allow_legacy_checkpoint "
            << (options.allow_legacy_checkpoint ? 1 : 0) << "\n";
        fsum << "res_tol_ns " << options.steady_ns_tolerance << "\n";
        fsum << "res_tol_sa " << options.steady_sa_tolerance << "\n";
        fsum << "res_tol_sa_relative_gate disabled\n";
        fsum << "curved_cell_mass_matrix physical_J_weighted\n";
        fsum << "geometry_storage mesh_vertex_backed_nodal_geom_dofs\n";
        fsum << "geometry_order Q" << GeomOrder << "\n";
        fsum << "geometry_nodes_1d " << GeomNodes1D << "\n";
        fsum << "geometry_dofs_per_cell " << GeomDimPk << "\n";
        fsum << "geometry_mapping "
#if defined(DG_CASE_NACA0012) && defined(DG_NACA_OFFICIAL_CGNS)
            << "nested_CGNS_Q4_points_to_DG_Q"
#else
            << (dg::case_config::use_analytic_wall_correction
                ? "mesh_vertex_plus_analytic_bump_correction_to_DG_Q"
                : "mesh_vertex_bilinear_to_DG_Q")
#endif
            << GeomOrder << "\n";
        fsum << "wall_face_geometry mesh_DG_geometry_basis_derivatives\n";
        fsum << "metric_terms "
            << (dg::case_config::use_analytic_wall_correction
                ? "analytic_bump_DG_geometry_basis_derivatives"
                : "mesh_DG_geometry_basis_derivatives")
            << "\n";
        fsum << "wall_distance "
            << (dg::case_config::use_analytic_wall_correction
                ? "analytic_bump_polyline_distance"
                : "mesh_bottom_boundary_polyline_distance")
            << "\n";
        fsum << "geometry_min_j_relative_to_bilinear " << GEOM_MIN_J_RELATIVE_TO_BILINEAR << "\n";
        fsum << "physical_convergence_relative_tolerance "
            << PHYSICAL_CONVERGENCE_RELATIVE_TOL << "\n";
        fsum << "physical_convergence_window_intervals "
            << PHYSICAL_CONVERGENCE_WINDOW_INTERVALS << "\n";
        fsum << "physical_convergence_window_steps "
            << PHYSICAL_CONVERGENCE_WINDOW_INTERVALS
                * DIAGNOSTIC_STEP_EVERY << "\n";
        fsum << "physical_convergence_hold_checks "
            << PHYSICAL_CONVERGENCE_HOLD_CHECKS << "\n";
        fsum << "physical_convergence_hold " << physical_convergence_hold << "\n";
        fsum << "physical_convergence_history_size "
            << physical_convergence_guard.history().size() << "\n";
        fsum << "last_physical_relative_change "
            << last_physical_relative_change << "\n";
        fsum << "br2_lifting_normal local_face_gauss_normal\n";
        fsum << "cell_average physical_J_weighted\n";
        fsum << "sa_limiter_avg_floor_factor " << SA_LIMITER_AVG_FLOOR_FACTOR << "\n";
        fsum << "sa_limiter_point_hard_floor_factor " << SA_LIMITER_POINT_HARD_FLOOR_FACTOR << "\n";
        fsum << "sa_limiter_point_relative_tolerance " << SA_LIMITER_POINT_REL_TOL << "\n";
        fsum << "outcome " << run_outcome << "\n";
        fsum << "converged " << (converged_marked ? 1 : 0) << "\n";
        fsum << "converged_step " << converged_step << "\n";
        fsum << "converged_time " << converged_time << "\n";
        fsum << "last_loop_step " << step << "\n";
        fsum << "accepted_steps_this_run " << accepted_steady_steps << "\n";
        fsum << "total_completed_steps "
            << (restart_step_offset + accepted_steady_steps) << "\n";
        fsum << "pseudo_time_final " << t << "\n";
        fsum << "cfl_final " << cfl_current << "\n";
        fsum << "best_max_l2scaled " << best_max_l2scaled << "\n";
        fsum << "best_max_ns_l2scaled " << best_max_ns_l2scaled << "\n";
        fsum << "best_sa_l2scaled " << best_sa_l2scaled << "\n";
        fsum << "sa_convergence_reference "
            << sa_convergence_reference << "\n";
        fsum << "sa_shrink_reference " << sa_shrink_reference << "\n";
        fsum << "last_sa_shrink_total_step "
            << last_sa_residual_shrink_total_step << "\n";
        fsum << "residual_growth_history_size "
            << residual_growth_guard.history().size() << "\n";
        fsum << "residual_convergence_hold " << res_hold << "\n";
        fsum << "last_max_l2scaled " << previous_max_l2scaled << "\n";
        fsum << "last_max_ns_l2scaled " << previous_max_ns_l2scaled << "\n";
        fsum << "last_sa_l2scaled " << previous_sa_l2scaled << "\n";
        fsum << "last_rho_update_l2 " << last_rho_update_l2 << "\n";
        fsum << "last_lg_rho_update_l2 " << last_lg_rho_update_l2 << "\n";
        fsum << "failure_reason " << run_failure_reason << "\n";
        fsum << "failure_step " << run_failure_step << "\n";
        fsum << "failure_time " << run_failure_time << "\n";
        fsum << "failure_cfl " << run_failure_cfl << "\n";
        fsum << "failure_dtau_min " << run_failure_dtau_min << "\n";
        fsum << "failure_dtau_max " << run_failure_dtau_max << "\n";
        fsum << "failure_max_l2scaled " << run_failure_max_l2scaled << "\n";
        fsum << "failure_max_ns_l2scaled " << run_failure_max_ns_l2scaled << "\n";
        fsum << "failure_sa_l2scaled " << run_failure_sa_l2scaled << "\n";
        fsum << "failure_rho_min " << run_failure_rho_min << "\n";
        fsum << "failure_p_min " << run_failure_p_min << "\n";
        fsum << "failure_nut_min " << run_failure_nut_min << "\n";
        fsum << "total_newton_iters " << run_total_newton_iters << "\n";
        fsum << "total_gmres_iters " << run_total_gmres_iters << "\n";
        fsum << "rejected_steps " << run_rejected_steps << "\n";
        fsum << "line_search_failures " << run_line_search_failures << "\n";
        fsum << "gmres_failures " << run_gmres_failures << "\n";
        fsum << "newton_failures " << run_newton_failures << "\n";
        fsum << "invalid_raw_failures " << run_invalid_raw_failures << "\n";
        fsum << "invalid_post_failures " << run_invalid_post_failures << "\n";
        fsum << "cfl_grows " << run_cfl_grows << "\n";
        fsum << "cfl_shrinks " << run_cfl_shrinks << "\n";
        fsum << "positivity_fix_rho " << g_rho_posfix << "\n";
        fsum << "positivity_fix_p " << g_p_posfix << "\n";
        fsum << "positivity_fix_nut " << g_nut_posfix << "\n";
        fsum << "limiter_hits_ns " << g_limiter_ns_hits << "\n";
        fsum << "limiter_hits_sa " << g_limiter_sa_hits << "\n";
        fsum << "solver_cpu_seconds " << solver_cpu_seconds << "\n";
        fsum << "solver_wall_seconds " << solver_wall_seconds << "\n";
        fsum << "total_cpu_seconds " << total_cpu_seconds << "\n";
        fsum << "total_wall_seconds " << total_wall_seconds << "\n";
        dg::io::write_study_guidance(
            fsum, run_outcome, run_failure_reason, run_rejected_steps);
        };

    // Phase 4: outer steady pseudo-time loop.
    const dg::nonlinear::NewtonConvergencePolicy newton_convergence{
        NEWTON_RES_TOL, NEWTON_RES_REDUCTION_TOL };
    const dg::nonlinear::SteadyResidualConvergencePolicy steady_convergence{
        options.steady_ns_tolerance, options.steady_sa_tolerance };
    const dg::nonlinear::JacobianReusePolicy jacobian_reuse{
        JAC_REBUILD_EVERY,
        JAC_REBUILD_CFL_REL_TOL,
        JAC_REBUILD_DTAU_REL_TOL,
        JAC_REBUILD_STATE_REL_TOL,
        NEWTON_JAC_STAGNATION_RATIO };
    const dg::nonlinear::InexactNewtonPolicy inexact_newton{
        GMRES_TOL, GMRES_INEXACT_TOL_MAX, 1e-2, 0.05 };
    const dg::nonlinear::BacktrackingConfig backtracking{
        LINE_SEARCH_MAX_ITERS, LINE_SEARCH_MIN_ALPHA, 0.5, 1e-4, 0.25 };
    const auto profile_loop_start = std::chrono::steady_clock::now();
    if (profile_mode) {
        g_profile_rhs_seconds = 0.0;
        g_profile_rhs_calls = 0;
        g_profile_rhs_timing_enabled = true;
        std::cout << "[PROFILE] starting real implicit solve steps="
            << steady_step_limit << std::endl;
    }
    for (step = 0; step < steady_step_limit; ++step) {
        if (profile_mode) ++profile_attempted_steps;
        secant_s_hist.clear();
        secant_e_hist.clear();
        secant_gram_hist.clear();
        const std::clock_t step_cpu_start = std::clock();
        auto step_cpu_seconds = [&]() {
            return (double)(std::clock() - step_cpu_start) / (double)CLOCKS_PER_SEC;
            };
        double dtau_min = 0.0, dtau_max = 0.0;
        const auto dtau_start = std::chrono::steady_clock::now();
        compute_local_dtau(solver, solver.K, cfl_current, dtau, &dtau_min, &dtau_max);
        if (profile_mode) {
            profile_dtau_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - dtau_start).count();
        }
        K_n = solver.K;
        Array4D<double> K_newton = K_n;
        double nonlinear_initial = 1e300;
        double nonlinear_final = 1e300;
        double step_rel_update = 0.0;
        int newton_iters_done = 0;
        int gmres_iters_last = 0;
        double gmres_rel_last = 1e300;
        double gmres_update_last = 1e300;
        double gmres_tol_last = 1e300;
        double accepted_alpha_last = 0.0;
        bool newton_converged = false;
        bool step_failed = false;
        std::string step_fail_reason;
        double previous_newton_be_scaled = std::numeric_limits<double>::infinity();

        // Phase 5: inner Newton loop solves one implicit backward-Euler step.
        for (int newton = 0; newton < NEWTON_MAX_ITERS; ++newton) {
            compute_be_residual_local(solver, K_newton, K_n, dtau,
                options.sa_cfl_scale, F);
            if (options.freeze_ns) restrict_residual_to_sa(F);
            const ResidualStats be_stats = compute_array_residual_stats(solver, F);
            const double be_scaled = max_scaled_l2(be_stats, solution_scales);
            if (newton == 0) nonlinear_initial = std::max(be_scaled, 1e-300);
            const double rel_be = be_scaled / nonlinear_initial;
            nonlinear_final = be_scaled;
            newton_iters_done = newton + 1;
            ++run_total_newton_iters;

            if (output_enabled) {
                std::vector<double> be_l2_scaled(NumEq);
                for (int e = 0; e < NumEq; ++e) {
                    be_l2_scaled[e] = be_stats.l2[e] / std::max(solution_scales[e], 1e-30);
                }
                nonlinear_history.write({
                    step,
                    newton,
                    t,
                    cpu_seconds(),
                    step_cpu_seconds(),
                    cfl_current,
                    dtau_min,
                    dtau_max,
                    std::vector<double>(be_stats.l2.begin(), be_stats.l2.end()),
                    std::move(be_l2_scaled),
                    be_scaled,
                    rel_be,
                    gmres_iters_last,
                    gmres_rel_last,
                    gmres_update_last,
                    gmres_tol_last,
                    accepted_alpha_last,
                    newton != 0 });
            }

            const bool outer_res_converged = options.freeze_ns
                ? steady_convergence.sa_converged(previous_sa_l2scaled)
                : steady_convergence.converged(previous_max_ns_l2scaled,
                    previous_sa_l2scaled);
            if (newton_convergence.converged(
                newton, be_scaled, rel_be, outer_res_converged)) {
                newton_converged = true;
                break;
            }

            const bool jacobian_matrix_needed =
                JACOBIAN_MATVEC_MODE == JacobianMatvecMode::AD_BLOCK ||
                JACOBIAN_MATVEC_MODE == JacobianMatvecMode::MULTI_SECANT_CORRECTED_AD ||
                (PRECONDITIONER_MODE != PreconditionerMode::NONE);
            const bool jacobian_matrix_available = jacobian_last_build_step >= 0;
            const double jacobian_state_relative_change =
                (newton == 0 && jacobian_matrix_available)
                ? relative_array_change(K_newton, K_jacobian_base)
                : 0.0;
            const dg::nonlinear::JacobianReuseState jacobian_reuse_state{
                jacobian_matrix_needed,
                jacobian_matrix_available,
                active_preconditioner_ready(),
                step,
                newton,
                jacobian_last_build_step,
                cfl_current,
                jacobian_build_cfl,
                dtau_min,
                dtau_max,
                jacobian_build_dtau_min,
                jacobian_build_dtau_max,
                jacobian_state_relative_change,
                be_scaled,
                previous_newton_be_scaled };
            const dg::nonlinear::JacobianReuseDecision jacobian_decision =
                dg::nonlinear::decide_jacobian_rebuild(
                    jacobian_reuse, jacobian_reuse_state);
            auto assemble_current_jacobian = [&](const char* reason) {
                const auto jacobian_start = std::chrono::steady_clock::now();
                if (!jacobian.assemble_ad(solver, K_newton, K_n, dtau,
                        options.sa_cfl_scale, F, true, options.freeze_ns)) {
                    std::cerr << "[WARN] AD block Jacobian assembly failed at step="
                        << step << " newton=" << newton << std::endl;
                    step_failed = true;
                    step_fail_reason = "AD block Jacobian assembly failed";
                    return false;
                }
                const auto jacobian_done = std::chrono::steady_clock::now();
                if (!build_active_preconditioner()) {
                    std::cerr << "[WARN] preconditioner build failed at step="
                        << step << " newton=" << newton << std::endl;
                    step_failed = true;
                    step_fail_reason = "preconditioner build failed";
                    return false;
                }
                const auto preconditioner_done = std::chrono::steady_clock::now();
                const double jacobian_seconds = std::chrono::duration<double>(
                    jacobian_done - jacobian_start).count();
                const double preconditioner_seconds = std::chrono::duration<double>(
                    preconditioner_done - jacobian_done).count();
                if (profile_mode) {
                    profile_jacobian_seconds += jacobian_seconds;
                    profile_preconditioner_seconds += preconditioner_seconds;
                    ++profile_jacobian_builds;
                }
                std::cout << "[PERF] Jacobian reason=" << reason
                    << " colors=" << g_ad_coloring.color_count
                    << " assembly_s=" << jacobian_seconds
                    << " preconditioner_s=" << preconditioner_seconds << std::endl;
                if (benchmark_mode) {
                    std::cout << "[BENCH] jacobian_fingerprint=0x" << std::hex
                        << jacobian.bitwise_fingerprint() << std::dec << std::endl;
                }
                jacobian_last_build_step = step;
                jacobian_build_cfl = cfl_current;
                jacobian_build_dtau_min = dtau_min;
                jacobian_build_dtau_max = dtau_max;
                jacobian_build_dtau = dtau;
                K_jacobian_base = K_newton;
                secant_s_hist.clear();
                secant_e_hist.clear();
                secant_gram_hist.clear();
                return true;
                };
            if (jacobian_decision.rebuild && !assemble_current_jacobian(
                dg::nonlinear::rebuild_reason_name(jacobian_decision.reason))) {
                break;
            }
            if (benchmark_mode) {
                std::cout << "[BENCH] completed residual and first Jacobian/preconditioner build; exiting."
                    << std::endl;
                return 0;
            }

            bool accepted = false;
            double accepted_alpha_current = 0.0;
            Array4D<double> K_accept = K_newton;
            double accepted_res = be_scaled;
            double best_rejected_be_ratio = std::numeric_limits<double>::infinity();
            const bool sa_dominant = options.freeze_ns ||
                previous_max_l2scaled > 2.0 * std::max(previous_max_ns_l2scaled, 1e-300);
            const double gmres_linear_tol =
                inexact_newton.linear_tolerance(rel_be, sa_dominant);
            const double gmres_update_res_gate =
                inexact_newton.update_residual_gate(sa_dominant);
            gmres_tol_last = gmres_linear_tol;
            const int jac_retry_limit = NEWTON_LINESEARCH_REBUILD_RETRIES;
            for (int jac_retry = 0; jac_retry <= jac_retry_limit && !accepted; ++jac_retry) {
                if (jac_retry > 0) {
                    set_active_preconditioner_not_ready();
                    if (!assemble_current_jacobian("line_search_retry")) break;
                }

                array_to_vector_inplace(K_newton, Kbase_vec);
                array_to_vector_inplace(F, Fbase_vec);
                bvec.resize(NumDofs);
                for (int i = 0; i < NumDofs; ++i) bvec[i] = -Fbase_vec[i];
                auto apply_A = [&](const std::vector<double>& v, std::vector<double>& out) {
                    const auto matvec_start = profile_mode
                        ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
                    if (JACOBIAN_MATVEC_MODE == JacobianMatvecMode::FINITE_DIFFERENCE) {
                        std::vector<double> active_v = v;
                        if (options.freeze_ns) restrict_vector_to_sa(active_v);
                        finite_difference_be_matvec(solver, K_newton, K_n,
                            dtau, options.sa_cfl_scale, F, active_v, out);
                        if (options.freeze_ns) {
                            restrict_vector_to_sa(out);
                            for (int c = 0; c < NumCells; ++c) {
                                const int base = c * BlockSize;
                                for (int k = 0; k < dimPk; ++k) for (int e = 0; e < 4; ++e) {
                                    out[base + local_dof(k, e)] = v[base + local_dof(k, e)];
                                }
                            }
                        }
                    }
                    else {
                        jacobian.matvec_with_current_dtau(v, out, dtau, jacobian_build_dtau);
                        if (JACOBIAN_MATVEC_MODE == JacobianMatvecMode::MULTI_SECANT_CORRECTED_AD &&
                            !secant_s_hist.empty() &&
                            secant_s_hist.size() == secant_e_hist.size() &&
                            secant_gram_hist.size() == secant_s_hist.size() * secant_s_hist.size()) {
                            const int mh = (int)secant_s_hist.size();
                            std::vector<double> gram = secant_gram_hist;
                            std::vector<double> rhs(mh, 0.0);
                            double diag_max = 0.0;
                            for (int i = 0; i < mh; ++i) {
                                rhs[i] = dot_vec(secant_s_hist[i], v);
                                diag_max = std::max(diag_max, std::abs(gram[i * mh + i]));
                            }
                            const double reg = SECANT_GRAM_REG * std::max(diag_max, 1.0);
                            for (int i = 0; i < mh; ++i) gram[i * mh + i] += reg;

                            std::vector<double> coeff;
                            if (solve_small_dense_system(gram, rhs, coeff)) {
                                for (int i = 0; i < mh; ++i) {
                                    axpy_vec(SECANT_CORRECTION_DAMPING * coeff[i], secant_e_hist[i], out);
                                }
                            }
                        }
                    }
                    if (profile_mode) {
                        profile_gmres_matvec_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - matvec_start).count();
                        ++profile_gmres_matvec_calls;
                    }
                    };
                auto apply_preconditioner_profiled = [&](const std::vector<double>& in,
                    std::vector<double>& out) {
                    const auto apply_start = profile_mode
                        ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
                    apply_active_preconditioner(in, out);
                    if (profile_mode) {
                        profile_gmres_preconditioner_apply_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - apply_start).count();
                        ++profile_gmres_preconditioner_apply_calls;
                    }
                    };

                delta_vec.assign(NumDofs, 0.0);
                // Solve J * delta = -F; delta_vec is the proposed Newton correction.
                const auto gmres_start = std::chrono::steady_clock::now();
                GMRESResult gmres = dg::linear::gmres_solve_right_preconditioned(
                    apply_A, apply_preconditioner_profiled, bvec, delta_vec,
                    GMRES_RESTART, GMRES_MAX_ITERS,
                    gmres_linear_tol, GMRES_UPDATE_TOL, gmres_update_res_gate);
                if (profile_mode) {
                    profile_gmres_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - gmres_start).count();
                    ++profile_gmres_calls;
                }
                gmres_iters_last = gmres.iterations;
                gmres_rel_last = gmres.rel_res;
                gmres_update_last = gmres.rel_update;
                run_total_gmres_iters += gmres.iterations;
                const bool gmres_strict_ok =
                    std::isfinite(gmres.rel_res) &&
                    gmres.rel_res <= GMRES_STRICT_TOL_SLACK * gmres_linear_tol;
                if (!gmres_strict_ok) {
                    if (jac_retry < jac_retry_limit) continue;
                    std::cerr << "[WARN] GMRES did not reach tolerance at step=" << step
                        << " newton=" << newton
                        << " retry=" << jac_retry
                        << " iters=" << gmres.iterations
                        << " rel=" << gmres.rel_res
                        << " rel_update=" << gmres.rel_update
                        << " strict_tol=" << gmres_linear_tol
                        << " slack_tol=" << (GMRES_STRICT_TOL_SLACK * gmres_linear_tol)
                        << std::endl;
                    step_failed = true;
                    step_fail_reason = "GMRES did not converge";
                    break;
                }

                // Backtracking line search accepts a valid state that decreases the BE residual.
                const auto line_search_start = std::chrono::steady_clock::now();
                auto line_search_result = dg::nonlinear::backtracking_line_search(
                    K_newton,
                    be_scaled,
                    backtracking,
                    [&](Array4D<double>& trial, double alpha) {
                        for (int c = 0; c < NumCells; ++c) {
                            const int ii = cell_i_from_id(c);
                            const int jj = cell_j_from_id(c);
                            const int base = c * BlockSize;
                            for (int p = 0; p < BlockSize; ++p) {
                                int pk, pe;
                                dof_from_local(p, pk, pe);
                                if (!options.freeze_ns || pe == 4) {
                                    trial(ii, jj, pk, pe) += alpha * delta_vec[base + p];
                                }
                            }
                        }
                    },
                    [&](const Array4D<double>& trial) {
                        return cellavg_state_valid(solver, trial);
                    },
                    [&](const Array4D<double>& trial) {
                        compute_be_residual_local(solver, trial, K_n, dtau,
                            options.sa_cfl_scale, Ftrial);
                        if (options.freeze_ns) restrict_residual_to_sa(Ftrial);
                        return scaled_l2_from_array(solver, Ftrial, solution_scales);
                    });
                if (profile_mode) {
                    profile_line_search_trials += line_search_result.trials;
                }
                if (profile_mode) {
                    profile_line_search_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - line_search_start).count();
                }
                best_rejected_be_ratio = std::min(best_rejected_be_ratio,
                    line_search_result.best_rejected_ratio);
                if (line_search_result.accepted) {
                    K_accept = std::move(line_search_result.state);
                    accepted_res = line_search_result.residual;
                    accepted_alpha_current = line_search_result.step;
                    accepted = true;
                }

                if (!accepted && jac_retry < jac_retry_limit) {
                    std::cerr << "[JAC] Newton line search failed at step=" << step
                        << " newton=" << newton
                        << " retry=" << jac_retry
                        << " bestRejectedBERatio=" << best_rejected_be_ratio
                        << ". Rebuilding Jacobian at current Newton state and retrying direction."
                        << std::endl;
                }
            }

            if (step_failed) break;
            if (!accepted) {
                std::cerr << "[WARN] Newton line search failed at step=" << step
                    << " newton=" << newton
                    << " relBE=" << rel_be
                    << " previousMaxL2scaled=" << std::scientific << previous_max_l2scaled
                    << " bestRejectedBERatio=" << best_rejected_be_ratio
                    << std::endl;
                step_failed = true;
                step_fail_reason = "Newton line search failed";
                break;
            }
            if (JACOBIAN_MATVEC_MODE == JacobianMatvecMode::MULTI_SECANT_CORRECTED_AD &&
                jacobian_last_build_step >= 0 &&
                accepted_alpha_current > 0.0) {
                std::vector<double> s_vec(NumDofs, 0.0);
                for (int c = 0; c < NumCells; ++c) {
                    const int ii = cell_i_from_id(c);
                    const int jj = cell_j_from_id(c);
                    const int base = c * BlockSize;
                    for (int p = 0; p < BlockSize; ++p) {
                        int pk, pe;
                        dof_from_local(p, pk, pe);
                        s_vec[base + p] = K_accept(ii, jj, pk, pe) - K_newton(ii, jj, pk, pe);
                    }
                }
                const double s_dot_s = dot_vec(s_vec, s_vec);
                if (std::isfinite(s_dot_s) && s_dot_s > 1e-300) {
                    Array4D<double> F_accept = Ftrial;
                    std::vector<double> F_accept_vec;
                    array_to_vector_inplace(F_accept, F_accept_vec);

                    std::vector<double> Js_vec;
                    jacobian.matvec_with_current_dtau(s_vec, Js_vec, dtau, jacobian_build_dtau);
                    std::vector<double> e_vec(NumDofs, 0.0);
                    for (int i = 0; i < NumDofs; ++i) {
                        e_vec[i] = (F_accept_vec[i] - Fbase_vec[i]) - Js_vec[i];
                    }
                    if (norm_vec(e_vec) > 1e-300) {
                        secant_s_hist.push_back(std::move(s_vec));
                        secant_e_hist.push_back(std::move(e_vec));
                        while ((int)secant_s_hist.size() > SECANT_HISTORY_SIZE) {
                            secant_s_hist.erase(secant_s_hist.begin());
                            secant_e_hist.erase(secant_e_hist.begin());
                        }
                        const int mh = (int)secant_s_hist.size();
                        secant_gram_hist.assign(mh * mh, 0.0);
                        for (int ii = 0; ii < mh; ++ii) {
                            for (int jj = 0; jj < mh; ++jj) {
                                secant_gram_hist[ii * mh + jj] = dot_vec(secant_s_hist[ii], secant_s_hist[jj]);
                            }
                        }
                    }
                }
            }
            K_newton = K_accept;
            accepted_alpha_last = accepted_alpha_current;
            nonlinear_final = accepted_res;
            previous_newton_be_scaled = be_scaled;
        }

        if (!step_failed && !newton_converged) {
            const double final_rel_be = nonlinear_final / std::max(nonlinear_initial, 1e-300);
            if (newton_convergence.converged_after_update(
                nonlinear_final, final_rel_be)) {
                newton_converged = true;
                std::cout << "[NEWTON] step=" << step
                    << " accepted after final update"
                    << " maxBEscaled=" << std::scientific << nonlinear_final
                    << " relBE=" << final_rel_be << std::endl;
            }
        }

        if (!step_failed && !newton_converged) {
            std::cerr << "[WARN] Newton reached max iterations without meeting BE residual tolerance at step="
                << step
                << " final maxBEscaled=" << nonlinear_final
                << " initial maxBEscaled=" << nonlinear_initial << std::endl;
            step_failed = true;
            step_fail_reason = "Newton did not converge";
        }

        if (!step_failed && newton_converged) {
            step_rel_update = relative_array_change(K_newton, K_n);
            const bool outer_res_converged = options.freeze_ns
                ? steady_convergence.sa_converged(previous_sa_l2scaled)
                : steady_convergence.converged(previous_max_ns_l2scaled,
                    previous_sa_l2scaled);
            if (!outer_res_converged
                && (!std::isfinite(step_rel_update)
                    || step_rel_update <= NOOP_RELATIVE_UPDATE_TOL)) {
                std::cerr << "[WARN] Newton produced a no-op update before coupled "
                    << "NS/SA convergence at step=" << step
                    << " relative_update=" << step_rel_update
                    << " NS=" << previous_max_ns_l2scaled
                    << " SA=" << previous_sa_l2scaled << std::endl;
                step_failed = true;
                step_fail_reason = "Newton produced no-op update before coupled convergence";
            }
        }

        if (step_failed) {
            ++run_rejected_steps;
            count_nonlinear_failure(step_fail_reason);
            const dg::steady::CflUpdate cfl_update = cfl_controller.reject();
            set_active_preconditioner_not_ready();
            if (cfl_update.decreased()) ++run_cfl_shrinks;
            const char* cfl_action =
                (step_fail_reason == "GMRES did not converge") ? "gmres-shrink" : "retry-shrink";
            log_event(step, t, cfl_action, cfl_update.old_value, cfl_update.new_value,
                dtau_min, dtau_max,
                previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                run_nan, run_nan, run_nan, step_fail_reason);
            std::cerr << "[RETRY] Implicit steady step failed at step=" << step
                << " pseudo_time=" << std::setprecision(16) << t
                << " reason=" << step_fail_reason
                << " cfl_action=" << cfl_action
                << ". Reducing CFL from " << cfl_update.old_value
                << " to " << cfl_current << std::endl;
            if (cfl_update.old_value <= CFL_MIN * (1.0 + 1e-12)) {
                std::cerr << "[FATAL] CFL reached CFL_MIN after repeated nonlinear failures.\n";
                log_event(step, t, "nonlinear_failure_fatal", cfl_update.old_value, cfl_current,
                    dtau_min, dtau_max,
                    previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                    run_nan, run_nan, run_nan, step_fail_reason);
                mark_run_failure(step_fail_reason, step, t, cfl_current, dtau_min, dtau_max,
                    previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                    run_nan, run_nan, run_nan);
                break;
            }
            continue;
        }

        double rho_min = 1e300, p_min = 1e300, nut_min = 1e300;
        if (!cellavg_state_valid(solver, K_newton, &rho_min, &p_min, &nut_min)) {
            const double old_cfl = cfl_current;
            ++run_invalid_raw_failures;
            ++run_rejected_steps;
            std::cerr << "[STOP] raw implicit Newton step produced an invalid state at step="
                << step
                << " pseudo_time=" << std::setprecision(16) << t
                << " rho_min=" << rho_min
                << " p_min=" << p_min
                << " nut_min=" << nut_min
                << " rollback=0"
                << " cfl_mode=" << ROBUST_CFL_MODE_NAME << std::endl;
            log_event(step, t, "invalid_raw_stop", old_cfl, cfl_current,
                dtau_min, dtau_max,
                previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                rho_min, p_min, nut_min, "raw_newton_state");
            mark_run_failure("invalid_raw_implicit_state", step, t, cfl_current, dtau_min, dtau_max,
                previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                rho_min, p_min, nut_min);
            break;
        }

        // Phase 6: accept the implicit state, then apply positivity/limiter stabilization.
        solver.K = K_newton;
        const long long ns_before = g_limiter_ns_hits;
        const long long sa_before = g_limiter_sa_hits;
        const auto stabilization_start = std::chrono::steady_clock::now();
        const dg::SALimiterReport sa_limiter_report =
            apply_solution_stabilization(solver, solver.K,
                options.limit_negative_sa);
        if (options.freeze_ns) restore_ns_state(solver.K, K_n);
        if (profile_mode) {
            profile_stabilization_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stabilization_start).count();
        }
        if (g_limiter_ns_hits != ns_before || g_limiter_sa_hits != sa_before) {
            log_event(step, t, "limiter", cfl_current, cfl_current,
                dtau_min, dtau_max,
                previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                run_nan, run_nan, run_nan, "solution_stabilization");
        }
        if (output_enabled && sa_limiter_report.activated()) {
            sa_limiter_history.write(step + 1, t, cpu_seconds(),
                cfl_current, sa_limiter_report);
        }
        if (!cellavg_state_valid(solver, solver.K, &rho_min, &p_min, &nut_min)) {
            const double old_cfl = cfl_current;
            ++run_invalid_post_failures;
            ++run_rejected_steps;
            std::cerr << "[STOP] accepted implicit step became invalid after stabilization/check at step="
                << step
                << " pseudo_time=" << std::setprecision(16) << t
                << " rho_min=" << rho_min
                << " p_min=" << p_min
                << " nut_min=" << nut_min
                << " stabilization=1"
                << " rollback=0"
                << " cfl_mode=" << ROBUST_CFL_MODE_NAME << std::endl;
            log_event(step, t, "invalid_post_stop", old_cfl, cfl_current,
                dtau_min, dtau_max,
                previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                rho_min, p_min, nut_min, "post_stabilization");
            mark_run_failure("invalid_post_stabilization_state", step, t, cfl_current, dtau_min, dtau_max,
                previous_max_l2scaled, previous_max_ns_l2scaled, previous_sa_l2scaled,
                rho_min, p_min, nut_min);
            break;
        }
        ++accepted_steady_steps;
        if (profile_mode) ++profile_accepted_steps;

        if (step % RES_CHECK_EVERY == 0) {
            computeRHS(solver, solver.K);
            const ResidualStats res_now = compute_rhs_residual_stats(solver);
            std::array<double, NumEq> l2scaled = { 0,0,0,0,0 };
            std::array<double, NumEq> linfscaled = { 0,0,0,0,0 };
            double max_l2scaled = 0.0, max_linfscaled = 0.0;
            double max_ns_l2scaled = 0.0;
            for (int e = 0; e < NumEq; ++e) {
                l2scaled[e] = res_now.l2[e] / std::max(residual_scales[e], 1e-30);
                linfscaled[e] = res_now.linf[e] / std::max(residual_scales[e], 1e-30);
                max_l2scaled = std::max(max_l2scaled, l2scaled[e]);
                max_linfscaled = std::max(max_linfscaled, linfscaled[e]);
                if (e < 4) max_ns_l2scaled = std::max(max_ns_l2scaled, l2scaled[e]);
            }
            const double sa_l2scaled = l2scaled[4];
            const double sa_l2scaled_bl = res_now.l2[4] / std::max(residual_scales_bl[4], 1e-30);
            const double sa_rel_initial = sa_l2scaled
                / std::max(sa_convergence_reference, 1e-300);
            const double ratio_l2 = max_l2scaled / std::max(previous_max_l2scaled, 1e-300);
            const double ratio_ns = max_ns_l2scaled / std::max(previous_max_ns_l2scaled, 1e-300);
            const double ratio_sa = sa_l2scaled / std::max(previous_sa_l2scaled, 1e-300);
            const double rho_update_l2 = compute_density_update_l2(solver, K_newton, K_n);
            const double lg_rho_update_l2 = std::log10(std::max(rho_update_l2, 1e-300));
            last_rho_update_l2 = rho_update_l2;
            last_lg_rho_update_l2 = lg_rho_update_l2;
            t += dtau_min;
            if (output_enabled) residual_history.write({
                step + 1,
                t,
                cpu_seconds(),
                step_cpu_seconds(),
                cfl_current,
                dtau_min,
                dtau_max,
                rho_update_l2,
                lg_rho_update_l2,
                std::vector<double>(res_now.l2.begin(), res_now.l2.end()),
                std::vector<double>(l2scaled.begin(), l2scaled.end()),
                std::vector<double>(res_now.linf.begin(), res_now.linf.end()),
                std::vector<double>(linfscaled.begin(), linfscaled.end()),
                max_l2scaled,
                max_linfscaled,
                sa_l2scaled,
                sa_l2scaled_bl,
                sa_rel_initial });

            if (output_enabled
                && ((step + 1) % DIAGNOSTIC_STEP_EVERY) == 0) {
                const long long total_step = restart_step_offset
                    + accepted_steady_steps;
                const dg::io::SAResidualDiagnosticRecord sa_diagnostic =
                    compute_sa_residual_diagnostic(solver, solver.K,
                        step + 1, total_step, t, cpu_seconds(), cfl_current,
                        sa_l2scaled, residual_scales[4]);
                sa_residual_diagnostics.write(sa_diagnostic);
                dg::io::SATermBalanceRecord sa_balance =
                    compute_sa_term_balance(solver, solver.K, sa_diagnostic,
                        step + 1, total_step, t, cpu_seconds(), cfl_current,
                        residual_scales[4]);
                finalize_sa_term_trend(sa_balance);
                sa_term_balance.write(sa_balance);
                std::cout << "[SA-BALANCE] total_step=" << total_step
                    << " hotspot=(" << sa_balance.hotspot_i << ","
                    << sa_balance.hotspot_j << ") conv="
                    << sa_balance.hotspot_terms[SA_CONVECTION] << " visc_volume="
                    << sa_balance.hotspot_terms[SA_VISCOSITY_VOLUME]
                    << " visc_interior_consistent="
                    << sa_balance.hotspot_terms[SA_VISCOSITY_INTERIOR_CONSISTENT]
                    << " visc_interior_lifting="
                    << sa_balance.hotspot_terms[SA_VISCOSITY_INTERIOR_LIFTING]
                    << " visc_wall_face="
                    << sa_balance.hotspot_terms[SA_VISCOSITY_WALL_FACE]
                    << " visc_other_boundary="
                    << sa_balance.hotspot_terms[SA_VISCOSITY_OTHER_BOUNDARY]
                    << " visc_total=" << sa_balance.hotspot_viscosity_total << " prod="
                    << sa_balance.hotspot_terms[SA_PRODUCTION] << " dest="
                    << sa_balance.hotspot_terms[SA_DESTRUCTION] << " cross="
                    << sa_balance.hotspot_terms[SA_CROSS_DIFFUSION] << " rhs="
                    << sa_balance.hotspot_total_rhs << " dominant_face="
                    << sa_balance.hotspot_dominant_interior_face
                    << " face_consistent="
                    << sa_balance.hotspot_dominant_face_consistent
                    << " face_lifting=" << sa_balance.hotspot_dominant_face_lifting
                    << " nut_jump=" << sa_balance.hotspot_dominant_face_nu_tilde_jump
                    << " dominant_term=" << sa_term_component_name(
                        sa_balance.hotspot_dominant_component)
                    << " dominant_fraction="
                    << sa_balance.hotspot_dominant_abs_fraction
                    << " cancellation_ratio="
                    << sa_balance.hotspot_cancellation_ratio
                    << " vorticity=" << sa_balance.hotspot_vorticity
                    << " mu_eff_over_mu_inf="
                    << sa_balance.hotspot_mu_effective_over_mu_inf
                    << " region_wall=" << sa_balance.hotspot_touches_wall
                    << " region_wake="
                    << sa_balance.hotspot_touches_wake_interface
                    << " region_open="
                    << sa_balance.hotspot_touches_open_boundary
                    << " fastest_growth_term=" << sa_term_component_name(
                        sa_balance.global_fastest_growth_component)
                    << " fastest_growth_ratio="
                    << sa_balance.global_fastest_growth_ratio
                    << " reconstruction_error="
                    << sa_balance.hotspot_reconstruction_error << std::endl;

                const WallOutputMonitor current_wall_output =
                    compute_wall_output_monitor(solver, solver.K);
                const double cf_relative_change = have_previous_wall_output
                    ? wall_output_relative_change(current_wall_output,
                        previous_wall_output)
                    : std::numeric_limits<double>::quiet_NaN();
#if defined(DG_CASE_NACA0012)
                const NacaForceCoefficients aerodynamic =
                    compute_naca_force_coefficients(solver, solver.K);
                const double aerodynamic_relative_change =
                    have_previous_aerodynamic_output
                        ? aerodynamic_output_relative_change(aerodynamic,
                            previous_aerodynamic_output)
                        : std::numeric_limits<double>::quiet_NaN();
#else
                const double aerodynamic_relative_change =
                    std::numeric_limits<double>::quiet_NaN();
#endif
                const dg::steady::PhysicalConvergenceDecision
                    physical_decision = physical_convergence_guard.observe(
                        make_physical_sample(current_wall_output
#if defined(DG_CASE_NACA0012)
                            , aerodynamic
#endif
                            ));
                const double physical_relative_change =
                    physical_decision.history_ready
                    ? physical_decision.maximum_relative_change
                    : std::numeric_limits<double>::quiet_NaN();
                last_physical_relative_change = physical_relative_change;
                if (physical_decision.history_ready
                    && std::isfinite(physical_relative_change)
                    && physical_relative_change <= PHYSICAL_CONVERGENCE_RELATIVE_TOL) {
                    ++physical_convergence_hold;
                }
                else {
                    physical_convergence_hold = 0;
                }
                physical_convergence_ready =
                    physical_convergence_hold >= PHYSICAL_CONVERGENCE_HOLD_CHECKS;
                dg::io::PhysicalConvergenceRecord physical_record;
                physical_record.local_step = step + 1;
                physical_record.total_step = total_step;
                physical_record.pseudo_time = t;
                physical_record.solver_cpu_seconds = cpu_seconds();
                physical_record.cfl = cfl_current;
                physical_record.cf_x1 = current_wall_output.cf[0];
                physical_record.cf_x2 = current_wall_output.cf[1];
                physical_record.cf_x3 = current_wall_output.cf[2];
                physical_record.cf_wall_mean = current_wall_output.mean_cf;
                physical_record.cf_relative_change = cf_relative_change;
                physical_record.aerodynamic_relative_change =
                    aerodynamic_relative_change;
                physical_record.convergence_relative_change =
                    physical_relative_change;
                physical_record.convergence_window_intervals =
                    PHYSICAL_CONVERGENCE_WINDOW_INTERVALS;
                physical_record.cf_x1_window_relative_change =
                    physical_decision.component_relative_change[0];
                physical_record.cf_x2_window_relative_change =
                    physical_decision.component_relative_change[1];
                physical_record.cf_x3_window_relative_change =
                    physical_decision.component_relative_change[2];
                physical_record.cf_mean_window_relative_change =
                    physical_decision.component_relative_change[3];
                physical_record.cd_window_relative_change =
                    physical_decision.component_relative_change[4];
                physical_record.cl_window_relative_change =
                    physical_decision.component_relative_change[5];
                physical_record.cm_window_relative_change =
                    physical_decision.component_relative_change[6];
#if defined(DG_CASE_NACA0012)
                physical_record.cd = aerodynamic.cd;
                physical_record.cl = aerodynamic.cl;
                physical_record.cm_c4 = aerodynamic.cm_c4;
#else
                physical_record.cd = std::numeric_limits<double>::quiet_NaN();
                physical_record.cl = std::numeric_limits<double>::quiet_NaN();
                physical_record.cm_c4 = std::numeric_limits<double>::quiet_NaN();
#endif
                physical_convergence.write(physical_record);
                previous_wall_output = current_wall_output;
                have_previous_wall_output = true;

#if defined(DG_CASE_NACA0012)
                previous_aerodynamic_output = aerodynamic;
                have_previous_aerodynamic_output = true;
                aerodynamic_convergence << (step + 1) << " " << total_step << " "
                    << std::setprecision(16) << t << " " << cpu_seconds() << " "
                    << cfl_current << " " << aerodynamic.cd << " " << aerodynamic.cl
                    << " " << aerodynamic.cm_c4 << " " << aerodynamic.cd_pressure
                    << " " << aerodynamic.cl_pressure << " " << aerodynamic.cd_viscous
                    << " " << aerodynamic.cl_viscous << " " << aerodynamic.cm_pressure
                    << " " << aerodynamic.cm_viscous << "\n";
                aerodynamic_convergence.flush();
#endif

                std::cout << "[SA-DIAG] step=" << (step + 1)
                    << " total_step=" << total_step
                    << " max_cell=(" << sa_diagnostic.maximum_cell_i
                    << "," << sa_diagnostic.maximum_cell_j << ")"
                    << " max_point=(" << sa_diagnostic.maximum_x
                    << "," << sa_diagnostic.maximum_y << ")"
                    << " wall_distance="
                    << sa_diagnostic.maximum_wall_distance
                    << " near_wall_L2="
                    << sa_diagnostic.near_wall_l2_scaled
                    << " airfoil_band_L2="
                    << sa_diagnostic.airfoil_band_l2_scaled
                    << " wake_band_L2="
                    << sa_diagnostic.wake_band_l2_scaled
                    << " wake_pair_common_L2="
                    << sa_diagnostic.wake_pair_common_l2_scaled
                    << " wake_pair_antisymmetric_L2="
                    << sa_diagnostic.wake_pair_antisymmetric_l2_scaled
                    << " outer_L2=" << sa_diagnostic.outer_l2_scaled
                    << " cf_change=" << cf_relative_change
#if defined(DG_CASE_NACA0012)
                    << " aero_change=" << aerodynamic_relative_change
                    << " Cd=" << aerodynamic.cd
                    << " Cl=" << aerodynamic.cl
#endif
                    << std::endl;
            }

            const bool ns_conv_ok = options.freeze_ns ||
                steady_convergence.ns_converged(max_ns_l2scaled);
            const bool sa_conv_ok = steady_convergence.sa_converged(sa_l2scaled);
            const bool physical_conv_ok = options.freeze_ns
                || physical_convergence_ready;
            if ((step + 1) >= RES_MIN_STEP && ns_conv_ok && sa_conv_ok
                && physical_conv_ok) res_hold++;
            else res_hold = 0;
            if (!converged_marked && res_hold >= RES_HOLD_CHECKS) {
                converged_marked = true;
                converged_step = step + 1;
                converged_time = t;
                std::cout << "[CONVERGED] residual"
                    << (REQUIRE_PHYSICAL_CONVERGENCE
                        ? " and physical-output" : "")
                    << " criteria met at step=" << converged_step
                    << " time=" << std::setprecision(16) << converged_time
                    << " maxL2scaled=" << max_l2scaled
                    << " maxNSL2scaled=" << max_ns_l2scaled
                    << " saL2scaledInit=" << sa_l2scaled
                    << " saL2scaledBL=" << sa_l2scaled_bl
                    << " saRelInitial=" << sa_rel_initial
                    << " physicalRelativeChange=" << last_physical_relative_change
                    << " maxLinfscaled=" << max_linfscaled << std::endl;
            }
            const bool stop_after_residual_convergence =
                STOP_ON_RESIDUAL_CONVERGENCE && converged_marked;

            if (((step + 1) % CONSOLE_STEP_EVERY) == 0 || converged_marked) {
                const long long limiter_ns_window = g_limiter_ns_hits - limiter_ns_last_print;
                const long long limiter_sa_window = g_limiter_sa_hits - limiter_sa_last_print;
                std::cout << "Step: " << (step + 1)
                    << " Time: " << std::fixed << std::setprecision(5) << t
                    << " CFL: " << std::scientific << cfl_current
                    << " Newton: " << newton_iters_done
                    << " GMRES: " << gmres_iters_last
                    << " alpha: " << accepted_alpha_last
                    << " maxRes: " << max_l2scaled
                    << " NS: " << max_ns_l2scaled
                    << " SA: " << sa_l2scaled
                    << " SArel: " << sa_rel_initial
                    << " BE: " << nonlinear_final
                    << " rhoMin: " << rho_min
                    << " pMin: " << p_min
                    << " limiterNSWindow: " << limiter_ns_window
                    << " limiterSAWindow: " << limiter_sa_window
                    << std::endl;
                limiter_ns_last_print = g_limiter_ns_hits;
                limiter_sa_last_print = g_limiter_sa_hits;
            }

            // Phase 7: FCFL control grows CFL only after a healthy, sufficiently small update.
            const bool nonlinear_step_healthy =
                newton_converged &&
                accepted_alpha_last >= 0.5 &&
                std::isfinite(gmres_rel_last) &&
                gmres_rel_last <= std::max(2.0 * gmres_tol_last, GMRES_TOL);
            const double fcfl = step_rel_update;
            const dg::steady::ResidualGrowthDecision growth_decision =
                residual_growth_guard.observe(max_ns_l2scaled, sa_l2scaled,
                    best_max_ns_l2scaled, best_sa_l2scaled);
            const bool sa_not_rising_stepwise =
                ratio_sa <= CFL_SA_GROW_STEP_RATIO;
            const bool residual_allows_growth =
                !options.residual_aware_cfl
                || (growth_decision.allow_growth && sa_not_rising_stepwise);
            sa_shrink_reference = std::min(sa_shrink_reference,
                std::max(sa_l2scaled, 1e-300));
            const dg::steady::SAResidualShrinkDecision shrink_decision =
                dg::steady::sa_residual_shrink_decision(sa_l2scaled,
                    sa_shrink_reference, ratio_sa,
                    sa_limiter_report.severe(CFL_SA_LIMITER_SEVERE_THETA),
                    static_cast<int>(std::min<long long>(
                        restart_step_offset + accepted_steady_steps
                            - last_sa_residual_shrink_total_step,
                        std::numeric_limits<int>::max())),
                    { CFL_SA_REPEAT_SHRINK_FACTOR,
                        CFL_SA_EMERGENCY_SHRINK_FACTOR,
                        CFL_SA_SHRINK_COOLDOWN_STEPS });
            const bool residual_requests_shrink =
                options.residual_aware_cfl && shrink_decision.request;
            const dg::steady::CflUpdate cfl_update =
                cfl_controller.accept(nonlinear_step_healthy, fcfl,
                    residual_allows_growth, residual_requests_shrink);
            const char* accepted_cfl_action = shrink_decision.emergency
                && cfl_update.decreased()
                ? "emergency-residual-shrink"
                : cfl_update.action;
            if (cfl_update.effectively_increased()) {
                ++run_cfl_grows;
                log_event(step + 1, t, "cfl_grow", cfl_update.old_value, cfl_update.new_value,
                    dtau_min, dtau_max, max_l2scaled, max_ns_l2scaled, sa_l2scaled,
                    rho_min, p_min, nut_min, accepted_cfl_action);
            }
            else if (cfl_update.decreased()) {
                last_sa_residual_shrink_total_step = restart_step_offset
                    + accepted_steady_steps;
                sa_shrink_reference = std::max(sa_l2scaled, 1e-300);
                ++run_cfl_shrinks;
                log_event(step + 1, t, "cfl_shrink", cfl_update.old_value,
                    cfl_update.new_value, dtau_min, dtau_max, max_l2scaled,
                    max_ns_l2scaled, sa_l2scaled, rho_min, p_min, nut_min,
                    accepted_cfl_action);
                // Periodic diagnostics can miss the short SA excursion that
                // requested this shrink. Capture the accepted state that
                // triggered it, using the CFL under which the state was made.
                // Avoid a duplicate row when this is already a scheduled
                // diagnostic step.
                if (output_enabled
                    && ((step + 1) % DIAGNOSTIC_STEP_EVERY) != 0) {
                    const long long total_step = restart_step_offset
                        + accepted_steady_steps;
                    const dg::io::SAResidualDiagnosticRecord sa_diagnostic =
                        compute_sa_residual_diagnostic(solver, solver.K,
                            step + 1, total_step, t, cpu_seconds(),
                            cfl_update.old_value, sa_l2scaled,
                            residual_scales[4]);
                    sa_residual_diagnostics.write(sa_diagnostic);
                    dg::io::SATermBalanceRecord sa_balance =
                        compute_sa_term_balance(solver, solver.K,
                            sa_diagnostic, step + 1, total_step, t,
                            cpu_seconds(), cfl_update.old_value,
                            residual_scales[4]);
                    finalize_sa_term_trend(sa_balance);
                    sa_term_balance.write(sa_balance);
                    std::cout << "[SA-SHRINK-DIAG] total_step=" << total_step
                        << " old_cfl=" << cfl_update.old_value
                        << " new_cfl=" << cfl_update.new_value
                        << " hotspot=(" << sa_balance.hotspot_i << ","
                        << sa_balance.hotspot_j << ")"
                        << " conv="
                        << sa_balance.hotspot_terms[SA_CONVECTION]
                        << " visc_volume="
                        << sa_balance.hotspot_terms[SA_VISCOSITY_VOLUME]
                        << " visc_consistent="
                        << sa_balance.hotspot_terms[
                            SA_VISCOSITY_INTERIOR_CONSISTENT]
                        << " visc_lifting="
                        << sa_balance.hotspot_terms[
                            SA_VISCOSITY_INTERIOR_LIFTING]
                        << " production="
                        << sa_balance.hotspot_terms[SA_PRODUCTION]
                        << " destruction="
                        << sa_balance.hotspot_terms[SA_DESTRUCTION]
                        << " cross="
                        << sa_balance.hotspot_terms[SA_CROSS_DIFFUSION]
                        << " dominant_face="
                        << sa_balance.hotspot_dominant_interior_face
                        << " face_lifting="
                        << sa_balance.hotspot_dominant_face_lifting
                        << " nut_jump="
                        << sa_balance.hotspot_dominant_face_nu_tilde_jump
                        << " q2_fraction="
                        << sa_balance.hotspot_sa_q2_fraction
                        << " neighbor_q2_fraction="
                        << sa_balance.dominant_neighbor_sa_q2_fraction
                        << " dominant_term=" << sa_term_component_name(
                            sa_balance.hotspot_dominant_component)
                        << " dominant_fraction="
                        << sa_balance.hotspot_dominant_abs_fraction
                        << " cancellation_ratio="
                        << sa_balance.hotspot_cancellation_ratio
                        << " vorticity=" << sa_balance.hotspot_vorticity
                        << " mu_eff_over_mu_inf="
                        << sa_balance.hotspot_mu_effective_over_mu_inf
                        << " region_wall="
                        << sa_balance.hotspot_touches_wall
                        << " region_wake="
                        << sa_balance.hotspot_touches_wake_interface
                        << " region_open="
                        << sa_balance.hotspot_touches_open_boundary
                        << " fastest_growth_term=" << sa_term_component_name(
                            sa_balance.global_fastest_growth_component)
                        << " fastest_growth_ratio="
                        << sa_balance.global_fastest_growth_ratio
                        << std::endl;
                }
            }
            if (cfl_update.changed() &&
                ((step + 1) % CONSOLE_STEP_EVERY) == 0) {
                std::cout << "[CFL] step=" << (step + 1)
                    << " cfl_action=" << accepted_cfl_action
                    << " old=" << std::scientific << cfl_update.old_value
                    << " new=" << cfl_current
                    << " ratioMaxL2=" << ratio_l2
                    << " ratioNS=" << ratio_ns
                    << " ratioSA=" << ratio_sa
                    << std::endl;
            }
            if (max_l2scaled < best_max_l2scaled) {
                best_max_ns_l2scaled = std::min(best_max_ns_l2scaled, std::max(max_ns_l2scaled, 1e-300));
                best_max_l2scaled = std::max(max_l2scaled, 1e-300);
                best_sa_l2scaled = std::min(best_sa_l2scaled, std::max(sa_l2scaled, 1e-300));
            }
            else if (max_ns_l2scaled < best_max_ns_l2scaled) {
                best_max_ns_l2scaled = std::max(max_ns_l2scaled, 1e-300);
            }
            if (sa_l2scaled < best_sa_l2scaled) {
                best_sa_l2scaled = std::max(sa_l2scaled, 1e-300);
            }
            previous_max_l2scaled = std::max(max_l2scaled, 1e-300);
            previous_max_ns_l2scaled = std::max(max_ns_l2scaled, 1e-300);
            previous_sa_l2scaled = std::max(sa_l2scaled, 1e-300);
            const long long completed_steps = restart_step_offset
                + accepted_steady_steps;
            if (output_enabled && options.checkpoint_every > 0
                && completed_steps % options.checkpoint_every == 0) {
                write_binary_checkpoint(options.checkpoint_file, solver.K,
                    completed_steps, t, cfl_current, checkpoint_signature,
                    current_continuation_state());
                std::cout << "[CHECKPOINT] file=\"" << options.checkpoint_file
                    << "\" completed_steps=" << completed_steps
                    << " pseudo_time=" << t << " cfl=" << cfl_current
                    << std::endl;
            }
            if (stop_after_residual_convergence) break;
        }
        else {
            t += dtau_min;
        }
    }

    if (run_outcome == "running") {
        run_outcome = converged_marked ? "converged" : "max_steps";
    }

    g_profile_rhs_timing_enabled = false;
    const double profile_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - profile_loop_start).count();

    if (profile_mode) {
        const double profile_gmres_vector_seconds =
            profile_statistics.gmres_vector_overhead_seconds();
        std::cout << "[PROFILE] outcome=" << run_outcome
            << " attempted_steps=" << profile_attempted_steps
            << " accepted_steps=" << profile_accepted_steps
            << " rejected_steps=" << run_rejected_steps
            << " newton_iters=" << run_total_newton_iters
            << " gmres_calls=" << profile_gmres_calls
            << " gmres_iters=" << run_total_gmres_iters
            << " jacobian_builds=" << profile_jacobian_builds
            << " line_search_trials=" << profile_line_search_trials
            << " final_cfl=" << cfl_current
            << " cfl_grows=" << run_cfl_grows
            << " cfl_shrinks=" << run_cfl_shrinks << std::endl;
        std::cout << "[PROFILE] wall_s=" << profile_wall_seconds
            << " rhs_s=" << g_profile_rhs_seconds
            << " rhs_calls=" << g_profile_rhs_calls
            << " jacobian_s=" << profile_jacobian_seconds
            << " preconditioner_build_s=" << profile_preconditioner_seconds
            << " gmres_s=" << profile_gmres_seconds
            << " gmres_matvec_s=" << profile_gmres_matvec_seconds
            << " gmres_matvec_calls=" << profile_gmres_matvec_calls
            << " gmres_preconditioner_apply_s=" << profile_gmres_preconditioner_apply_seconds
            << " gmres_preconditioner_apply_calls=" << profile_gmres_preconditioner_apply_calls
            << " gmres_vector_overhead_s=" << profile_gmres_vector_seconds
            << " line_search_inclusive_s=" << profile_line_search_seconds
            << " dtau_s=" << profile_dtau_seconds
            << " stabilization_s=" << profile_stabilization_seconds << std::endl;
        std::cout << "[PROFILE] line_search_inclusive_s contains its trial RHS calls; "
            << "do not add it to rhs_s when computing totals." << std::endl;
        return run_outcome == "failed" ? 1 : 0;
    }

    residual_history.close();
    nonlinear_history.close();
    event_history.close();
    sa_limiter_history.close();
    sa_residual_diagnostics.close();
    sa_term_balance.close();
    physical_convergence.close();
    aerodynamic_convergence.close();
    write_binary_checkpoint(options.checkpoint_file, solver.K,
        restart_step_offset + accepted_steady_steps, t, cfl_current,
        checkpoint_signature, current_continuation_state());
    auto solver_end_time = std::chrono::steady_clock::now();
    const double solver_wall_seconds = std::chrono::duration<double>(solver_end_time - start_time).count();
    const double solver_cpu_seconds = cpu_seconds();

    std::cout << "Positivity fixes applied: rho=" << g_rho_posfix
        << "  p=" << g_p_posfix
        << "  nut=" << g_nut_posfix << "\n";
    std::cout << "Limiter activations: ns=" << g_limiter_ns_hits
        << "  sa=" << g_limiter_sa_hits << "\n";
    if (converged_marked) {
        std::cout << "Steady residual"
            << (REQUIRE_PHYSICAL_CONVERGENCE
                ? " and physical-output" : "")
            << " convergence first reached at step=" << converged_step
            << " time=" << std::setprecision(16) << converged_time << "\n";
    }
    else {
        std::cout << "Steady convergence criteria not reached by step limit="
            << steady_step_limit << ".\n";
    }

    // Phase 8: write final solution, verification quantities, and run summary.
    write_verification_outputs(solver, solver.K);
    write_solution_snapshot(solver, solver.K, "solution_tend.dat");
    auto total_end_time = std::chrono::steady_clock::now();
    const double total_wall_seconds = std::chrono::duration<double>(total_end_time - start_time).count();
    const double total_cpu_seconds = cpu_seconds();
    write_run_summary(solver_wall_seconds, solver_cpu_seconds, total_wall_seconds, total_cpu_seconds);
    std::cout << "Run summary written to run_summary.dat\n";
    std::cout << "Event history written to event_history.dat\n";
    std::cout << "\n--- Finished --- Solver wall-clock: "
        << solver_wall_seconds << " s, total wall-clock: " << total_wall_seconds << " s ---\n";
    return 0;
}


