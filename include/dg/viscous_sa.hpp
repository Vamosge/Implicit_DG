#pragma once

#include <algorithm>
#include <cmath>

#if defined(_MSC_VER)
#define DG_PHYSICS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define DG_PHYSICS_FORCE_INLINE inline __attribute__((always_inline))
#else
#define DG_PHYSICS_FORCE_INLINE inline
#endif

namespace dg {

struct ViscousSAParameters {
    double gamma;
    double reference_p_over_rho;
    double mu_reference;
    double prandtl;
    double turbulent_prandtl;
    double sutherland_ratio;
    double sigma;
    double cb1;
    double cb2;
    double kappa;
    double cw1;
    double cw2;
    double cw3;
    double cv1;
    double ct3;
    double ct4;
    double cn1;
};

template <class Scalar>
struct SASourceTerms {
    Scalar production = 0.0;
    Scalar destruction = 0.0;
    Scalar cross_diffusion = 0.0;
    Scalar total = 0.0;
};

struct DoubleMath {
    static double value(double x) { return x; }
    static double max(double a, double b) { return std::max(a, b); }
    static double min(double a, double b) { return std::min(a, b); }
    static double abs(double x) { return std::abs(x); }
    static double exp(double x) { return std::exp(x); }
    static double pow(double x, double exponent) { return std::pow(x, exponent); }
    static double freeze(double x) { return x; }
};

template <class Scalar, class Math>
inline Scalar sa_fv1(const Scalar& chi, const ViscousSAParameters& parameters)
{
    const Scalar chi3 = chi * chi * chi;
    const double cv13 = parameters.cv1 * parameters.cv1 * parameters.cv1;
    return chi3 / (chi3 + cv13 + 1e-30);
}

template <class Scalar, class Math>
inline Scalar sa_ft2(const Scalar& chi, const ViscousSAParameters& parameters)
{
    return parameters.ct3 * Math::exp((-parameters.ct4) * chi * chi);
}

template <class Scalar, class Math>
inline Scalar laminar_viscosity(const Scalar* U,
    const ViscousSAParameters& parameters)
{
    const double gamma_minus_one = parameters.gamma - 1.0;
    const Scalar rho = Math::max(U[0], 1e-12);
    const Scalar u = U[1] / rho;
    const Scalar v = U[2] / rho;
    const Scalar p = Math::max(gamma_minus_one *
        (U[3] - 0.5 * rho * (u * u + v * v)), 1e-12);
    const Scalar temperature_ratio = Math::max(
        (p / rho) / parameters.reference_p_over_rho, 1e-12);
    return parameters.mu_reference * Math::pow(temperature_ratio, 1.5)
        * (1.0 + parameters.sutherland_ratio)
        / (temperature_ratio + parameters.sutherland_ratio);
}

template <class Scalar, class Math>
inline Scalar sa_fn_negative(const Scalar& chi,
    const ViscousSAParameters& parameters)
{
    const Scalar chi3 = chi * chi * chi;
    return (parameters.cn1 + chi3) /
        Math::max(parameters.cn1 - chi3, 1e-30);
}

template <class Scalar, class Math>
inline Scalar sa_eddy_viscosity(const Scalar& rho, const Scalar& nut,
    const Scalar& mu_lam, const ViscousSAParameters& parameters)
{
    if (Math::value(nut) <= 0.0) return Scalar(0.0);
    const Scalar nu = mu_lam / Math::max(rho, 1e-12);
    const Scalar chi = nut / Math::max(nu, 1e-30);
    return rho * nut * sa_fv1<Scalar, Math>(chi, parameters);
}

template <class Scalar, class Math>
DG_PHYSICS_FORCE_INLINE void viscous_flux(const Scalar* U, const Scalar* dUx, const Scalar* dUy,
    const ViscousSAParameters& parameters, Scalar* Fv, Scalar* Gv)
{
    const double gamma_minus_one = parameters.gamma - 1.0;
    const Scalar rho = Math::max(U[0], 1e-12);
    const Scalar inv_rho = 1.0 / rho;
    const Scalar u = U[1] * inv_rho;
    const Scalar v = U[2] * inv_rho;
    const Scalar urx = (dUx[1] - u * dUx[0]) * inv_rho;
    const Scalar vrx = (dUx[2] - v * dUx[0]) * inv_rho;
    const Scalar ury = (dUy[1] - u * dUy[0]) * inv_rho;
    const Scalar vry = (dUy[2] - v * dUy[0]) * inv_rho;
    const Scalar dE_rho_dx =
        (dUx[3] * rho - U[3] * dUx[0]) * (inv_rho * inv_rho);
    const Scalar dE_rho_dy =
        (dUy[3] * rho - U[3] * dUy[0]) * (inv_rho * inv_rho);
    const Scalar temperature_x = gamma_minus_one *
        (dE_rho_dx - (u * urx + v * vrx));
    const Scalar temperature_y = gamma_minus_one *
        (dE_rho_dy - (u * ury + v * vry));
    const Scalar nut = U[4] * inv_rho;
    const Scalar mu_lam = laminar_viscosity<Scalar, Math>(U, parameters);
    const Scalar nu_lam = mu_lam * inv_rho;
    const Scalar mu_t = sa_eddy_viscosity<Scalar, Math>(
        rho, nut, mu_lam, parameters);
    const Scalar mu_effective = mu_lam + mu_t;
    const Scalar divergence = urx + vry;
    const Scalar tau_xx = mu_effective *
        (2.0 * urx - (2.0 / 3.0) * divergence);
    const Scalar tau_yy = mu_effective *
        (2.0 * vry - (2.0 / 3.0) * divergence);
    const Scalar tau_xy = mu_effective * (ury + vrx);
    const Scalar conductivity = parameters.gamma / gamma_minus_one *
        (mu_lam / parameters.prandtl + mu_t / parameters.turbulent_prandtl);
    const Scalar dnut_dx = (dUx[4] - nut * dUx[0]) * inv_rho;
    const Scalar dnut_dy = (dUy[4] - nut * dUy[0]) * inv_rho;
    Scalar diffusive_nut = nut;
    if (Math::value(nut) < 0.0) {
        const Scalar chi = nut / Math::max(nu_lam, 1e-30);
        diffusive_nut = nut * sa_fn_negative<Scalar, Math>(chi, parameters);
    }
    const Scalar mu_sa = Math::max(
        mu_lam + rho * diffusive_nut, 1e-14 * parameters.mu_reference)
        / parameters.sigma;

    Fv[0] = 0.0;
    Fv[1] = tau_xx;
    Fv[2] = tau_xy;
    Fv[3] = u * tau_xx + v * tau_xy + conductivity * temperature_x;
    Fv[4] = mu_sa * dnut_dx;
    Gv[0] = 0.0;
    Gv[1] = tau_xy;
    Gv[2] = tau_yy;
    Gv[3] = u * tau_xy + v * tau_yy + conductivity * temperature_y;
    Gv[4] = mu_sa * dnut_dy;
}

template <class Scalar, class Math>
DG_PHYSICS_FORCE_INLINE SASourceTerms<Scalar> sa_source_terms(const Scalar* U,
    const Scalar* dUx, const Scalar* dUy, double wall_distance,
    double destruction_distance, const ViscousSAParameters& parameters,
    bool freeze_positive_production_derivative = false)
{
    const Scalar rho = Math::max(U[0], 1e-12);
    const Scalar inv_rho = 1.0 / rho;
    const Scalar u = U[1] * inv_rho;
    const Scalar v = U[2] * inv_rho;
    const Scalar nut = U[4] * inv_rho;
    const Scalar mu_lam = laminar_viscosity<Scalar, Math>(U, parameters);
    const Scalar nu = mu_lam * inv_rho;
    const Scalar chi = nut / Math::max(nu, 1e-30);
    const Scalar ury = (dUy[1] - u * dUy[0]) * inv_rho;
    const Scalar vrx = (dUx[2] - v * dUx[0]) * inv_rho;
    const Scalar vorticity = Math::abs(vrx - ury);
    const Scalar dnut_dx = (dUx[4] - nut * dUx[0]) * inv_rho;
    const Scalar dnut_dy = (dUy[4] - nut * dUy[0]) * inv_rho;
    const Scalar grad_nut_squared =
        dnut_dx * dnut_dx + dnut_dy * dnut_dy;

    SASourceTerms<Scalar> terms;
    const bool positive_branch = Math::value(nut) >= 0.0;
    if (positive_branch) {
        const Scalar fv1 = sa_fv1<Scalar, Math>(chi, parameters);
        const Scalar fv2 = 1.0 - chi / (1.0 + chi * fv1 + 1e-30);
        const Scalar ft2 = sa_ft2<Scalar, Math>(chi, parameters);
        const double inv_kd2 = 1.0 /
            (parameters.kappa * parameters.kappa * wall_distance * wall_distance + 1e-30);
        const Scalar sbar = nut * fv2 * inv_kd2;
        constexpr double cv2 = 0.7;
        constexpr double cv3 = 0.9;
        Scalar stilde;
        if (Math::value(sbar) >= -cv2 * Math::value(vorticity)) {
            stilde = vorticity + sbar;
        }
        else {
            const Scalar denominator = Math::max(
                (cv3 - 2.0 * cv2) * vorticity - sbar, 1e-30);
            stilde = vorticity + vorticity *
                (cv2 * cv2 * vorticity + cv3 * sbar) / denominator;
        }
        const Scalar stilde_production = Math::max(stilde, 0.0);
        const Scalar stilde_safe = Math::max(stilde, 1e-12);
        Scalar r_sa = nut /
            (stilde_safe * parameters.kappa * parameters.kappa
                * wall_distance * wall_distance + 1e-30);
        r_sa = Math::min(Math::max(r_sa, 0.0), 10.0);
        const Scalar g = r_sa + parameters.cw2 *
            (Math::pow(r_sa, 6.0) - r_sa);
        const double cw3_sixth = std::pow(parameters.cw3, 6.0);
        const Scalar fw = g * Math::pow(
            (1.0 + cw3_sixth) /
                (Math::pow(g, 6.0) + cw3_sixth + 1e-30),
            1.0 / 6.0);
        const Scalar destruction_coefficient = parameters.cw1 * fw
            - parameters.cb1 / (parameters.kappa * parameters.kappa) * ft2;
        terms.production = rho * parameters.cb1 * (1.0 - ft2)
            * stilde_production * nut;
        terms.destruction = rho * destruction_coefficient
            * (nut / destruction_distance) * (nut / destruction_distance);
        terms.cross_diffusion = rho * (parameters.cb2 / parameters.sigma)
            * grad_nut_squared;
        if (freeze_positive_production_derivative) {
            terms.production = Math::freeze(terms.production);
        }
    }
    else {
        terms.production = rho * parameters.cb1 * (1.0 - parameters.ct3)
            * vorticity * nut;
        terms.destruction = -rho * parameters.cw1
            * (nut / destruction_distance) * (nut / destruction_distance);
        terms.cross_diffusion = rho * (parameters.cb2 / parameters.sigma)
            * grad_nut_squared;
    }

    const Scalar diffusive_nut_for_density = positive_branch
        ? nut
        : nut * sa_fn_negative<Scalar, Math>(chi, parameters);
    terms.cross_diffusion = terms.cross_diffusion
        - (nu + diffusive_nut_for_density)
            * (dUx[0] * dnut_dx + dUy[0] * dnut_dy) / parameters.sigma;
    terms.total = terms.production - terms.destruction + terms.cross_diffusion;
    return terms;
}

} // namespace dg

#undef DG_PHYSICS_FORCE_INLINE
