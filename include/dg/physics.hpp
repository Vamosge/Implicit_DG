#pragma once

#include <algorithm>
#include <cmath>

namespace dg {

struct PrimitiveState {
    double rho = 1.0;
    double u = 0.0;
    double v = 0.0;
    double p = 1.0;
    double nut = 0.0;
};

inline PrimitiveState conservative_to_primitive(const double* U, double gamma)
{
    const double gamma_minus_one = gamma - 1.0;
    PrimitiveState V;
    V.rho = std::max(U[0], 1e-12);
    V.u = U[1] / V.rho;
    V.v = U[2] / V.rho;
    V.p = std::max(gamma_minus_one *
        (U[3] - 0.5 * V.rho * (V.u * V.u + V.v * V.v)), 1e-12);
    V.nut = U[4] / V.rho;
    return V;
}

inline void primitive_to_conservative(const PrimitiveState& V, double gamma, double* U)
{
    const double rho = std::max(V.rho, 1e-12);
    const double p = std::max(V.p, 1e-12);
    U[0] = rho;
    U[1] = rho * V.u;
    U[2] = rho * V.v;
    U[3] = p / (gamma - 1.0) + 0.5 * rho * (V.u * V.u + V.v * V.v);
    U[4] = rho * V.nut;
}

inline double primitive_entropy(const PrimitiveState& V, double gamma)
{
    return std::max(V.p, 1e-12) /
        std::pow(std::max(V.rho, 1e-12), gamma);
}

inline double primitive_sound_speed(const PrimitiveState& V, double gamma)
{
    return std::sqrt(gamma * std::max(V.p, 1e-12) /
        std::max(V.rho, 1e-12));
}

inline void velocity_to_normal_tangent(double u, double v, double nx, double ny,
    double& un, double& ut)
{
    un = u * nx + v * ny;
    ut = -u * ny + v * nx;
}

inline void normal_tangent_to_velocity(double un, double ut, double nx, double ny,
    double& u, double& v)
{
    u = un * nx - ut * ny;
    v = un * ny + ut * nx;
}

inline void reflect_normal(const double* UL, double nx, double ny, double* UR)
{
    UR[0] = UL[0];
    const double rho = std::max(UL[0], 1e-12);
    const double u = UL[1] / rho;
    const double v = UL[2] / rho;
    const double un = u * nx + v * ny;
    UR[1] = rho * (u - 2.0 * un * nx);
    UR[2] = rho * (v - 2.0 * un * ny);
    UR[3] = UL[3];
    UR[4] = UL[4];
}

// Four-equation Euler HLLC flux. The transported SA variable is upwinded by
// the caller using flux[0], matching the solver's existing discretization.
inline void hllc_euler_flux(const double* UL, const double* UR,
    double nx, double ny, double gamma, double* flux)
{
    const double gamma_minus_one = gamma - 1.0;
    auto get_primitive = [gamma_minus_one](const double* U,
        double& rho, double& u, double& v, double& p) {
        rho = std::max(U[0], 1e-12);
        u = U[1] / rho;
        v = U[2] / rho;
        p = std::max(gamma_minus_one *
            (U[3] - 0.5 * rho * (u * u + v * v)), 1e-12);
    };

    double rhoL, uL, vL, pL, rhoR, uR, vR, pR;
    get_primitive(UL, rhoL, uL, vL, pL);
    get_primitive(UR, rhoR, uR, vR, pR);
    const double unL = uL * nx + vL * ny;
    const double unR = uR * nx + vR * ny;
    const double aL = std::sqrt(gamma * pL / rhoL);
    const double aR = std::sqrt(gamma * pR / rhoR);
    const double SL = std::min(unL - aL, unR - aR);
    const double SR = std::max(unL + aL, unR + aR);

    if (SL >= 0.0) {
        flux[0] = UL[0] * unL;
        flux[1] = UL[1] * unL + pL * nx;
        flux[2] = UL[2] * unL + pL * ny;
        flux[3] = (UL[3] + pL) * unL;
        return;
    }
    if (SR <= 0.0) {
        flux[0] = UR[0] * unR;
        flux[1] = UR[1] * unR + pR * nx;
        flux[2] = UR[2] * unR + pR * ny;
        flux[3] = (UR[3] + pR) * unR;
        return;
    }

    double denominator = rhoL * (SL - unL) - rhoR * (SR - unR);
    if (std::abs(denominator) < 1e-10) {
        denominator = denominator >= 0.0 ? 1e-10 : -1e-10;
    }
    const double Sstar = (pR - pL + rhoL * unL * (SL - unL)
        - rhoR * unR * (SR - unR)) / denominator;

    if (Sstar >= 0.0) {
        double dSL = SL - Sstar;
        if (std::abs(dSL) < 1e-10) dSL = dSL >= 0.0 ? 1e-10 : -1e-10;
        const double factor = rhoL * (SL - unL) / dSL;
        double dSL_un = SL - unL;
        if (std::abs(dSL_un) < 1e-10) dSL_un = dSL_un >= 0.0 ? 1e-10 : -1e-10;
        const double Us[4] = {
            factor,
            factor * (uL + (Sstar - unL) * nx),
            factor * (vL + (Sstar - unL) * ny),
            factor * (UL[3] / rhoL +
                (Sstar - unL) * (Sstar + pL / (rhoL * dSL_un)))
        };
        flux[0] = UL[0] * unL + SL * (Us[0] - UL[0]);
        flux[1] = UL[1] * unL + pL * nx + SL * (Us[1] - UL[1]);
        flux[2] = UL[2] * unL + pL * ny + SL * (Us[2] - UL[2]);
        flux[3] = (UL[3] + pL) * unL + SL * (Us[3] - UL[3]);
        return;
    }

    double dSR = SR - Sstar;
    if (std::abs(dSR) < 1e-10) dSR = dSR >= 0.0 ? 1e-10 : -1e-10;
    const double factor = rhoR * (SR - unR) / dSR;
    double dSR_un = SR - unR;
    if (std::abs(dSR_un) < 1e-10) dSR_un = dSR_un >= 0.0 ? 1e-10 : -1e-10;
    const double Us[4] = {
        factor,
        factor * (uR + (Sstar - unR) * nx),
        factor * (vR + (Sstar - unR) * ny),
        factor * (UR[3] / rhoR +
            (Sstar - unR) * (Sstar + pR / (rhoR * dSR_un)))
    };
    flux[0] = UR[0] * unR + SR * (Us[0] - UR[0]);
    flux[1] = UR[1] * unR + pR * nx + SR * (Us[1] - UR[1]);
    flux[2] = UR[2] * unR + pR * ny + SR * (Us[2] - UR[2]);
    flux[3] = (UR[3] + pR) * unR + SR * (Us[3] - UR[3]);
}

} // namespace dg
