#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

// A minimal forward-mode AD scalar for functions of two variables.
struct Dual2 {
    double value = 0.0;
    std::array<double, 2> derivative = {0.0, 0.0};

    Dual2() = default;
    Dual2(double v) : value(v) {}
};

Dual2 variable(double value, int index)
{
    Dual2 result(value);
    result.derivative[index] = 1.0;
    return result;
}

Dual2 operator+(const Dual2& a, const Dual2& b)
{
    Dual2 result(a.value + b.value);
    for (int p = 0; p < 2; ++p) {
        result.derivative[p] = a.derivative[p] + b.derivative[p];
    }
    return result;
}

Dual2 operator*(const Dual2& a, const Dual2& b)
{
    Dual2 result(a.value * b.value);
    for (int p = 0; p < 2; ++p) {
        result.derivative[p] =
            a.derivative[p] * b.value + a.value * b.derivative[p];
    }
    return result;
}

Dual2 exp(const Dual2& a)
{
    Dual2 result(std::exp(a.value));
    for (int p = 0; p < 2; ++p) {
        result.derivative[p] = result.value * a.derivative[p];
    }
    return result;
}

std::array<Dual2, 2> residual(const Dual2& x, const Dual2& y)
{
    // F_0 = x^2 + y, F_1 = x*y + exp(y).
    return {x * x + y, x * y + exp(y)};
}

int main()
{
    // Identity seeds return the complete 2x2 Jacobian in one evaluation.
    const Dual2 x = variable(2.0, 0); // dx/dx=1, dx/dy=0
    const Dual2 y = variable(0.0, 1); // dy/dx=0, dy/dy=1
    const auto F = residual(x, y);

    // At (2,0): F=(4,1), J=[[4,1],[0,3]].
    assert(std::abs(F[0].value - 4.0) < 1e-14);
    assert(std::abs(F[1].value - 1.0) < 1e-14);
    assert(std::abs(F[0].derivative[0] - 4.0) < 1e-14);
    assert(std::abs(F[0].derivative[1] - 1.0) < 1e-14);
    assert(std::abs(F[1].derivative[0] - 0.0) < 1e-14);
    assert(std::abs(F[1].derivative[1] - 3.0) < 1e-14);

    // A single directional seed (3,-1) returns J*v instead of all columns.
    Dual2 xd(2.0);
    Dual2 yd(0.0);
    xd.derivative[0] = 3.0;
    yd.derivative[0] = -1.0;
    const auto Fdirection = residual(xd, yd);
    assert(std::abs(Fdirection[0].derivative[0] - 11.0) < 1e-14);
    assert(std::abs(Fdirection[1].derivative[0] + 3.0) < 1e-14);

    std::cout << "F = [" << F[0].value << ", " << F[1].value << "]\n";
    std::cout << "J = [[" << F[0].derivative[0] << ", "
              << F[0].derivative[1] << "], ["
              << F[1].derivative[0] << ", "
              << F[1].derivative[1] << "]]\n";
    std::cout << "J*[3,-1] = [" << Fdirection[0].derivative[0]
              << ", " << Fdirection[1].derivative[0] << "]\n";
}
