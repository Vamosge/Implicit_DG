#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dg {

// Contiguous cell-major storage for DG fields:
//   (cell-i, cell-j, basis-node, equation/component).
// Keeping this ownership-neutral container outside the solver is the first
// step toward replacing nx/ny with rank-local extents and halo storage.
template <class T>
class Field4D {
public:
    std::size_t _Nx = 0;
    std::size_t _Ny = 0;
    std::size_t _dimPk = 0;
    std::size_t _NumEq = 0;
    std::vector<T> data;

    Field4D() = default;

    Field4D(std::size_t nx, std::size_t ny, std::size_t basis_size,
        std::size_t component_count)
    {
        reset(nx, ny, basis_size, component_count);
    }

    void reset(std::size_t nx, std::size_t ny, std::size_t basis_size,
        std::size_t component_count, const T& initial_value = T{})
    {
        _Nx = nx;
        _Ny = ny;
        _dimPk = basis_size;
        _NumEq = component_count;
        data.assign(nx * ny * basis_size * component_count, initial_value);
    }

    [[nodiscard]] bool empty() const noexcept { return data.empty(); }
    [[nodiscard]] std::size_t cell_count() const noexcept { return _Nx * _Ny; }
    [[nodiscard]] std::size_t values_per_cell() const noexcept { return _dimPk * _NumEq; }

    T& operator()(std::size_t i, std::size_t j, std::size_t basis,
        std::size_t component) noexcept
    {
        return data[index(i, j, basis, component)];
    }

    const T& operator()(std::size_t i, std::size_t j, std::size_t basis,
        std::size_t component) const noexcept
    {
        return data[index(i, j, basis, component)];
    }

    T* cell_data(std::size_t i, std::size_t j) noexcept
    {
        return data.data() + index(i, j, 0, 0);
    }

    const T* cell_data(std::size_t i, std::size_t j) const noexcept
    {
        return data.data() + index(i, j, 0, 0);
    }

private:
    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j,
        std::size_t basis, std::size_t component) const noexcept
    {
        return (((i * _Ny + j) * _dimPk + basis) * _NumEq + component);
    }
};

} // namespace dg
