#pragma once

#include <cstddef>
#include <vector>

namespace dg {

class WallGeometry {
public:
    void set_flat_wall(double x_begin, double x_end, double y = 0.0);

    void set_profile(std::vector<double> profile_x, std::vector<double> profile_y,
        std::size_t wall_begin, std::size_t wall_end);

    // General ordered polyline, including a non-monotone airfoil surface.
    void set_polyline(std::vector<double> wall_x, std::vector<double> wall_y);

    [[nodiscard]] bool empty() const noexcept { return wall_x_.size() < 2; }
    [[nodiscard]] bool has_profile() const noexcept { return profile_x_.size() >= 2; }
    [[nodiscard]] double x_begin() const noexcept;
    [[nodiscard]] double x_end() const noexcept;
    [[nodiscard]] double height(double x) const noexcept;
    [[nodiscard]] double distance(double x, double y, double floor = 0.0) const noexcept;

private:
    std::vector<double> profile_x_;
    std::vector<double> profile_y_;
    std::vector<double> wall_x_;
    std::vector<double> wall_y_;
    bool monotone_x_ = true;
};

} // namespace dg
