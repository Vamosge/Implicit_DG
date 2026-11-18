#include "dg/wall_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace dg {
namespace {

bool is_nondecreasing(const std::vector<double>& values)
{
    return std::adjacent_find(values.begin(), values.end(),
        [](double left, double right) { return right < left; }) == values.end();
}

} // namespace

void WallGeometry::set_flat_wall(double x_begin, double x_end, double y)
{
    if (!(x_end > x_begin)) throw std::invalid_argument("wall extent must have positive length");
    profile_x_ = {x_begin, x_end};
    profile_y_ = {y, y};
    wall_x_ = profile_x_;
    wall_y_ = profile_y_;
    monotone_x_ = true;
}

void WallGeometry::set_profile(std::vector<double> profile_x,
    std::vector<double> profile_y, std::size_t wall_begin, std::size_t wall_end)
{
    if (profile_x.size() != profile_y.size() || profile_x.size() < 2) {
        throw std::invalid_argument("wall profile coordinates are empty or inconsistent");
    }
    if (!is_nondecreasing(profile_x)) {
        throw std::invalid_argument("wall profile x coordinates must be nondecreasing");
    }
    if (wall_begin >= profile_x.size() || wall_end >= profile_x.size()
        || wall_end <= wall_begin) {
        throw std::invalid_argument("viscous-wall profile index range is invalid");
    }

    profile_x_ = std::move(profile_x);
    profile_y_ = std::move(profile_y);
    wall_x_.assign(profile_x_.begin() + static_cast<std::ptrdiff_t>(wall_begin),
        profile_x_.begin() + static_cast<std::ptrdiff_t>(wall_end + 1));
    wall_y_.assign(profile_y_.begin() + static_cast<std::ptrdiff_t>(wall_begin),
        profile_y_.begin() + static_cast<std::ptrdiff_t>(wall_end + 1));
    monotone_x_ = true;
}

void WallGeometry::set_polyline(std::vector<double> wall_x,
    std::vector<double> wall_y)
{
    if (wall_x.size() != wall_y.size() || wall_x.size() < 2) {
        throw std::invalid_argument("wall polyline coordinates are empty or inconsistent");
    }
    profile_x_.clear();
    profile_y_.clear();
    wall_x_ = std::move(wall_x);
    wall_y_ = std::move(wall_y);
    monotone_x_ = is_nondecreasing(wall_x_);
}

double WallGeometry::x_begin() const noexcept
{
    return wall_x_.empty() ? 0.0 : wall_x_.front();
}

double WallGeometry::x_end() const noexcept
{
    return wall_x_.empty() ? 0.0 : wall_x_.back();
}

double WallGeometry::height(double x) const noexcept
{
    if (profile_x_.empty()) return 0.0;
    if (x <= profile_x_.front()) return profile_y_.front();
    if (x >= profile_x_.back()) return profile_y_.back();
    const auto upper = std::upper_bound(profile_x_.begin(), profile_x_.end(), x);
    const std::size_t hi = static_cast<std::size_t>(upper - profile_x_.begin());
    const std::size_t lo = hi - 1;
    const double dx = profile_x_[hi] - profile_x_[lo];
    if (dx <= 0.0) return profile_y_[lo];
    const double fraction = (x - profile_x_[lo]) / dx;
    return profile_y_[lo] + fraction * (profile_y_[hi] - profile_y_[lo]);
}

double WallGeometry::distance(double x, double y, double floor) const noexcept
{
    if (wall_x_.size() < 2) return std::max(floor, 0.0);

    const auto segment_distance_squared = [&](std::size_t segment) {
        const double ax = wall_x_[segment];
        const double ay = wall_y_[segment];
        const double bx = wall_x_[segment + 1];
        const double by = wall_y_[segment + 1];
        const double vx = bx - ax;
        const double vy = by - ay;
        const double length_squared = vx * vx + vy * vy;
        double parameter = 0.0;
        if (length_squared > 1e-30) {
            parameter = ((x - ax) * vx + (y - ay) * vy) / length_squared;
            parameter = std::clamp(parameter, 0.0, 1.0);
        }
        const double dx = x - (ax + parameter * vx);
        const double dy = y - (ay + parameter * vy);
        return dx * dx + dy * dy;
    };

    if (!monotone_x_) {
        double best_squared = segment_distance_squared(0);
        for (std::size_t segment = 1; segment + 1 < wall_x_.size(); ++segment) {
            best_squared = std::min(best_squared, segment_distance_squared(segment));
        }
        return std::max(std::sqrt(std::max(best_squared, 0.0)), floor);
    }

    const auto upper = std::upper_bound(wall_x_.begin(), wall_x_.end(), x);
    std::size_t center = upper == wall_x_.begin()
        ? 0
        : static_cast<std::size_t>(upper - wall_x_.begin() - 1);
    center = std::min(center, wall_x_.size() - 2);
    double best_squared = segment_distance_squared(center);

    for (std::size_t segment = center; segment-- > 0;) {
        const double horizontal = x > wall_x_[segment + 1]
            ? x - wall_x_[segment + 1]
            : (x < wall_x_[segment] ? wall_x_[segment] - x : 0.0);
        if (horizontal * horizontal >= best_squared) break;
        best_squared = std::min(best_squared, segment_distance_squared(segment));
    }
    for (std::size_t segment = center + 1; segment + 1 < wall_x_.size(); ++segment) {
        const double horizontal = x > wall_x_[segment + 1]
            ? x - wall_x_[segment + 1]
            : (x < wall_x_[segment] ? wall_x_[segment] - x : 0.0);
        if (horizontal * horizontal >= best_squared) break;
        best_squared = std::min(best_squared, segment_distance_squared(segment));
    }
    return std::max(std::sqrt(std::max(best_squared, 0.0)), floor);
}

} // namespace dg
