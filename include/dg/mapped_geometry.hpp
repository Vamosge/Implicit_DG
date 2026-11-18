#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dg {

struct VolumeMetric {
    double x = 0.0;
    double y = 0.0;
    double xr = 0.0;
    double xs = 0.0;
    double yr = 0.0;
    double ys = 0.0;
    double jacobian = 0.0;
};

struct FaceQuadratureMetric {
    double x = 0.0;
    double y = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    double measure = 0.0;
};

// Immutable-after-construction metric storage. Cell-major indexing matches
// Field4D and the block matrix, which keeps later rank-local partitioning
// straightforward.
class MappedGeometryCache {
public:
    void reset(int nx, int ny, int volume_points, int face_count,
        int face_points)
    {
        if (nx <= 0 || ny <= 0 || volume_points <= 0 || face_count <= 0 ||
            face_points <= 0) {
            throw std::invalid_argument("mapped-geometry dimensions must be positive");
        }
        nx_ = nx;
        ny_ = ny;
        volume_points_ = volume_points;
        face_count_ = face_count;
        face_points_ = face_points;
        volume_.assign(static_cast<std::size_t>(nx) * ny * volume_points, {});
        face_.assign(static_cast<std::size_t>(nx) * ny * face_count * face_points, {});
    }

    bool empty() const noexcept { return volume_.empty(); }

    VolumeMetric& volume(int i, int j, int q) noexcept
    {
        return volume_[volume_index(i, j, q)];
    }

    const VolumeMetric& volume(int i, int j, int q) const noexcept
    {
        return volume_[volume_index(i, j, q)];
    }

    FaceQuadratureMetric& face(int i, int j, int f, int g) noexcept
    {
        return face_[face_index(i, j, f, g)];
    }

    const FaceQuadratureMetric& face(int i, int j, int f, int g) const noexcept
    {
        return face_[face_index(i, j, f, g)];
    }

    std::size_t memory_bytes() const noexcept
    {
        return volume_.size() * sizeof(VolumeMetric) +
            face_.size() * sizeof(FaceQuadratureMetric);
    }

private:
    std::size_t volume_index(int i, int j, int q) const noexcept
    {
        return (static_cast<std::size_t>(i) * ny_ + j) * volume_points_ + q;
    }

    std::size_t face_index(int i, int j, int f, int g) const noexcept
    {
        return ((static_cast<std::size_t>(i) * ny_ + j) * face_count_ + f) *
            face_points_ + g;
    }

    int nx_ = 0;
    int ny_ = 0;
    int volume_points_ = 0;
    int face_count_ = 0;
    int face_points_ = 0;
    std::vector<VolumeMetric> volume_;
    std::vector<FaceQuadratureMetric> face_;
};

} // namespace dg
