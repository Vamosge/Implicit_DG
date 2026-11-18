#include "dg/mapped_geometry.hpp"

#include <cassert>
#include <stdexcept>

int main()
{
    dg::MappedGeometryCache cache;
    cache.reset(3, 2, 4, 4, 3);
    assert(!cache.empty());

    cache.volume(2, 1, 3).jacobian = 7.5;
    cache.face(2, 1, 3, 2).measure = 4.25;
    assert(cache.volume(2, 1, 3).jacobian == 7.5);
    assert(cache.face(2, 1, 3, 2).measure == 4.25);
    assert(cache.volume(0, 0, 0).jacobian == 0.0);
    assert(cache.memory_bytes() > 0);

    bool rejected = false;
    try {
        dg::MappedGeometryCache invalid;
        invalid.reset(0, 2, 4, 4, 3);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
