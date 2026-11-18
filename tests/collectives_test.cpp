#include "dg/collectives.hpp"

#include <cstdlib>

int main()
{
    const dg::SerialCollectives collectives;
    if (collectives.rank() != 0 || collectives.size() != 1) return EXIT_FAILURE;
    if (collectives.sum(2.5) != 2.5) return EXIT_FAILURE;
    if (collectives.minimum(-1.0) != -1.0) return EXIT_FAILURE;
    if (collectives.maximum(8.0) != 8.0) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
