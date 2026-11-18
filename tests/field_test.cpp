#include "dg/field.hpp"

#include <cstdlib>

int main()
{
    dg::Field4D<double> field(3, 2, 4, 5);
    field(2, 1, 3, 4) = 7.5;
    if (field.data.size() != 3U * 2U * 4U * 5U) return EXIT_FAILURE;
    if (field.cell_count() != 6U || field.values_per_cell() != 20U) return EXIT_FAILURE;
    if (field(2, 1, 3, 4) != 7.5) return EXIT_FAILURE;
    if (field.cell_data(2, 1)[19] != 7.5) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
