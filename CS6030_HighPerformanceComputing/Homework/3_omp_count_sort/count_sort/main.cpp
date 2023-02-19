#include "count_sort.hpp"
#include "helpers.hpp"
#include "print.hpp"

#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

/**
 * @brief generates random data and computes the histogram
 *
 * in: see usage for list of input arguments
 *
 * out:
 * * bin_maxes  - a list containing the upper bound of each bin
 * * bin_counts - a list containing the number of elements in each bin
 *
 * @param argc commandline argument count
 * @param argv commandline arguments
 * @return int exit status
 */
int main(int argc, char **argv)
{
    const auto [thread_count, data_count, random_seed] = parse_args(argc, argv);

    std::srand(random_seed);

    auto data = generate_data(data_count);

    print(data);
    count_sort(begin(data), end(data), thread_count);
    // count_sort(data.data(), data.size(), thread_count);
    print(data);

    assertm(std::is_sorted(begin(data), end(data)), "data is sorted");

    return EXIT_SUCCESS;
}