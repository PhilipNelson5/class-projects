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

    const auto data_raw = generate_data(data_count);
    const auto trials = 20;
    double total_time_ms = 0.0;

    for (auto i = 0; i < trials; ++i)
    {
        auto data = data_raw;

        auto start = std::chrono::high_resolution_clock::now();
        count_sort(begin(data), end(data), thread_count);
        // count_sort(data.data(), data.size(), thread_count);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = finish - start;
        const double time_ms = ms.count();
        total_time_ms += time_ms;
    }
    const auto average_time_ms = total_time_ms / trials;
    print(thread_count, ",", average_time_ms);

    return EXIT_SUCCESS;
}