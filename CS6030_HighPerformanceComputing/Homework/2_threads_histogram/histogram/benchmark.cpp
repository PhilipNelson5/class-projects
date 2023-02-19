#include "helpers.hpp"
#include "histogram.hpp"
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
int main(int argc, char** argv)
{
  const auto [thread_count, bin_count, min_meas, max_meas, data_count, random_seed] = parse_args(argc, argv);

  std::srand(random_seed);

  auto data = generate_data(data_count, min_meas, max_meas);
  
  const auto trials = 50;
  double total_time_ms = 0;
  for (auto i = 0; i < trials; ++i)
  {
    auto start = std::chrono::high_resolution_clock::now();
    auto [bin_maxes, bin_counts] = histogram(bin_count, min_meas, max_meas, data, thread_count);
    (void)bin_maxes;
    (void)bin_counts;
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = finish - start;
    const double time_ms = ms.count();
    total_time_ms += time_ms;
#ifndef NDEBUG
    auto bin_sum = std::accumulate(begin(bin_counts), end(bin_counts), 0.0, std::plus<double>());
    assertm(bin_sum == data_count, "all elements are binned");
#endif
  }
  const auto average_time_ms = total_time_ms / trials;
  print(thread_count, ",", average_time_ms);

  return EXIT_SUCCESS;
}