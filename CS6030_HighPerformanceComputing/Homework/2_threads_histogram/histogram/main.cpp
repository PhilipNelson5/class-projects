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
  
  auto [bin_maxes, bin_counts] = histogram(bin_count, min_meas, max_meas, data, thread_count);

#ifndef NDEBUG
  auto bin_sum = std::accumulate(begin(bin_counts), end(bin_counts), 0.0, std::plus<double>());
  assertm(bin_sum == data_count, "all elements are binned");
#endif

  print("bin_maxes:", bin_maxes);
  print("bin_counts:", bin_counts);

  return EXIT_SUCCESS;
}