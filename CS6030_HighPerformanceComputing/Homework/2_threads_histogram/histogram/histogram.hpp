#pragma once

#include "print.hpp"
#include <algorithm>
#include <iterator>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

/**
 * @brief creates a vector of bin maxes based on inputs
 * 
 * @tparam T type of the measurements
 * @param min_meas minimum measurement
 * @param max_meas maximum measurement
 * @param bin_count number of bins
 * @return auto vector of bin maxes
 */
template<typename T>
auto calculate_bin_maxes(const T min_meas, const T max_meas, const int bin_count)
{
  const double bin_size = (max_meas - min_meas) / bin_count;
  std::vector<double> bin_maxes;
  bin_maxes.reserve(bin_count);

  // generator function
  const auto g = [min_meas, bin_size](){
    T last = min_meas;
    return [=]() mutable {
      return last += bin_size;
    };
  }();

  std::generate_n(std::back_inserter(bin_maxes), bin_count, g);
  return bin_maxes;
}

/**
 * @brief Get the bin that the element goes in
 * 
 * @tparam T type of element
 * @param elem element to bin
 * @param bin_maxes the vector of bin maxes
 * @return int the bin that the element goes in
 */
template<typename T>
int get_bin(const T elem, const std::vector<T>& bin_maxes)
{
  const auto bin_it = std::find_if(begin(bin_maxes), end(bin_maxes), [elem](const T bin_max){
    return elem < bin_max;
  });
  return std::distance(begin(bin_maxes), bin_it);
}

/**
 * @brief calculate the histogram of the input data set
 * 
 * @tparam T the type of the input data
 * @param bin_count number of bins
 * @param data the data
 * @param bin_maxes a vector containing the maximin value of each bin
 * @return auto bin_maxes and bin_counts: vector of bin maxes, vector of bin counts
 */
template<typename T, typename ForwardIt>
auto histogram(ForwardIt first, ForwardIt last, const int bin_count, const std::vector<T>& bin_maxes)
{
  std::vector<int> bin_counts;
  bin_counts.resize(bin_count);
  
  std::for_each(first, last, [bin_maxes, &bin_counts](const T elem){
    const int bin = get_bin(elem, bin_maxes);
    ++bin_counts[bin];
  });
  
  return bin_counts;
}

/**
 * @brief calculate the histogram of the input data set in parallel
 * 
 * @tparam T type of elements in input data
 * @param bin_count number of bins
 * @param min_meas minimum measurement
 * @param max_meas maximum measurement
 * @param data the data
 * @param thread_count number of threads
 * @return auto bin_maxes and bin_counts: vector of bin maxes, vector of bin counts
 */
template<typename T>
auto histogram(const int bin_count, const T min_meas, const T max_meas, const std::vector<T>& data, const int thread_count = 1)
{
  std::mutex mutex; 

  std::vector<double> bin_maxes = calculate_bin_maxes(min_meas, max_meas, bin_count);

  std::vector<int> bin_counts;
  bin_counts.resize(bin_count);
  
  std::vector<std::thread> threads;

  for (int i = 0; i < thread_count; ++i)
  {
    threads.emplace_back([=, &data, &bin_maxes, &bin_counts, &mutex](){
      const int rank = i;
      const int count = data.size() / thread_count;
      auto my_bin_counts = histogram(
        cbegin(data) + rank * count,
        rank != thread_count - 1 
          ? cbegin(data) + (rank + 1) * count 
          : cend(data),
        bin_count, bin_maxes);
      
      {
        std::lock_guard<std::mutex> lock(mutex);
        std::transform(begin(bin_counts), end(bin_counts), begin(my_bin_counts), begin(bin_counts),
          [](const int count1, const int count2) -> int {
            return count1 + count2;
          }
        );
      }
    });
  }

  std::for_each(begin(threads), end(threads), [](std::thread& thread){ thread.join(); });
  return std::make_tuple(bin_maxes, bin_counts);
}