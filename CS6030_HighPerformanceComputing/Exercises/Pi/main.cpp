#include <iostream>
#include <cmath>
#include <numeric>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <iomanip>
#include <thread>
#include "Barrier.hpp"
using std::cout, std::endl;

double term(const int n, const double sign)
{
  return sign / (2.0 * n + 1.0);
}

/**
 * @brief serial program
 * 
 * @param n number of iterations
 * @return double calculated value of PI
 */
double a(const unsigned n)
{
  double sum = 0.0;
  double sign = 1.0;
  for (auto i = 0u; i < n; ++i, sign = -sign)
  {
    sum += term(i, sign);
  }
  return 4.0 * sum;
}

/**
 * @brief parallel implementation with single thread adding partial sums
 * 
 * @param n number of iterations
 * @return double calculated value of PI
 */
double b(const unsigned n, const int thread_count)
{
  std::vector<std::thread> threads;
  std::vector<double> partial_sums;
  Barrier barrier(thread_count);
  partial_sums.resize(thread_count);
  double pi = 0;
  for (auto rank = 0; rank < thread_count; ++rank) 
  {
    threads.emplace_back(
      [rank, n, thread_count, &partial_sums, &pi, &barrier]()
      {
        const unsigned my_n = n / thread_count;
        const unsigned start = my_n * rank;
        const unsigned end = start + my_n;
        
        double sign = start % 2 == 0 ? 1.0 : -1.0;
        double sum = 0.0;

        for (auto i = start; i < end; ++i, sign = -sign)
        {
          sum += term(i, sign);
        }
        partial_sums[rank] = sum;
        barrier.wait();
        if (rank == 0)
        {
          pi = 4.0 * std::accumulate(std::begin(partial_sums), std::end(partial_sums), 0.0, [](const double a, const double b){
            return a + b;
          });
        }
      }
    );
  }
  for (auto & thread : threads)
  {
    thread.join();
  }
  return pi;
}

void global_sum(std::vector<double>& values, const int rank, const unsigned size, const int level)
{
	const int mask = 1 << level;
	const int dest = rank ^ mask;

	if (dest > size - 1 || rank % mask != 0) return;
	if (rank > dest) 
	{
    values[dest] += values[rank];
	}
}

void global_sum(std::vector<double>& values, const int rank, const unsigned size, Barrier& barrier)
{
  const unsigned n = std::log2(size);
  for (int level = 0; level < n; ++level)
  {
    global_sum(values, rank, size, level);
    barrier.wait();
  }
}

/**
 * @brief parallel implementation with all threads reducing partial sums in a binary tree pattern
 * 
 * @param n number of iterations
 * @return double calculated value of PI
 */
double c(const unsigned n, const int thread_count)
{
  std::vector<std::thread> threads;
  std::vector<double> partial_sums;
  Barrier barrier(thread_count);
  partial_sums.resize(thread_count);
  double pi = 0;
  for (auto rank = 0; rank < thread_count; ++rank) 
  {
    threads.emplace_back(
      [rank, n, thread_count, &partial_sums, &pi, &barrier]()
      {
        const unsigned my_n = n / thread_count;
        const unsigned start = my_n * rank;
        const unsigned end = start + my_n;
        
        double sign = start % 2 == 0 ? 1.0 : -1.0;
        double sum = 0.0;

        for (auto i = start; i < end; ++i, sign = -sign)
        {
          sum += term(i, sign);
        }
        partial_sums[rank] = sum;

        barrier.wait();
        
        global_sum(partial_sums, rank, thread_count, barrier);
        
        if (rank == 0) pi = 4.0 * partial_sums[0];
      }
    );
  }
  for (auto & thread : threads)
  {
    thread.join();
  }
  return pi;
}

/**
 * @brief parallel implementation with all threads adding to global, mutex protected sum
 * 
 * @param n number of iterations
 * @return double calculated value of PI
 */
double d(const unsigned n, const int thread_count)
{
  std::vector<std::thread> threads;
  std::vector<double> partial_sums;
  std::mutex m;
  double sum = 0;
  double pi = 0;
  for (auto rank = 0; rank < thread_count; ++rank) 
  {
    threads.emplace_back(
      [rank, n, thread_count, &sum, &m, &pi]()
      {
        const unsigned my_n = n / thread_count;
        const unsigned start = my_n * rank;
        const unsigned end = start + my_n;
        
        double sign = start % 2 == 0 ? 1.0 : -1.0;
        double partial_sum = 0.0;

        for (auto i = start; i < end; ++i, sign = -sign)
        {
          partial_sum += term(i, sign);
        }
        
        {
          std::unique_lock<std::mutex> lk(m);
          sum += partial_sum;
        }

        if (rank == 0) pi = 4.0 * sum;
      }
    );
  }
  for (auto & thread : threads)
  {
    thread.join();
  }
  return pi;
}

/**
 * @brief time the execution of a function
 * 
 * @tparam T return type of the function
 * @tparam F function
 * @param f the function to be timed
 * @return std::tuple<T, double> [function return value, execution time in ms]
 */
template<typename T, typename F>
std::tuple<T, double> time(F f)
{
  auto start = std::chrono::high_resolution_clock::now();
  auto ret_val = f();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> ms = end - start;

  return {ret_val, ms.count()};
}

int main()
{
  double total = 0.0;
  
  // for (int thread_count = 2; thread_count < 100; thread_count += 1)
  for (int thread_count = 2; thread_count < 2049; thread_count *= 2)
  // for (int thread_count = 5; thread_count < 1001; thread_count += 5)
  {
    const unsigned n = 2147483647 - 1;
    const unsigned iterations = n / thread_count * thread_count;

    cout << iterations << ',' << thread_count << ',';
    {
      auto [pi, ms] = time<double>([iterations]() -> double {return a(iterations);});
      total += pi; // If you don't do something with the result, pi, the optimizer will skip the computation
      cout << ms << ',';
    }{
      auto [pi, ms] = time<double>([iterations, thread_count]() -> double {return b(iterations, thread_count);});
      total += pi;
      cout << ms << ',';
    }{
      auto [pi, ms] = time<double>([iterations, thread_count]() -> double {return c(iterations, thread_count);});
      total += pi;
      cout << ms << ',';
    }{
      auto [pi, ms] = time<double>([iterations, thread_count]() -> double {return d(iterations, thread_count);});
      total += pi;
      cout << ms << ',';
    }
    cout << endl;
  }
  cout << total << endl;
  // cout << "PI: " << std::setprecision(10) << a(iterations) << endl;
  // cout << "PI: " << std::setprecision(10) << b(iterations, thread_count) << endl;
  // cout << "PI: " << std::setprecision(10) << c(iterations, thread_count) << endl;
  // cout << "PI: " << std::setprecision(10) << d(iterations, thread_count) << endl;
  
  return EXIT_SUCCESS;
}