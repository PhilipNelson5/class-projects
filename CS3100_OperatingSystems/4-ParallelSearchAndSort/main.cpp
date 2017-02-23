#include "parallelLinearSearch.hpp"
#include "parallelQuickSort.hpp"
#include "progressBar.hpp"
#include "random.hpp"
#include "threadPool.hpp"
#include "threadSafeQueue.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

// Run with 2-8 threads in the thread pool.
// Use std::sort and std::find as serial algorithm to compare for speedup.
// Search and sort vectors of 100 to 1,000,000 elements (use a log scale).
// Include a report of timing, speedup, and efficiency.

namespace
{
  int tests = 5;
  int min_threads = 2;
  int max_threads = 8;
  int min_size = 1;
  int max_size = 6;
  int low = 2;
  int high = 100;
  bool verbose = false;
  int prog = 0;
  int total = 0;
}

std::vector<int> generate(int size, int low, int high)
{
  std::vector<int> v;
  for (int i = 0; i < size; ++i)
    v.push_back(rand(low, high));
  return v;
}

void testSort(std::ofstream &fout)
{
  Timer t;
  int threads = 0;
  int size = 10000;
  std::vector<int> list;
  std::vector<int> listCopy;
  fout << "Threads,Average,Standard Dev" << std::endl;
  fout << "Parallel Quick Sort" << std::endl;

  for (int i = min_size; i <= max_size; ++i) // SIZE
  {
    high = size = pow(10, i);
    fout << "size: " << size << std::endl;
    if (verbose) printMsg(prog, total, ("size: " + std::to_string(size)));

    for (int j = min_threads; j <= max_threads; ++j) // THREADS
    {
      threads = j;
      if (verbose) printMsg(prog, total, ("threads: " + std::to_string(threads)));

      std::sort(listCopy.begin(), listCopy.end());
      parallelQuickSort(listCopy, threads);

      for (int k = 0; k < tests; ++k) // TESTS
      {
        list = generate(size, low, high);
        listCopy = list;
        incProg(prog, total, "");
        t.time([&]() { parallelQuickSort(listCopy, threads); });
        // t.time([&]() { std::sort(listCopy.begin(), listCopy.end()); });
      }

      auto stdev = t.getStdDev();
      auto ave = t.getAverage();

      if (verbose) printMsg(prog, total, ("           Average: " + std::to_string(t.getAverage())));
      if (verbose) printMsg(prog, total, ("Standard Deviation: " + std::to_string(t.getStdDev())));
      fout << threads << "," << ave << "," << stdev << "," << std::endl;
      t.reset();
      std::string output = isSorted(listCopy) ? "Sorted" : "Unsorted";
      if (verbose) printMsg(prog, total, output);
    }
  }
}

void testSearch(std::ofstream &fout)
{
  Timer t;
  int threads = 0;
  int size = 10000;
  int found = -1;
  std::vector<int> list;
  fout << "Parallel Search" << std::endl;
  fout << "Threads,Average,Standard Dev" << std::endl;

  for (int i = min_size; i <= max_size; ++i) // SIZE
  {
    high = size = pow(10, i);
    fout << "size: " << size << std::endl;
    if (verbose) printMsg(prog, total, ("size: " + std::to_string(size)));

    for (int j = min_threads; j <= max_threads; ++j) // THREADS
    {
      threads = j;
      if (verbose) printMsg(prog, total, ("threads: " + std::to_string(threads)));

      for (int k = 0; k < tests; ++k) // TESTS
      {
        incProg(prog, total, "");
        list = generate(size, low, high);
        list[rand(0, size - 1)] = 1;
        found = -1;
        t.time([&found, &list, j]() { found = parallelLinearSearch(list, 1, j); });
        // t.time([=]() { std::find(list.begin(), list.end(), 1); });
      }
      auto stdev = t.getStdDev();
      auto ave = t.getAverage();

      if (verbose) printMsg(prog, total, ("           Average: " + std::to_string(ave)));
      if (verbose) printMsg(prog, total, ("Standard Deviation: " + std::to_string(stdev)));
      if (verbose)
        printMsg(prog, total, ((found >= 0) ? (std::string) "Found" : (std::string) "Not Found"));
      fout << threads << "," << ave << "," << stdev << std::endl;
      t.reset();
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc > 1)
    for (int i = 0; i < argc; ++i)
      switch (argv[i][1])
      {
      case 'v':
        verbose = true;
        break;
      }
  auto n = std::thread::hardware_concurrency();
  if (verbose) printMsg(prog, total, (std::to_string(n) + " concurrent threads are supported."));
  std::ofstream fout("data.csv");

  total = tests * (max_threads - min_threads + 1) * (max_size - min_size + 1) * 2;

  printMsg(prog, total, "");

  testSearch(fout);
  testSort(fout);

  std::cout << std::setw(87) << std::left << "\033[32m[DONE!]\033[0m" << std::endl;
}
