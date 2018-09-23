#include "random.hpp"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
void random_fill(std::vector<int>::iterator b,
                 std::vector<int>::iterator e,
                 int low = 0,
                 int high = 1000)
{
  std::for_each(b, e, [&](int& a) { a = randInt(low, high); });
}
int main(int argc, char** argv)
{
    int n = 11;
    if (argc >= 2)
    {
      n = std::stoi(argv[1]);
    }
    std::cout << argc << std::endl;

    /* ---------------------- */
    /* Generate Unsorted Data */
    /* ---------------------- */
    std::vector<int> unsorted;
    unsorted.resize(n);
    random_fill(std::begin(unsorted), std::end(unsorted));

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(begin(unsorted), end(unsorted));
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time =
      std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Time: " << total_time << " ms\n";
}
