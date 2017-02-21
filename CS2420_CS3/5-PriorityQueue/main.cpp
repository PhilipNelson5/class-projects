#include <chrono>
#include <iostream>
#include <random>
#include <thread>

#include "dynamicMedian.hpp"

namespace
{
  DynamicMedian med;
}

// pause thread for "sec" seconds
void wait(int sec)
{
  std::chrono::seconds time(sec);
  std::this_thread::sleep_for(time);
}

// returns a random int on the range low to high
int rand(int low, int high)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(low, high);
  return dist(mt);
}

// insert random numbers and display result for each insert
// good for small n
void randInsertVerbose(int n)
{
  int next;
  for (int i = 0; i < n; ++i)
  {
    next = rand(-1000, 1000);
    std::cout << "INSERT: " << next << std::endl;
    med.insert(next);
    std::cout << med.toString();
    wait(1);
  }
  std::cout << med.report();
  med.clear();
}

// insert random numbers and display report at the end
// good for large n
void randInsertQuiet(int n)
{
  for (int i = 0; i < n; ++i)
    med.insert(rand(-100, 100));
  std::cout << med.report();
  med.clear();
}

void line()
{
  std::cout << "--------------------------------------------------------" << std::endl;
}

int main()
{
  int n;
  std::string response = "y";
  while (true)
  {
    std::cout << "Quiet will only print a report at the end,\nVerbose will print after each insert"
              << std::endl
              << "[q] Quiet" << std::endl
              << "[v] Verbose" << std::endl
              << "[m] Manual entry" << std::endl
              << "[x] Quit" << std::endl;
    std::getline(std::cin, response);

    if (response[0] == 'q')
    {
      std::cout << "How many numbers will be added to the dynamic median? ";
      std::cin >> n;
      randInsertQuiet(n);
      std::cin.ignore();
    }
    if (response[0] == 'v')
    {
      std::cout << "How many numbers will be added to the dynamic median? ";
      std::cin >> n;
      randInsertVerbose(n);
      std::cin.ignore();
    }
    if (response[0] == 'm') break;

    if (response[0] == 'x') return EXIT_SUCCESS;

    line();
  }

  std::cout << "Enter in your own numbers, any other char to exit." << std::endl;
  int num;
  while (std::cin >> num)
  {
    med.insert(num);
    std::cout << med.toString();
  }
  std::cout << med.report();
  med.clear();
  line();
  return EXIT_SUCCESS;
}
