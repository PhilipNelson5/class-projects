#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using std::cout, std::cerr, std::endl;

/**
 * @brief wrapper around assert macro to allow nicer syntax for adding a message
 */
#define assertm(exp, msg) assert(((void)msg, exp));

/**
 * @brief display usage message about this program
 * 
 * @param program_name the first commandline argument containing the name used to invoke the program
 * @return * void 
 */
void usage(char* program_name)
{
  const int w = 27;
  cerr << "usage: " << program_name << " <bin count> <min meas> <max meas> <data count> [random_seed = 100]\n"
    "  " << std::setw(w) << std::left << "bin count (int) " << "number of bins in the histogram\n"
    "  " << std::setw(w) << std::left << "min meas (float)" << "minimum value to use in pseudo random number generation\n"
    "  " << std::setw(w) << std::left << "max meas (float)" << "maximum value to use in pseudo random number generation\n"
    "  " << std::setw(w) << std::left << "data count (int)" << "number of random values to feed into histogram\n"
    "  " << std::setw(w) << std::left << "random seed (unsigned)" << "value to provide to srand\n"
    << endl;
  
  std::exit(EXIT_FAILURE);
}

struct Args
{
  int bin_count, data_count;
  unsigned random_seed = 100u;
  float min_meas, max_meas;
};

/**
 * @brief parse commandline arguments
 * 
 * @param argc commandline argument count
 * @param argv commandline arguments
 * @return auto all the commandline arguments
 */
auto parse_args(int argc, char** argv)
{
  Args args;
  if (argc < 5) usage(argv[0]);
  try
  {
    args.bin_count = std::stoi(argv[1]);
    args.min_meas = std::stof(argv[2]);
    args.max_meas = std::stof(argv[3]);
    args.data_count = std::stoi(argv[4]);
    if (argc > 5) args.random_seed = std::stoul(argv[5]);
  }
  catch(const std::exception& e)
  {
    cerr << e.what() << endl;
    usage(argv[0]);
  }
  return args;
}

/**
 * @brief generate a vector of <data_count> random double values between <min_meas> and <max_meas>
 * 
 * @param data_count number of values to generate
 * @param min_meas minimum value
 * @param max_meas maximum value
 * @return auto vector of data
 */
template <typename T>
auto generate_data(const int data_count, const T min_meas, const T max_meas)
{
  std::vector<T> data;
  data.reserve(data_count);
  const auto g = [=](){ return min_meas + rand() / static_cast<T>(RAND_MAX) * (max_meas - min_meas); };
  std::generate_n(std::back_inserter(data), data_count, g);
  return data;
}
