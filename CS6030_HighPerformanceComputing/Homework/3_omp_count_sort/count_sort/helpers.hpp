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
#define assertm(exp, msg) assert(((void)msg, exp))

/**
 * @brief display usage message about this program
 *
 * @param program_name the first commandline argument containing the name used to invoke the program
 * @return * void
 */
void usage(char *program_name)
{
    const int w = 27;
    /* clang-format off */
    cerr << "usage: " << program_name << " <thread count> <data count> [random_seed = 100]\n"
        "  " << std::setw(w) << std::left << "thread count (int)" << "number of threads to use in histogram calculation \n"
        "    " << "hardware concurrency: " << std::thread::hardware_concurrency() << "\n"
        "  " << std::setw(w) << std::left << "data count (int)" << "number of random values to feed into histogram\n"
        "  " << std::setw(w) << std::left << "random seed (unsigned)" << "value to provide to srand\n"
        << endl;
    /* clang-format on */

    std::exit(EXIT_FAILURE);
}

/**
 * @brief parse commandline arguments
 *
 * @param argc commandline argument count
 * @param argv commandline arguments
 * @return auto all the commandline arguments
 */
auto parse_args(int argc, char **argv)
{
    int thread_count, data_count;
    unsigned random_seed = 100u;
    if (argc < 3)
        usage(argv[0]);
    try
    {
        thread_count = std::stoi(argv[1]);
        data_count = std::stoi(argv[2]);
        if (argc > 3)
            random_seed = std::stoul(argv[3]);
    }
    catch (const std::exception &e)
    {
        cerr << e.what() << endl;
        usage(argv[0]);
    }
    return std::make_tuple(thread_count, data_count, random_seed);
}

/**
 * @brief generate a vector of <data_count> random double values between <min_meas> and <max_meas>
 *
 * @param data_count number of values to generate
 * @param min_meas minimum value
 * @param max_meas maximum value
 * @return auto vector of data
 */
auto generate_data(const int data_count)
{
    std::vector<int> data;
    data.reserve(data_count);
    const auto g = [=]() { return 1 + rand() % data_count; };
    std::generate_n(std::back_inserter(data), data_count, g);
    return data;
}
