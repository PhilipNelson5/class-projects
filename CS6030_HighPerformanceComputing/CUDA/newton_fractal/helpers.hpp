#pragma once

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;

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
    cerr << "usage: " << program_name << " <width> <height> [frames = 300]\n"
        "  " << std::setw(w) << std::left << "width (int)" << "image width\n"
        "  " << std::setw(w) << std::left << "height (int)" << "image height\n"
        "  " << std::setw(w) << std::left << "frames (unsigned)" << "number of frames to generate\n"
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
std::tuple<int, int, int> parse_args(int argc, char **argv)
{
    int width, height, frames = 300;
    if (argc < 3)
        usage(argv[0]);
    try
    {
        width = std::stoi(argv[1]);
        height = std::stoi(argv[2]);
        if (argc > 3)
            frames = std::stoi(argv[3]);
    }
    catch (const std::exception &e)
    {
        cerr << e.what() << endl;
        usage(argv[0]);
    }
    return std::make_tuple(width, height, frames);
}

