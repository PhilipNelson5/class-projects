#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <tuple>

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
    cerr << "usage: " << program_name << "[image = gc_1024x1024.raw] [width = 1024] [height = 1024]\n"
        "  " << std::setw(w) << std::left << "image (string)" << "image file path\n"
        "  " << std::setw(w) << std::left << "width (uint)" << "image width\n"
        "  " << std::setw(w) << std::left << "height (uint)" << "image height\n"
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
std::tuple<std::string, unsigned, unsigned> parse_args(int argc, char **argv)
{
    std::string filename = "gc_1024x1024.raw";
    unsigned width = 1024, height = 1024;

    try
    {
        if(argc > 1) filename = argv[1];
        if(argc > 2) width = std::stoul(argv[2]);
        if(argc > 3) height = std::stoul(argv[3]);
    }
    catch (const std::exception &e)
    {
        cerr << e.what() << endl;
        usage(argv[0]);
    }
    return std::make_tuple(filename, width, height);
}

