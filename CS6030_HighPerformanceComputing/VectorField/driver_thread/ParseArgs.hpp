#pragma once

#include <cxxopts.hpp>

struct Args
{
    int nThreads, nCurves, maxCurveLen, imgScaleFactor, vfH, vfW;
    std::string inputFileName;
    bool makeImage;
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
    cxxopts::Options options(argv[0], "Compute streamlines in a vector field with multiple threads");
    options.add_options()
        ("h,help", "Show usage")
        ("n,num_threads", "Number of threads", cxxopts::value<int>()->default_value("4"))
        ("s,num_streamlines", "Number of streamlines", cxxopts::value<int>()->default_value("100"))
        ("m,max_line_length", "Maximum number of line segments for each streamline", cxxopts::value<int>()->default_value("10000"))
        ("i,image_scale_factor", "Number of times larger to make the output image than the input field", cxxopts::value<int>()->default_value("2"))
        ("height", "input file height", cxxopts::value<int>()->default_value("600"))
        ("width", "input file width", cxxopts::value<int>()->default_value("1300"))
        ("f,file", "input file", cxxopts::value<std::string>()->default_value("cyl2d_1300x600_float32.raw"))
        ("no_image", "skip image generation", cxxopts::value<bool>())
        ;

    try {

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(EXIT_SUCCESS);
        }
        const int nThreads = result["num_threads"].as<int>();
        const int nCurves = result["num_streamlines"].as<int>();
        const int maxCurveLen = result["max_line_length"].as<int>();
        const int imgScaleFactor = result["image_scale_factor"].as<int>();
        const int vfH = result["height"].as<int>();
        const int vfW = result["width"].as<int>();
        const std::string inputFileName = result["file"].as<std::string>();
        const bool makeImage = !result.count("no_image");
        return Args { nThreads, nCurves, maxCurveLen, imgScaleFactor, vfH, vfW, inputFileName, makeImage };

    } catch (cxxopts::argument_incorrect_type e) {
        std::cout << e.what() << std::endl;
    } catch (cxxopts::option_not_exists_exception e) {
        std::cout << e.what() << std::endl;
    }
    std::cout << options.help() << std::endl;
    exit(EXIT_FAILURE);
}