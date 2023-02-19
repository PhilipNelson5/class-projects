#pragma once

#include <algorithm>
#include <fstream>
#include <vector>
#include <iterator>
#include "print.hpp"

namespace Image
{
    /**
     * @brief representation of a pixel in an image
     */
    struct Pixel {
        uint8_t r, g, b;
    };

    bool operator==(const Pixel& a, const Pixel& b)
    {
      return a.r == b.r && a.g == b.g && a.b == b.b;
    }

    bool operator!=(const Pixel& a, const Pixel& b)
    {
      return !(a == b);
    }

    /**
     * @brief read a binary file to an image
     * 
     * @param filename name of the file to read from
     * @param size number of Pixels to read
     * @return new array with read data
     */
    auto read(const std::string &filename, const unsigned size)
    {
        std::ifstream fin(filename, std::ios::binary);
        Pixel *image = new Pixel[size];
        fin.read((char *)image, sizeof(Pixel) * size);
        return image;
    }

    /**
     * @brief read an image from a binary file
     * 
     * @tparam T type of elements in the image
     * @param filename name of the file to read from
     * @param width width of the image
     * @param height height of the image
     * @param channels number of channels per pixel
     * @return new image as a 1D array
     */
    auto read(const std::string &filename, const unsigned width, const unsigned height)
    {
        return read(filename, width * height);
    }

    /**
     * @brief write an image to a file in the PPM format 
     * see https://en.wikipedia.org/wiki/Netpbm#PPM_example
     * ppm can be viewed directly in some image editing programs
     * or converted with imagemagick `convert out.ppm out.png`
     * 
     * @param filename name of the file to write to
     * @param image array to write
     * @param width width of the image
     * @param height height of the image
     * @param depth thei maximum value for each channel
     */
    void write_ppm(
        const std::string &filename, const Pixel *image,
        const unsigned width, const unsigned height, const unsigned depth = 255)
    {
        std::ofstream fout(filename);
        fout << "P3\n"
             << width << " " << height << " " << depth << "\n";
        for (unsigned row = 0; row < height; ++row)
        {
            for (unsigned col = 0; col < width; ++col)
            {
                const auto p = image[row*width+col];
                fout << (int)p.r << " " << (int)p.g << " " << (int)p.b << "  ";
            }
            fout << "\n";
        }
        fout.close();
    }

    /**
     * @brief write an array to a binary file
     * 
     * @param filename name of the file to write to
     * @param output array to write
     * @param size size of the array
     */
    void write_binary(
        const std::string &filename, const Pixel *output,
        const unsigned size)
    {
        std::ofstream fout(filename, std::ios::binary);
        fout.write((char *) output, sizeof (Pixel) * size);
        fout.close();
    }

    /**
     * @brief write an image to a binary file
     * 
     * @tparam T type of elements in the image
     * @param filename name of the file to write to
     * @param image array to write
     * @param width width of the image
     * @param height height of the image
     * @param channels number of channels per pixel
     */
    void write_binary(
        const std::string &filename, const Pixel *image,
        const unsigned width, const unsigned height)
    {
        write_binary(filename, image, width * height);
    }

    /**
     * @brief compares two arrays for equivelance, not safe for floating point types
     * 
     * @tparam T type of the input matrix
     * @param a first input array
     * @param b second input array
     * @param size size of input arrays
     * @return true input arrays are the same
     * @return false input arrays differ by one or more elements
     */
    template <typename T>
    bool is_same(const T *a, const T *b, const unsigned size)
    {
        for (auto i = 0u; i < size; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

} // namespace Image
