#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "cell.hpp"
#include "writePNG.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>

enum Output
{
  ASCII, /**< print in ascii to the terminal */
  GIF    /**< save as sequentially named pngs and converted to a gif */
};

using World = std::vector<std::vector<Cell>>;

/**
 * Print a row of the world in ascii to the terminal
 *
 * @param row The row of the world to print
 */
void print_ascii_row(std::vector<Cell> const& row)
{
  for (auto i = 0u; i < row.size(); ++i)
  {
    std::cout << (row[i] == Cell::ALIVE ? "*" : ".");
  }
  std::cout << '\n';
}

/**
 * Print the world in ascii to the terminal
 *
 * @param world The representation of the whole world
 */
void print_ascii_world(World const& world)
{
  std::cout << "\033[2J\033[1;1H";
  for (auto i = 0u; i < world.size(); ++i)
  {
    for (auto j = 0u; j < world[0].size(); ++j)
    {
      std::cout << (world[i][j] == Cell::ALIVE ? "*" : ".");
    }
    std::cout << '\n';
  }
  std::cout << std::setw(world[0].size()) << std::setfill('-') << '-'
            << std::endl;
}

/**
 * Save the world as a png image
 *
 * @param world The representation of the whole world
 */
void print_png_world(World const& world)
{
  static int image_num = 0;
  int scale = 3;
  std::vector<uint8_t> image;
  std::for_each(begin(world), end(world), [&](auto row) {
    for (int l = 0; l < scale; ++l)
    {
      std::for_each(begin(row), end(row), [&](auto c) {
        auto color = c == Cell::ALIVE ? 0 : 255;
        for (auto i = 0; i < scale; ++i)
        {
          image.push_back(color);
          image.push_back(color);
          image.push_back(color);
        }
      });
    }
  });
  std::string filename = "images/" + std::to_string(++image_num) + ".png";
  save_png_libpng(
    filename, image.data(), world[0].size() * scale, world.size() * scale);
}

/**
 * Print the world in the given output type
 *
 * @param world       The representation of the whole world
 * @param output_type The output type
 */
void print_world(World world, Output output_type)
{
  switch (output_type)
  {
  case Output::ASCII:
    print_ascii_world(world);
    sleep(1);
    break;
  case Output::GIF:
    print_png_world(world);
    break;
  }
}

#endif
