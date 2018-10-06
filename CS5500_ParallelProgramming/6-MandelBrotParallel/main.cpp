#include "controller.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <vector>

/*
 *std::vector<int> render_contiguous(int start,
 *                                   int end,
 *                                   double X_MIN,
 *                                   double X_MAX,
 *                                   double Y_MIN,
 *                                   double Y_MAX,
 *                                   int IMAGE_HIGHT,
 *                                   int IMAGE_WIDTH,
 *                                   int MAX_ITERS)
 *{
 *  std::vector<int> imagebuf;
 *  imagebuf.reserve(end - start + 1);
 *  for (auto i = start / IMAGE_WIDTH; i < IMAGE_HIGHT; ++i)
 *  {
 *    for (auto j = start % IMAGE_WIDTH; j < IMAGE_WIDTH; ++j)
 *    {
 *      imagebuf.push_back(mandelbrot(
 *        i, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS, X_MIN, X_MAX, Y_MIN,
 *Y_MAX));
 *    }
 *  }
 *
 *  return imagebuf;
 *}
 */

/**
 * Write the iteration data as a ppm image
 *
 * @param imagebuf The array containing all the iteration data
 * @param width    The width of the image in pixels
 * @param height   The height of the image in pixels
 * @param filename The name of the ppm image
 */
void write_ppm_image(std::vector<int> const& imagebuf,
                     const int width,
                     const int height,
                     std::string fileName)
{
  std::ofstream fout(fileName);
  fout << "P3\n" << width << " " << height << "\n255\n";
  std::for_each(begin(imagebuf), end(imagebuf), [&fout](int num) {
    fout << num << " " << num << " " << num << " ";
  });
  fout << std::endl;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 6)
  {
    if (0 == rank)
    {
      std::cerr << "incorrect number of arguments\n"
                << "mandelbrot height width max_iters x_min x_max y_min\n";
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (0 == rank)
  {
    master(argv);
  }
  else
  {
    slave(argv);
  }

  MPI_Finalize();

  return (EXIT_SUCCESS);
}
