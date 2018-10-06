#include "calculator.hpp"
#include "color.hpp"
#include "ppmToBmp.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <vector>

/**
 * Renders the Mandelbrot set
 *
 * @param X_MIN       The minimum real (x) value of the image
 * @param X_MAX       The maximum real (x) value of the image
 * @param Y_MIN       The minimum imaginary (y) value of the image
 * @param Y_MAX       The maximum imaginary (y) value of the image
 * @param IMAGE_HIGHT The height in pixels of the image
 * @param IMAGE_WIDTH The width in pixels of the image
 * @param MAX_ITERS   The maximum number of iterations to attempt
 * @return All the iteration data
 */
std::vector<int> render(double X_MIN,
                        double X_MAX,
                        double Y_MIN,
                        double Y_MAX,
                        int IMAGE_HIGHT,
                        int IMAGE_WIDTH,
                        int MAX_ITERS)
{
  std::vector<int> imagebuf;
  imagebuf.reserve(IMAGE_HIGHT * IMAGE_WIDTH);
  for (int i = 0; i < IMAGE_HIGHT; ++i)
  {
    for (int j = 0; j < IMAGE_WIDTH; ++j)
    {
      imagebuf.push_back(mandelbrot(
        i, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX));
    }
  }

  return imagebuf;
}

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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (0 == rank)
  {
    if (argc < 5)
    {
      std::cerr << "incorrect number of arguments\n"
                << "mandelbrot HEIGHT WIDTH ITERATIONS X_MIN X_MAX Y_MIN\n";
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    int IMAGE_HIGHT = std::stoi(argv[1]);
    int IMAGE_WIDTH = std::stoi(argv[2]);
    int MAX_ITERS = std::stoi(argv[3]);
    double X_MIN = std::stod(argv[4]);
    double X_MAX = std::stod(argv[5]);
    double Y_MAX = std::stod(argv[6]);
    double Y_MIN =
      Y_MAX - (X_MAX - X_MIN) * ((double)(IMAGE_HIGHT)) / IMAGE_WIDTH;

    std::vector<int> imagebuf;

    auto t1 = MPI_Wtime();

    // auto tests = 10u;
    // for (int size = 256; size < 2000; size *= 2)
    // {
      // for (auto i = 0u; i < tests; ++i)
        imagebuf = render(
          X_MIN, X_MAX, Y_MIN, Y_MAX, IMAGE_HIGHT, IMAGE_WIDTH, MAX_ITERS);

      auto t2 = MPI_Wtime();

      ppmToBmp(imagebuf,
      IMAGE_WIDTH,
      IMAGE_HIGHT,
      std::bind(color_scheme_2, std::placeholders::_1, MAX_ITERS),
      "brot.bmp");

      auto t3 = MPI_Wtime();

      std::cout << IMAGE_HIGHT << " x " << IMAGE_WIDTH << '\n'
                // << "Time to compute: " << (t2 - t1) / tests << '\n'
                << "Time to compute: " << (t2 - t1) << '\n'
                << "Time to write image: " << t3 - t2 << '\n'
                << "Total time: " << t3 - t1 << '\n'
                << '\n';
    // }
  }

  MPI_Finalize();

  return (EXIT_SUCCESS);
}
