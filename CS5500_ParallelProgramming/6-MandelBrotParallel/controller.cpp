#include "controller.hpp"
#include "calculator.hpp"
#include "color.hpp"
#include "ppmToBmp.hpp"
#include "writePNG.hpp"
#include <algorithm>
#include <mpi.h>
#include <vector>

enum TAG
{
  RENDER,
  STOP
};

void master(char** argv)
{
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int IMAGE_HIGHT = std::stoi(argv[1]);
  int IMAGE_WIDTH = std::stoi(argv[2]);
  int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HIGHT;
  const int MAX_ITERS = std::stoi(argv[3]);
  // const double X_MIN = std::stod(argv[4]);
  // const double X_MAX = std::stod(argv[5]);
  // const double Y_MAX = std::stod(argv[6]);
  // const double Y_MIN =
  // Y_MAX - (X_MAX - X_MIN) * ((double)(IMAGE_HIGHT)) / IMAGE_WIDTH;

  std::vector<int> imagebuf(IMAGE_SIZE);

  auto tests = 1;
  auto t1 = MPI_Wtime();
  for (auto i = 0; i < tests; ++i)
  {

    auto dest = 1;
    std::vector<int> buf(1);
    for (auto i = 0; i < IMAGE_HIGHT; ++i)
    {
      buf[0] = i;
      MPI_Send(buf.data(), 1, MPI_INT, dest++, TAG::RENDER, MPI_COMM_WORLD);

      if (dest > world_size - 1) dest = 1;
    }

    buf.resize(IMAGE_WIDTH);
    MPI_Status stat;
    for (auto i = 0; i < IMAGE_HIGHT; ++i)
    {
      MPI_Recv(buf.data(),
               buf.size(),
               MPI_INT,
               MPI_ANY_SOURCE,
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &stat);

      std::copy(
        begin(buf), end(buf), begin(imagebuf) + IMAGE_WIDTH * stat.MPI_TAG);
    }
  }

  auto t2 = MPI_Wtime();

  /*
  ppmToBmp(imagebuf,
           IMAGE_WIDTH,
           IMAGE_HIGHT,
           std::bind(color_scheme_2, std::placeholders::_1, MAX_ITERS),
           "brot.bmp");
  */

  std::vector<uint8_t> pixles;
  pixles.reserve(IMAGE_SIZE * 3);
  std::for_each(
    begin(imagebuf), end(imagebuf), [&MAX_ITERS, &pixles](int iter) {
      auto [r, g, b] = color_scheme_1(iter, MAX_ITERS);
      pixles.push_back(r);
      pixles.push_back(g);
      pixles.push_back(b);
    });

  std::string filename = "brot" + std::to_string(IMAGE_WIDTH) + ".png";
  if (save_png_libpng(filename, pixles.data(), IMAGE_WIDTH, IMAGE_HIGHT))
  {
    std::cout << "Successfully wrote " << filename << '\n';
  }
  else
  {
    std::cout << "Failed to write " << filename << '\n';
  }

  auto t3 = MPI_Wtime();

  std::cout << IMAGE_HIGHT << " x " << IMAGE_WIDTH << '\n'
            << "Time to compute: " << (t2 - t1) / tests
            << '\n'
            //<< "Time to compute: " << (t2 - t1) << '\n'
            << "Time to write image: " << t3 - t2 << '\n'
            << '\n'
            << '\n';
  for (auto i = 1; i < world_size; ++i)
  {
    std::vector<int> buf(1);
    MPI_Send(buf.data(), 0, MPI_INT, i, TAG::STOP, MPI_COMM_WORLD);
  }
}

void slave(char** argv)
{
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int IMAGE_HIGHT = std::stoi(argv[1]);
  int IMAGE_WIDTH = std::stoi(argv[2]);
  // int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HIGHT;
  const int MAX_ITERS = std::stoi(argv[3]);
  const double X_MIN = std::stod(argv[4]);
  const double X_MAX = std::stod(argv[5]);
  const double Y_MAX = std::stod(argv[6]);
  const double Y_MIN =
    Y_MAX - (X_MAX - X_MIN) * ((double)(IMAGE_HIGHT)) / IMAGE_WIDTH;

  MPI_Status stat;
  std::vector<int> line(1);
  std::vector<int> imagebuf;
  imagebuf.resize(IMAGE_WIDTH);

  auto ct = 0;
  do
  {
    ++ct;
    MPI_Recv(line.data(), 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    switch (stat.MPI_TAG)
    {
    case TAG::RENDER:

      render_row(imagebuf,
                 line[0],
                 X_MIN,
                 X_MAX,
                 Y_MIN,
                 Y_MAX,
                 IMAGE_HIGHT,
                 IMAGE_WIDTH,
                 MAX_ITERS);

      MPI_Send(
        imagebuf.data(), IMAGE_WIDTH, MPI_INT, 0, line[0], MPI_COMM_WORLD);

      break;
    }

  } while (stat.MPI_TAG != TAG::STOP);
  std::cout << rank << " finished, completed " << ct << " lines of the image\n";
}
