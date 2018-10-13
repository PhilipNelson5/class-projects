#include "cell.hpp"
#include "communication.hpp"
#include "output.hpp"
#include "random.hpp"
#include "rules.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <vector>

inline void help(std::string msg, int rank)
{
  if (rank == 0)
  {
    std::cout << msg << std::endl;

    std::cout
      << "usage gameOfLife width height generations makeOutput outputType"
      << std::endl;
  }
}

inline void end()
{
  MPI_Finalize();
  exit(EXIT_SUCCESS);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (argc < 4)
  {
    help("", rank);
    end();
  }

  auto shouldOutput = (bool)std::stoi(argv[4]);
  Output output_type = Output::ASCII;
  if (shouldOutput)
  {
    if (argc > 5)
    {
      output_type = static_cast<Output>(std::stoi(argv[5]));
    }
    else
    {
      help("Not enough arguments provided", rank);
      end();
    }
  }

  if (output_type > 1)
  {
    help("invalid outputType:\n  0 - term ascii\n  1 - gif\n", rank);
    end();
  }

  auto width = std::stoi(argv[1]);
  auto hight = std::stoi(argv[2]);
  auto iters = std::stoi(argv[3]);
  auto rpp = hight / world_size;
  auto aspectRatio = (double)width / hight;
  width = rpp * world_size;
  hight = width / aspectRatio;

  //=========================================================================//
  //                                  Master                                 //
  //=========================================================================//
  if (0 == rank)
  {
    std::cout << "Simulating:\n"
              << "---------------------\n"
              << "\n"
              << width << " x " << hight << " world\n"
              << "Processes: " << world_size << "\n"
              << "Rows Per Process: " << rpp << "\n"
              << "Make output: " << std::boolalpha << shouldOutput << "\n"
              << "---------------------\n"
              << "\n";

    /* setup world */
    World world(hight);
    std::for_each(begin(world), end(world), [&](std::vector<Cell>& row) {
      for (auto i = 0; i < width; ++i)
      {
        row.push_back(random_int(0, 6) == 0 ? Cell::ALIVE : Cell::DEAD);
      }
    });

    /* print the first generation */
    if (shouldOutput) print_world(world, output_type);

    /* send strips to other processes */
    for (int dest = 1, row = rpp; dest < world_size; ++dest)
    {
      for (int tag = 0; tag < rpp; ++tag, ++row)
      {
        MPI_Send(world[row].data(), width, MPI_INT, dest, tag, MPI_COMM_WORLD);
      }
    }

    /* copy out master's strip */
    World strip(rpp + 2);
    std::for_each(begin(strip), end(strip), [&](std::vector<Cell>& row) {
      row.resize(width);
    });
    for (int row = 0; row < rpp; ++row)
    {
      strip[row + 1] = world[row];
    }

    /* Run the simulation */
    auto simulationTime = 0.0;
    auto imageTime = 0.0;
    double t1, t2, t3;
    for (auto i = 0; i < iters; ++i)
    {
      t1 = MPI_Wtime();

      send_recv(strip, rank, world_size);
      natural_selection(strip);
      gatherMaster(world, strip, rpp, world_size);

      t2 = MPI_Wtime();

      if (shouldOutput)
      {
        print_world(world, output_type);
        std::cout << "generation " << i << " complete\n";
      }

      t3 = MPI_Wtime();

      simulationTime += t2 - t1;
      imageTime += t3 - t2;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    /* convert the images in ./images/ to a gif using imagemagick */
    auto t4 = MPI_Wtime();
    if (shouldOutput)
      system("convert -loop 0 -delay 25 `ls images | sort -g | sed "
             "'s-^-images/-'` out.gif");
    auto t5 = MPI_Wtime();
    std::cout << "Simulation Time: " << simulationTime
              << "\nImage Write Time: " << imageTime
              << "\ngif Creating time: " << t5 - t4 << std::endl;
    std::ofstream fout;
    // fout.open("benchmark.csv", std::fstream::app);
    // fout << world_size << ',' << simulationTime << std::endl;
  }
  //=========================================================================//
  //                                  Slave                                  //
  //=========================================================================//
  else
  {
    World strip(rpp + 2);

    /* resize rows to receive from neighbors */
    std::for_each(begin(strip), end(strip), [&](std::vector<Cell>& row) {
      row.resize(width);
    });

    /* receive strip from master */
    MPI_Status stat;
    for (auto row = 0; row < rpp; ++row)
    {
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
      MPI_Recv(strip[stat.MPI_TAG + 1].data(),
               width,
               MPI_INT,
               0,
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    /* run the simulation */
    for (auto i = 0; i < iters; ++i)
    {
      send_recv(strip, rank, world_size);
      natural_selection(strip);
      gatherSlave(strip, rpp);

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
