#include "random.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

#define MCW MPI_COMM_WORLD

void print1per(int data, std::string title)
{
  int rank;
  int size;

  MPI_Comm_rank(MCW, &rank);
  MPI_Comm_size(MCW, &size);

  int* dArray = new int[size];
  MPI_Gather(&data, 1, MPI_INT, dArray, 1, MPI_INT, 0, MCW);

  if (rank == 0)
  {
    std::cout << title << '\n';
    for (int i = 0; i < size; ++i)
    {
      std::cout << std::setw(5) << i << std::setw(5) << dArray[i] << "\n";
    }
    std::cout << std::endl;
  }
}

int cube(int c, int sendData, int rank)
{
  int recvData;
  auto dest = rank ^ (1 << c);

  MPI_Send(&sendData, 1, MPI_INT, dest, 0, MCW);
  MPI_Recv(&recvData, 1, MPI_INT, dest, 0, MCW, MPI_STATUS_IGNORE);

  return recvData;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MCW, &rank);
  MPI_Comm_size(MCW, &size);

  if (0 != (size & (size - 1)))
  {
    if (rank == 0)
    {
      std::cerr << "There must be a power of 2 number of threads\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  int data = random_int(0, 100);
  print1per(data, "unsorted");
  int steps = log2(size);
  for (int i = 0; i < steps; ++i)
  {
    for (int j = i; j >= 0; --j)
    {
      auto recv = cube(j, data, rank);
      auto dest = rank ^ (1 << j);
      if (rank % (int)pow(2, i + 2) < pow(2, i + 1))
      {
        // ascending
        if (rank < dest)
          data = std::min(recv, data);
        else
          data = std::max(recv, data);
      }
      else
      {
        // descending
        if (rank < dest)
          data = std::max(recv, data);
        else
          data = std::min(recv, data);
      }
    }
  }
  print1per(data, "sorted");

  MPI_Finalize();

  return EXIT_SUCCESS;
}
