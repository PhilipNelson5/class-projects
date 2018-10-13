#include "random.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <vector>

#define slep 100000

void print1per(int data, std::string title = "")
{
  int rank;
  int word_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &word_size);

  if (0 == rank)
  {
    int* dArray = new int[word_size];
    MPI_Gather(&data, 1, MPI_INT, dArray, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << title << '\n';
    for (int i = 0; i < word_size; ++i)
    {
      std::cout << std::setw(5) << i << std::setw(5) << dArray[i] << "\n";
    }
    std::cout << std::endl;
  }
  else
  {
    MPI_Gather(&data, 1, MPI_INT, nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

int cube(int c, int sendData, int rank)
{
  int recvData;
  auto dest = rank ^ (1 << c);

  MPI_Send(&sendData, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Recv(&recvData, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return recvData;
}

int ring(int dir, int sendData, int rank, int world_size)
{
  int recvData;
  auto dest = (rank + 1 * dir) % world_size;
  auto src = (rank - 1 * dir) % world_size;

  MPI_Send(&sendData, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Recv(&recvData, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return recvData;
}

void cubeSum(int num, int rank, int world_size)
{
  int log2n = log2(world_size);
  for (auto i = 0; i < log2n; ++i)
  {
    num += cube(i, num, rank);
    usleep(slep);
  }
  print1per(num, "cube sum");
}

void ringSum(int num, int rank, int world_size)
{
  int next, prev = num;

  for (auto i = 0; i < world_size - 1; ++i)
  {
    next = ring(1, prev, rank, world_size);
    num += next;
    prev = next;
    usleep(slep);
  }
  print1per(num, "ring sum");
}

void masterSlaveSum(int num, int rank, int world_size)
{
  if (0 == rank)
  {
    int recvData;
    for (auto i = 1; i < world_size; ++i)
    {
      MPI_Recv(&recvData,
               1,
               MPI_INT,
               MPI_ANY_SOURCE,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      num += recvData;
      usleep(slep);
    }
    std::cout << "master slave sum\n    0   " << num << "\n\n";
  }
  else
  {
    MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

void mpiAllReduce(int num)
{
  MPI_Allreduce(&num, &num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  usleep(slep);
  print1per(num, "all reduce");
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (0 != (world_size & (world_size - 1)))
  {
    if (rank == 0)
    {
      std::cerr << "There must be a power of 2 number of threads\n";
    }

    MPI_Finalize();
    exit(EXIT_SUCCESS);
  }

  int num;
  if (0 == rank)
  {
    std::vector<int> data(world_size);
    random_double_fill(begin(data), end(data), 0, 10);
    MPI_Scatter(data.data(), 1, MPI_INT, &num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    print1per(num, "original data");
  }
  else
  {
    MPI_Scatter(nullptr, 1, MPI_INT, &num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    print1per(num);
  }

  auto t1 = MPI_Wtime();
  cubeSum(num, rank, world_size);
  auto t2 = MPI_Wtime();
  ringSum(num, rank, world_size);
  auto t3 = MPI_Wtime();
  masterSlaveSum(num, rank, world_size);
  auto t4 = MPI_Wtime();
  mpiAllReduce(num);
  auto t5 = MPI_Wtime();

  if (0 == rank)
    std::cout << "cube: " << t2 - t1 << "\nring: " << t3 - t2
              << "\nmaster slave: " << t4 - t3 << "\nall reduce: " << t5 - t4
              << "\n";

  MPI_Finalize();

  return (EXIT_SUCCESS);
}
