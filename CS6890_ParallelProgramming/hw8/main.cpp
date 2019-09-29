#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

enum class pm2iIcf
{
  PLUS,
  MINUS
};

int mask(std::string maskStr)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (auto itMask = std::rbegin(maskStr); itMask != std::rend(maskStr);
       ++itMask, rank >>= 1)
  {
    if (*itMask == 'x' | *itMask == 'X') continue;

    bool ithBitZero = rank % 2 == 0;

    if (ithBitZero != (*itMask == '0')) return false;
  }

  return true;
}

int pm2i(std::string const& m, enum pm2iIcf icf, int i, int data)
{
  if (!mask(m)) return data;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int source, dest;
  switch (icf)
  {
  case pm2iIcf::PLUS:
    dest = (rank + (int)std::pow(2, i)) % size;
    source = (rank + size - (int)std::pow(2, i)) % size;
    break;
  case pm2iIcf::MINUS:
    dest = (rank + size - (int)std::pow(2, i)) % size;
    source = (rank + (int)std::pow(2, i)) % size;
    break;
  }

  MPI_Send(&data, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Recv(&data, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return data;
}

int cube(std::string const& m, int icf, int data)
{
  if (!mask(m))
    return data;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string pm = "XXXXXXXX";
  data = pm2i(pm, pm2iIcf::PLUS, icf, data);

  pm[pm.length()-1-icf] = '0';
  data = pm2i(pm, pm2iIcf::MINUS, icf+1, data);

  return data;
}

void printRunning(const char* m, int running[64])
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << m << ": ";
  for (int i = 0; i < size; ++i)
  {
    std::cout << std::setw(3) << running[i];
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  int rank, size;
  int data;
  int running[64];
  const char* m = "XXXXXXXX";
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if ((size & (size - 1)))
  {
    if (rank == 0)
    {
      std::cerr
        << "\n\n\tThere must be a perfect power of 2 number of threads: "
        << size << "\n\n\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (!rank)
    std::cout << "\ncube 0:\n";
  data = cube(m, 0, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printRunning(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank)
    std::cout << "\ncube 1:\n";
  data = cube(m, 1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printRunning(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank)
    std::cout << "\ncube 2:\n";
  data = cube(m, 2, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printRunning(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  const char* m1 = "XXXXXXX0";
  if (!rank)
    std::cout << "\ncube 1:\n";
  data = cube(m1, 1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printRunning(m1, running);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
