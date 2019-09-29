#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

enum class IlliacIcf
{
  PLUS_1,
  MINUS_1,
  PLUS_N,
  MINUS_N
};

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

int illiac(const std::string mask, IlliacIcf icf, int data)
{
  (void)mask; // ignore mask

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int m = std::sqrt(size);

  if (icf == IlliacIcf::PLUS_1) { data = pm2i(mask, pm2iIcf::PLUS, 0, data); }
  else if (icf == IlliacIcf::MINUS_1)
  {
    data = pm2i(mask, pm2iIcf::MINUS, 0, data);
  }
  else if (icf == IlliacIcf::PLUS_N)
  {
    auto pm2i_i = log2(m);
    data = pm2i(mask, pm2iIcf::PLUS, pm2i_i, data);
  }
  else if (icf == IlliacIcf::MINUS_N)
  {
    auto pm2i_i = log2(m);
    data = pm2i(mask, pm2iIcf::MINUS, pm2i_i, data);
  }
  else
  {
    if (rank == 0) std::cout << "Invalid ICF. Aborting..." << std::endl;
    return data;
  }

  return data;
}

void printRunningGrid(const char* msg, int running[64])
{
  (void)msg;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int m = std::sqrt(size);
  for (int i = 0, j = 0; j < m; ++j)
  {
    for (int k = 0; k < m; ++i, ++k)
      std::cout << std::setw(2) << running[i] << " ";
    std::cout << '\n';
  }
}


int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, size, data;
  int running[64];
  const char* m = "XXXXXXXX";

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (std::sqrt(size) != (int)std::sqrt(size))
  {
    if (rank == 0)
    {
      std::cerr
        << "\n\n\tThere must be a perfect square number of threads\n\n\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (!rank) std::cout << "\nplus 1:\n";
  data = illiac(m, IlliacIcf::PLUS_1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nminus 1:\n";
  data = illiac(m, IlliacIcf::MINUS_1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nplus n:\n";
  data = illiac(m, IlliacIcf::PLUS_N, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nminus n:\n";
  data = illiac(m, IlliacIcf::MINUS_N, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
