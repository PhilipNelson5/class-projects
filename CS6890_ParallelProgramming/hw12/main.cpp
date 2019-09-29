#include <mpi.h>     // for MPI_COMM_WORLD, MPI_Comm_rank, MPI_Barrier, MPI_...
#include <stdlib.h>  // for exit, EXIT_FAILURE, EXIT_SUCCESS
#include <cmath>     // for log2
#include <iomanip>   // for operator<<, setw
#include <iostream>  // for operator<<, ostream, cout, endl, basic_ostream
#include <iterator>  // for rbegin, rend
#include <string>    // for basic_string, string, allocator

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

void printRunningGrid(int running[64])
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int m = std::sqrt(size);
  for (int i = 0, j = 0; j < m; ++j)
  {
    for (int k = 0; k < m; ++i, ++k)
      std::cout << std::setw(2) << running[i] << " ";
    std::cout << '\n';
  }
  std::cout << std::endl;
}

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

int cube(std::string const& m, int icf, int data)
{
  if (!mask(m)) return data;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto dest = rank ^ (1 << icf);

  MPI_Send(&data, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Recv(&data, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return data;
}

int pm2i(std::string msk, enum pm2iIcf icf, int i, int data)
{
  auto origionalData = data;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int m = log2(size);

  auto digit = (icf == pm2iIcf::PLUS) ? '1' : '0';

  auto xMask = std::string(m, 'X');
  for (auto j = m - 1; j >= i; --j)
  {
    msk = xMask;

    for (int c = i; c < j; ++c)
      msk[msk.length() - 1 - c] = digit;

    auto tmp = cube(xMask, j, data);

    if (mask(msk)) data = tmp;
  }

  if (mask(msk))
    return data;
  else
    return origionalData;
}

int illiac(const std::string mask, IlliacIcf icf, int data)
{
  (void)mask; // ignore mask

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int m = std::sqrt(size);

  if (icf == IlliacIcf::PLUS_1)
  {
    data = pm2i(mask, pm2iIcf::PLUS, 0, data);
  }
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

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, size, data;
  int running[64];
  const char* m = "XXXXXXXXX";

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (std::sqrt(size) != (int)std::sqrt(size))
  {
    if (rank == 0)
    {
      std::cerr << "\n\n\tThere must be a perfect square number of threads\n\n\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (!rank) std::cout << "\nplus 1:\n";
  data = illiac(m, IlliacIcf::PLUS_1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nminus 1:\n";
  data = illiac(m, IlliacIcf::MINUS_1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nplus n:\n";
  data = illiac(m, IlliacIcf::PLUS_N, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nminus n:\n";
  data = illiac(m, IlliacIcf::MINUS_N, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(running);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
