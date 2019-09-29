#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <mpi.h>

enum class ShuffExIcf
{
  shffle,
  xchange
};

int mask(std::string const& maskStr)
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

int shuffleExchange(const char* m, ShuffExIcf icf, int data)
{
  if (!mask(m)) return data;

  int rank, size, dest /*, source*/;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  switch (icf)
  {
  case ShuffExIcf::shffle:
  {
    int n = log2(size) - 1;    // number of bits in the address space
    dest = rank & ~(1 << n);   // clear the nth
    dest = dest << 1;          // shift left
    dest = dest | (rank >> n); // replace nth bit
    break;
  }
  case ShuffExIcf::xchange:
    dest = rank ^ 1;
    break;
  }


  MPI_Send(&data, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Recv(
    &data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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

  if (!rank) std::cout << "\nshuffle:\n";
  data = shuffleExchange(m, ShuffExIcf::shffle, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) std::cout << "\nexchange:\n";
  data = shuffleExchange(m, ShuffExIcf::xchange, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m, running);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}
