#include <iostream>
#include <mpi.h>

/**
 * @brief mask takes a process rank and a string bit mask
 * to determine if that process was selected to do work.
 * @param maskStr The string representation of the bitmask
 * @return 1 (true), 0 (false)
 */
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

void printRunning(const char* m, int running[64])
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << m << ": ";

  for (auto i = size - 1; i >= 0; --i)
  {
    std::cout << running[i];
  }

  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  int running[64];
  int rank, size;
  int localrunning;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char* m1 = "XXXXXXXX";
  localrunning = mask(m1);
  MPI_Gather(&localrunning, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m1, running);

  const char* m2 = "XXXXXXX0";
  localrunning = mask(m2);
  MPI_Gather(&localrunning, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m2, running);

  const char* m3 = "XXXXXXX1";
  localrunning = mask(m3);
  MPI_Gather(&localrunning, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m3, running);

  const char* m4 = "XXXXXX00";
  localrunning = mask(m4);
  MPI_Gather(&localrunning, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m4, running);

  const char* m5 = "XXXXX1X1";
  localrunning = mask(m5);
  MPI_Gather(&localrunning, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m5, running);

  const char* m6 = "XXXXX110";
  localrunning = mask(m6);
  MPI_Gather(&localrunning, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m6, running);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
