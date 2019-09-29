#include <iostream>
#include <mpi.h>
#include <string>

enum RingIcf
{
  PLUS_ONE = 1,
  MINUS_ONE = -1
};

int ring(const std::string mask, int icf, int data)
{
  (void)mask; // ignore mask

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dest, source;

  if (icf == RingIcf::PLUS_ONE)
  {
    dest = (rank + 1) % size;
    source = (rank + size - 1) % size;
  }
  else if (icf == RingIcf::MINUS_ONE)
  {
    dest = (rank + size - 1) % size;
    source = (rank + 1) % size;
  }
  else
  {
    if (rank == 0) std::cout << "Invalid ICF. Aborting..." << std::endl;
    return data;
  }

  MPI_Send(&data, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  MPI_Recv(&data, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return data;
}

void printRunning(const char* m, int running[64])
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << m << ": ";
  for (int i = 0; i < size; ++i)
    std::cout << running[i] << " ";
  std::cout << std::endl;
}
using namespace std;

int main(int argc, char** argv)
{

  int rank, size, data;

  int running[64];

  const char* m = "11111111";

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  data = ring(m, RingIcf::PLUS_ONE, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m, running);

  data = ring(m, RingIcf::MINUS_ONE, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m, running);

  data = ring(m, -4, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunning(m, running);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
