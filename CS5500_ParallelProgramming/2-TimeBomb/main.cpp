#include <iostream>
#include <mpi.h>

#define MCW MPI_COMM_WORLD

int main(int argc, char** argv)
{
  int rank, size, data;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MCW, &rank); // what is your process index in the world
  MPI_Comm_size(MCW, &size); // how many processes are there

  // locally blocks until it is safe to write to the data buffer
  MPI_Send(&rank, 1, MPI_INT, (rank + 1) % size, 0, MCW);

  // locally blocks until it receives a message with the correct tag
  MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);

  std::cout << "I am process " << rank << " of " << size
            << " and received a message from process " << data << std::endl;

  MPI_Finalize();

  return EXIT_SUCCESS;
}
