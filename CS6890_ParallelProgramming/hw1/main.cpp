#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, world_size, data;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Send(&rank, 1, MPI_INT, (rank + 1) % world_size, 0, MPI_COMM_WORLD);
  MPI_Recv(&data, 1, MPI_INT, ((rank - 1) + world_size) % world_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::cout << "I am " << rank << " of " << world_size << " and received " << data << ".\n";

  MPI_Finalize();

  return EXIT_SUCCESS;
}
