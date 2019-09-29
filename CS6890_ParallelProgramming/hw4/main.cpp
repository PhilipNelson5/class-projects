#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

enum class IlliacIcf
{
  PLUS_1,
  MINUS_1,
  PLUS_3,
  MINUS_3
};

enum class Mesh2DIcf
{
  RIGHT,
  LEFT,
  UP,
  DOWN
};

void printRunningGrid(const char* msg, int running[64])
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int m = std::sqrt(size);
  std::cout << msg << '\n';
  for (int i = 0, j = 0; j < m; ++j)
  {
    for (int k = 0; k < m; ++i, ++k)
      std::cout << std::setw(2) << running[i] << " ";
    std::cout << '\n';
  }
  std::cout << std::endl;
}

int illiac(const std::string mask, IlliacIcf icf, int data)
{
  (void)mask; // ignore mask

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int m = std::sqrt(size);

  int dest, source;

  if (icf == IlliacIcf::PLUS_1)
  {
    dest = (rank + 1) % size;
    source = (rank + size - 1) % size;
  }
  else if (icf == IlliacIcf::MINUS_1)
  {
    dest = (rank + size - 1) % size;
    source = (rank + 1) % size;
  }
  else if (icf == IlliacIcf::PLUS_3)
  {
    dest = (rank + m) % size;
    source = (rank + size - m) % size;
  }
  else if (icf == IlliacIcf::MINUS_3)
  {
    source = (rank + m) % size;
    dest = (rank + size - m) % size;
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

int mesh2d(const std::string mask, Mesh2DIcf icf, int data)
{
  (void)mask; // ignore mask

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int m = std::sqrt(size);

  int dest, source;

  if (icf == Mesh2DIcf::RIGHT)
  {
    int row = 0;
    if (rank != 0) row = rank / m;

    dest = rank + 1;
    if (dest == (row + 1) * m) dest -= m;

    source = rank - 1;
    if (source == row * m - 1) source += m;
    //std::cout << source << " --> " << rank << " (" << row << ")"
              //<< " --> " << dest << std::endl;
  }
  else if (icf == Mesh2DIcf::LEFT)
  {
    dest = (rank + size - 1) % size;
    source = (rank + 1) % size;
  }
  else if (icf == Mesh2DIcf::UP)
  {
    dest = (rank + m) % size;
    source = (rank + size - m) % size;
  }
  else if (icf == Mesh2DIcf::DOWN)
  {
    source = (rank + m) % size;
    dest = (rank + size - m) % size;
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

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  int rank, size, data;
  int running[64];
  const char* m = "11111111";

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

  data = illiac(m, IlliacIcf::PLUS_1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);

  data = illiac(m, IlliacIcf::MINUS_1, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);

  data = illiac(m, IlliacIcf::PLUS_3, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);

  data = illiac(m, IlliacIcf::MINUS_3, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);

  data = mesh2d(m, Mesh2DIcf::RIGHT, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) printRunningGrid(m, running);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
