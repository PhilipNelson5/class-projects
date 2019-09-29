#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

enum class ShuffExIcf
{
  shffle,
  xchange
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

int shuffleExchange(std::string msk, ShuffExIcf icf, int data)
{
  if (!mask(msk)) return data;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int m = log2(size);

  if (icf == ShuffExIcf::shffle)
  {
    int A;
    int tmp;
    if (mask("XXXXXXX0")) { A = data; }
    tmp = pm2i("XXXXXXXX", pm2iIcf::PLUS, 0, data);
    if (!mask("XXXXXXX1")) { data = tmp; }
    for (auto j = 1; j < m; ++j)
    {
      msk = "XXXXXXXX";
      msk[msk.size() - 1 - j] = '1';
      if (mask(msk)) { std::swap(data, A); }
      tmp = pm2i("XXXXXXXX", pm2iIcf::PLUS, j, data);
      if (mask("XXXXXXX0")) { data = tmp; }
    }
    tmp = pm2i("XXXXXXXX", pm2iIcf::PLUS, 0, data);
    if (!mask("XXXXXXX0")) { data = tmp; }
    if (mask("XXXXXXX0")) { data = A; }
  }
  else if (icf == ShuffExIcf::xchange)
  {
    data = pm2i("XXXXXXXX", pm2iIcf::PLUS, 0, data);
    data = pm2i("XXXXXXX0", pm2iIcf::MINUS, 1, data);
  }

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
