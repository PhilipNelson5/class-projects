#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
//#include "/usr/local/include/mpi.h"
#define MCW MPI_COMM_WORLD

using namespace std;

enum illiacMeshICF
{
  plus1,
  minus1,
  plusn,
  minusn,
  up,
  down,
  lft,
  rght
};
enum pm2iICF
{
  pls,
  mins
};
enum ShuffExICF
{
  shffle,
  xchange
};
int mask(const char* m)
{
  int r;
  bool match = true;
  MPI_Comm_rank(MCW, &r);
  for (int bit = 7; bit >= 0; --bit)
  {
    if (m[bit] != 'X')
    {
      if (r % 2 == 0 && m[bit] == '1') match = false;
      if (r % 2 == 1 && m[bit] == '0') match = false;
    }
    r /= 2;
  }
  if (match) { return 1; }
  else
  {
    return 0;
  }
}

int shuffleExchange(const char* m, ShuffExICF icf, int data)
{
  // TODO: write your code here

  return data;
}

void printRunning(const char* m, int running[64])
{
  int rank, size;
  MPI_Comm_rank(MCW, &rank);
  MPI_Comm_size(MCW, &size);
  cout << m << ": ";
  for (int i = 0; i < size; ++i)
  {
    cout << setw(3) << running[i];
  }
  cout << endl;
}

int main(int argc, char** argv)
{
  int rank, size;
  int data;
  int running[64];
  const char* m = "XXXXXXXX";
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MCW, &rank);
  MPI_Comm_size(MCW, &size);

  if (!rank) cout << "\nshuffle:\n";
  data = shuffleExchange(m, shffle, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MCW);
  if (rank == 0) printRunning(m, running);
  MPI_Barrier(MCW);

  if (!rank) cout << "\nexchange:\n";
  data = shuffleExchange(m, xchange, rank);
  MPI_Gather(&data, 1, MPI_INT, running, 1, MPI_INT, 0, MCW);
  if (rank == 0) printRunning(m, running);
  MPI_Barrier(MCW);

  MPI_Finalize();

  return 0;
}

