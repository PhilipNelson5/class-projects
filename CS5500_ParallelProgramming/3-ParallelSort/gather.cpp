#include "random.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

#define MCW MPI_COMM_WORLD

enum Tag
{
  UNSORTED,
  SORTED,
  FINALIZE
};

std::vector<int> merge(std::vector<int> const& a, std::vector<int> const& b)
{
  std::vector<int> merged;
  unsigned int i = 0, j = 0;
  while (i < a.size() && j < b.size())
  {
    if (a[i] < b[j])
    {
      merged.push_back(a[i++]);
    }
    else
    {
      merged.push_back(b[j++]);
    }
  }

  std::copy(begin(a) + i, end(a), std::back_inserter(merged));
  std::copy(begin(b) + j, end(b), std::back_inserter(merged));

  return merged;
}

void random_fill(std::vector<int>::iterator b,
                 std::vector<int>::iterator e,
                 int low = 0,
                 int high = 1000)
{
  std::for_each(b, e, [&](int& a) { a = randInt(low, high); });
}

int main(int argc, char** argv)
{
  int rank, world_size;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MCW, &rank);
  MPI_Comm_size(MCW, &world_size);

  if (0 == rank)
  {
    int n = 11;
    if (argc >= 2)
    {
      n = std::stoi(argv[1]);
    }

    /* ---------------------- */
    /* Generate Unsorted Data */
    /* ---------------------- */
    std::vector<int> data;
    data.resize(n);
    random_fill(std::begin(data), std::end(data));

    const int chunksize = n / world_size;
    std::vector<int> chunk;
    chunk.resize(chunksize);

    auto start = std::chrono::high_resolution_clock::now();

    /* --------------------- */
    /* Scatter Unsorted Data */
    /* --------------------- */
    MPI_Scatter(data.data(),
                chunksize,
                MPI_INT,
                chunk.data(),
                chunksize,
                MPI_INT,
                0,
                MCW);

    /* ---------------------- */
    /* Sort Any Leftover Data */
    /* ---------------------- */
    std::vector<int> leftover;
    if (n % world_size != 0)
    {
      std::copy_n(rend(data), n % world_size, std::back_inserter(leftover));
      std::sort(begin(leftover), end(leftover));
    }

    /* ------------------- */
    /* Sort Unsorted Chunk */
    /* ------------------- */
    std::sort(begin(chunk), end(chunk));

    /* ------------------ */
    /* Gather Sorted Data */
    /* ------------------ */
    MPI_Gather(chunk.data(),
               chunksize,
               MPI_INT,
               data.data(),
               chunksize,
               MPI_INT,
               0,
               MCW);

    /* ----------------- */
    /* Merge Sorted Data */
    /* ----------------- */
    for (auto i = 0; i < world_size - 1; ++i)
    {
      std::inplace_merge(begin(data) + i * chunksize,
                         begin(data) + (i + 1) * chunksize,
                         begin(data) + (i + 2) * chunksize);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_time =
      std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Time: " << total_time << " ms\n";
  }
  else
  {
    int n;
    MPI_Status stat;
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
