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
    std::vector<int> unsorted;
    unsorted.resize(n);
    random_fill(std::begin(unsorted), std::end(unsorted));

    auto start = std::chrono::high_resolution_clock::now();
    /* ------------------------- */
    /* Send Unsorted Data Chunks */
    /* ------------------------- */
    const int chunksize = n / (world_size - 1);
    for (auto i = 0; i < world_size - 2; ++i)
    {
      MPI_Send((begin(unsorted) + (chunksize * i)).base(),
               chunksize,
               MPI_INT,
               i + 1,
               Tag::UNSORTED,
               MCW);
    }

    MPI_Send((begin(unsorted) + (chunksize * (world_size - 2))).base(),
             (chunksize + (n % chunksize)),
             MPI_INT,
             world_size - 1,
             Tag::UNSORTED,
             MCW);

    /* ------------------- */
    /* Receive Sorted Data */
    /* ------------------- */
    MPI_Status stat;
    std::vector<int> data;
    std::vector<int> sorted;
    int size;
    for (auto i = 0; i < world_size - 1; ++i)
    {
      MPI_Probe(MPI_ANY_SOURCE, Tag::SORTED, MCW, &stat);
      MPI_Get_count(&stat, MPI_INT, &size);
      data.resize(size);
      MPI_Recv(data.data(),
               size,
               MPI_INT,
               MPI_ANY_SOURCE,
               Tag::SORTED,
               MCW,
               MPI_STATUS_IGNORE);

      sorted = merge(sorted, data);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time =
      std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Time: " << total_time << " ms\n";

    /* -------------------------------- */
    /* Tell Other Processes to Finalize */
    /* -------------------------------- */
    const int x = 0;
    for (auto i = 0; i < world_size - 1; ++i)
    {
      MPI_Send(&x, 1, MPI_INT, i+1, Tag::FINALIZE, MCW);
    }
  }
  else
  {
    int n;
    MPI_Status stat;

    do
    {
      MPI_Probe(0, MPI_ANY_TAG, MCW, &stat);
      if (Tag::UNSORTED == stat.MPI_TAG)
      {
        MPI_Get_count(&stat, MPI_INT, &n);

        std::vector<int> data;
        data.resize(n);

        MPI_Recv(
          data.data(), n, MPI_INT, 0, Tag::UNSORTED, MCW, MPI_STATUS_IGNORE);
        std::sort(begin(data), end(data));
        MPI_Send(data.data(), n, MPI_INT, 0, Tag::SORTED, MCW);
      }
    } while (Tag::FINALIZE != stat.MPI_TAG);
    std::cout << "Process " << rank << " finalized" << std::endl;
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
