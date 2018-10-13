#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP

#include "cell.hpp"
#include <mpi.h>
#include <vector>

using World = std::vector<std::vector<Cell>>;

/**
 * Share border information with neighbors
 *
 * @param world      The representation of the world known to the process
 * @param rank       The process rank
 * @param world_size The number of processes
 */
void send_recv(World& world, int rank, int world_size)
{
  auto destN = ((rank - 1) + world_size) % world_size;
  auto destS = (rank + 1) % world_size;
  auto ct = world[1].size();
  /* clang-format off */
  MPI_Request request1, request2;
  MPI_Isend(
    world[1].data(), ct, MPI_INT, destN, 0, MPI_COMM_WORLD, &request1);
  MPI_Recv(
    world[world.size()-1].data(), ct, MPI_INT, destS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Isend(
    world[world.size()-2].data(), ct, MPI_INT, destS, 0, MPI_COMM_WORLD, &request2);
  MPI_Recv(
    world[0].data(), ct, MPI_INT, destN, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  /* clang-format on */
  int flag1, flag2;
  MPI_Test(&request1, &flag1, MPI_STATUS_IGNORE);
  if (!flag1) std::cout << "Request 1 not finished\n";
  MPI_Test(&request1, &flag2, MPI_STATUS_IGNORE);
  if (!flag2) std::cout << "Request 2 not finished\n";
}

/**
 * Gather all the strips to the master, (called by the master)
 *
 * @param world      The representation of the whole world
 * @param strip      The representation of the world known to the process
 * @param rpp        The number of rows given to each process
 * @param world_size The number of processes
 */
void gatherMaster(World& world, World& strip, int rpp, int world_size)
{
  MPI_Status stat;
  for (auto src = 1, row = rpp; src < world_size; ++src)
  {
    for (auto recvd = 0; recvd < rpp; ++recvd, ++row)
    {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
      auto r = stat.MPI_SOURCE * rpp + stat.MPI_TAG;
      MPI_Recv(world[r].data(),
               world[r].size(),
               MPI_INT,
               stat.MPI_SOURCE,
               stat.MPI_TAG,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
  for (auto row = 0; row < rpp; ++row)
  {
    world[row] = strip[row + 1];
  }
}

/**
 * Gather all the strips to the master (called by the slaves)
 *
 * @param strip      The representation of the world known to the process
 * @param rpp        The number of rows given to each process
 */
void gatherSlave(World& strip, int rpp)
{
  for (auto row = 0; row < rpp; ++row)
  {
    MPI_Send(strip[row + 1].data(),
             strip[row + 1].size(),
             MPI_INT,
             0,
             row,
             MPI_COMM_WORLD);
  }
}

#endif
