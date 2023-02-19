#pragma once

#include <Helpers/AssertHelper.hpp>
#include <mpi.h>

#define MCW MPI_COMM_WORLD

/**
 * @brief Get the size of a communicator
 * 
 * @param comm communicator
 * @return int size
 */
int get_comm_size(MPI_Comm comm)
{
    int size = -1;
    MPI_Comm_size(comm, &size);
    assertm(size >= 0, "Failed to get comm size");
    return size;
}

/**
 * @brief Get the rank within a communicator
 * 
 * @param comm communicator
 * @return int rank
 */
int get_comm_rank(MPI_Comm comm)
{
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    assertm(rank >= 0, "Failed to get comm rank");
    return rank;
}