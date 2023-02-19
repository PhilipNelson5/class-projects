/**
 * @file global_sum.cpp
 * @author Philip Nelson (A01666904)
 * @version 0.1
 * @date 2021-07-26
 */

#include <cmath>
#include <iostream>
#include <mpi.h>

#include "mpi_get_type.hpp"
#include "print.hpp"

#define MCW MPI_COMM_WORLD

/**
 * @brief sum values across all members of a communicator. The result 
 * ends up with process 0
 * 
 * @tparam T type of the value to be summed
 * @param value the value to be summed
 * @param comm the communicator
 * @param rank the rank of the process
 * @param size the size of the communicator
 * @param level the level of the send operation
 * @return T the sum thus far
 */
template<typename T>
T global_sum(const T value, MPI_Comm comm, const int rank, const int size, const int level)
{
	const int mask = 1 << level;
	const int dest = rank ^ mask;

	if (dest > size - 1 || rank % mask != 0) return value;
	if (rank > dest) 
	{
		MPI_Send(&value, 1, mpi_get_type<T>(), dest, 0, comm);
		return 0;
	}
	else
	{
		T recv;
		MPI_Recv(&recv, 1, mpi_get_type<T>(), dest, 0, comm, MPI_STATUS_IGNORE);
		return value + recv;
	}
}

/**
 * @brief sum values across all members of a communicator. The result
 * ends up in process 0
 * 
 * @tparam T type of the value to be summed
 * @param value the value to be summed
 * @param comm the communicator
 * @return T the sum thus far
 */
template<typename T>
T global_sum(const T value, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	const int n = std::ceil(std::log2(size));
	double sum = value;
	for (int level = 0; level < n; ++level)
	{
		sum = global_sum(sum, comm, rank, size, level);
	}
	return sum;
}

int main(int argc, char** argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MCW, &rank);
	MPI_Comm_size(MCW, &size);

	const auto value = rank + 0.25;
	print_sync_value(value, MCW);
	const auto sum = global_sum(value, MCW);

	if (rank == 0) print("sum:", sum);
	
	MPI_Finalize();
}
