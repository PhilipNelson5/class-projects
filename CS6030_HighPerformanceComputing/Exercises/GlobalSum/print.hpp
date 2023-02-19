#pragma once

#include <string>
#include <mpi.h>

template<typename T>
void print(T t) { std::cout << t << '\n'; };

template<typename T, typename... Args>
void print(T t, Args... args) { std::cout << t << ' '; print(args...); };

void print_sync_msg(std::string msg, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	MPI_Status s;
	
	if (rank == 0) 
	{
		print(msg);
		int len;
		for (int i = 1; i < size; ++i)
		{
			MPI_Probe(i, MPI_ANY_TAG, comm, &s);
			MPI_Get_count(&s, MPI_CHAR, &len);

			char* buf = new char[len];

			MPI_Recv(buf, len, MPI_CHAR, i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

			print(buf);
			delete[] buf;
		}
	}
	else
	{
		const int len = msg.length() + 1;
		MPI_Send(msg.c_str(), len, MPI_CHAR, 0, 0, comm);
	}
}

template<typename T>
void print_sync_value(T value, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	
	if (rank == 0) 
	{
		std::cout << "[ " << value << " ";
		for (int i = 1; i < size; ++i)
		{
			T recv;

			MPI_Recv(&recv, 1, mpi_get_type<T>(), i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

			std::cout << recv << " ";
		}
		std::cout << "]" << std::endl;
	}
	else
	{
		MPI_Send(&value, 1, mpi_get_type<T>(), 0, 0, comm);
	}
}
