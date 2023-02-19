#include "mpi_get_type.hpp"
#include <mpi.h>

/**
 * @brief scatter data across a communicator
 * 
 * @tparam T type of the data
 * @param data data array
 * @param count total numver of elements
 * @param comm communicator
 * @return std::vector<T> local data
 */
template <typename T>
std::vector<T> scatter(const T* data, const int count, MPI_Datatype type, MPI_Comm comm)
{
    std::vector<T> local_data(count);
    MPI_Scatter(
        data, count, type,
        local_data.data(), count, type,
        0, comm
    );
    return local_data;
}

/**
 * @brief scatter data as evenly as possible across a communicator
 * 
 * @tparam T type of the data
 * @param data data array
 * @param sendcounts array of send counts for each rank in the communicator
 * @param displs array of displations into data to start sending to each rank in the communicator
 * @param comm communicator
 * @return std::vector<T> local data
 */
template <typename T>
std::vector<T> scatterv(const T* data, const int* sendcounts, const int* displs, MPI_Datatype type, MPI_Comm comm)
{
    const int recvcount = sendcounts[get_comm_rank(comm)];
    std::vector<T> local_data(recvcount);
    MPI_Scatterv(
        data, sendcounts, displs, type, 
        local_data.data(), recvcount, type,
        0, comm
    );
    return local_data;
}

/**
 * @brief smart scatter data across a communicator using MPI_Scatter or MPI_Scatterv
 * 
 * @tparam T type of the data
 * @param sendbuff buffer of data to send
 * @param data_count total number of elements in sendbuff
 * @param send_count number of elements to send to each rank
 * @param comm communicator
 * @return std::vector<T> local data
 */
template <typename T>
std::vector<T> smart_scatter(const T* sendbuff, const int data_count, const int send_count, MPI_Datatype type, MPI_Comm comm)
{
    const int comm_size = get_comm_size(comm);
    if (data_count % comm_size == 0)
    {
        return scatter(sendbuff, send_count, type, comm);
    }
    else
    {
        std::vector<int> displs;
        displs.reserve(comm_size);
        std::vector<int> sendcounts(comm_size, send_count);
        int disp = 0, i = 1;
        displs.push_back(0);
        sendcounts[0] += 1;
        for (;i < data_count % comm_size; ++i)
        {
            displs.push_back(disp += send_count + 1);
            sendcounts[i] += 1;
        }
        displs.push_back(disp += send_count+1);
        ++i;
        for (;i < comm_size; ++i)
        {
            displs.push_back(disp += send_count);
        }

        return scatterv(sendbuff, sendcounts.data(), displs.data(), type, comm);
    }
}

