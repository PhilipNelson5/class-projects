#include <Histogram/Histogram.hpp>
#include <helpers.hpp>
#include <mpi_get_type.hpp>
#include <print.hpp>

#include <chrono>
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
std::vector<T> scatter(const T* data, const int count, MPI_Comm comm)
{
    std::vector<T> local_data(count);
    MPI_Scatter(
        data, count, mpi_get_type<T>(),
        local_data.data(), count, mpi_get_type<T>(),
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
std::vector<T> scatterv(const T* data, const int* sendcounts, const int* displs, MPI_Comm comm)
{
    const int recvcount = sendcounts[get_comm_rank(comm)];
    std::vector<T> local_data(recvcount);
    MPI_Scatterv(
        data, sendcounts, displs, mpi_get_type<T>(), 
        local_data.data(), recvcount, mpi_get_type<T>(),
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
std::vector<T> smart_scatter(const T* sendbuff, const int data_count, const int send_count, MPI_Comm comm)
{
    const int comm_size = get_comm_size(comm);
    if (data_count % comm_size == 0)
    {
        return scatter(sendbuff, send_count, MCW);
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

        return scatterv(sendbuff, sendcounts.data(), displs.data(), MCW);
    }
}

/**
 * @brief initialize the random data and distribute local copies across the world communicator
 * 
 * @param data_count total number of elements in sendbuff
 * @param min_meas minimum value for random data
 * @param max_meas maximum value for random data
 * @return std::vector<double> local data
 */
std::vector<double> init_local_data(const int data_count, const double min_meas, const double max_meas)
{
    const int rank = get_comm_rank(MCW);
    const int comm_size = get_comm_size(MCW);

    const int send_count = data_count / comm_size;
    if (rank == 0)
    {
        auto data = generate_data(data_count, min_meas, max_meas);
        return smart_scatter(data.data(), data_count, send_count, MCW);
    }
    else
    {
        return smart_scatter<double>(NULL, data_count, send_count, MCW);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    const int rank = get_comm_rank(MCW);
    const int comm_size = get_comm_size(MCW);

    Args args;
    if (rank == 0) args = parse_args(argc, argv);
    MPI_Bcast(&args, sizeof(args), MPI_CHAR, 0, MCW);
    std::srand(args.random_seed);
    
    const auto local_data = init_local_data(args.data_count, args.min_meas, args.max_meas);

    const auto bin_maxes = calculate_bin_maxes(args.min_meas, args.max_meas, args.bin_count);
    const auto bin_counts = histogram(begin(local_data), end(local_data), args.bin_count, bin_maxes);

    if (rank == 0)
    {
        std::vector<int> final_bin_counts(bin_counts.size());
        MPI_Reduce(bin_counts.data(), final_bin_counts.data(), args.bin_count, MPI_INT, MPI_SUM, 0, MCW);

        print("bin_maxes:", bin_maxes);
        print("bin_counts:", final_bin_counts);
    }
    else
    {
        MPI_Reduce(bin_counts.data(), NULL, args.bin_count, MPI_INT, MPI_SUM, 0, MCW);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
