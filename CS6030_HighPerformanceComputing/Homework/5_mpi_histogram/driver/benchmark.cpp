#include <Histogram/Histogram.hpp>
#include <helpers.hpp>
#include <mpi_get_type.hpp>
#include <print.hpp>

#include <chrono>
#include <mpi.h>

#define MCW MPI_COMM_WORLD

int get_comm_size(MPI_Comm comm)
{
    int size = -1;
    MPI_Comm_size(comm, &size);
    assertm(size >= 0, "Failed to get comm size");
    return size;
}

int get_comm_rank(MPI_Comm comm)
{
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    assertm(rank >= 0, "Failed to get comm rank");
    return rank;
}

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

template <typename T>
std::vector<T> init_local_data(const int data_count, const T min_meas, const T max_meas)
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
        return smart_scatter<T>(NULL, data_count, send_count, MCW);
    }
}

// 1. have process 0 read in the inputs data and distribute them among all the processes
// 2. have process 0 populate an array (data) of <data_count> float elements between <min_meas> and <max_meas>.
//      Use srand(100) to initialize your pseudorandom sequence.
// 3. have process 0 distribute portions of the pseudorandom sequence to the other processors
//      (note: do not share the entire array with all other processes)
// 4. compute the histogram 
// 5. have process 0 print out the outputs (i.e., bin_maxes and bin_counts)

// B. Scaling studies (20 points)
// Plot the execution time and speedup using 1, 4, 16, 32 and 64 cores
//  and 3 different problem size:
// 50'000'000 elements
// 100'000'000 elements
// 200'000'000 elements

// You will need to report 3 timings for each experiment:
// 1. total execution time (from step 1 to 5)
// 2. application time (from 3 to 5)
// 3. local histogram computation time (only step 3)

int millis(const auto diff)
{
    return std::chrono::duration <double, std::milli>(diff).count();
}

int main(int argc, char** argv)
{
    const auto _total_time_a = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);

    const int comm_size = get_comm_size(MCW);
    const int rank = get_comm_rank(MCW);

    Args args;
    if (rank == 0) args = parse_args(argc, argv);
    MPI_Bcast(&args, sizeof(args), MPI_CHAR, 0, MCW);
    if(!rank) std::cout << comm_size << "," << args.data_count << ",";
    std::srand(args.random_seed);
    
    const int send_count = args.data_count / comm_size;
    if (rank == 0)
    {
        auto data = generate_data(args.data_count, args.min_meas, args.max_meas);

        const auto _app_time_a = std::chrono::high_resolution_clock::now();
        const auto local_data = smart_scatter(data.data(), args.data_count, send_count, MCW);

        const auto _hist_time_a = std::chrono::high_resolution_clock::now();
        const auto bin_maxes = calculate_bin_maxes(args.min_meas, args.max_meas, args.bin_count);
        const auto bin_counts = histogram(std::begin(local_data), std::end(local_data), args.bin_count, bin_maxes);
        const auto _hist_time_b = std::chrono::high_resolution_clock::now();

        std::vector<int> final_bin_counts(bin_counts.size());
        MPI_Reduce(bin_counts.data(), final_bin_counts.data(), args.bin_count, MPI_INT, MPI_SUM, 0, MCW);
        const auto _app_time_b = std::chrono::high_resolution_clock::now();

        std::cout << millis(_app_time_b - _app_time_a) << "," <<
                     millis(_hist_time_b - _hist_time_a) << ",";
    }
    else
    {
        const auto local_data = smart_scatter<float>(NULL, args.data_count, send_count, MCW);

        const auto bin_maxes = calculate_bin_maxes(args.min_meas, args.max_meas, args.bin_count);
        const auto bin_counts = histogram(begin(local_data), end(local_data), args.bin_count, bin_maxes);

        MPI_Reduce(bin_counts.data(), NULL, args.bin_count, MPI_INT, MPI_SUM, 0, MCW);
    }

    
    MPI_Finalize();
    const auto _total_time_b = std::chrono::high_resolution_clock::now();
    if (!rank) std::cout << millis(_total_time_b - _total_time_a) << std::endl;

    return EXIT_SUCCESS;
}
