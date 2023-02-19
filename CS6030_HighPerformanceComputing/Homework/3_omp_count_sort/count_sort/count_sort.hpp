#pragma once
#include "print.hpp"
#include <chrono>
#include <cstring>
#include <omp.h>
#include <thread>

/**
 * @brief a sort algorithim that operates in parallel
 *
 * @tparam ForwardIt input iterator
 * @tparam T type of the value of the iterator
 * @param first where to start sorting
 * @param last where to end sorting
 * @param thread_count number of threads to use
 * @param chunksize chunksize to pass to the omp scheduler
 */
template <typename ForwardIt, typename T = typename std::iterator_traits<ForwardIt>::value_type>
void count_sort(ForwardIt first, ForwardIt last, const int thread_count = 1, const int chunksize = 1)
{
    const auto n = std::distance(first, last);
    std::vector<T> temp(n);

    /**
     * parallel - create a new parallel context
     * num_threads - specify the number of threads to create
     * default - do not implicitly assign a data sharing attribute
     * shared - temp, first, and last will be shared between threads
     */
#pragma omp parallel num_threads(thread_count) default(none) shared(temp, first, last)
    {
        /**
         * for - parallelize the following for loop
         * schedule - assign each thread chunksize iterations of the loop at a time
         */
#pragma omp for schedule(static, chunksize)
        for (auto it_i = first; it_i < last; ++it_i)
        {
            int count = 0;
            for (auto it_j = first; it_j < last; ++it_j)
            {
                if (*it_j < *it_i)
                {
                    ++count;
                }
                else if (*it_j == *it_i && it_j < it_i)
                {
                    ++count;
                }
            }
            temp[count] = *it_i;
        }
        const int rank = omp_get_thread_num();
        const int n_per_thread = n / thread_count;
        const int my_n = (rank == thread_count - 1) ? n - (rank * n_per_thread) : n_per_thread;

        auto start_it = cbegin(temp) + rank * n_per_thread;
        auto end_it = start_it + my_n;
        auto start_dest_it = first + rank * n_per_thread;

        std::move(start_it, end_it, start_dest_it);
    }
    // std::move(begin(temp), end(temp), first);
}

/**
 * @brief a sort algorithim that operates in parallel
 *
 * @param a array of integers to sort
 * @param n size of input array
 * @param thread_count number of threads to use
 * @param chunksize chunksize to pass to the omp scheduler
 */
void count_sort(int a[], int n, int thread_count, const int chunksize = 1)
{
    int i, j, count;
    int *temp = (int *)malloc(n * sizeof(int));
    /**
     * parallel - create a new parallel context
     * num_threads - specify the number of threads to create
     * default - do not implicitly assign a data sharing attribute
     * shared - temp, n, a, and thread_count will be shared between threads
     * private - i, j, and count will have have their own copies in each thread
     */
#pragma omp parallel num_threads(thread_count) default(none) shared(temp, n, a, thread_count) private(i, j, count)
    {
        /**
         * for - parallelize the following for loop
         * schedule - assign each thread chunksize iterations of the loop at a time
         */
#pragma omp for schedule(static, chunksize)
        for (i = 0; i < n; i++)
        {
            count = 0;
            for (j = 0; j < n; j++)
                if (a[j] < a[i])
                    count++;
                else if (a[j] == a[i] && j < i)
                    count++;
            temp[count] = a[i];
        }

        const int rank = omp_get_thread_num();
        const int n_per_thread = n / thread_count;
        const int start = rank * n_per_thread;
        const int my_n = (rank == thread_count - 1) ? n - (rank * n_per_thread) : n_per_thread;
        memcpy(a + start, temp + start, my_n * sizeof(int));
    }
    // memcpy(a, temp, n * sizeof(int));

    free(temp);
} /* count_sort */