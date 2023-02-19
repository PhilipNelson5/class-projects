#pragma once

#include "print.hpp"

/**
 * @brief check for errors in a cuda result
 * 
 * @param result the result from calling a cuda function
 * @return cudaError_t the result from calling a cuda function
 */
cudaError_t check(cudaError_t result, std::string msg = "")
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR: %s %s\n", cudaGetErrorString(result), msg.c_str());
        assert(result == cudaSuccess);
    }
    return result;
}

/**
 * @brief check for errors the last cuda function
 * 
 * @return cudaError_t the result from calling the last cuda function
 */
cudaError_t check()
{
    return check(cudaGetLastError());
}

/**
 * @brief time the execution of a cuda kernel in milliseconds
 * 
 * @tparam K the type of the kernel function
 * @param kernel the kernel to execute and time
 * @return float the execution time of the kernel in milliseconds
 */
template <typename K>
float time_kernel(K kernel)
{
    float ms;
    cudaEvent_t startEvent, stopEvent;

    check(cudaEventCreate(&startEvent));
    check(cudaEventCreate(&stopEvent));
    check(cudaEventRecord(startEvent));
    kernel();
    check();
    check(cudaEventRecord(stopEvent));
    check(cudaEventSynchronize(stopEvent));
    check(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return ms;
}

/**
 * @brief time the execution of a cuda kernel over multiple trials
 * 
 * @tparam K the type of the kernel function
 * @param kernel the kernel to execute and time
 * @param n_trials the number of trials to average over
 * @return float the average execution time of the kernel in milliseconds
 */
template <typename K>
float benchmark_kernel(K kernel, const int n_trials = 100)
{
    float t_total = 0;
    for (int _ = 0; _ < n_trials; ++_)
    {
        t_total += time_kernel(kernel);
    }
    return t_total / n_trials;
}
