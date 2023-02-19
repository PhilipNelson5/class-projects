#pragma once

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
cudaError_t check(std::string msg = "")
{
    return check(cudaGetLastError(), msg);
}
