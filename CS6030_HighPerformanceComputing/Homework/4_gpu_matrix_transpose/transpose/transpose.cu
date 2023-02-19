#pragma once

#include "image.cu"
using Image::Pixel;
extern __shared__ uint8_t smem[];

/**
 * @brief copy the input to the output
 * 
 * @tparam T type of the input and output matrices
 * @param input the input matrix
 * @param output the output matrix
 * @param width width of the matrix
 * @param height height of the matrix
 * @param channels number of channels per pixel
 */
template <typename T>
__global__ void copy(const T *input, T *output, const unsigned width, const unsigned height)
{
    // determine the grid id
    dim3 gid(
        (blockIdx.x * blockDim.x) + threadIdx.x,
        (blockIdx.y * blockDim.y) + threadIdx.y);

    // kill excess threads
    if (gid.x >= width || gid.y >= height)
        return;

    // copy elements
    output[gid.y * width + gid.x] = input[gid.y * width + gid.x];
}

/**
 * @brief Calculate the transpose of an input matrix on the host
 * 
 * @tparam T type of the input and output matrices
 * @param image the input matrix
 * @param width width of the matrix
 * @param height height of the matrix
 * @param channels number of channels per pixel
 * @return T* new transposed matrix
 */
template <typename T>
T *transpose(const T *image, const unsigned width, const unsigned height)
{
    const unsigned size = width * height;
    T *imageT = new T[size];
    for (auto row = 0u; row < height; ++row)
    {
        for (auto col = 0u; col < width; ++col)
        {
            imageT[col * height + row] = image[row * width + col];
        }
    }
    return imageT;
}

/**
 * @brief Calculate the transpose of an input matrix using global memory
 * 
 * @tparam T type of the input and output matrices
 * @param input the input matrix
 * @param output the output matrix
 * @param width width of the matrix
 * @param height height of the matrix
 * @param channels number of channels per pixel
 */
template <typename T>
__global__ void transpose(const T *input, T *output, const unsigned width, const unsigned height)
{
    // determine the grid id
    dim3 gid(
        (blockIdx.x * blockDim.x) + threadIdx.x,
        (blockIdx.y * blockDim.y) + threadIdx.y);

    // kill excess threads
    if (gid.x >= width || gid.y >= height)
        return;

    // transpose elements
    output[gid.x * width + gid.y] = input[gid.y * width + gid.x];
}

/**
 * @brief Calculate the transpose of an input matrix using shared memory
 * 
 * @tparam T type of the input and output matrices
 * @param input the input matrix
 * @param output the output matrix
 * @param width width of the matrix
 * @param height height of the matrix
 * @param channels number of channels per pixel
 */
template <typename T>
__global__ void transpose_tiled(const T *input, T *output, const unsigned width, const unsigned height)
{
    // "allocate" shared array
    T *tile = reinterpret_cast<T *>(smem);

    // determine the grid id
    dim3 gid(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    // transpose the block, then get the grid id
    const dim3 gidT(
        blockIdx.y * blockDim.x + threadIdx.x,
        blockIdx.x * blockDim.y + threadIdx.y);

    // kill excess threads
    if (gid.x >= width || gid.y >= height)
        return;

    const auto id_in = gid.y * width + gid.x;
    const auto id_tile = threadIdx.y * blockDim.x + threadIdx.x;
    const auto id_tileT = threadIdx.x * blockDim.x + threadIdx.y;
    const auto id_out= gidT.y * width + gidT.x;

    tile[id_tile] = input[id_in];
    __syncthreads();
    output[id_out] = tile[id_tileT];
}
