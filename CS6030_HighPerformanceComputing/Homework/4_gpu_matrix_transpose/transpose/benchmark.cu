#include "helpers.hpp"
#include "transpose.cu"
#include "helpers_cuda.hpp"
#include "image.cu"
#include "print.hpp"
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <tuple>

/**
 * @brief calculate the effective bandwidth of an operation
 * 
 * @param R the number of bytes read
 * @param W the number of bytes written
 * @param t_ms the ammount of time elapsed in miliseconds
 * 
 * @return the effective bandwidth
 */
float effecticeBW(const unsigned R, const unsigned W, const float t_ms)
{
    return (R + W) / t_ms * 1000 / 1e9;
}

/**
 * @brief test a kernel and compute the effective bandwidth
 * 
 * @tparam K the type of the kernel to test
 * @param k kernel to test
 * @param size the number of bytes read/written
 * (assumes same number of bytes are read and written)
 * @param name name of the test
 */
template <typename K>
std::tuple<float, float> test(K k, const unsigned size)
{
    const float time = benchmark_kernel(k);
    const float ebw = effecticeBW(size, size, time);
    return std::make_tuple(ebw, time);
}

template <typename K>
void run_trials(K k, const unsigned width, const unsigned height, const unsigned size, const std::string name)
{
    for (int i = 0; i <= std::log2(32); ++i)
    {
        const dim3 block(std::pow(2, i), std::pow(2, i));
        const dim3 grid((int)ceil((double)width / block.x), (int)ceil((double)height / block.y));
        const float time = benchmark_kernel( [grid, block, k](){ k(grid, block); });
        // const float time = time_kernel( [grid, block, k](){ k(grid, block); });
        const float ebw = effecticeBW(size, size, time);
        print(name, i, time, ebw);
    }
    
}

int main(int argc, char **argv)
{
    using Image::Pixel;
    int width, height;
    std::string filename;
    std::tie(filename, width, height) = parse_args(argc, argv);
    const unsigned size = width * height;
    const unsigned size_bytes = size * sizeof(Pixel);

    print("Transposing", filename, "[", width, "x", height, "]\n");

    // Read input file
    const Pixel *input_h = Image::read(filename, size);
    
    // Transpose input file on the cpu
    const Pixel *expected = transpose(input_h, width, height);

    // Allocate host and device buffers
    Pixel *output_h = new Pixel[size];
    Pixel *input_d, *output_d;
    check(cudaMalloc((void **)&input_d, size_bytes), "malloc input_d");
    check(cudaMalloc((void **)&output_d, size_bytes), "malloc output_d");
    
    check(cudaMemcpy((void *)input_d, (void *)input_h, size_bytes, cudaMemcpyHostToDevice), "memcpy input");
    
    print("kernel dim time ebw");

    // 0: copy
    const auto k0 = [input_d, output_d, width, height](const dim3 grid, const dim3 block)
    { copy<<<grid, block>>>(input_d, output_d, width, height); };

    // 1: Naive transpose
    const auto k1 = [input_d, output_d, width, height](const dim3 grid, const dim3 block)
    { transpose<<<grid, block>>>(input_d, output_d, width, height); };

    // 2: tiled transpose
    const auto k2 = [input_d, output_d, width, height](const dim3 grid, const dim3 block)
    { transpose_tiled<<<grid, block, block.x * block.y * sizeof(Pixel)>>>(input_d, output_d, width, height); };

    // Test the 3 kernels
    run_trials(k0, width, height, size_bytes, "Copy");
    check(cudaMemcpy((void *)output_h, (void *)output_d, size_bytes, cudaMemcpyDeviceToHost), "memcpy output");

    run_trials(k1, width, height, size_bytes, "Naive_Transpose");
    check(cudaMemcpy((void *)output_h, (void *)output_d, size_bytes, cudaMemcpyDeviceToHost), "memcpy output");
    
    run_trials(k2, width, height, size_bytes, "Tiled_Transpose");
    check(cudaMemcpy((void *)output_h, (void *)output_d, size_bytes, cudaMemcpyDeviceToHost), "memcpy output");

    // cleanup
    delete[] input_h;
    delete[] output_h;
    delete[] expected;
    check(cudaFree((void *)input_d), "free input");
    check(cudaFree((void *)output_d)), "free output";

    return EXIT_SUCCESS;
}
