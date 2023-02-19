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
void test(K k, const unsigned size, std::string name)
{
    const float time = benchmark_kernel(k);
    const float ebw = effecticeBW(size, size, time);
    print(name);
    print("Execution Time     :", time, "ms");
    print("Effective Bandwidth:", ebw, "GB/s\n");
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
    
    // Copy input to device
    check(cudaMemcpy((void *)input_d, (void *)input_h, size_bytes, cudaMemcpyHostToDevice), "memcpy input");

    // prepare 3 different kernels to compare effective bandwith
    dim3 grid, block;
    
    // 0: Copy the input to the output
    block = dim3(32, 32);
    grid = dim3((int)ceil((double)width / block.x), (int)ceil((double)height / block.y));
    const auto k0 = [=]()
    { copy<<<grid, block>>>(input_d, output_d, width, height); };

    // 1: Naive transpose
    block = dim3(8, 8);
    grid = dim3((int)ceil((double)width / block.x), (int)ceil((double)height / block.y));
    const auto k1 = [=]()
    { transpose<<<grid, block>>>(input_d, output_d, width, height); };

    // 2: tiled transpose
    block = dim3(16, 16);
    grid = dim3((int)ceil((double)width / block.x), (int)ceil((double)height / block.y));
    const auto k2 = [=]()
    { transpose_tiled<<<grid, block, block.x * block.y * sizeof(Pixel)>>>(input_d, output_d, width, height); };

    // Test the 3 kernels
    test(k0, size_bytes, "Copy");
    check(cudaMemcpy((void *)output_h, (void *)output_d, size_bytes, cudaMemcpyDeviceToHost), "memcpy output");
    print(Image::is_same(input_h, output_h, size) ? "SUCCESS\n" : "FAIL\n");

    test(k1, size_bytes, "Transpose Naive");
    check(cudaMemcpy((void *)output_h, (void *)output_d, size_bytes, cudaMemcpyDeviceToHost), "memcpy output");
    print(Image::is_same(expected, output_h, size) ? "SUCCESS\n" : "FAIL\n");
    
    test(k2, size_bytes, "Transpose Tile");
    check(cudaMemcpy((void *)output_h, (void *)output_d, size_bytes, cudaMemcpyDeviceToHost), "memcpy output");
    print(Image::is_same(expected, output_h, size) ? "SUCCESS\n" : "FAIL\n");

    // write output image
    Image::write_binary("out.raw", output_h, height, width);
    // Image::write_ppm("out.ppm", output_h, height, width);

    // cleanup
    delete[] input_h;
    delete[] output_h;
    delete[] expected;
    check(cudaFree((void *)input_d), "free input");
    check(cudaFree((void *)output_d)), "free output";

    return EXIT_SUCCESS;
}
