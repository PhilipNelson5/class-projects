#include "cuda_helpers.cu"
#include "helpers.hpp"
#include "image.cu"
#include "newton_solver.cu"
#include "ppm.h"
#include "print.hpp"
#include <chrono>
#include <cuda.h>
#include <thrust/complex.h>
#include <unistd.h>
#include <vector>
#include <stdio.h>

/**
 * linear interpolate
 * @param i value from zero to n
 * @param n max value
 * @param min of new range
 * @param max of new range
 */
__device__
double lerp(const int i, const double n, const double min, const double max) {
    return (i * (max - min) / n) + min;
}

__device__
int closest(const thrust::complex<double> z, const thrust::complex<double> roots[], const int num)
{
  int n = 0;
  double val = abs(z - roots[n]);
  for (int i = 1; i < num; ++i)
  {
    double next = abs(z - roots[i]);
    if (next < val)
    {
      val = next;
      n = i;
    }
  }
  return n;
}

__global__
void run(
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int image_width, const int image_height,
    uint8_t* image, const int frame)
{
    const int z_size = 1;

    const int pallet[5][3] = {
      {38,70,83}, {42,157,143}, {233,196,106}, {244,162,}, {231,111,81}
    };

    thrust::complex<double> roots[5] = {
      thrust::complex<double>(-1.3247,0),
      thrust::complex<double>(0,-1),
      thrust::complex<double>(0,1),
      thrust::complex<double>(0.66236,-0.56228),
      thrust::complex<double>(0.66236,0.56228)
    };

    dim3 gid (
      (blockIdx.x * blockDim.x) + threadIdx.x,
      (blockIdx.y * blockDim.y) + threadIdx.y,
      (blockIdx.z * blockDim.z) + threadIdx.z
    );

    if (gid.x >= image_width || gid.y >= image_height || gid.z >= z_size) return;
    const double a = lerp(gid.x, image_width, x_min, x_max);
    const double b = lerp(gid.y, image_height, y_min, y_max);
    const thrust::complex<double> c(a,b);

    const thrust::complex<double> root = root_finder_newton(
        [](thrust::complex<double> z){return z*z*z*z*z + z*z - z + 1;},
        [](thrust::complex<double> z){return 5*z*z*z*z + 2*z - 1;},
        c, 1.0e-6, frame
    );


    int n = closest(root, roots, 5); 
    double dist = abs(root - roots[n]);

    const float red = pallet[n][0] * 1/(1-dist);
    const float grn = pallet[n][1] * 1/(1-dist);
    const float blu = pallet[n][2] * 1/(1-dist);


    SetPixel(image, image_width, gid.x, gid.y, red, grn, blu );
}

__host__
int main(int argc, char **argv)
{
  int image_width, image_height, frames;
  double x_center = 1.0;
  double y_center = 0.0;
  double x_width = 10.0;
  double y_height = 10.0;

  std::tie(image_width, image_height, frames) = parse_args(argc, argv);
  //print(image_width, "x", image_height);
  const int image_size = image_width * image_height * 4;
  uint8_t* image_d;
  uint8_t* image_h = new uint8_t[image_size];

  dim3 grid, thread;
  std::tie(grid, thread) = get_dim(image_width, image_height);
  //print("grid:  ", grid);
  //print("thread:", thread);

  cudaMalloc((void**) &image_d, image_size * sizeof(uint8_t));

  double total_time_ms = 0;
  for(int frame = 0; frame < frames; ++frame)
  {
    //const double p = (double)frame / frames;
    
    const double x_min = x_center - x_width / 2.0;
    const double x_max = x_center + x_width / 2.0;
    const double y_min = y_center - y_height / 2.0;
    const double y_max = y_center + y_height / 2.0;

    auto start = std::chrono::high_resolution_clock::now();
    run<<<grid,thread>>>(x_min, x_max, y_min, y_max, image_width, image_height, image_d, frame);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = finish - start;
    const double time_ms = ms.count();
    total_time_ms += time_ms;

    cudaMemcpy((void*)image_h, (void*)image_d, image_size, cudaMemcpyDeviceToHost);

    fwrite(image_h, 1, image_size*sizeof(uint8_t), stdout);
    //write_ppm("out.ppm", image_h, image_width, image_height);
  }
  fflush (stdout);
  fclose(stdout);
  const auto average_time_ms = total_time_ms / frames;
  std::cerr << "\n\naverage frame generation time: " << average_time_ms << "\n" << std::endl;

  cudaFree(image_d);
  delete image_h;

  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}
