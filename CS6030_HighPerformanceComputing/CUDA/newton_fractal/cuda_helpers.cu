#include <cuda.h>
#include <tuple>

__host__
std::tuple<dim3, dim3> get_dim(const int x_size = 1, const int y_size = 1, const int z_size = 1)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const auto max_t = deviceProp.maxThreadsDim;
  const auto max_t_block = deviceProp.maxThreadsPerBlock;
  int x = 1; // min(max_t[0], x_size);
  int y = 1; // min(max_t[1], y_size);
  int z = 1; // min(max_t[2], z_size);

  int x0 = 0;
  int y0 = 0;
  int z0 = 0;
  while(min(x_size, x+1) * min(y_size, y+1) * min(z_size, z+1) <= max_t_block
      && (x != x0 || y != y0 || z != z0))
  {
    x0 = x; y0 = y; z0 = z;
    x = min(min(x_size, x+1), max_t[0]);
    y = min(min(y_size, y+1), max_t[1]);
    z = min(min(z_size, z+1), max_t[2]);
  }

  if (x * y * z > max_t_block)
  {
    printf("ERROR max thread dim: %d > %d\n", x*y*z, max_t_block);
    exit(EXIT_FAILURE);
  }
  if (x > max_t[0] || y > max_t[1] || z > max_t[2])
  {
    printf("ERROR max thread per block: x: %d > %d || y: %d > %d || z: %d > %d\n",
        x, max_t[0], y, max_t[1], z, max_t[2]);
    exit(EXIT_FAILURE);
  }

  dim3 grid(
    max((int)ceil((double)x_size/x), 1),
    max((int)ceil((double)y_size/y), 1),
    max((int)ceil((double)z_size/z), 1)
  );
  dim3 thread(x, y, z);

  if(grid.x * thread.x < x_size || grid.y * thread.y < y_size || grid.z * thread.z < z_size)
  {
    printf("ERROR not enough thread: x: %d < %d || y: %d < %d || z: %d < %d\n",
        grid.x * thread.x, x_size, grid.y * thread.y, y_size, grid.z * thread.z, z_size);
    exit(EXIT_FAILURE);
  }

  return std::make_tuple(grid, thread);
}
