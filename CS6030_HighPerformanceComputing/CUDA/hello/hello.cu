/*
 * Hello world cuda
 *
 * compile: nvcc hello_cuda.cu -o hello
 *  
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

__global__
void cuda_hello(){
    // thread id of current block (on x axis)
    int tid = threadIdx.x;

    // block id (on x axis)
    int bid = blockIdx.x;

    printf("Ciao belli from block %d core %d!\n", bid, tid);
}

int main() {
    // Launch GPU kernel

    dim3 g(2,2,1);
    dim3 t(2,4,1);
    cuda_hello<<<g,t>>>();

    // cuda synch barrier
    cudaDeviceSynchronize();

    return 0;
}
