#include <iostream>
#include <omp.h>

int main()
{
  int tid = 0;

// Fork a team of threads giving them their own copies of variables
#pragma omp parallel private(tid)
  {
    // Obtain thread id number
    tid = omp_get_thread_num();

    // Only master thread does this
    if (tid == 0)
    {
      int thread_ct = omp_get_num_threads();
      std::cout << "Number of threads is " << thread_ct << '\n';
    }

    std::cout << "Hello World from thread " << tid << '\n';

  } // All threads join master thread and disband

  return EXIT_SUCCESS;
}
