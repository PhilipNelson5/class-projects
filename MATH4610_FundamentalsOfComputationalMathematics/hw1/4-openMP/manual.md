---
title: OpenMP Hello World
layout: default
---
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# OpenMP Hello World Software Manual

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

    make

will produce an executable **./hello.out** that can be executed.

**Description/Purpose:** This code shows that you have OpenMP installed and are able to compile/run programs using OpenMP.

**Input:** There are no inputs needed in this case.

**Output:** This code will print out the number of availiable threads of execution availiale on the system and each thread will
print "Hello World from thread #" where # is the thread's id.

**Usage/Example:**

The code can be run on a Linux commandline as shown below:

```
# ./hello.out
```

**Output** from the lines above
```
Number of threads is 4
Hello World from thread 0
Hello World from thread 3
Hello World from thread 1
Hello World from thread 2
```

The first line displays the number of threads of execution available on the system.
The following lines are messages from each thread ending in their thread id.

**Implementation/Code:** The following is the code for `helloWorld.cpp`

``` c++
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
```

**Last Modified:** September 2018
