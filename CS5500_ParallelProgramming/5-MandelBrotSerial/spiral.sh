#!/usr/bin/env bash
mpiexec -n 1 ./mandelbrot.out 512 512 1000 -.760574 -.762574 -.0837596; feh brot.bmp
