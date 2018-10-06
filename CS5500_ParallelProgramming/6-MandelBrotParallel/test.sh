#!/usr/bin/env bash

COUNTER=256
while [  $COUNTER -lt 9000 ]; do
  mpiexec --oversubscribe -n 4 ./mandelbrot.out $COUNTER $COUNTER 1000 -.760574 -.762574 -.0837596
  let COUNTER=COUNTER*2
done
