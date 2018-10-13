#!/usr/bin/env bash

for i in `seq 2 10`;
do
  mpiexec --oversubscribe -n $i release 1024 1024 500 0
done
