#!/usr/bin/env python3
import os
import time

trials = 10
elements = [50000000, 100000000, 200000000]
processes = [2**i for i in range(7)]

filename = 'timing.csv'
with open(filename, 'w') as f:
    f.write("rank_count,data_count,app_time,histogram_time,total_time\n")

for n_processes in processes:
    for n_elements in elements:
        print(f"mpiexec -n {n_processes} ./build/driver/bench 10 0 100 {n_elements}")
        for trial in range(trials):
            os.system(f"mpiexec -n {n_processes} ./build/driver/bench 10 0 100 {n_elements} >> {filename}")
            time.sleep(1)
