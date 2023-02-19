import os

threads = [2**i for i in range(1, 6+1)]
curves = [2**i for i in range(6, 10+1)]
trials = 10

filename = 'timing_mpi.csv'
with open(filename, 'w') as f:
    f.write("threads,curves,program_time,curve_time\n")

for t in threads:
    for c in curves:
        for _ in range(trials):
            os.system(f"mpiexec -n {t} ./build/driver_mpi/run_mpi -s {c} --no_image >> {filename}")