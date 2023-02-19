import os

threads = [2**i for i in range(0, 10+1)] # threads per block
curves = [2**i for i in range(6, 10+1)]
trials = 10

filename = 'timing_cuda.csv'
with open(filename, 'w') as f:
    f.write("threads,curves,program_time,curve_time\n")

for t in threads:
    for c in curves:
        print(f"./build/driver_cuda/run_cuda -s {c} -n {t} --no_image >> {filename}")
        for _ in range(trials):
            os.system(f"./build/driver_cuda/run_cuda -s {c} -n {t} --no_image >> {filename}")