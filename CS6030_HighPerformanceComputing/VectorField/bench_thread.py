import os

threads = [2**i for i in range(4+1)] + [ 20, 23, 26, 29, 32 ]
curves = [2**i for i in range(6, 10+1)]
trials = 10

filename = 'timing_thread.csv'
with open(filename, 'w') as f:
    f.write("threads,curves,program_time,curve_time\n")

for t in threads:
    for c in curves:
        for _ in range(trials):
            os.system(f"./build/driver_thread/run_thread -n {t} -s {c} --no_image >> {filename}")