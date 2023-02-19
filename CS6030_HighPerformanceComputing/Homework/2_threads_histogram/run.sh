#!/usr/bin/env bash
set -e
echo "threads,time_ms" > timings.csv
for t in {1..12}
do
    echo ./build/histogram/histogram_benchmark $t 10 0.0 5.0 500000000 >&2
    ./build/histogram/histogram_benchmark $t 10 0.0 5.0 500000000 >> timings.csv
done

[ ! -d ./images ] && mkdir images
./plot.py