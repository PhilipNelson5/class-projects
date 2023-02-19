#!/usr/bin/env bash
set -e

function data_count() {
    t=1
    file="timings.csv"
    echo "data_count,time_ms" > $file
    for n in {1000..50000..1000}
    do
        echo ./build/count_sort/count_sort_benchmark $t $n >&2
        ./build/count_sort/count_sort_benchmark $t $n >> $file
    done

    ./plot_data_count.py
}

function thread_count() {
    n=10000
    file="timings.csv"
    echo "threads,time_ms" > $file
    for t in {1..12}
    do
        echo ./build/count_sort/count_sort_benchmark $t $n >&2
        ./build/count_sort/count_sort_benchmark $t $n >> $file
    done

    ./plot_thread_count.py
}    

function chunksize() {
    t=8
    n=10000
    file="timings.csv"
    echo "chunksize,time_ms" > $file

    # echo ./build/count_sort/count_sort_benchmark $t $n 100 1 >&2
    # ./build/count_sort/count_sort_benchmark $t $n 100 10 >> $file

    for c in {1..100}
    do
        echo ./build/count_sort/count_sort_benchmark $t $n 100 $c >&2
        ./build/count_sort/count_sort_benchmark $t $n 100 $c >> $file
    done

    # ./plot_thread_count.py
}    

[ ! -d ./images ] && mkdir images
# data_count
thread_count
# chunksize