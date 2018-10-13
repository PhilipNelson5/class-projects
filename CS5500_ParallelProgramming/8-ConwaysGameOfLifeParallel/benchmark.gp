set datafile separator ","
set term png

set output "report/images/benchmark.png"
set title "Simulation Time vs Number of Processes"
set xlabel "Number of Processes" offset 0,-1.1
set ylabel "Simulation Time (s)"
set key outside autotitle columnhead
set xtics axis nomirror in
set ytics axis nomirror in
plot "benchmark.csv" using 1:2 with linespoints lw 2
