set datafile separator ","
set term png

set output "images/benchmark.png"
set title "Time to Render vs Image Size"
set xlabel "Image Size (px x px)" offset 0,-1.1
set ylabel "Image Render Time (ms)"
set y2label "File Write Time (ms)" offset 3.5,0
set key outside autotitle columnhead
set xtics axis nomirror in rotate by 60 right
set ytics axis nomirror in
set y2tics axis nomirror in
plot "benchmark/lineByLine4p.csv" using 1:2 with linespoints lw 2, \
"benchmark/lineByLine4p.csv" using 1:3 with linespoints lw 2 axes x1y2, \

set output "images/pixelsPerSec.png"
set title "Pixels Rendered Per Second vs Image Size"
set xlabel "Image Size (px x px)" offset 0,-1.1
set ylabel "Pixels Rendered per Second"
set key outside autotitle columnhead
set xtics axis nomirror in rotate by 60 right
set ytics axis nomirror in
unset y2tics
unset y2label
plot "benchmark/lineByLine4p.csv" using 1:5 with linespoints lw 2, \

set output "images/speedup.png"
set title "Image Rendering Speedup"
set xlabel "Image Size (px x px)" offset 0,-1.1
set ylabel "Speedup"
set key outside autotitle columnhead
set xtics axis nomirror in rotate by 60 right
set ytics axis nomirror in
plot "benchmark/combinedBenchmark.csv" using 1:($6/$2) title "Speedup" with linespoints lw 2, \
