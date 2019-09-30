set term png
set output "images/parallelMatrixVector.png"
set title "Parallel Matrix Vector Multiply"
set xlabel "time (ms)" offset 0,-1
set ylabel "number of elements M*N"
set key outside
set xtics axis nomirror in
set ytics axis nomirror in
plot \
"data/data1.txt" using 2:1 title "serial" with linespoints lw 2, \
"data/data1.txt" using 3:1 title "parallel 1" with linespoints lw 2, \
"data/data1.txt" using 4:1 title "parallel 2" with linespoints lw 2
