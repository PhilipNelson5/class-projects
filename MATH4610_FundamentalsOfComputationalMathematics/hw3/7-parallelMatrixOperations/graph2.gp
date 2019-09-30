set term png
set output "images/parallelMatrixMatrix.png"
set title "Parallel Matrix Matrix Multiply"
set xlabel "time (ms)" offset 0,-1
set ylabel "number of elements M*N"
set key outside
set xtics axis nomirror in rotate by 45 offset -0.8,-1.6
set ytics axis nomirror in
plot \
"data/data2.txt" using 2:1 title "serial" with linespoints lw 2, \
"data/data2.txt" using 3:1 title "parallel 1" with linespoints lw 2, \
"data/data2.txt" using 4:1 title "parallel 2" with linespoints lw 2
