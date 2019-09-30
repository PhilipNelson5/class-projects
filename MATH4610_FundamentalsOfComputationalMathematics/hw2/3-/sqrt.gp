set term png
set output "images/sqrtApprox.png"
set title "approximation of d/dx sqrt (3)"
set xlabel "value of h" offset 0,-1
set ylabel "d/dx sqrt (3)"
set key outside
set xtics axis nomirror in
set ytics axis nomirror in
plot "data/sqrt.dat" using 1:2 title "calculated value" with linespoints lw 3, \
"data/sqrt.dat" using 1:3 title "approximated value" with linespoints lw 3

set output "images/sqrtError.png"
set title "error of approximation of d/dx sqrt (3)"
set xlabel "value of h" offset 0,-1
set ylabel "error"
plot "data/sqrt.dat" using 1:4 title "absolute error" with linespoints lw 3, \
"data/sqrt.dat" using 1:5 title "relative error" with linespoints lw 3
