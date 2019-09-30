set term png
set output "images/newSqrtApprox.png"
set title "new approximation of d/dx sqrt (3)"
set xlabel "value of h" offset 0,-1
set ylabel "d/dx sqrt (3)"
set key outside
set xtics axis nomirror in
set ytics axis nomirror in
plot "data/newSqrt.dat" using 1:2 title "calculated value" with linespoints lw 3, \
"data/newSqrt.dat" using 1:3 title "approximated value" with linespoints lw 3

set output "images/newSqrtError.png"
set title "error of new approximation of d/dx sqrt (3)"
set xlabel "value of h" offset 0,-1
set ylabel "error"
plot "data/newSqrt.dat" using 1:4 title "absolute error" with linespoints lw 3, \
"data/newSqrt.dat" using 1:5 title "relative error" with linespoints lw 3
