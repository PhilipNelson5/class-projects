set term png
set output "images/polyApprox.png"
set title "approximation of d/dx x^3 + 2x^2 + 10x + 7"
set xlabel "value of h" offset 0,-1
set ylabel "d/dx x^3 + 2x^2 + 10x + 7"
set key outside
set xtics rotate
set xtics axis nomirror in
set ytics axis nomirror in
plot "data/poly.csv" using 1:2 title "calculated value" with linespoints, \
"data/poly.csv" using 1:3 title "approximated value" with linespoints

set output "images/polyError.png"
set title "error of approximation of d/dx x^3 + 2x^2 + 10x + 7"
set xlabel "value of h" offset 0,-1
set ylabel "error"
plot "data/poly.csv" using 1:4 title "absolute error" with linespoints, \
"data/poly.csv" using 1:5 title "relative error" with linespoints
