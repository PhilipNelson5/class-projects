#ifndef CALCULATOR_HPP
#define CALCULATOR_HPP

double interpolate(int i, double n, double min, double max);

int mandelbrot(int i,
               int j,
               int IMAGE_WIDTH,
               int IMAGE_HIGHT,
               int MAX_ITERS,
               double X_MIN,
               double X_MAX,
               double Y_MIN,
               double Y_MAX);

#endif
