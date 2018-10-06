#ifndef CALCULATOR_HPP
#define CALCULATOR_HPP

#include <vector>

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

std::vector<int> render(double X_MIN,
                        double X_MAX,
                        double Y_MIN,
                        double Y_MAX,
                        int IMAGE_HIGHT,
                        int IMAGE_WIDTH,
                        int MAX_ITERS);

void render_row(std::vector<int>& imagebuf,
                int row,
                double X_MIN,
                double X_MAX,
                double Y_MIN,
                double Y_MAX,
                int IMAGE_HIGHT,
                int IMAGE_WIDTH,
                int MAX_ITERS);

#endif
