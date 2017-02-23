#ifndef MANDELBROT_HPP
#define MANDELBROT_HPP
#include <string>

bool import(std::string file);

double interpolate(int i, double n, double min, double max);

int mandelbrot(int i, int j);

void render(int beg, int end);

void renderPixels(int beg, int end);

int imagesize();

void write(std::string);

int height();

int width();

#endif
