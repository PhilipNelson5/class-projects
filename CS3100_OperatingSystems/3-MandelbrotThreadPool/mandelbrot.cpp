#include "mandelbrot.hpp"
#include "image.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace
{
  int IMAGE_HIGHT, IMAGE_WIDTH, MAX_ITERS, COLOR_DEPTH, COLOR_SCHEME;
  double X_MIN, X_MAX, Y_MIN, Y_MAX;
  Image pic;
}

// import settings from file
bool import(std::string file)
{
  std::ifstream fin(file);
  if (!fin)
  {
    std::cout << "Bad File..." << std::endl;
    return false;
  }
  std::string ignore;

  fin >> ignore >> IMAGE_HIGHT;
  fin >> ignore >> IMAGE_WIDTH;
  fin >> ignore >> COLOR_DEPTH;
  fin >> ignore >> MAX_ITERS;

  fin >> ignore >> X_MIN;
  fin >> ignore >> X_MAX;
  fin >> ignore >> Y_MAX;

  fin >> ignore >> COLOR_SCHEME;

  fin.close();

  Y_MIN =
    Y_MAX - (X_MAX - X_MIN) * (static_cast<double>(IMAGE_HIGHT) / IMAGE_WIDTH);

  pic.set(IMAGE_HIGHT, IMAGE_WIDTH, MAX_ITERS, COLOR_DEPTH, COLOR_SCHEME);

  return true;
}

// linear interpolation
double interpolate(int i, double n, double min, double max)
{
  return (i * (max - min) / n) + min;
}

// calculate iterations for an (x,y) coordinate
int mandelbrot(int i, int j)
{
  double xtemp;
  double x0 = interpolate(j, IMAGE_WIDTH, X_MIN, X_MAX);
  double y0 = interpolate(i, IMAGE_HIGHT, Y_MIN, Y_MAX);
  double x = 0.0;
  double y = 0.0;
  int iters = 0;
  while (x * x + y * y < 4 && iters < MAX_ITERS)
  {
    xtemp = x * x - y * y + x0;
    y = 2 * x * y + y0;
    x = xtemp;
    iters += 1;
  }
  return iters;
}

// fills a range of the image vector
void render(int beg, int end)
{
  int iters;
  for (int i = beg; i <= end; ++i)
  {
    for (int j = 0; j < IMAGE_WIDTH; ++j)
    {
      iters = mandelbrot(i, j);
      pic.image[(i * IMAGE_WIDTH) + j] = iters;
    }
  }
}

void renderPixels(int beg, int end)
{
  int iters;
  for (int i = beg; i <= end; ++i)
  {
    iters = mandelbrot(i / IMAGE_WIDTH, i % IMAGE_HIGHT);
    pic.image[i] = iters;
  }
}

int imagesize()
{
  return IMAGE_HIGHT * IMAGE_WIDTH;
}

void write(std::string file)
{
  pic.writeImage(file);
}

int width()
{
  return IMAGE_WIDTH;
}

int height()
{
  return IMAGE_HIGHT;
}
