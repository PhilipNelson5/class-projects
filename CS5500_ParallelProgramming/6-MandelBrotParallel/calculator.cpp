#include <vector>

/**
 * Linear interpolation
 * Used to determine the complex coordinate from the pixel location
 *
 * @param i    The pixel x or y
 * @param n    The width or height of the image
 * @param min  The X or Y min
 * @param max  The X or Y max
 * @return     The result of the linear interpolation
 */
inline double interpolate(const int i,
                          const double n,
                          const double min,
                          const double max)
{
  return (i * (max - min) / n) + min;
}

/**
 * Calculates the number of iterations it takes for any pixel
 * in the image to diverge
 *
 * @param i           The x value of the pixel
 * @param j           The y value of the pixel
 * @param IMAGE_WIDTH The width in pixels of the image
 * @param IMAGE_HIGHT The height in pixels of the image
 * @param MAX_ITERS   The maximum number of iterations to attempt
 * @param X_MIN       The minimum real (x) value of the image
 * @param X_MAX       The maximum real (x) value of the image
 * @param Y_MIN       The minimum imaginary (y) value of the image
 * @param Y_MAX       The maximum imaginary (y) value of the image
 * @return            The number of iterations for the specified pixel
 */
int mandelbrot(const int i,
               const int j,
               const int IMAGE_WIDTH,
               const int IMAGE_HIGHT,
               const int MAX_ITERS,
               const double X_MIN,
               const double X_MAX,
               const double Y_MIN,
               const double Y_MAX)
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
    ++iters;
  }

  return iters;
}

/**
 * Renders the Mandelbrot set
 *
 * @param X_MIN       The minimum real (x) value of the image
 * @param X_MAX       The maximum real (x) value of the image
 * @param Y_MIN       The minimum imaginary (y) value of the image
 * @param Y_MAX       The maximum imaginary (y) value of the image
 * @param IMAGE_HIGHT The height in pixels of the image
 * @param IMAGE_WIDTH The width in pixels of the image
 * @param MAX_ITERS   The maximum number of iterations to attempt
 * @return All the iteration data
 */
std::vector<int> render(double X_MIN,
                        double X_MAX,
                        double Y_MIN,
                        double Y_MAX,
                        int IMAGE_HIGHT,
                        int IMAGE_WIDTH,
                        int MAX_ITERS)
{
  std::vector<int> imagebuf;
  imagebuf.reserve(IMAGE_HIGHT * IMAGE_WIDTH);
  for (int i = 0; i < IMAGE_HIGHT; ++i)
  {
    for (int j = 0; j < IMAGE_WIDTH; ++j)
    {
      imagebuf.push_back(mandelbrot(
        i, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX));
    }
  }

  return imagebuf;
}

void render_row(std::vector<int>& imagebuf,
                int row,
                double X_MIN,
                double X_MAX,
                double Y_MIN,
                double Y_MAX,
                int IMAGE_HIGHT,
                int IMAGE_WIDTH,
                int MAX_ITERS)
{
  for (int j = 0; j < IMAGE_WIDTH; ++j)
  {
    imagebuf[j] = (mandelbrot(
      row, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX));
  }
}
