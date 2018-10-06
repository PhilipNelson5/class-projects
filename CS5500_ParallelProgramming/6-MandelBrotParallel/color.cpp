#include "color.hpp"
#include <cmath>
#include <tuple>

/**
 * A simple gray scale color scheme
 *
 * @param iters     The number of iterations to turn into a color
 * @param max_iters The maximum number of iterations
 * @return          A tuple containing the [R,G,B] triple for that iteration
 */
std::tuple<int, int, int> color_scheme_0(int iters, int max_iters)
{
  if (iters == max_iters)
  {
    return {0, 0, 0};
  }

  auto c = log(iters) / log(max_iters) * 255;

  return {c, c, c};
}

/**
 * A simple linear gradient color scheme
 *
 * @param iters     The number of iterations to turn into a color
 * @param max_iters The maximum number of iterations
 * @return          A tuple containing the [R,G,B] triple for that iteration
 */
std::tuple<int, int, int> color_scheme_1(int iters, int max_iters)
{
  if (iters == max_iters)
  {
    return {0, 0, 0};
  }

  int r, g, b;

  r = 100 + log(max_iters / iters) * 155;
  g = log(max_iters / iters) * 255;
  b = 50 + log(max_iters / iters) * 205;

  return {r, g, b};
}

/**
 * A simple logarithmic gradient color scheme
 *
 * @param iters     The number of iterations to turn into a color
 * @param max_iters The maximum number of iterations
 * @return          A tuple containing the [R,G,B] triple for that iteration
 */
std::tuple<int, int, int> color_scheme_2(int iters, int max_iters)
{
  if (iters == max_iters)
  {
    return {0, 0, 0};
  }

  int r, g, b;

  r = log(max_iters / iters) * 255;
  // r = log(iters) / log(max_iters) * 255;
  g = 0;
  b = 0;

  return {r, g, b};
}
