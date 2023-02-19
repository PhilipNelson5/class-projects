#pragma once

#include <Image/Pixel.hpp>
#include "plot.hpp"

namespace Graphics {
namespace Draw {
    
/**
 * @brief draw an anti-aliased circle on an image
 * 
 * @param image image to draw on
 * @param xm center x point
 * @param ym center y point
 * @param r radius
 * @param color color
 */
inline void aaCircle(Image &image, int xm, int ym, int r, const Pixel &color)
{
    float x = r, y = 0;
    float i, x2, e2, err = 2 - 2 * r;
    r = 1 - err;
    for (;;)
    {
        i = 1 - abs(err + 2 * (x + y) - 2) / r;
        plot(image, xm + x, ym - y, i, color);
        plot(image, xm + y, ym + x, i, color);
        plot(image, xm - x, ym + y, i, color);
        plot(image, xm - y, ym - x, i, color);
        if (x == 0)
            break;
        e2 = err;
        x2 = x;
        if (err > y)
        {
            i = 1 - (err + 2 * x - 1) / r;
            if (i <= 1)
            {
                plot(image, xm + x, ym - y + 1, i, color);
                plot(image, xm + y - 1, ym + x, i, color);
                plot(image, xm - x, ym + y - 1, i, color);
                plot(image, xm - y + 1, ym - x, i, color);
            }
            err -= --x * 2 - 1;
        }
        if (e2 <= x2--)
        {
            i = 1 - (1 - 2 * y - e2) / r;
            if (i <= 1)
            {
                plot(image, xm + x2, ym - y, i, color);
                plot(image, xm + y, ym + x2, i, color);
                plot(image, xm - x2, ym + y, i, color);
                plot(image, xm - y, ym - x2, i, color);
            }
            err -= --y * 2 - 1;
        }
    }
}
/**
 * @brief draw an anti-aliased circle on an image
 * 
 * @param image image to draw on
 * @param p center
 * @param r radius
 * @param color color
 */
inline void aaCircle(Image &image, Math::Pnt2<int> p, int r, const Pixel &color)
{ aaCircle(image, p.x, p.y, r, color); }

} // namespace Draw
} // namespace Graphics
