#pragma once

#include <Image/Pixel.hpp>
#include "plot.hpp"

namespace Graphics {
namespace Draw {
    
inline int   ipart(float x)  { return int(std::floor(x)); }
inline float round(float x)  { return std::round(x);      }
inline float fpart(float x)  { return x - std::floor(x);  }
inline float rfpart(float x) { return 1.0f - fpart(x);    }

/**
 * @brief Bresenham's line algorithm for drawing aliased lines on a image
 * 
 * @param image image to draw on
 * @param x0 start x point
 * @param y0 start y point
 * @param x1 end x point
 * @param y1 end y point
 * @param c color
 */
inline void line(Image &image, int x0, int y0, int x1, int y1, const Pixel &c)
{
    const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    const int dy = std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2, e2;

    while (true)
    {
        if (x0 >= 0 && x0 < image.width() && y0 >= 0 && y0 < image.height())
            image.at({x0, y0}) = c;
        if (x0 == x1 && y0 == y1)
            break;
        e2 = err;
        if (e2 > -dx)
        {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dy)
        {
            err += dx;
            y0 += sy;
        }
    }
}
/**
 * @brief Bresenham's line algorithm for drawing aliased lines on a image
 * 
 * @param image image to draw on
 * @param p0 start point
 * @param p1 end point
 * @param c color
 */
inline void line(Image &image, Math::Pnt2<int> p0, Math::Pnt2<int> p1, const Pixel &c)
{ line(image, p0.x, p0.y, p1.x, p1.y, c); }

/**
 * @brief Xiaolin Wu's line algorithm for drawing anti-aliased lines on an image
 * 
 * @param image image to draw on
 * @param x0 start x point
 * @param y0 start y point
 * @param x1 end x point
 * @param y1 end y point
 * @param c color
 */
inline void aaLine(Image &image, int x0, int y0, int x1, int y1, const Pixel &c)
{
    // x0 = std::clamp(x0, 0, image.width);
    // y0 = std::clamp(y0, 0, image.height);
    // x1 = std::clamp(x1, 0, image.width);
    // y1 = std::clamp(y1, 0, image.height);
    const bool steep = std::abs(y1 - y0) > abs(x1 - x0);

    if (steep)
    {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }

    if (x0 > x1)
    {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }

    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float gradient = (dx == 0.0f) ? 1.0f : dy / dx;

    // handle first endpoint
    int xpxl1;
    float intery;
    {
        const float xend = round(x0);
        const float yend = y0 + gradient * (xend - x0);
        const float xgap = rfpart(x0 + 0.5);
        xpxl1 = xend;
        const int ypxl1 = ipart(yend);

        if (steep)
        {
            plot(image, ypxl1, xpxl1, rfpart(yend) * xgap, c);
            plot(image, ypxl1 + 1, xpxl1, fpart(yend) * xgap, c);
        }
        else
        {
            plot(image, xpxl1, ypxl1, rfpart(yend) * xgap, c);
            plot(image, xpxl1, ypxl1 + 1, fpart(yend) * xgap, c);
        }
        intery = yend + gradient; // first y-intersection for the main loop
    }

    int xpxl2;
    {
        // handle second endpoint
        const float xend = round(x1);
        const float yend = y1 + gradient * (xend - x1);
        const float xgap = fpart(x1 + 0.5);
        xpxl2 = xend;
        const int ypxl2 = ipart(yend);
        if (steep)
        {
            plot(image, ypxl2, xpxl2, rfpart(yend) * xgap, c);
            plot(image, ypxl2 + 1, xpxl2, fpart(yend) * xgap, c);
        }
        else
        {
            plot(image, xpxl2, ypxl2, rfpart(yend) * xgap, c);
            plot(image, xpxl2, ypxl2 + 1, fpart(yend) * xgap, c);
        }
    }

    // main loop
    if (steep)
    {
        for (int x = xpxl1 + 1; x <= xpxl2 - 1; ++x)
        {
            plot(image, ipart(intery), x, rfpart(intery), c);
            plot(image, ipart(intery) + 1, x, fpart(intery), c);
            intery += gradient;
        }
    }
    else
    {
        for (int x = xpxl1 + 1; x <= xpxl2 - 1; ++x)
        {
            plot(image, x, ipart(intery), rfpart(intery), c);
            plot(image, x, ipart(intery) + 1, fpart(intery), c);
            intery += gradient;
        }
    }
}
/**
 * @brief Xiaolin Wu's line algorithm for drawing anti-aliased lines on an image
 * 
 * @param image image to draw on
 * @param p0 start point
 * @param p1 end point
 * @param c color
 */
inline void aaLine(Image &image, Math::Pnt2<int> p0, Math::Pnt2<int> p1, const Pixel &c)
{ aaLine(image, p0.x, p0.y, p1.x, p1.y, c); }

/**
 * @brief an attempt to draw a thick line by combining multiple aliased and anti-aliased lines
 * 
 * @param image image to draw on
 * @param x0 start x point
 * @param y0 start y point
 * @param x1 end x point
 * @param y1 end y point
 * @param wd width
 * @param color color
 */
inline void aaLineWidth(Image& image, int x0, int y0, int x1, int y1, int wd, const Pixel& color)
{
    if (wd == 1)
    {
        aaLine(image, x0, y0, x1, y1, color);
        return;
    }
    wd = wd / 2;
    for (int i = -wd+1; i < wd; ++i)
    {
        line(image, x0, y0 + i, x1, y1 + i, color);
    }
    aaLine(image, x0, y0 - wd, x1, y1 - wd, color);
    aaLine(image, x0, y0 + wd , x1, y1 + wd , color);
}
/**
 * @brief an attempt to draw a thick line by combining multiple aliased and anti-aliased lines
 * 
 * @param image image to draw on
 * @param p0 start point
 * @param p1 end point
 * @param wd width
 * @param color color
 */
inline void aaLineWidth(Image &image, Math::Pnt2<int> p0, Math::Pnt2<int> p1, int wd, const Pixel &c)
{ aaLineWidth(image, p0.x, p0.y, p1.x, p1.y, wd, c); }


} // namespace Draw
} // namespace Graphics