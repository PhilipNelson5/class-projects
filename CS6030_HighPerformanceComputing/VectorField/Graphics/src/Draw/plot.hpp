#pragma once

#include <Image/Image.hpp>

namespace Graphics {
namespace Draw {

/**
 * @brief Mix two colors
 * 
 * @param c1 first color
 * @param c2 second color
 * @param p percentage of first color
 * @return Pixel mixed color
 */
inline Pixel mixColors(const Pixel &c1, const Pixel &c2, const float p)
{
    return {static_cast<uint8_t>(std::sqrt(p * c1.r * c1.r + (1 - p) * c2.r * c2.r)),
            static_cast<uint8_t>(std::sqrt(p * c1.g * c1.g + (1 - p) * c2.g * c2.g)),
            static_cast<uint8_t>(std::sqrt(p * c1.b * c1.b + (1 - p) * c2.b * c2.b))};
}

/**
 * @brief Helper function to plot a pixel on an image with mixed colors
 * 
 * @param image image to draw on
 * @param x x location
 * @param y y location
 * @param p percentage of color to plot
 * @param c color
 */
inline void plot(Image &image, const int x, const int y, const double p, const Pixel &c)
{
    if (x < 0 || x >= image.width() || y < 0 || y >= image.height()) return;
    image.at({x,y}) = mixColors(c, image.at({x, y}), p);
}

} // namespace Draw
} // namespace Graphics