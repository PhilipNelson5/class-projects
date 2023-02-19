#pragma once

#include <vector>

#include <Image/Pixel.hpp>
#include <Image/Image.hpp>
#include "plot.hpp"

namespace Graphics {
namespace Draw {

/**
 * @brief draw a curve to an image
 * 
 * @param image image to draw on
 * @param points list of points
 * @param color color
 */
inline void curve(Image& image, std::vector<Math::Pnt2<int>> const& points, const Pixel color)
{
    for(auto i = 1u; i < points.size(); ++i)
    {
        line(image, points[i-1], points[i], color);
    }
}

/**
 * @brief draw an anti-aliased curve to an image
 * 
 * @param image image to draw on
 * @param points list of points
 * @param color color
 */
inline void aaCurve(Image& image, std::vector<Math::Pnt2<int>> const& points, const Pixel color)
{
    for(auto i = 1u; i < points.size(); ++i)
    {
        aaLine(image, points[i-1], points[i], color);
    }
}

/**
 * @brief draw an anti-aliased curve with width to an image
 * 
 * @param image image to draw on
 * @param points list of points
 * @param wd width
 * @param color color
 */
inline void aaCurveWidth(Image& image, std::vector<Math::Pnt2<int>> const& points, const int wd, const Pixel color)
{
    for(auto i = 1u; i < points.size(); ++i)
    {
        aaLineWidth(image, points[i-1], points[i], wd, color);
    }
}

} // namespace Draw
} // namespace Graphics
