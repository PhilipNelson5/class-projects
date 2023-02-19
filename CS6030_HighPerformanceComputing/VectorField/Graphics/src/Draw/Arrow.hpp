#pragma once

#include <iostream>

#include <Image/Pixel.hpp>
#include <Point/Pnt2.hpp>
#include <Vector/Vec2.hpp>
#include <Draw/Line.hpp>
#include "plot.hpp"

#include <vector>

namespace Math
{

// TODO: find the right place to put this
template <typename T>
Vec2<T> makeVec2(const Pnt2<T> head, const Pnt2<T> tail)
{
    return { tail.x - head.x, tail.y - head.y };
}

} // namespace Math

namespace Graphics {
namespace Draw {

/**
 * @brief draw an anti-aliased arrow on an image from hail to head
 * 
 * @param image image to draw on
 * @param x0 tail x point
 * @param y0 tail y point
 * @param x1 head x point
 * @param y1 head y point
 * @param c color
 */
inline void aaArrow(Image& image, int x0, int y0, int x1, int y1, const Pixel &c)
{
    using Vec = Math::Vec2<float>;
    using Pnt = Math::Pnt2<float>;
    aaLine(image, x0, y0, x1, y1, c);

    const Pnt pHead = Pnt{(float)x1, (float)y1};
    const Pnt pTail = Pnt{(float)x0, (float)y0};
    const float len = Math::distance(pHead, pTail) / 3.0;
    const Vec v = len * Math::normalize(makeVec2(pHead, pTail));
    const Pnt p1 = pHead + Math::rotate(v, 25.0f);
    const Pnt p2 = pHead + Math::rotate(v, -25.0f);

    aaLine(image, pHead.x, pHead.y, p1.x, p1.y, c);
    aaLine(image, pHead.x, pHead.y, p2.x, p2.y, c);
}
/**
 * @brief draw an anti-aliased arrow on an image from tail to head
 * 
 * @param image image to draw on
 * @param p0 tail
 * @param p1 head
 * @param c color
 */
inline void aaArrow(Image& image, Math::Pnt2<int> p0, Math::Pnt2<int> p1, const Pixel &c)
{ aaArrow(image, p0.x, p0.y, p1.x, p1.y, c); }

/**
 * @brief draw an anti-aliased arrow with width on an image from hail to head
 * 
 * @param image image to draw on
 * @param x0 tail x point
 * @param y0 tail y point
 * @param x1 head x point
 * @param y1 head y point
 * @param wd width
 * @param c color
 */
inline void aaArrowWidth(Image& image, int x0, int y0, int x1, int y1, const int wd, const Pixel &c)
{
    using Vec = Math::Vec2<float>;
    using Pnt = Math::Pnt2<float>;
    aaLineWidth(image, x0, y0, x1, y1, wd, c);

    const Pnt pHead = Pnt{(float)x1, (float)y1};
    const Pnt pTail = Pnt{(float)x0, (float)y0};
    const float len = Math::distance(pHead, pTail) / 3.0;
    const Vec v = len * Math::normalize(makeVec2(pHead, pTail));
    const Pnt p1 = pHead + Math::rotate(v, 25.0f);
    const Pnt p2 = pHead + Math::rotate(v, -25.0f);

    aaLineWidth(image, pHead.x, pHead.y, p1.x, p1.y, wd, c);
    aaLineWidth(image, pHead.x, pHead.y, p2.x, p2.y, wd, c);
}
/**
 * @brief draw an anti-aliased arrow with width on an image from hail to head
 * 
 * @param image image to draw on
 * @param p0 tail
 * @param p1 head
 * @param wd width
 * @param c color
 */
inline void aaArrowWidth(Image& image, Math::Pnt2<int> p0, Math::Pnt2<int> p1, const int wd, const Pixel &c)
{ aaArrowWidth(image, p0.x, p0.y, p1.x, p1.y, wd, c); }

/**
 * @brief draw an evenly distributed field of arrows on an image
 * 
 * @tparam T 
 * @param image image to draw on
 * @param arrows list of arrows, pair<point, direction>
 * @param nX number of arrows along the width of the image
 * @param c color
 */
template <typename T>
inline void aaArrowField(Image& image, const std::vector<std::pair<Math::Pnt2<int>, Math::Vec2<T>>> & arrows, const int nX, const Pixel &c)
{
    const int cell_wd = image.width() / nX / 3 * 2;
    for(auto i = 0u; i < arrows.size(); ++i)
    {
        const auto & arrow = arrows[i];
        Math::Pnt2<int> head {
            arrow.first.x + static_cast<int>(cell_wd * arrow.second.x),
            arrow.first.y + static_cast<int>(cell_wd * arrow.second.y)
        };
        aaArrow(image, arrow.first, head, c);
    }
}

} // namespace Draw
} // namespace Graphics
