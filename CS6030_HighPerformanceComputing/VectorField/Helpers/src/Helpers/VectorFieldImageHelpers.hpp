#pragma once

#include <Color/Color.hpp>
#include <Misc/lerp.hpp>
#include <Draw/Arrow.hpp>
#include <Draw/Curve.hpp>
#include <Draw/Line.hpp>
#include <Image/Image.hpp>
#include <Point/Pnt2.hpp>
#include <VectorField/VectorField.hpp>

#include <algorithm>
#include <chrono>
#include <vector>

namespace Helpers {

/**
 * @brief transform list of arrows from vector field space to image pixel space
 * 
 * @tparam T 
 * @param arrows list of arrows, pair<point, direction>
 * @param image image
 * @param vf vector field
 * @return std::vector<std::pair<Math::Pnt2<int>, Math::Vec2<T>>> transformed arrows
 */
template <typename T>
inline std::vector<std::pair<Math::Pnt2<int>, Math::Vec2<T>>>
transform_to_pixel_space(const std::vector<std::pair<Math::Pnt2<T>, Math::Vec2<T>>>& arrows, const Graphics::Image& image, const VectorField<T>& vf)
{
    std::vector<std::pair<Math::Pnt2<int>, Math::Vec2<T>>> t_arrows;
    t_arrows.reserve(arrows.size());
    for (const auto& arrow : arrows)
    {
        t_arrows.emplace_back(Math::Pnt2<int> {
            static_cast<int>(Math::lerp(0.0f, (T)image.width(), arrow.first.x / vf.width())),
            static_cast<int>(Math::lerp(0.0f, (T)image.height(), arrow.first.y / vf.height()))
        }, arrow.second);
    }
    return t_arrows;
}

/**
 * @brief transform an iterator of points from vector field space to image pixel space
 * 
 * @tparam T 
 * @tparam ForwardIt 
 * @param first first point iterator
 * @param last end iterator
 * @param image image
 * @param vf vector field
 * @return std::vector<Math::Pnt2<int>> transformed points
 */
template <typename T, typename ForwardIt>
inline std::vector<Math::Pnt2<int>>
transform_to_pixel_space(ForwardIt first, ForwardIt last, Graphics::Image const& image, VectorField<T> const& vf)
{
    std::vector<Math::Pnt2<int>> t_points;
    t_points.reserve(std::distance(first, last));
    std::for_each(first, last, [&t_points, &image, &vf](const Math::Pnt2<T>& point){
        t_points.emplace_back(Math::Pnt2<int> {
            static_cast<int>(Math::lerp(0.0f, (T)image.width(), point.x / vf.width())),
            static_cast<int>(Math::lerp(0.0f, (T)image.height(), point.y / vf.height()))
        });
    });
    return t_points;
}

/**
 * @brief draw a curve, iterator of points, in vector field space, on an image in pixel space
 * 
 * @tparam T 
 * @tparam ForwardIt 
 * @param first first point iterator
 * @param last end iterator
 * @param image image
 * @param vf vector field
 * @param color color
 */
template <typename T, typename ForwardIt>
inline void draw_curve_from_vector_space(ForwardIt first, ForwardIt last, Graphics::Image& image, VectorField<T> const& vf, const Graphics::Pixel color)
{
    const auto t_points = transform_to_pixel_space(first, last, image, vf);
    Graphics::Draw::aaCurveWidth(image, t_points, 3, color);
}

} // namespace Helpers