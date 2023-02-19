#include <Color/Color.hpp>
#include <Draw/Arrow.hpp>
#include <Draw/Curve.hpp>
#include <Draw/Line.hpp>
#include <Helpers/VectorFieldImageHelpers.hpp>
#include <Image/Image.hpp>
#include <Misc/lerp.hpp>
#include <Point/Pnt2.hpp>
#include <VectorField/VectorField.hpp>

#include "ParseArgs.hpp"

#include <chrono>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

template <typename F>
double time(F f)
{
    auto a = std::chrono::high_resolution_clock::now();
    f();
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = b - a;
    return ms.count();
}

template <typename T>
using Curve = std::vector<Math::Pnt2<T>>;

void example_1(const Args args)
{
    const int nX = args.vfW/20, nY = args.vfH/20;

    const auto vf = VectorField<float>::read(args.inputFileName, args.vfW, args.vfH);

    std::vector<Curve<float>> curvesRK;
    std::vector<Curve<float>> curvesEuler;
    auto a = std::chrono::high_resolution_clock::now();

    for (float y = 0; y < args.nCurves; ++y)
    {
        float ds = (float)vf.height()/args.nCurves/2.0;
        const Math::Pnt2<float>p {0, ds + (float)vf.height()/args.nCurves*y};
        curvesRK.push_back(vf.calculateStreamLineRK(p, args.maxCurveLen));
        curvesEuler.push_back(vf.calculateStreamLineEuler(p, args.maxCurveLen));
    }
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = b - a;
    std::cout << "time: " << ms.count() << std::endl;

    /** Draw Image **/
    if (args.makeImage)
    {
        Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);
        img.fill_image(Graphics::Color::BLACK);

        const auto arrows = vf.calculateArrowField(nX, nY);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX, Graphics::Color::WHITE);
        
        for (const auto& curve : curvesRK)
        {
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::RED);
        }

        for (const auto& curve : curvesEuler)
        {
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        }

        img.write_ppm("out.ppm");
        auto err = system("convert out.ppm out.png");
        (void)err;
    }
}

void example_2(Args args)
{
    const int nX = args.vfW/20, nY = args.vfH/20;
    args.vfH = 1000;
    args.vfW = args.vfH * 2;

    const auto vf = VectorField<float>::generate(
        [](const Math::Pnt2<float> p) -> Math::Vec2<float> { return { p.x*p.y, 10 * std::sin(p.x) }; },
        0, 40, -10, 10, args.vfW, args.vfH
    );

    std::vector<Curve<float>> curvesRK;
    std::vector<Curve<float>> curvesEuler;
    auto a = std::chrono::high_resolution_clock::now();

    const float start_y = 0;
    const float end_y = args.vfH/2.0f;
    for (float y = 0; y < args.nCurves; ++y)
    {
        float ds = (float)vf.height()/args.nCurves/2.0;
        const Math::Pnt2<float>p {
            (float)args.vfW - 10,
            ds + (end_y-start_y) / args.nCurves * y
        };
        curvesRK.push_back(vf.calculateStreamLineRK(p, args.maxCurveLen*4));
        // curvesEuler.push_back(vf.calculateStreamLineEuler(p, args.maxCurveLen*4));
    }
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = b - a;
    std::cout << "time: " << ms.count() << std::endl;

    /** Draw Image **/
    if (args.makeImage)
    {
        Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);
        img.fill_image(Graphics::Color::BLACK);

        const auto arrows = vf.calculateArrowField(nX, nY);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX, Graphics::Color::WHITE);
        
        for (const auto& curve : curvesRK)
        {
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {88, 214, 141});
        }

        for (const auto& curve : curvesEuler)
        {
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        }

        img.write_ppm("out.ppm");
        auto err = system("convert out.ppm out.png");
        (void)err;
    }
}

void example_3(Args args)
{
    const int nX = args.vfW/20, nY = args.vfH/20;
    args.vfH = 1000;
    args.vfW = args.vfH * 2;

    const auto vf = VectorField<float>::generate(
        [](const Math::Pnt2<float> p) -> Math::Vec2<float> { return { std::cos(p.y), std::sin(p.x) }; },
        -20, 20, -10, 10, args.vfW, args.vfH
    );

    std::vector<Curve<float>> curvesRK;
    std::vector<Curve<float>> curvesEuler;
    auto a = std::chrono::high_resolution_clock::now();

    for (float y = 0; y < args.nCurves; ++y)
        for (float x = 0; x < args.nCurves; ++x)
        {
            const Math::Pnt2<float>p {
                (float)args.vfW / args.nCurves * x,
                (float)args.vfH / args.nCurves * y,
            };
            curvesRK.push_back(vf.calculateStreamLineRK(p, args.maxCurveLen*4));
            // curvesEuler.push_back(vf.calculateStreamLineEuler(p, args.maxCurveLen*4));
        }
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = b - a;
    std::cout << "time: " << ms.count() << std::endl;

    /** Draw Image **/
    if (args.makeImage)
    {
        Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);
        img.fill_image(Graphics::Color::BLACK);

        const auto arrows = vf.calculateArrowField(nX, nY);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX, Graphics::Color::WHITE);
        
        for (const auto& curve : curvesRK)
        {
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {255, 87, 51 });
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {88, 214, 141});
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {93, 173, 226});
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::RED);
        }

        for (const auto& curve : curvesEuler)
        {
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        }

        img.write_ppm("out.ppm");
        auto err = system("convert out.ppm out.png");
        (void)err;
    }
}

void example_4(Args args)
{
    const int nX = args.vfW/20, nY = args.vfH/20;
    args.vfH = 1000;
    args.vfW = args.vfH * 2;

    const auto vf = VectorField<float>::generate(
        [](const Math::Pnt2<float> p) -> Math::Vec2<float> {
            std::complex<float> c {p.x, p.y};
            std::complex<float> z {0, 0};
            z = z * z + c;
            z = z * z + c;
            z = z * z + c;
            return { z.real(), z.imag() };
        },
        -20, 20, -10, 10, args.vfW, args.vfH
    );

    std::vector<Curve<float>> curvesRK;
    std::vector<Curve<float>> curvesEuler;
    auto a = std::chrono::high_resolution_clock::now();

    for (float y = 0; y < args.nCurves; ++y)
        for (float x = 0; x < args.nCurves; ++x)
        {
            const Math::Pnt2<float>p {
                (float)args.vfW / args.nCurves * x,
                (float)args.vfH / args.nCurves * y,
            };
            curvesRK.push_back(vf.calculateStreamLineRK(p, args.maxCurveLen*4));
            // curvesEuler.push_back(vf.calculateStreamLineEuler(p, args.maxCurveLen*4));
        }
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = b - a;
    std::cout << "time: " << ms.count() << std::endl;

    /** Draw Image **/
    if (args.makeImage)
    {
        Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);
        img.fill_image(Graphics::Color::BLACK);

        const auto arrows = vf.calculateArrowField(nX, nY);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX, Graphics::Color::WHITE);
        
        for (const auto& curve : curvesRK)
        {
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {255, 87, 51 });
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {88, 214, 141});
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {93, 173, 226});
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, {175, 122, 197});
            // Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::RED);
        }

        for (const auto& curve : curvesEuler)
        {
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        }

        img.write_ppm("out.ppm");
        auto err = system("convert out.ppm out.png");
        (void)err;
    }
}

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    example_4(args);

    return EXIT_SUCCESS;
}