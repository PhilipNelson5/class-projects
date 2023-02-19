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
#include <cstddef>
#include <iostream>
#include <mutex>
#include <thread>
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
double duration(T a, T b)
{
    std::chrono::duration<double, std::milli> ms = b - a;
    return ms.count();
}

template <typename T>
using Curve = std::vector<Math::Pnt2<T>>;

int main(int argc, char** argv)
{
    const auto program_start = std::chrono::high_resolution_clock::now();
    const auto args = parse_args(argc, argv);

    const auto vf = VectorField<float>::read(args.inputFileName, args.vfW, args.vfH);
    const int nX = vf.width()/20, nY = vf.height()/20;

    std::vector<Curve<float>> curves;
    std::vector<Math::Pnt2<float>> streamlineStarts;
    for (int y = 0; y < args.nCurves; ++y)
    {
        float ds = (float)vf.height()/args.nCurves / 2;
        streamlineStarts.push_back({1, ds+(float)vf.height()/(float)args.nCurves*(float)y});
    }
    
    std::vector<std::thread> threads;
    // if (args.nThreads > streamlineStarts.size()) args.nThreads = streamlineStarts.size();
    std::mutex m;
    const auto streamline_start = std::chrono::high_resolution_clock::now();
    auto first = std::begin(streamlineStarts);
    for (auto i = 0u; i < args.nThreads-1; ++i)
    {
        auto last = first + streamlineStarts.size() / args.nThreads;
        threads.emplace_back([&args, &m, &curves, &vf, first, last](){
            for(auto curr = first; curr < last; ++curr)
            {
                const auto curve = vf.calculateStreamLineRK(*curr, args.maxCurveLen); 
                {
                    std::lock_guard<std::mutex> lock(m);
                    curves.push_back(std::move(curve));
                }
            }
        });
        first = last;
    }

    const auto last = std::end(streamlineStarts);
    for(auto curr = first; curr < last; ++curr)
    {
        const auto curve = vf.calculateStreamLineRK(*curr, args.maxCurveLen); 
        {
            std::lock_guard<std::mutex> lock(m);
            curves.push_back(std::move(curve));
        }
    }
    
    for (auto& thread : threads) { thread.join(); }

    const auto streamline_end = std::chrono::high_resolution_clock::now();
    const auto program_end = std::chrono::high_resolution_clock::now();
    const double streamline = duration(streamline_start, streamline_end);
    const double program = duration(program_start, program_end);
    std::cout << args.nThreads << "," << args.nCurves << "," << program << "," << streamline << std::endl;
    
    if (args.makeImage)
    {
        Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);
        img.fill_image(Graphics::Color::BLACK);

        const auto arrows = vf.calculateArrowField(nX, nY);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX, Graphics::Color::WHITE);
        
        for (const auto& curve : curves)
        {
            Helpers::draw_curve_from_vector_space(
                std::begin(curve), std::end(curve), img, vf, {236, 112, 99}
            );
        }

        img.write_ppm("out.ppm");
        auto err = system("convert out.ppm out.png");
        (void)err;
    }
    return EXIT_SUCCESS;
}