#include <Color/Color.hpp>
#include <Draw/Arrow.hpp>
#include <Draw/Curve.hpp>
#include <Draw/Line.hpp>
#include <Helpers/VectorFieldImageHelpers.hpp>
#include <Helpers/CudaHelpers.hpp>
#include <Image/Image.hpp>
#include <Misc/lerp.hpp>
#include <Point/Pnt2.hpp>
#include <VectorField/VectorField.hpp>

#include "ParseArgs.hpp"

#include <chrono>
#include <cstddef>
#include <cuda.h>
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
double duration(T a, T b)
{
    std::chrono::duration<double, std::milli> ms = b - a;
    return ms.count();
}

template <typename T>
using Curve = std::vector<Math::Pnt2<T>>;

__global__ void calculate_streamlines(Math::Vec2<float> *input, Math::Pnt2<float> *output, const size_t width, const size_t height, const int nCurves, const int max_curve_len)
{
    const int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id >= nCurves) return;

    VectorField<float> vf(width, height, input, false);
    const Math::Pnt2<float>p {0, (float)height / nCurves * id};
    vf.calculateStreamLineRK(p, &output[id*max_curve_len], max_curve_len);
}

int main(int argc, char** argv)
{
    const auto program_start = std::chrono::high_resolution_clock::now();
    const auto args = parse_args(argc, argv);

    const int nX = args.vfW/10, nY = args.vfH/10;

    const auto vf = VectorField<float>::read(args.inputFileName, args.vfW, args.vfH);
    
    const auto streamline_start = std::chrono::high_resolution_clock::now();

    // Allocate host and device buffers
    Math::Pnt2<float> *output_h = new Math::Pnt2<float>[args.maxCurveLen*args.nCurves];
    Math::Vec2<float> *input_d;
    Math::Pnt2<float> *output_d;
    const size_t output_size_bytes = args.maxCurveLen * args.nCurves * sizeof(Math::Pnt2<float>);
    check(cudaMalloc((void **)&input_d, vf.size() * sizeof(VectorField<float>::value_type)), "malloc input_d");
    check(cudaMalloc((void **)&output_d, output_size_bytes), "malloc output_d");
        
    // Copy input to device
    check(cudaMemcpy((void *)input_d, (void *)vf.data(), vf.size() * sizeof(VectorField<float>::value_type), cudaMemcpyHostToDevice), "memcpy input");
    
    const int nBlocks = std::ceil((double)args.nCurves / args.nThreadsPerBlock);
    const int nThreads = args.nThreadsPerBlock > args.nCurves ? args.nCurves : args.nThreadsPerBlock;

    calculate_streamlines<<<nBlocks, nThreads>>>(input_d, output_d, vf.width(), vf.height(), args.nCurves, args.maxCurveLen);
    check("call calculate streamlines");
    
    // Copy output to host
    check(cudaMemcpy((void *)output_h, (void *)output_d, output_size_bytes, cudaMemcpyDeviceToHost), "memcpy output");

    const auto streamline_end = std::chrono::high_resolution_clock::now();
    const auto program_end = std::chrono::high_resolution_clock::now();
    const double program = duration(program_start, program_end);
    const double streamline = duration(streamline_start, streamline_end);
    std::cout << args.nThreadsPerBlock << "," << args.nCurves << "," << program << "," << streamline << std::endl;

    if (args.makeImage)
    {
        /** Draw Image **/
        Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);

        img.fill_image(Graphics::Color::WHITE);

        const auto arrows = vf.calculateArrowField(nX/2, nY/2);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX/2, Graphics::Color::BLACK);
        
        for (auto i = 0u; i < args.nCurves; ++i)
        {
            const int start = i*args.maxCurveLen;
            // length of the curve is the first "point" in the list
            const int length = output_h[start].x;
            const int first = start + 1;
            const int last = first + length - 1;
            Helpers::draw_curve_from_vector_space(&output_h[first], &output_h[last], img, vf, Graphics::Color::BLUE);
        }

        img.write_ppm("out.ppm");
        auto err = system("convert out.ppm out.png");
        (void)err;
    }

    delete[] output_h;
    check(cudaFree((void *)input_d), "free input");
    check(cudaFree((void *)output_d)), "free output";

    return EXIT_SUCCESS;
}
