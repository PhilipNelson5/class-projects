#include <Color/Color.hpp>
#include <Draw/Arrow.hpp>
#include <Draw/Curve.hpp>
#include <Draw/Line.hpp>
#include <Helpers/VectorFieldImageHelpers.hpp>
#include <Image/Image.hpp>
#include <Misc/lerp.hpp>
#include <Point/Pnt2.hpp>
#include <VectorField/VectorField.hpp>

#include "mpi_utils.hpp"
#include "scatter.hpp"
#include "ParseArgs.hpp"

#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
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

Args get_args(int argc, char** argv, int rank)
{
    if (rank == 0)
    {
        auto args = parse_args(argc, argv);

        // broadcast everything but the string at the end of the Arg struct
        MPI_Bcast(&args, sizeof(args)-sizeof(std::string), MPI_CHAR, 0, MCW);

        int size = args.inputFileName.size() + 1;
        MPI_Bcast(&size, 1, MPI_INT, 0, MCW);

        MPI_Bcast(const_cast<char*>(args.inputFileName.data()), size, MPI_CHAR, 0, MCW);

        return args;
    }
    else
    {
        Args args;

        // broadcast everything but the string at the end of the Arg struct
        MPI_Bcast(&args, sizeof(args)-sizeof(std::string), MPI_CHAR, 0, MCW);

        int size;
        MPI_Bcast(&size, 1, MPI_INT, 0, MCW);

        args.inputFileName.resize(size);
        MPI_Bcast(const_cast<char*>(args.inputFileName.data()), size, MPI_CHAR, 0, MCW);

        return args;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const auto program_start = std::chrono::high_resolution_clock::now();
    const int world_size = get_comm_size(MCW);
    const int world_rank = get_comm_rank(MCW);
    const auto args = get_args(argc, argv, world_rank);

    if (world_size < 2)
    {
        if (world_rank == 0) std::cout << "requires at least two processes" << std::endl;
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    const int nX = args.vfW/20, nY = args.vfH/20;

    // === Create 2 float datatype === //
    MPI_Datatype two_float_type;
    MPI_Type_contiguous(2, MPI_FLOAT, &two_float_type);
    MPI_Type_commit(&two_float_type);
    
    if (world_rank == 0)
    {
        MPI_Comm dummy;
        MPI_Comm_split(MCW, MPI_UNDEFINED, 0, &dummy);

        const auto vf = VectorField<float>::read(args.inputFileName, args.vfW, args.vfH);

        // === Receive Curves === //
        MPI_Status status;
        int count;
        std::vector<Curve<float>> curves;
        curves.reserve(args.nCurves);
        for (int i = 0; i < args.nCurves; ++i)
        {
            MPI_Probe(MPI_ANY_SOURCE, 0, MCW, &status);
            MPI_Get_count(&status, two_float_type, &count);
            std::vector<Math::Pnt2<float>> curve(count);
            MPI_Recv(curve.data(), count, two_float_type, status.MPI_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
            curves.push_back(std::move(curve));
        }
        const auto program_end = std::chrono::high_resolution_clock::now();
        const double program = duration(program_start, program_end);
        double streamline;
        MPI_Recv(&streamline, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
        std::cout << world_size << "," << args.nCurves << "," << program << "," << streamline << std::endl;

        if (args.makeImage)
        {
            // === Draw Image === //
            Graphics::Image img(args.vfW*args.imgScaleFactor, args.vfH*args.imgScaleFactor);
            img.fill_image(Graphics::Color::WHITE);

            const auto arrows = vf.calculateArrowField(nX/2, nY/2);
            const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
            Graphics::Draw::aaArrowField(img, t_arrows, nX/2, Graphics::Color::BLACK);

            for (const auto & curve : curves)
            {
                Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
            }

            img.write_ppm("out.ppm");
            auto err = system("convert out.ppm out.png");
            (void)err;
        }
        
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // === Make Worker Communicator === //
    MPI_Comm worker_comm;
    MPI_Comm_split(MCW, 0, world_rank, &worker_comm);
    const int worker_size = get_comm_size(worker_comm);
    const int worker_rank = get_comm_rank(worker_comm);
 
    // === Calculate Streamlines === //
    const auto vf = VectorField<float>::read(args.inputFileName, args.vfW, args.vfH);
    
    std::vector<Curve<float>> curves;
    curves.reserve(args.nCurves);
    const int send_count = args.nCurves/worker_size;

    if (worker_rank == 0)
    {
        MPI_Barrier(worker_comm);
        const auto streamline_start = std::chrono::high_resolution_clock::now();

        std::vector<Math::Pnt2<float>> streamlineStarts;
        for (int y = 0; y < args.nCurves; ++y)
        {
            streamlineStarts.push_back({0, (float)vf.height()/args.nCurves*y});
        }

        const auto my_starts = smart_scatter(streamlineStarts.data(), streamlineStarts.size(), send_count, two_float_type, worker_comm);
        for (const auto & point : my_starts)
        {
            const auto curve = vf.calculateStreamLineRK(point, args.maxCurveLen);
            MPI_Send(curve.data(), curve.size(), two_float_type, 0, 0, MCW);
        }
        const auto streamline_end = std::chrono::high_resolution_clock::now();
        const double streamline = duration(streamline_start, streamline_end);
        double average;
        MPI_Reduce(&streamline, &average, 1, MPI_DOUBLE, MPI_SUM, 0, worker_comm);
        average /= worker_size;
        MPI_Send(&average, 1, MPI_DOUBLE, 0, 0, MCW);
    }
    else
    {
        MPI_Barrier(worker_comm);
        const auto streamline_start = std::chrono::high_resolution_clock::now();

        const auto my_starts = smart_scatter<Math::Pnt2<float>>(NULL, args.nCurves, send_count, two_float_type, worker_comm);
        for (const auto & point : my_starts)
        {
            const auto curve = vf.calculateStreamLineRK(point, args.maxCurveLen);
            MPI_Send(curve.data(), curve.size(), two_float_type, 0, 0, MCW);
        }
        const auto streamline_end = std::chrono::high_resolution_clock::now();
        const double streamline = duration(streamline_start, streamline_end);
        MPI_Reduce(&streamline, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, worker_comm);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}