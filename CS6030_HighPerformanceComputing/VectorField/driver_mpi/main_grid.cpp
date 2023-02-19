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
using Curve = std::vector<Math::Pnt2<T>>;

enum DIRECTIONS {UP, DOWN, LEFT, RIGHT};
// enum class TAG { NEW_};
struct Args {
    int max_curve_length;
};
// struct CurveInfo

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const int size = get_comm_size(MCW);
    const int world_rank = get_comm_rank(MCW);

    const int vfW = 1300, vfH = 600;
    const int max_curve_len = 20000;
    const int nX = vfW/10, nY = vfH/10;
    const int nCurves = 4;nY/10;

    // Create 2 float datatype
    MPI_Datatype two_float_type;
    MPI_Type_contiguous(2, MPI_FLOAT, &two_float_type);
    MPI_Type_commit(&two_float_type);
    
    if (world_rank == 0)
    {
        MPI_Comm dummy;
        MPI_Comm_split(MCW, MPI_UNDEFINED, 0, &dummy);

        MPI_Finalize();
        return EXIT_SUCCESS;

        const auto vf = VectorField<float>::read("cyl2d_1300x600_float32.raw", vfW, vfH);

        /** Draw Image **/
        Graphics::Image img(vfW*4, vfH*4);
        img.fill_image(Graphics::Color::WHITE);

        const auto arrows = vf.calculateArrowField(nX/2, nY/2);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, nX/2, Graphics::Color::BLACK);

        // receive all curves
        MPI_Status status;
        int count;
        for (int i = 0; i < nCurves; ++i)
        {
            MPI_Probe(MPI_ANY_SOURCE, 0, MCW, &status);
            MPI_Get_count(&status, two_float_type, &count);
            std::vector<Math::Pnt2<float>> curve(count);
            MPI_Recv(curve.data(), count, two_float_type, status.MPI_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
            Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        }

        std::string file_name_prefix = "out";
        std::string ppm_name = file_name_prefix + ".ppm";
        std::string png_name = file_name_prefix + ".png";
        std::cout << "writing " << ppm_name << std::endl;
        const auto f = [&]() { img.write_ppm(ppm_name); };
        std::cout << "ppm write time: " << time(f) << std::endl;
        std::string command = "convert " + ppm_name + " " + png_name;
        auto err = system(command.c_str());
        
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // === Make Cartesian Communicator === //
    // === Make Worker Communicator === //
    MPI_Comm worker_comm;
    MPI_Comm_split(MCW, 0, world_rank, &worker_comm);
    const int worker_size = get_comm_size(worker_comm);
    // Ask MPI to decompose our processes in a 2D cartesian grid
    int neighbors[4], coords[2], dims[2]{0,0};
    int periods[2] = {false, false};
    int reorder = true;
    MPI_Comm grid_comm;
    MPI_Dims_create(worker_size, 2, dims);
    MPI_Cart_create(worker_comm, 2, dims, periods, reorder, &grid_comm);
    const int worker_rank = get_comm_rank(worker_comm);
    MPI_Cart_coords(grid_comm, worker_rank, 2, coords);
    MPI_Cart_shift(grid_comm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(grid_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    
    const int bW = vfW / dims[1];
    const int bH = vfH / dims[0];
 
    // === Calculate Streamlines === //
    const auto vf = VectorField<float>::read_block("cyl2d_1300x600_float32.raw", vfW, vfH, bW, bH, coords[1], coords[0]);
    std::cout << "read" << std::endl;
    
    // std::vector<Curve<float>> curves;
    // curves.reserve(nCurves);
    // const int send_count = nCurves/worker_size;

    if (worker_rank == 0)
    {
        // std::vector<Math::Pnt2<float>> streamlineStarts;
        // for (float y = 0; y < nCurves; ++y)
        // {
        //     streamlineStarts.push_back({0, (float)vf.height()/nCurves*y});
        // }

        // const auto my_starts = smart_scatter(streamlineStarts.data(), streamlineStarts.size(), send_count, two_float_type, worker_comm);
        // std::vector<Curve<float>> curves;
        // for (const auto & point : my_starts)
        // {
        //     printf("%d:\tp(%d, %d)\n", worker_rank, point.x, point.y);
        //     const auto curve = vf.calculateStreamLineRK(point, max_curve_len);
        //     curves.push_back(curve);
        //     // MPI_Send(curve.data(), curve.size(), two_float_type, 0, 0, MCW);
        // }

        /** Draw Image **/
        Graphics::Image img(bW*4, bH*4);
        img.fill_image(Graphics::Color::WHITE);

        const auto arrows = vf.calculateArrowField(bW/10, bH/10);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, bH/10, Graphics::Color::BLACK);

        // for (const auto & curve : curves)
        // {
        //     Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        // }

        std::string file_name_prefix = "out" + std::to_string(worker_rank);
        std::string ppm_name = file_name_prefix + ".ppm";
        std::string png_name = file_name_prefix + ".png";
        std::cout << "writing " << ppm_name << std::endl;
        const auto f = [&]() { img.write_ppm(ppm_name); };
        std::cout << "ppm write time: " << time(f) << std::endl;
        std::string command = "convert " + ppm_name + " " + png_name;
        auto err = system(command.c_str());
    }
    else
    {
        // const auto my_starts = smart_scatter<Math::Pnt2<float>>(NULL, nCurves, send_count, two_float_type, worker_comm);
        // std::vector<Curve<float>> curves;
        // for (const auto & point : my_starts)
        // {
        //     printf("%d:\tp(%d, %d)\n", worker_rank, point.x, point.y);
        //     const auto curve = vf.calculateStreamLineRK(point, max_curve_len);
        //     curves.push_back(curve);
        //     // MPI_Send(curve.data(), curve.size(), two_float_type, 0, 0, MCW);
        // }

        /** Draw Image **/
        Graphics::Image img(bW*4, bH*4);
        img.fill_image(Graphics::Color::WHITE);

        const auto arrows = vf.calculateArrowField(bW/10, bH/10);
        const auto t_arrows = Helpers::transform_to_pixel_space(arrows, img, vf);
        Graphics::Draw::aaArrowField(img, t_arrows, bH/10, Graphics::Color::BLACK);

        // for (const auto & curve : curves)
        // {
        //     Helpers::draw_curve_from_vector_space(std::begin(curve), std::end(curve), img, vf, Graphics::Color::BLUE);
        // }

        std::string file_name_prefix = "out" + std::to_string(worker_rank);
        std::string ppm_name = file_name_prefix + ".ppm";
        std::string png_name = file_name_prefix + ".png";
        std::cout << "writing " << ppm_name << std::endl;
        const auto f = [&]() { img.write_ppm(ppm_name); };
        std::cout << "ppm write time: " << time(f) << std::endl;
        std::string command = "convert " + ppm_name + " " + png_name;
        auto err = system(command.c_str());
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}