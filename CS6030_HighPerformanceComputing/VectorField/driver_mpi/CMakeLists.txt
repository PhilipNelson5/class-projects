find_package(MPI REQUIRED)

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.0.0
)
FetchContent_MakeAvailable(cxxopts)


set(TARGET run_mpi)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PUBLIC cxxopts::cxxopts MPI::MPI_CXX VectorField Graphics Helpers)

set(TARGET_GRID run_mpi_grid)
add_executable(${TARGET_GRID} main_grid.cpp)
target_link_libraries(${TARGET_GRID} PUBLIC cxxopts::cxxopts MPI::MPI_CXX VectorField Graphics Helpers)
