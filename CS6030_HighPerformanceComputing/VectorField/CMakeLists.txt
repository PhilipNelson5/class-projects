cmake_minimum_required(VERSION 3.14)

project(
    FinalProject
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")
# add_compile_options(-Wall -Wextra -Wpedantic -Werror)

set (MAKE_TESTS FALSE)
if (${MAKE_TESTS})
    enable_testing()
endif()

add_subdirectory(Expected)
add_subdirectory(Math)
add_subdirectory(VectorField)
add_subdirectory(Graphics)
add_subdirectory(Helpers)
add_subdirectory(driver_cuda)
add_subdirectory(driver_mpi)
add_subdirectory(driver_serial)
add_subdirectory(driver_thread)
