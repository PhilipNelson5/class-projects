cmake_minimum_required(VERSION 3.21)

# set the project name
project(
    Homework5
    LANGUAGES CXX
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")

enable_testing()
add_subdirectory(Helpers)
add_subdirectory(Histogram)
add_subdirectory(driver)
