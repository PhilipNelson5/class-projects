cmake_minimum_required(VERSION 3.14)

# set the project name
project(
    Homework2
    LANGUAGES CXX
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")

include(CTest)
include(FetchContent)

add_subdirectory(histogram)
add_subdirectory(tests)
