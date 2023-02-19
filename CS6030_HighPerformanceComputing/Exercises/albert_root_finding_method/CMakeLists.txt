cmake_minimum_required(VERSION 3.14)

project(
    Aberth
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

include(CTest)
include(FetchContent)

add_subdirectory(function)
add_subdirectory(aberth)
add_subdirectory(aberth-exe)
add_subdirectory(tests)
