cmake_minimum_required(VERSION 3.14)

project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(hello hello.cu)
add_executable(info cuda_info.cu)


