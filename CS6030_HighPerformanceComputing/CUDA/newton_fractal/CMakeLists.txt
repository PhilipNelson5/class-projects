cmake_minimum_required(VERSION 3.14)

project(gif LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror --expt-relaxed-constexpr")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(${PROJECT_NAME} main.cu)

#target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  #--expt-relaxed-constexpr
#>)

