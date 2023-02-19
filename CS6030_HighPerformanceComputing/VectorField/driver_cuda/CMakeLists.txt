include(CheckLanguage)

check_language(CUDA)

if (CMAKE_CUDA_COMPILER)

    message(STATUS "CUDA AVAILIABLE")
	  enable_language(CUDA)

	  if(NOT DEFINED CMAKE_CUDA_STANDARD)
		    set(CMAKE_CUDA_STANDARD 14)
		    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	  endif()

    include(FetchContent)
    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v3.0.0
    )
    FetchContent_MakeAvailable(cxxopts)

	  set(TARGET run_cuda)
	  add_executable(${TARGET} main.cu)
	  set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
	  target_link_libraries(${TARGET} PUBLIC cxxopts::cxxopts VectorField Graphics Helpers)

else()

    message(STATUS "CUDA UNAVAILIABLE")

endif()
