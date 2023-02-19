set(TARGET run_serial)
add_executable(${TARGET} main.cpp)

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.0.0
)
FetchContent_MakeAvailable(cxxopts)

target_link_libraries(${TARGET} PUBLIC cxxopts::cxxopts VectorField Graphics Helpers)