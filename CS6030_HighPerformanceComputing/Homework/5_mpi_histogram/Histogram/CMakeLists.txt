set(TARGET Histogram)

add_library(${TARGET} INTERFACE
    src/Histogram/Histogram.hpp
)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(test)