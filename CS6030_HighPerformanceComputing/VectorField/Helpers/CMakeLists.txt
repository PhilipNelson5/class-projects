set(TARGET Helpers)

add_library(${TARGET} INTERFACE
    src/Helpers/AssertHelper.hpp
    src/Helpers/CudaHelpers.hpp
    src/Helpers/VectorFieldImageHelpers.hpp
)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")