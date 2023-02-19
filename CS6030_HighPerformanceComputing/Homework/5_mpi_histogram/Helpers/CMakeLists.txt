set(TARGET Helpers)

add_library(${TARGET} INTERFACE
    src/helpers.hpp
    src/print.hpp
    src/mpi_get_type.hpp
)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")