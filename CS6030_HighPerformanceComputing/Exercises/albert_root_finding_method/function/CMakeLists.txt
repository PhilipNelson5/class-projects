set(TARGET function)

add_library(${TARGET}
    src/function/function.hpp
    src/function/function.cpp
)

target_include_directories(${TARGET} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")