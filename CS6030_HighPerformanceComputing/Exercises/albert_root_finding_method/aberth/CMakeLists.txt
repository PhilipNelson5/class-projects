set(TARGET aberth)

add_library(${TARGET}
    src/aberth/aberth.hpp
    src/aberth/aberth.cpp
)

target_link_libraries(${TARGET} PUBLIC function)

target_include_directories(${TARGET} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")