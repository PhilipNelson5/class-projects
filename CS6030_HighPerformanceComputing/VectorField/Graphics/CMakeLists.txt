set(TARGET Graphics)

add_library(${TARGET} INTERFACE
    src/Color/Color.hpp
    src/Draw/Arrow.hpp
    src/Draw/Circle.hpp
    src/Draw/Curve.hpp
    src/Draw/Line.hpp
    src/Image/Image.hpp
    src/Image/Pixel.hpp
)

target_link_libraries(${TARGET} INTERFACE Math)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")