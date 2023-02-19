set(TARGET Math)

add_library(${TARGET} INTERFACE
    src/Point/Pnt2.hpp
    src/Solvers/RungeKutta.hpp
    src/Solvers/Euler.hpp
    src/Vector/Vec2.hpp
    src/Misc/lerp.hpp
)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(${TARGET} INTERFACE Expected)

if (${MAKE_TESTS})
    add_subdirectory(test)
endif()
