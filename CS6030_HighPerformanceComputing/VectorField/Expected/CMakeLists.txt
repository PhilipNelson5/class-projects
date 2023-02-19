set(TARGET Expected)

add_library(${TARGET} INTERFACE
    src/Expected/Expected.hpp
)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if (${MAKE_TESTS})
    add_subdirectory(test)
endif()
