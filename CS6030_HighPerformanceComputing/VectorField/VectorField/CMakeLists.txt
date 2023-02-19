set(TARGET VectorField)

add_library(${TARGET} INTERFACE
    src/VectorField/VectorField.hpp
)

target_include_directories(${TARGET} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${TARGET} INTERFACE Expected Math)

if (${MAKE_TESTS})
    add_subdirectory(test)
endif()
