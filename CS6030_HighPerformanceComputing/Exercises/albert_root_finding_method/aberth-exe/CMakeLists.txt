set(TARGET aberth-exe)

add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE aberth)
