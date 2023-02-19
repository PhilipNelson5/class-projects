project(histogram)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

# add the executable
add_executable(${PROJECT_NAME} main.cpp)
add_executable(${PROJECT_NAME}_benchmark benchmark.cpp)
set(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_benchmark)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE Threads::Threads)