project(count_sort)

add_executable(${PROJECT_NAME} main.cpp)
add_executable(${PROJECT_NAME}_benchmark benchmark.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()