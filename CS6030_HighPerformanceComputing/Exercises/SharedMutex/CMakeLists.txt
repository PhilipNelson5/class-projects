cmake_minimum_required(VERSION 3.10)

# set the project name
project(SharedMutex VERSION 1.0.0)

# add the executable
add_executable(release main.cpp)

# specify the C++ standard
set_target_properties(release PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(release PRIVATE Threads::Threads)