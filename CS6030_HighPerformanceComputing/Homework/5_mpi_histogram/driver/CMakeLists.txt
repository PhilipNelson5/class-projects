find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

set(TARGET run)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PUBLIC Histogram Helpers MPI::MPI_CXX)

set(TARGET bench)
add_executable(${TARGET} benchmark.cpp)
target_link_libraries(${TARGET} PUBLIC Histogram Helpers MPI::MPI_CXX)
