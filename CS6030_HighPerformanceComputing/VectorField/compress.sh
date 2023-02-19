cp writeup/writeup.pdf .
tar -cvzf VectorField.tar.gz writeup.pdf \
    bench_* \
    writeup.pdf \
    CMakeLists.txt \
    cyl2d_1300x600_float32.raw \
    VectorField \
    Math \
    Helpers \
    Graphics \
    Expected \
    driver_thread \
    driver_serial \
    driver_mpi \
    driver_cuda