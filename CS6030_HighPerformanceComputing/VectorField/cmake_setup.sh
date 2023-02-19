mkdir -p build

/uufs/chpc.utah.edu/sys/installdir/cmake/3.21.4/bin/cmake \
    --no-warn-unused-cli \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCUDA_TOOLKIT_ROOT_DIR:STRING=/uufs/chpc.utah.edu/sys/installdir/cuda/10.0.130/ \
    -H/uufs/chpc.utah.edu/common/home/u6039417/CS6030/VectorField \
    -B/uufs/chpc.utah.edu/common/home/u6039417/CS6030/VectorField/build \
    -G "Unix Makefiles"

    # -DCMAKE_C_COMPILER:FILEPATH=/uufs/chpc.utah.edu/sys/installdir/gcc/9.2.0/bin/gcc \
    # -DCMAKE_CXX_COMPILER:FILEPATH=/uufs/chpc.utah.edu/sys/installdir/gcc/9.2.0/bin/g++ \