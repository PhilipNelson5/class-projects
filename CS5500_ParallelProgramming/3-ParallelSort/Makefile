OBJS = main.cpp
EXECS=release
MPICC?=mpic++

all: ${EXECS}

release: ${OBJS}
	${MPICC} ${OBJS} -O3 -o release.out

stdsort: stdsort.cpp
	g++ -std=c++17 -O3 stdsort.cpp -o stdsort.out

clean:
	rm -f ${EXECS}
