EXECS=release
MPICC?=mpic++

all: ${EXECS}

release: global_sum.cpp
	${MPICC} -std=c++1z -o release global_sum.cpp

clean:
	rm -f ${EXECS}
