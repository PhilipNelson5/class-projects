EXECS=release
MPICC?=mpic++

all: ${EXECS}

release: *.hpp *.cpp
	${MPICC} -std=c++1z -Wall -o release histogram.cpp

clean:
	rm -f ${EXECS}
