EXECS=release
MPICC?=g++

all: ${EXECS}

release: *.cpp
	${MPICC} -std=c++1z -pthread -Wall -o release main.cpp

clean:
	rm -f ${EXECS}
