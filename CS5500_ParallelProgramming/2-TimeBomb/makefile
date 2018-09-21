EXECS=release
MPICC?=mpic++

all: ${EXECS}

release: main.cpp
	${MPICC} -o release main.cpp

clean:
	rm -f ${EXECS}
