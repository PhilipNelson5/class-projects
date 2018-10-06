OBJS = bitonic.cpp
EXECS=release
MPICC?=mpic++
FLAGS=-O3

all: ${EXECS}

release: ${OBJS}
	${MPICC} ${FLAGS} ${OBJS} -o release.out

clean:
	rm -f ${EXECS}
