OBJS = main.cpp
EXECS=release
MPICC?=mpic++

all: ${EXECS}

release: ${OBJS}
	${MPICC} -O3 ${OBJS} -o release

clean:
	rm -f ${EXECS}
