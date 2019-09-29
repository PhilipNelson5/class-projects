OBJS=main.cpp
MPICC?=mpic++

release: ${OBJS}
	${MPICC} -std=c++14 -O3 ${OBJS} -o release.out

run:
	mpiexec --oversubscribe -n 8 release.out

clean:
	rm -f *.out
