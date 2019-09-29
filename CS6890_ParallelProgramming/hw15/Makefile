OBJS=main.cpp
MPICC?=mpic++

release: ${OBJS}
	${MPICC} -std=c++17 -O3 ${OBJS} -o release.out

run:
	mpiexec --oversubscribe -n 20 release.out

clean:
	rm -f *.out
