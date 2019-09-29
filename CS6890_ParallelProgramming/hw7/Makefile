OBJS=main.cpp
MPICC?=mpic++

release: ${OBJS}
	${MPICC} -O3 ${OBJS} -o release.out

run:
	mpiexec --oversubscribe -n 16 release.out

clean:
	rm -f *.out
