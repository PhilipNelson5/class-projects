OBJS=main.cpp
MPICC?=mpic++

release: ${OBJS}
	${MPICC} -O3 ${OBJS} -o release.out

run:
	mpiexec -n 8 --oversubscribe release.out

clean:
	rm -f *.out
