OBJS = helloWorld.cpp

CC = g++
FLAGS = -fopenmp -std=c++17
DEBUG_FLAGS = -Og -g3 -Wall -Wextra -Werror
RELEASE_FLAGS = -O3

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o hello.out

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug.out

clean:
	rm *.gch *.out *.o &> /dev/null
