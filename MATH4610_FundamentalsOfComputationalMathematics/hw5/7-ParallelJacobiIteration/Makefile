OBJS = main.cpp

CC = g++
FLAGS = -std=c++17 -fopenmp
DEBUG_FLAGS = -Og -g3 -Wall -Wextra
RELEASE_FLAGS = -O3

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o parallelJacobiIteration.out

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug.out
