OBJS = main.cpp logisticSolver.hpp

CC = g++
FLAGS = -std=c++17
DEBUG_FLAGS = -Og -g3 -Wall -Wextra -Werror
RELEASE_FLAGS = -O3

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o logistic.out

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug.out
