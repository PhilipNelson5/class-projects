OBJS = main.cpp ../matrix/matrix.hpp

CC = g++
FLAGS = -std=c++17
DEBUG_FLAGS = -Og -g3 -Wall -Wextra
RELEASE_FLAGS = -O3

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o test.out

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug.out
