OBJS = main.cpp expected.hpp

CC = g++
DEBUG_FLAGS = -Og -g3 -Wall -Wextra -Werror
RELEASE_FLAGS = -O3
FLAGS = -std=c++14

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug.out

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release.out
