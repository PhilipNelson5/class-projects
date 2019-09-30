OBJS = ../matrix/matrix.hpp ../matrix/matrix_util.hpp ../matrix/vector_util.hpp main.cpp

CC = g++
FLAGS = -std=c++17
DEBUG_FLAGS = -g3 -Og
RELEASE_FLAGS = -g0 -O3
WARNINGS = -Wall -Wextra -Werror

release: $(OBJS)
	$(CC) $(FLAGS) $(RELEASE_FLAGS) $(WARNINGS) $(OBJS) -o explicitEulerTest.out

debug: $(OBJS)
	$(CC) $(FLAGS) $(DEBUG_FLAGS) $(WARNINGS) $(OBJS) -o debug.out
