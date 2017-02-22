OBJS = main.cpp edge.hpp graph.hpp graph.cpp 
CC = g++
DEBUG_FLAGS = -O0 -g -fsanitize=address
RELEASE_FLAGS = -O3
FLAGS = -std=c++14 -Wall -Wextra -Werror

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release
