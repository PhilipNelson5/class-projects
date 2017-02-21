OBJS = main.cpp avlTree.hpp wordLadderTree.hpp wordLadderTree.cpp wordLadderQueue.hpp wordLadderQueue.cpp linkedList.hpp Ladder.hpp
CC = g++
DEBUG_FLAGS = -O0 -g -fsanitize=address -o debug
RELEASE_FLAGS = -O3 -o release
FLAGS = -Wall -Wextra -Werror -std=c++14

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS)

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS)
