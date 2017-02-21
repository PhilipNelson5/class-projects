OBJS = main.cpp game.hpp game.cpp hashTable.hpp hashTable.cpp record.hpp
CC = g++
DEBUG_FLAGS = -O0 -g -fsanitize=address
RELEASE_FLAGS = -O3
FLAGS = -Wall -Wextra -Werror -std=c++14

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug

hash: hashTable.hpp hashTable.cpp hashTest.cpp
	$(CC) $(DEBUG_FLAGS) $(FLAGS) hashTable.hpp hashTable.cpp hashTest.cpp -o hash

test: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -lpthread -lgtest -o test 
