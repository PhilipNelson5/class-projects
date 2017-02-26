OBJS = main.cpp path.hpp termColors.hpp command.hpp command.cpp

CC = g++
DEBUG_FLAGS = -O0 -g3 -fsanitize=thread
RELEASE_FLAGS = -O3 -g0
FLAGS = -std=c++14 -Wall -Wextra #-Werror #-pie

default: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o conch && ./conch

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug && cgdb ./debug

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release

clean:
	rm debug release conch

all: $(OBJS)
	make debug release
