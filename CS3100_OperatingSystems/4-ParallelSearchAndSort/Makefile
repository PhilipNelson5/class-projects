OBJS = main.cpp timer.hpp threadPool.hpp threadSafeQueue.hpp parallelLinearSearch.hpp parallelQuickSort.hpp 

CC = g++
DEBUG_FLAGS = -O0 -g3 -fsanitize=thread
RELEASE_FLAGS = -O3 -g0
FLAGS = -std=c++14 -pthread -Wall -Wextra -Werror #-pie

default: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o PhilipNelson && ./PhilipNelson -v

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release

clean:
	rm debug release PhilipNelson

all: $(OBJS)
	make debug release
