OBJS = main.cpp mandelbrot.hpp mandelbrot.cpp image.hpp image.cpp timer.hpp threadPool.hpp threadSafeQueue.hpp

CC = g++
DEBUG_FLAGS = -O0 -g3 -fsanitize=thread -Wall -Wextra -Werror
RELEASE_FLAGS = -O3 -g0
FLAGS = -std=c++14 -pthread -Wall -Wextra -Werror -pie

default: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o PhilipNelson && ./PhilipNelson 

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release

all: $(OBJS)
	make debug release
