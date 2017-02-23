OBJS = main.cpp Mandelbrot.hpp Mandelbrot.cpp Calculator.hpp Calculator.cpp

CC = g++
DEBUG_FLAGS = -Og -g3 -fsanitize=address -Wall -Wextra -Werror
RELEASE_FLAGS = -O3
FLAGS = -std=c++14

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o release
