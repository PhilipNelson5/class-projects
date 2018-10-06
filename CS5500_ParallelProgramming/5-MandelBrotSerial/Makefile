OBJS = main.cpp calculator.cpp color.cpp

CC = mpic++
FLAGS = -std=c++17
DEBUG_FLAGS = -O0 -g3
RELEASE_FLAGS = -g0 -O3
WARNINGS = -Wall -Wextra -Werror

release: $(OBJS)
	$(CC) $(FLAGS) $(RELEASE_FLAGS) $(WARNINGS) $(OBJS) -o mandelbrot.out

debug: $(OBJS)
	$(CC) $(FLAGS) $(DEBUG_FLAGS) $(WARNINGS) $(OBJS) -o debug.out
