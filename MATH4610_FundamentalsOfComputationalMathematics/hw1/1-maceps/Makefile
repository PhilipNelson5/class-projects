OBJS = main.cpp maceps.hpp

CC = g++
FLAGS = -std=c++17
DEBUG_FLAGS = -Og -g3 -Wall -Wextra -Werror
RELEASE_FLAGS = -O3

release: $(OBJS)
	$(CC) $(RELEASE_FLAGS) $(FLAGS) $(OBJS) -o maceps.out

debug: $(OBJS)
	$(CC) $(DEBUG_FLAGS) $(FLAGS) $(OBJS) -o debug.out

clean:
	rm *.gch *.out *.o &> /dev/null
