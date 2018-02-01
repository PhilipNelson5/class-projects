OBJS = main.cpp
release: $(OBJS)
	g++ -std=c++14 -Wall -Wextra -Werror -O3 -g0 -o release $(OBJS)

debug: $(OBJS)
	g++ -std=++14 -Wall -Wextra -Werror -O0 -g3 -o debug $(OBJS)
