OBJS = main.cpp tree.hpp tree.cpp

release: $(OBJS)
	g++ -std=c++14 -Wall -Wextra -Werror -O3 -o release $(OBJS)

debug: $(OBJS)
	g++ -std=c++14 -Wall -Wextra -Werror -fsanitize=address -O0 -g3 -o debug $(OBJS)
