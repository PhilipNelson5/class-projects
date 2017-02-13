OBJS = main.cpp cell.hpp display.hpp display.cpp rules.hpp rules.cpp world.hpp world.cpp

release: $(OBJS)
	g++ -std=c++14 -Wall -Wextra -O3 -g0 -o release $(OBJS)


debug: $(OBJS)
	g++ -std=c++14 -Wall -Wextra -Werror -O0 -g3 -o debug $(OBJS)
