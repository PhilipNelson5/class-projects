OBJS = TestTree.cpp wordTree.hpp wordTree.cpp Node.hpp 

release: $(OBJS)
	g++ -std=c++14 -O3 -Wall -Wextra -Werror -o release $(OBJS)

debug: $(OBJS)
	g++ -std=c++14 -fsanitize=address -O0 -g3 -Wall -Wextra -Werror -o debug $(OBJS)
