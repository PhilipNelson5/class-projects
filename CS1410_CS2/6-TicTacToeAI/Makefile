OBJS = main.cpp Board.h Board.cpp DefensiveRandomPlayer.h DefensiveRandomPlayer.cpp HumanPlayer.h HumanPlayer.cpp PerfectPlayer.h PerfectPlayer.cpp Player.h Player.cpp RandomPlayer.h RandomPlayer.cpp

defualt: $(OBJS)
	g++ -std=c++14 -O3 -o ticTacToe $(OBJS)

release: $(OBJS)
	g++ -std=c++14 -O3 -o release $(OBJS)

debug: $(OBJS)
	g++ -std=c++14 -fsanitize=address -O0 -g3 -o debug $(OBJS)
