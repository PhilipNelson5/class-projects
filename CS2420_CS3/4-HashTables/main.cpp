#include "game.hpp"

int main()
{
	//list of files to open
	std::vector<std::string> games = {"gameTest.txt", "game0.txt", "game1.txt", "game2.txt", "game3.txt", "game4.txt"};
	Game test(games);
	test.play();
}
