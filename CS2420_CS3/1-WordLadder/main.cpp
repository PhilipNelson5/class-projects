#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include "wordLadder.hpp"

namespace
{
	const char QUIT = '0';
	const char RAND = '1';
}

void instructions()
{
			std::cout << "--------------------------------------------------------------------------------------------------------------" << std::endl;
			std::cout << "Choose two words of the same length to discover the word ladder that connects them!!" << std::endl;
			std::cout << "Enter '" << RAND << "' to have two random words chosen or after your first word to have the second word be choosen for you." << std::endl;
			std::cout << "Enter '" << QUIT << "' to exit." << std::endl;
}

int main(int argc, char* argv[])
{
	std::string filePath = "words.txt";
	if (argc > 1) filePath = argv[1];

	std::string beg, end;
	WordLadder game(filePath);
	auto run = true;
	while (run)
	{
		instructions();
		try
		{
			std::cin >> beg;
			std::transform(beg.begin(), beg.end(), beg.begin(), tolower);
			switch (beg[0])
			{
			case QUIT:
				run = false;
				break;
			case RAND:
				game.findRandom();
				break;
			default:
				std::cin >> end;
				std::transform(end.begin(), end.end(), end.begin(), tolower);
				switch (end[0])
				{
				case RAND:
					game.findRandom(beg);
					break;
				default:
					game.findLadder(beg, end);
					break;
				}
				break;
			}
		}
		catch (std::exception & e)
		{
			std::cerr << e.what() << std::endl << std::endl;
		}
	}
	return EXIT_SUCCESS;
}
