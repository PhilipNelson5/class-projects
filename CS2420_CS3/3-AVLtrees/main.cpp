#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include "wordLadderTree.hpp"
#include "wordLadderQueue.hpp"

namespace
{
	const char QUIT = '0';
	const char RAND = '1';
}

//demonstration of the avl tree code
void avlTest()
{
	AvlTree<int> t;
	std::vector<int> nums1 {1, 3, 5, 7, 9, 9, 9, 11, 2, 9, 4, 8};
	std::vector<int> nums2 {30, 50, 30, 30, 15, 18};

	for(auto && e : nums1)
		t.insert(e);

	std::cout << t.toString("Tree 1") << std::endl;

	std::cout << "Remove: 7" << std::endl; t.remove(7);
	std::cout << "Remove: 8" << std::endl << std::endl; t.remove(9);

	std::cout << t.toString("Tree 1") << std::endl;

	for(auto && e : nums2)
		t.insert(e);

	std::cout << t.toString("Tree 1") << std::endl;

	std::cout << "Removed: " << t.removeMin() << std::endl;
	std::cout << "Removed: " << t.removeMin() << std::endl;
	std::cout << "Removed: " << t.removeMin() << std::endl << std::endl;

	t.insert(17);
	
	std::cout << t.toString("Tree 1") << std::endl;
}

//The instructions for starting the word ladder finder
void instructions()
{
	std::cout << "--------------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << "Choose two words of the same length to discover the word ladder that connects them!!" << std::endl;
	//std::cout << "Enter '" << RAND << "' to have two random words chosen or after your first word to have the second word be choosen for you." << std::endl;
	std::cout << "Enter '" << QUIT << "' to exit." << std::endl;
}

int main(int argc, char* argv[])
{
	avlTest();

	std::string filePath = "words.txt";
	if (argc > 1) filePath = argv[1];

	std::string beg, end;
	WordLadderTree astar(filePath);
	WordLadderQueue brute(filePath);
	auto run = true;
	while (run)
	{
		instructions();
		try
		{
			std::cin >> beg;
			switch (beg[0])
			{
				case QUIT:
					run = false;
					break;
				case RAND:
					astar.findRandom();
					break;
				default:
					std::cin >> end;
					if(end[0] == RAND)
						astar.findRandom(beg);
					else 
					{
						astar.findLadder(beg, end);
						brute.findLadder(beg, end);
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
