#include "network.hpp"
#include <iostream>
#include <cstdlib>

void instructions()
{
	std::cout << "Enter a population to simulate\n"
		<< "any other char to quit\n"
		<< "population: ";
}

int main()
{
	Network net;
	int size;

	instructions();
	while (std::cin >> size)
	{
		net.simulate(size);
		instructions();
	}

	std::cin.get();
	return EXIT_SUCCESS;
}
