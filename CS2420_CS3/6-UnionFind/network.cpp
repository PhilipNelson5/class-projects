#include "network.hpp"
#include <iostream>
#include <random>

//returns a random int on the range low to high
int rand(int low, int high) {
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(low, high);
	return dist(mt);
}

void Network::simulate(int population)
{
	init();

	std::cout << "-----------------------------------------" << std::endl
		<< "Simulaion size: " << population << std::endl;
	network.init(population);
	size = population;

	while (!network.isComplete())
	{
		makeFriend();
		++days;
	}

	report();
}

void Network::makeFriend()
{
	int p1 = rand(0, size - 1);
	int p2 = rand(0, size - 1);

	while (p1 == p2)
		p2 = rand(0, size - 1);

	network._union(p1, p2);
}

void Network::init()
{
	days = 0;
}

void Network::report()
{
	if (size < 10000)
	{
		std::cout << "Simulation Complete" << std::endl << std::endl
			<< "-------------------------------------------" << std::endl
			<< "  population\tdays\tfinds\titems\t" << std::endl
			<< "  " << size << "\t\t" << days << "\t" << network.getFinds() << "\t"
			<< network.getTouches() << "\t" << std::endl
			<< "-------------------------------------------" << std::endl << std::endl;
	}

	else
	{
		if (network.getFinds() < 10000000)
		{
			std::cout << "Simulation Complete" << std::endl << std::endl
				<< "-------------------------------------------" << std::endl
				<< "  population\tdays\tfinds\titems\t" << std::endl
				<< "  " << size << "\t" << days << "\t" << network.getFinds() << "\t"
				<< network.getTouches() << "\t" << std::endl
				<< "-------------------------------------------" << std::endl << std::endl;
		}
		else
		{
			std::cout << "Simulation Complete" << std::endl << std::endl
				<< "-------------------------------------------" << std::endl
				<< "  population\tdays\tfinds\t items\t" << std::endl
				<< "  " << size << "\t" << days << "\t" << network.getFinds() << " "
				<< network.getTouches() << std::endl
				<< "-------------------------------------------" << std::endl << std::endl;
		}

	}
}
