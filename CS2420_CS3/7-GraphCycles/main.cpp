#include <iostream>
#include "graph.hpp"

void menu()
{
	std::cout << "----[MENU]----" << std::endl
		<< "[0] prog7A.txt" << std::endl
		<< "[1] prog7B.txt" << std::endl
		<< "[2] prog7C.txt" << std::endl
		<< "[3] manual entry" << std::endl
		<< "[q] to quit" << std::endl;
}

void print(Graph & g)
{
	std::cout << g.toString() << std::endl;
	std::cout << (g.isConnected() ? "Connected" : "Unconnected") << std::endl;
}

int main()
{
	int num;
	std::string file;
	menu();
	while(std::cin >> num)
	{
		if(num == 0)
		{
			Graph g("prog7A.txt");
			print(g);
		}   
		else if(num == 1)
		{ 
			Graph g("prog7B.txt");
			print(g);
		}   
		else if(num == 2)
		{ 
			Graph g("prog7C.txt");
			print(g);
		} 
		else if(num == 3)
		{
			std::cin >> file;
			Graph g(file); 
			print(g);
		}
		menu();
	}
}
