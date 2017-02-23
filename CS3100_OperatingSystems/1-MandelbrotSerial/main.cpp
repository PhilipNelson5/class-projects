#include <cstdlib>
#include <iostream>
#include "Mandelbrot.hpp"
#include "Calculator.hpp"

int main(int argc, char* argv[])
{
	Mandelbrot m("settings.txt");
	if(argc > 1)
	{
		m.render(atoi(argv[1]));
		m.print();
	}
	else
	{
		m.render(11);
		m.print();
		auto grey = m.getColor(100, 100);
		std::cout << "color at pixel (100, 100) is: " 
			<< grey << " " << grey << " " << grey << std::endl;
	}

	return EXIT_SUCCESS;
}
