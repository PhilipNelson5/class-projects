#include "mandelbrot.hpp"
#include "timer.hpp"
#include <iostream>
#include <fstream>
#include <thread>

void runt(int div)
{
	int sec = hight()/div;
	std::vector<std::thread> threads;

	for(int i = 0; i < div; ++i)
	{
		if(i == div-1)
			threads.emplace_back([=](){render((i*width()*sec), imagesize()-1);});
		else
			threads.emplace_back([=](){render((i*width()*sec), ((i+1)*width()*sec)-1);});
	}

	for(auto && e:threads)
		e.join();
}

int main(int argc, char* argv[])
{
	//check if settings.txt exists
	if(!import("settings.txt"))
		return EXIT_FAILURE;
	std::ofstream fout("data.csv");

	Timer t;
	t.reset();
	auto threads = 1;
	auto tests = 10;
	auto n = std::thread::hardware_concurrency();

	//specify number of tests per thread
	if(argc > 1)
	{
		tests = atoi(argv[1]);
	}
	std::cout << n << " concurrent threads are supported.\n";
	std::cout << tests << " tests per thread\n";

	fout << "Threads,Average,Standard Dev" << std::endl;

	//tests
	for(int i = 1; i < 10; ++i)
	{
		threads = i;
		std::cout << "Threads: " << threads << std::endl;
		fout << threads << ",";

		runt(threads);

		for(int i = 0; i < tests; ++i)
		{
			t.time([=](){runt(threads);});
		}

		auto stdev = t.getStdDev();
		auto ave = t.getAverage();

		std::cout << "           Average: " << t.getAverage() << std::endl;
		fout << ave << "," << std::endl;
		std::cout << "Standard Deviation: " << t.getStdDev() << std::endl;
		fout << stdev << ",";

		t.reset();
	}

	//write out image to file
	write("SAVE.ppm");
}
