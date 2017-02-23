#include <fstream>
#include <iostream>
#include "Mandelbrot.hpp"
#include "Calculator.hpp"

int Mandelbrot::getColor(int i, int j)
{
	auto iters = mandelbrot(i, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX);
	return interpalate(iters, MAX_ITERS, 0, COLOR_DEPTH);
}

void Mandelbrot::render(int sample_size)
{
	std::cout << IMAGE_HIGHT << "x" << IMAGE_WIDTH << std::endl;
	for(int k = 0; k < sample_size; ++k)
	{
		for (int i = 0; i < IMAGE_HIGHT; ++i)
		{
			for (int j = 0; j < IMAGE_WIDTH; ++j)
			{
				timer.start();
				mandelbrot(i, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX);
				timer.end();
			}
		}
		std::cout << timer.getTime() << " milliseconds" << std::endl;
		timer.save();
	}
	std::cout << "Standard Deviation: " << timer.getStdDev() << std::endl;
	std::cout << "           Average: " << timer.getAverage() << std::endl;
	timer.reset();
}

void Mandelbrot::print()
{
	header();
	timer.start();
	for (int i = 0; i < IMAGE_HIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			auto iters = mandelbrot(i, j, IMAGE_WIDTH, IMAGE_HIGHT, MAX_ITERS,
					X_MIN, X_MAX, Y_MIN, Y_MAX);
			//color_scheme(iters, i, j);
			color_scheme(iters);
		}
		//progress_bar(i, IMAGE_HIGHT, PROGRESS_SCALE);
	}
	timer.end();
std::cout << std::endl << "Time to generate and write file: " << timer.getTime() << " milliseconds" << std::endl << std::endl;
}

bool Mandelbrot::import(std::string file) {
	std::ifstream fin(file);
	if(!fin)
	{
		std::cout << "Bad File..." << std::endl;
		return false;
	}
	std::string ignore;

	fin >> ignore >> IMAGE_HIGHT;
	fin >> ignore >> IMAGE_WIDTH;
	fin >> ignore >> MAX_ITERS;

	fin >> ignore >> X_MIN;
	fin >> ignore >> X_MAX;
	fin >> ignore >> Y_MAX;

	fin >> ignore >> COLOR_SCHEME;

	Y_MIN = Y_MAX - (X_MAX - X_MIN) * (static_cast<double>(IMAGE_HIGHT) / IMAGE_WIDTH);
	fin.close();
	return true;
}

void Mandelbrot::progress_bar(int i, int IMAGE_HIGHT, int PROGRESS_SCALE) {
	for (int k = 0; k <= PROGRESS_SCALE; ++k) {
		if (i == IMAGE_HIGHT * k / PROGRESS_SCALE) {
			std::cout << (k * 100 / PROGRESS_SCALE) 
				<< "% (" << i << " lines) complete." << std::endl;
			return;
		}
		if (i == IMAGE_HIGHT - 1 * k / PROGRESS_SCALE) {
			std::cout << (k * 100 / PROGRESS_SCALE) 
				<< "% (" << i + 1 << " lines) complete." << std::endl;
			return;
		}
	}
}

//void Mandelbrot::color_scheme(int iters, int i, int j)
void Mandelbrot::color_scheme(int iters)
{
	switch (COLOR_SCHEME) {
		case 1:
			gray_scale(iters);
			break;
	}
}

/**********************************************************************************************************/
/*                                            [Color Scheme 1]                                            */
/**********************************************************************************************************/
void Mandelbrot::gray_scale(int iters)
{
	if (iters == MAX_ITERS)
		iters = 0;
		//iters = interpalate(iters, MAX_ITERS, 0, COLOR_DEPTH);
	ppm(iters, iters, iters);
}

void Mandelbrot::header()
{
	save.open("SAVE.txt");
	save << "P3" << std::endl;
	save << IMAGE_WIDTH << " " << IMAGE_HIGHT << std::endl;
	save << COLOR_DEPTH << std::endl;
}

void Mandelbrot::ppm(int c1, int c2, int c3)
{
	save << c1 << " " << c2 << " " << c3 << " ";
}
