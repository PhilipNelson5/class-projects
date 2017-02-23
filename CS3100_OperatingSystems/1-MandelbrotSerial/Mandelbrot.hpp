#ifndef MANDELBROT_HPP
#define MANDELBROT_HPP

#include <vector>
#include <fstream>
#include <string>

#include "timer.hpp"

class Mandelbrot
{
	public:
		Mandelbrot(std::string file)
		//{ import(file); image.reserve(IMAGE_HIGHT*IMAGE_WIDTH); }
		{ import(file); }

		void render(int);
		void print();
		bool import(std::string);
		void header();
		int getColor(int, int);

	private:
		int IMAGE_HIGHT, IMAGE_WIDTH, MAX_ITERS, COLOR_SCHEME;
		const int PROGRESS_SCALE = 10;
		const int COLOR_DEPTH = 255;
		double X_MIN, X_MAX;
		double Y_MAX, Y_MIN;
		std::vector<double> image;
		std::ofstream save;
		Timer timer;

		//void color_scheme(int, int, int);
		void color_scheme(int);
		void ppm(int, int, int);
		void progress_bar(int, int, int);

		void gray_scale(int);
};

#endif
