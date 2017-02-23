#include "image.hpp"
#include <fstream>

void Image::writeImage(std::string file)
{
	std::ofstream fout(file);

	fout << "P3" << std::endl;
	fout << IMAGE_WIDTH << " " << IMAGE_HIGHT << std::endl;
	fout << COLOR_DEPTH << std::endl;

	int c = 0;
	for(auto && e:image)
	{ 
		fout << e << " " << e << " "  << e << "   ";
		if(c%IMAGE_WIDTH == 0)
			fout << std::endl;
		++c;
	}
	//fout << color(e) << " " << color(e) << " "  << color(e) << " ";
	fout.close();
}

void Image::set(int h, int w, int i, int d, int c)
{
	IMAGE_HIGHT = h;
	IMAGE_WIDTH = w;
	MAX_ITERS = i;
	COLOR_DEPTH = d;
	COLOR_SCHEME = c;
	image.resize(IMAGE_HIGHT*IMAGE_WIDTH);
}
