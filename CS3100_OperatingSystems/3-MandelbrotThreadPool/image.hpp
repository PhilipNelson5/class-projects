#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#include <utility>
#include <vector>

class Image
{
	public:
		Image() : IMAGE_HIGHT(0), IMAGE_WIDTH(0),
							MAX_ITERS(0), COLOR_DEPTH(0), COLOR_SCHEME(0) {}

		Image(int h, int w, int i, int d, int c) : IMAGE_HIGHT(h), IMAGE_WIDTH(w),
																				MAX_ITERS(i), COLOR_DEPTH(d), COLOR_SCHEME(c)
	{image.reserve(IMAGE_HIGHT*IMAGE_WIDTH);}

		void set(int h, int w, int i, int d, int c);

		std::vector<int> image;
		void writeImage(std::string);
		int IMAGE_HIGHT, IMAGE_WIDTH, MAX_ITERS, COLOR_DEPTH, COLOR_SCHEME;
};

#endif
