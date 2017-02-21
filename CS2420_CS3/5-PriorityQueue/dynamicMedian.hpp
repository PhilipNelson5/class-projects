#ifndef DYNAMIC_MEDIAN_HPP
#define DYNAMIC_MEDIAN_HPP

#include "skewHeap.hpp"

class DynamicMedian
{
	public:
		DynamicMedian() : currMedian(0), maxHeap([](int a, int b){return a > b ? true : false;}) {}
		void insert(int);
		int getMedian() const { return currMedian; }
		std::string toString() const;
		std::string report() const;
		void clear();

	private:
		int currMedian;
		SkewHeap minHeap;
		SkewHeap maxHeap;
};

#endif
