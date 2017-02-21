#include <iostream>
#include <sstream>
#include <string>

#include "dynamicMedian.hpp"

//insert into the dynamic median system
void DynamicMedian::insert(int key)
{
	//first insert
	if (maxHeap.isEmpty() && minHeap.isEmpty() && currMedian == 0)
	{
		currMedian = key;
		return;
	}

	//which heap?
	if(key < currMedian)
		maxHeap.insert(key);
	else //(key >= currMedian)
		minHeap.insert(key);

	//rearrange heaps?
	if(maxHeap.size() > minHeap.size()+1)
	{
		minHeap.insert(currMedian);
		currMedian = maxHeap.pop();
	}
	if(maxHeap.size() < minHeap.size()-1)
	{
		maxHeap.insert(currMedian);
		currMedian = minHeap.pop();
	}
}

//returns a string with:
//the small number heap
//the currMedian
//the large number heap
std::string DynamicMedian::toString() const
{
	std::ostringstream oss;
	oss << "----------------------" << std::endl
		<< "Max Heap: " << maxHeap.size() << std::endl
		<< maxHeap.toString() << std::endl
		<< "[" << currMedian << "]" << std::endl << std::endl
		<< "Min Heap: " << minHeap.size() << std::endl
		<< minHeap.toString() << std::endl
		<< "----------------------" << std::endl;
	return oss.str();
}

//returns a report of the internal status of the system
std::string DynamicMedian::report() const
{
	std::ostringstream oss;
	oss 
		<< "\t\tsize\ttop element\tmerges" << std::endl
		<< "MaxHeap:\t" << maxHeap.size() << "\t" << maxHeap.getTop() << "\t\t"	<< maxHeap.getMergeCount() << std::endl
		<< "Median [" << currMedian << "]" << std::endl
		<< "MinHeap:\t" << minHeap.size() << "\t" << minHeap.getTop() << "\t\t" << minHeap.getMergeCount() << std::endl
		<< "--------------------------------------------------------" << std::endl;
	return oss.str();
}

void DynamicMedian::clear()
{
	minHeap.clear();
	maxHeap.clear();
	currMedian = 0;
	std::cout << "Dynamic Median Cleared" << std::endl;
}
