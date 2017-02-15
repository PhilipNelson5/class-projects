#include <algorithm>
#include <limits>
#include <iostream>
#include "sort.hpp"
// merge and quick recursively

std::ostream& operator<< (std::ostream& o, std::vector<double> const & v){
	for(auto & e:v)
		o << e << ", ";
	return o;
}
  
std::ostream& operator<< (std::ostream& o, std::vector<int> const & v){
   for(auto & e:v)
		o << e << ", ";
   return o;
}

//----------------------------------------------------//
//--------------------[Bubble Sort]-------------------//
//----------------------------------------------------//
void bubbleSort	(std::vector<int>& list){
	bool done = false;
	while (!done){
		done = true;
		for(auto i = 0u; i < list.size()-1; ++i){
			if(list[i] > list[i+1]){
				std::swap(list[i], list[i+1]);
				done = false;
			}
		}
	}
}

//----------------------------------------------------//
//-------------------[Insertion Sort]-----------------//
//----------------------------------------------------//
void insertionSort (std::vector<int>& list){
	for(auto i = 1u; i < list.size(); ++i)
		for(auto j = 0u; j < i; ++j)
			if(list[j] > list[i])
				std::swap(list[j], list[i]);
}
//----------------------------------------------------//
//------------------[Selection Sort]------------------//
//----------------------------------------------------//
int findMin (std::vector<int>& list, int start){
	auto min = std::numeric_limits<int>::max();
	auto cur = -1;

	for (unsigned int i = start; i < list.size(); ++i)
		if (min > list[i]){
			 min = list[i];
			 cur = i;
		}
	return cur;
}

void selectionSort (std::vector<int>& list){
	for (auto i = 0u; i < list.size(); ++i){
		int small = findMin(list, i);
		std::swap(list[i], list[small]);
   }
}

//----------------------------------------------------//
//--------------------[Merge Sort]--------------------//
//----------------------------------------------------//
//merge 2 sorted lists
void mergeList(std::vector<int>& v, int start, int mid, int end){
	std::vector<int> merged;
	int i = start;
	int j = mid;
	//for(int z = start; z < end; z++)
		while(i < mid && j < end){
			if (v[i] < v[j]){
				merged.push_back(v[i]);
				++i;
			}
			else{
				merged.push_back(v[j]);
				++j;
			}
		}		
	while(i<mid){
		merged.push_back(v[i]);
		++i;
	}
	while(j<end){
		merged.push_back(v[j]);
		++j;
	}
	for(unsigned int x = start, y = 0; y < merged.size(); ++x, ++y ){
		v[x] = merged[y];
	}
}

//recursive merge sort
void mergeSort(std::vector<int>& v, int start, int end){
	if (end-start<=1) return;
	
	auto mid = (start + end) / 2;
	mergeSort(v, start, mid);
	mergeSort(v, mid, end);
	mergeList(v, start, mid, end);
}

//call to recursive merge sort
void mergeSort (std::vector<int>& list){
	mergeSort(list, 0, list.size());
}

//----------------------------------------------------//
//--------------------[Quick Sort]--------------------//
//----------------------------------------------------//
//choose the median of 3 for the pivot
int med(std::vector<int>& v, int start, int end){
	//std::cout << "s: " << start << ", e: " << end << std::endl;

	auto first = v[start];
	auto mid = v[end / 2];
	auto last = v[end - 1];

	if (first<mid && mid>last) return end/2;
	if (mid<last && last>first) return end;
	return start;
}

//divide the list and sort about the pivot
int partition(std::vector<int>& v, int pivot, int start, int end){
	std::swap(v[pivot], v[end-1]);
	int check = start;
	for (int i = start; i < end-1; ++i){
		if (v[i] <= v[end-1]){
			std::swap(v[i], v[check]);
			++check; 
		}
	}
	std::swap(v[end-1], v[check]);
	return check;
}

//recursive quick sort
void quick(std::vector<int>& v, int start, int end){
	if (end - start <= 1) return;
	auto pivot = (start+end)/2;//med(v, start, end) ;
	pivot = partition(v, pivot, start, end);
	quick(v, start, pivot);
	quick(v, pivot, end);
}

//call to recursive quick sort
void quickSort(std::vector<int>& list){
	//std::cout << "quick sort started on list size " << list.size() <<std::endl;
	quick(list, 0, list.size());
}
