#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <typeinfo>
#include <string>
#include <vector>
#include "sort.hpp"

namespace{
	int algF = 4; //first algorighm to test
	int algL = 6;  //last algorithm to test
	int min = 1000000; //initial list size to tests
	int max = 10000000; //max list size to test
	int sample = 7 + 1;  //number of times to run test per algorithm
	int inc = 1000000;  //increment of list size
	std::ofstream fout("data.csv");  //where to write test data
}

void init(std::vector<int>& list, int listSize){
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(1, 2 * listSize);

	for (int i = 0; i < listSize; ++i)
		list.push_back(dist(mt));
	//list.push_back(rand()%100);
}

void isSorted(std::vector<int>& list){
	bool sorted = true;
	for (auto i = 0u; i < list.size() - 1; ++i){
		if (list[i]>list[i + 1]){
			sorted = false;
		}
	}
	if (sorted)
		std::cout << "Sorted: YES!" << std::endl;
	else std::cout << "Sorted: NO!" << std::endl;
}

void ave(std::vector<double>& v){
	//std::cout << v << std::endl;
	for (auto&&e : v)
		e /= (sample - 1);
}

std::string algName(int n){
	switch (n){
	case 1: return "Bubble Sort";
	case 2: return "Insertion Sort";
	case 3: return "Selection Sort";
	case 4: return "Merg Sort";
	case 5: return "Quick Sort";
	case 6: return "std::sort";
	}
	return "not an algorithm";
}

void test(std::ostream& o){

	auto start = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	bool first = true;

	std::cout << "begin" << std::endl;
	for (int j = algF; j <= algL; ++j){//loops through all sorting algorithms
		std::vector<double> average(max / inc, 0);
		
		std::cout << algName(j) << ": start" << std::endl;
		if (typeid(o) != typeid(std::cout))
			o << algName(j) << std::endl;

		first = true;
		for (int k = 0; k < sample; ++k){//runs each test 'sample' times
			int x = 0;
			for (int i = min; i <= max; i += inc){//tests each algorithm on data sets between 'mix' and 'max'
				std::vector<int> list;
				list.reserve(i);
				init(list, i);
				//std::cout << "unsorted: " << list << std::endl << std::endl;
				switch (j) {
				case 1:
					start = std::chrono::high_resolution_clock::now();
					bubbleSort(list);
					end = std::chrono::high_resolution_clock::now();
					break;
				case 2:
					start = std::chrono::high_resolution_clock::now();
					insertionSort(list);
					end = std::chrono::high_resolution_clock::now();
					break;
				case 3:
					start = std::chrono::high_resolution_clock::now();
					selectionSort(list);
					end = std::chrono::high_resolution_clock::now();
					break;
				case 4:
					start = std::chrono::high_resolution_clock::now();
					mergeSort(list);
					end = std::chrono::high_resolution_clock::now();
					break;
				case 5:
					start = std::chrono::high_resolution_clock::now();
					quickSort(list);
					end = std::chrono::high_resolution_clock::now();
					break;
				case 6:
					start = std::chrono::high_resolution_clock::now();
					std::sort(list.begin(), list.end());
					end = std::chrono::high_resolution_clock::now();
					break;
				}
				if (!first)//ignore the first test
					average[x] += (std::chrono::duration <double, std::milli>(end - start).count());
				++x;
				//std::cout << std::endl << "sorted: " << list <<std::endl;
				//isSorted(list);
				if (i%inc == 0)
				std::cout << static_cast<float>(i) / max * 100 << "%, ";	
			}
			std::cout << "run: " << k << " finished" << std::endl;
			first = false;
		}

		std::cout << algName(j) << ": done" << std::endl;
		std::cout << "-------------------------" << std::endl;
		ave(average);
		int x = 0;
		for (int i = min; i <= max; i += inc){
			std::cout << "size " << i << " list -> " << average[x] << " ms" << std::endl;
			++x;
		}
		if (typeid(o) != typeid(std::cout)){
			x = 0;
			for (int i = min; i <= max; i += inc){
				o << i << "," << average[x] << std::endl;
				++x;
			}
		}
		std::cout << std::endl << std::endl;
	}
	std::cout << "done" << std::endl << std::endl;
}


int main(){
	test(fout);
}
