#ifndef SORT_HPP
#define SORT_HPP

#include <vector>
#include <iostream>

std::ostream& operator<< (std::ostream& o, std::vector<int> const & v);
std::ostream& operator<< (std::ostream& o, std::vector<double> const & v);

void bubbleSort (std::vector<int>& list);
void insertionSort (std::vector<int>& list);
void selectionSort (std::vector<int>& list);
void mergeSort(std::vector<int>& list);
void quickSort (std::vector<int>& list);

#endif
