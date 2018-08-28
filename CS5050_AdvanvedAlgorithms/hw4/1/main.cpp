#include <iostream>
#include <iomanip>
#include "minHeap.hpp"

int main()
{
  minHeap<int> h;
  h.insert(4);
  h.insert(1);
  h.insert(13);
  h.insert(9);
  h.insert(3);
  h.insert(8);
  h.insert(2);
  h.insert(11);
  h.insert(6);
  h.insert(7);
  h.insert(5);
  h.insert(12);
  h.insert(10);


  std::cout << h.toString() << "\n";

  while(true)
  {
    auto k = 0;
    auto x = 0;
    std::cin >> k;
    std::cin >> x;
    std::cout << std::boolalpha << h.kthLessThanX(k, x) << "\n";
  }

  std::cout << std::endl;
}
