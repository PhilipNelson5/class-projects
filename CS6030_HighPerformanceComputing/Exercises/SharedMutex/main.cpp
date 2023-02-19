#include <iostream>
#include "ThreadSafeOrderedLinkedList.hpp"
#include "OrderedLinkedList.hpp"
using std::cout, std::endl;


int main()
{
  std::function<bool(int, int)> comp = std::greater<int>();
  // OrderedLinkedList<int> ll(comp);
  OrderedLinkedList<int> ll;
  ll.insert(7);
  ll.insert(5);
  ll.insert(4);
  ll.insert(5);
  ll.insert(5);
  ll.insert(6);
  ll.remove(5);
  ll.remove(5);
  cout << ll << endl;
  cout << ll.member(4) << endl;
  cout << ll.member(5) << endl;
  ll.remove(5);
  cout << ll.member(5) << endl;
  cout << ll << endl;
  return EXIT_SUCCESS;
}