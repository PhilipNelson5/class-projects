#include "linkedlist.hpp"
#include <iostream>

template <typename T>
Linkedlist<T> copy(Linkedlist<T> &source)
{
  return Linkedlist<T>(source);
}

int main()
{
  Linkedlist<int> a;

  //-----[insert elements]-----//
  a.insert(5);
  a.insert(3);
  a.insert(1);
  a.insert(5);
  a.insert(2);
  a.insert(6);
  std::cout << "a insert 5, 3, 1, 5, 2" << std::endl;
  std::cout << "a: " << a << std::endl << std::endl;

  //-----[remove elements]-----//
  a.remove(5);
  std::cout << "remove 5" << std::endl;
  std::cout << "a: " << a << std::endl << std::endl;

  //-----[size function]-----/
  std::cout << "a.size()" << std::endl;
  std::cout << "size a: " << a.size() << std::endl << std::endl;

  //-----[copy construction]-----//
  auto b = a; // auto b(a);
  a.remove(2);
  std::cout << "b copy constructed a, remove 2 from a" << std::endl;
  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl << std::endl;

  //-----[copy assignment]-----//
  Linkedlist<int> c;
  c = a;
  a.remove(3);
  std::cout << "c = a, remove 3 from a" << std::endl;
  std::cout << "a: " << a << std::endl;
  std::cout << "c: " << c << std::endl << std::endl;

  //-----[move construction]-----//
  Linkedlist<int> d = copy(a);
  a.insert(10);
  a.insert(1);

  std::cout << "move construction d constructed with copy(a)" << std::endl;
  std::cout << "a insert 10, 1" << std::endl;
  std::cout << "a: " << a << std::endl;
  std::cout << "d: " << d << std::endl << std::endl;

  //-----[move assignment]-----//
  Linkedlist<int> e;
  e = copy(a);
  a.insert(21);
  a.insert(4);

  std::cout << "move assignment e = copy(a)" << std::endl;
  std::cout << "a insert 21, 4" << std::endl;
  std::cout << "a: " << a << std::endl;
  std::cout << "e: " << e << std::endl << std::endl;

  //-----[for each]-----//
  a.forEach([](int &e) { e += 1; });
  std::cout << "add one to each element" << std::endl;
  std::cout << "a: " << a << std::endl << std::endl;

  //-----[index operator]-----//
  std::cout << "0th element of a" << std::endl;
  std::cout << "a[0] = " << a[0] << std::endl << std::endl;

  return EXIT_SUCCESS;
}
