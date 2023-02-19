#pragma once
#include <iostream>
#include <memory>
using std::cout, std::endl;

template<typename T>
struct Node
{
  Node(T data, std::shared_ptr<Node<T>> next = nullptr): data(data), next(next) {};
  T data;
  std::shared_ptr<Node> next;
};

template<typename T>
std::ostream& operator<<(std::ostream& o, std::shared_ptr<Node<T>> node)
{
  o << "<" << node->data << ">";
  return o;
}

