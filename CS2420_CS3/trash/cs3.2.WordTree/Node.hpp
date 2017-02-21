#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <string>

struct Node
{
  Node(std::string w = "",
       int numC = 0,
       int subT = 0,
       std::shared_ptr<Node> p = nullptr,
       std::shared_ptr<Node> s = nullptr,
       std::shared_ptr<Node> c = nullptr)
    : word(w), subTree(subT), numChildren(numC), siblings(s), children(c), parent(p)
  {
  }

  std::string word;
  int subTree;
  int numChildren;

  std::shared_ptr<Node> siblings;
  std::shared_ptr<Node> children;
  std::shared_ptr<Node> parent;
};

#endif
