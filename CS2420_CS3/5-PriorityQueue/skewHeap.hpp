#ifndef SKEW_HEAP_HPP
#define SKEW_HEAP_HPP

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

struct node
{
  node(int d, std::shared_ptr<node> l = nullptr, std::shared_ptr<node> r = nullptr)
    : data(d), left(l), right(r)
  {
  }
  int data;
  std::shared_ptr<node> left;
  std::shared_ptr<node> right;
};

class SkewHeap
{
public:
  SkewHeap(std::function<bool(int, int)> f = [](int a, int b) { return a < b ? true : false; })
    : heapSize(0), mergeCount(0), compare(f){};
  void insert(int data);
  int pop();
  int getTop() const { return !root ? -80000000 : root->data; }
  int getMergeCount() const { return mergeCount; }
  int size() const { return heapSize; }
  bool isEmpty() const { return heapSize == 0; }
  void clear();

  std::string toString() const
  {
    std::ostringstream oss;
    if (isEmpty())
    {
      oss << "[EMPTY]" << std::endl;
      return oss.str();
    }
    toString(root, oss, "");
    return oss.str();
  }

private:
  int heapSize;
  int mergeCount;
  std::function<bool(int, int)> compare;
  std::shared_ptr<node> root;
  std::shared_ptr<node> merge(std::shared_ptr<node> root1, std::shared_ptr<node> root2);
  void toString(std::shared_ptr<node> curr, std::ostringstream &oss, std::string tab) const;
};

#endif
