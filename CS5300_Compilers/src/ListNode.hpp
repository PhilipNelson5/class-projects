#ifndef LIST_NODE_HPP
#define LIST_NODE_HPP

#include <algorithm> // for reverse
#include <memory>    // for shared_ptr
#include <vector>    // for vector

template<typename T>
class ListNode
{
public:
  ListNode(T* data)
    : data(std::shared_ptr<T>(data))
    , next(nullptr)
  {}

  ListNode(T* data, ListNode<T>* next)
    : data(std::shared_ptr<T>(data))
    , next(std::shared_ptr<ListNode<T>>(next))
  {}

  static std::vector<std::shared_ptr<T>> makeVector(ListNode<T>*& list)
  {
    std::vector<std::shared_ptr<T>> vec;
    for (auto cur = std::shared_ptr<ListNode<T>>(list); cur != nullptr;
         cur = cur->next)
    {
      if (cur->data != nullptr) vec.push_back(cur->data);
    }
    std::reverse(std::begin(vec), std::end(vec));
    return vec;
  }

  static std::vector<T> makeDerefVector(ListNode<T>*& list)
  {
    std::vector<T> vec;
    for (auto cur = std::shared_ptr<ListNode<T>>(list); cur != nullptr;
         cur = cur->next)
    {
      if (cur->data != nullptr) vec.push_back(*(cur->data));
    }
    std::reverse(std::begin(vec), std::end(vec));
    return vec;
  }

  const std::shared_ptr<T> data;

  const std::shared_ptr<ListNode<T>> next;
};

#endif
