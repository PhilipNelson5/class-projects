#pragma once
#include <iostream>
#include <functional>
#include <memory>
#include "Node.hpp"
using std::cout, std::endl;

template<typename T>
class OrderedLinkedList
{
public:
  OrderedLinkedList(std::function<bool(T,T)> comp = std::less<T>()): head(nullptr), comp(comp) {};

  void insert(T new_data)
  {
    if (head == nullptr)
    {
      head = std::make_shared<Node<T>>(new_data);
      return;
    }

    auto curr = head;
    if (comp(new_data, head->data))
    {
      head = std::make_shared<Node<T>>(new_data, head);
      return;
    }
    while (curr->next != nullptr && comp(curr->next->data, new_data))
    {
      curr = curr->next;
    }
    curr->next = std::make_shared<Node<T>>(new_data, curr->next);
  }
  
  void remove(T remove_data)
  {
    if (head == nullptr) return;
    if (head->data == remove_data)
    {
      head = head->next;
      return;
    }

    auto curr = head;
    std::shared_ptr<Node<T>> prev = nullptr;
    while (curr->next != nullptr && comp(curr->data, remove_data))
    {
      prev = curr;
      curr = curr->next;
    }
    if(curr->data != remove_data) return;
    prev->next = curr->next;
  }
  
  bool member(T member_data)
  {
    if (head == nullptr) return false;
    auto curr = head;
    while (curr != nullptr && comp(curr->data, member_data))
    {
      curr = curr->next;
    }
    if(curr->data == member_data) return true;
    return false;
  }
  
  friend std::ostream& operator<<(std::ostream& o, const OrderedLinkedList& ll)
  {
    cout << "[ ";
    for (auto curr = ll.head; curr != nullptr; curr = curr->next)
    {
      cout << curr->data << " "; 
    }
    cout << "]";
    return o;
  }

private: 
  std::shared_ptr<Node<T>> head;
  std::function<bool(T,T)> comp;
};
