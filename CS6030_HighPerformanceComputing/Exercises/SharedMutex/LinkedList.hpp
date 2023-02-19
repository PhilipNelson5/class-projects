#pragma once
#include <iostream>
#include <memory>
#include "Node.hpp"
using std::cout, std::endl;

template<typename T>
class LinkedList
{
public:
  LinkedList(): head(nullptr) {};
  void insert(T data)
  {
    if (head == nullptr)
    {
      head = std::make_shared<Node<T>>(data);
      return;
    }

    auto curr = head;
    while (curr->next != nullptr)
    {
      curr = curr->next;
    }
    curr->next = std::make_shared<Node<T>>(data);
  }
  
  void set(int i, T data)
  {
    get_node(i)->data = data;
  }
  
  T get(int i)
  {
    return get_node(i)->data;
  }
  
  friend std::ostream& operator<<(std::ostream& o, const LinkedList& ll)
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
  std::shared_ptr<Node<T>> get_node(int i)
  {
    auto curr = head;
    while(curr != nullptr && i > 0)
    {
      curr = curr->next;
      --i;
    }
    if (curr != nullptr && i == 0)
    {
      return curr;
    }
    throw std::out_of_range("attempt to access element beyond the end of the list");
  }
  std::shared_ptr<Node<T>> head;
};
