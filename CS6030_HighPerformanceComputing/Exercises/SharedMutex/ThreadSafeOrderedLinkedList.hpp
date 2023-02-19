#pragma once
#include <iostream>
#include <functional>
#include <memory>
#include "Node.hpp"
using std::cout, std::endl;

template<typename T>
class ThreadSafeOrderedLinkedList
{
public:
  ThreadSafeOrderedLinkedList(std::function<bool(T,T)> comp = std::less<T>()): head(nullptr), comp(comp) {};

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
    auto curr = head;
    std::shared_ptr<Node<T>> prev = nullptr;
    while (curr->next != nullptr && curr->data < remove_data)
    {
      prev = curr;
      curr = curr->next;
    }
    if(curr->data == remove_data)
    prev->next = curr->next;
  }
  
  void set(int i, T data)
  {
    get_node(i)->data = data;
  }
  
  T get(int i)
  {
    return get_node(i)->data;
  }
  
  friend std::ostream& operator<<(std::ostream& o, const ThreadSafeOrderedLinkedList& ll)
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
  std::function<bool(T,T)> comp;
};
