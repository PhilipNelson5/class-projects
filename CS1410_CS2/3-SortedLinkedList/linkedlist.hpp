#ifndef LINKEDLIST_HPP
#define LINKEDLIST_HPP

#include <functional>
#include <memory>

template <typename T>
struct Node
{
  Node(T d) : data(d), next(nullptr), prev(nullptr) {}

  T data;
  std::shared_ptr<Node> next;
  std::shared_ptr<Node> prev;
};

template <typename T>
int size(std::shared_ptr<Node<T>>);

template <typename T>
std::shared_ptr<Node<T>> destroy(std::shared_ptr<Node<T>>);

template <typename T>
class Linkedlist
{

public:
  Linkedlist() : head(nullptr) {}
  Linkedlist(Linkedlist const &other) : head(clone(other.head, nullptr)) {} // deep copy
  Linkedlist(Linkedlist const &&other) : head(other.head) {}                // shallow copy
  ~Linkedlist()
  {
    destroy(head);
    head = nullptr;
  }

  void insert(T);
  std::shared_ptr<Node<T>> find(T target) const { return findR(target, head); }
  void remove(T);
  void forEach(std::function<void(T &)>);
  int size() { return sizeR(head); }

  T operator[](int);
  friend std::ostream &operator<<(std::ostream &o, Linkedlist<T> const &list)
  {
    print(o, list.head);
    return o;
  }

  Linkedlist<T> operator=(Linkedlist<T> &other)
  {
    head = clone(other.head, nullptr);
    return *this;
  }
  Linkedlist<T> operator=(Linkedlist<T> &&other)
  {
    head = other.head;
    return *this;
  }

private:
  std::shared_ptr<Node<T>> clone(std::shared_ptr<Node<T>> cur, std::shared_ptr<Node<T>> prev);
  std::shared_ptr<Node<T>> head;
};

template <typename T>
std::shared_ptr<Node<T>> destroy(std::shared_ptr<Node<T>> cur)
{
  if (!cur) return nullptr;
  cur->prev = nullptr;
  return cur->next;
}

template <typename T>
void Linkedlist<T>::insert(T item)
{
  if (!head)
  {
    head = std::make_shared<Node<T>>(item);
    return;
  }

  if (head->data > item)
  {
    auto oldhead = head;
    head = std::make_shared<Node<T>>(item);
    head->next = oldhead;
    oldhead->prev = head;
    return;
  }
  auto cur = head;
  for (; cur->next; cur = cur->next)
  {
    if (cur->next->data > item)
    {
      auto newNode = std::make_shared<Node<T>>(item);
      cur->next->prev = newNode;
      newNode->next = cur->next;
      newNode->prev = cur;
      cur->next = newNode;
      return;
    }
  }
  auto newNode = std::make_shared<Node<T>>(item);
  newNode->prev = cur;
  cur->next = newNode;
  return;
}

template <typename T>
std::shared_ptr<Node<T>> Linkedlist<T>::clone(std::shared_ptr<Node<T>> cur,
                                              std::shared_ptr<Node<T>> prev)
{
  if (!cur) return nullptr;
  auto copy = std::make_shared<Node<T>>(cur->data);
  copy->prev = prev;
  copy->next = clone(cur->next, copy);
  return copy;
}

template <typename T>
std::shared_ptr<Node<T>> findR(T target, std::shared_ptr<Node<T>> cur)
{
  if (!cur) return nullptr;
  if (cur->data == target) return cur;
  return findR(target, cur->next);
}

template <typename T>
void Linkedlist<T>::remove(T target)
{
  auto cur = find(target);
  while (cur)
  {
    if (cur->next) cur->next->prev = cur->prev;
    if (cur->prev)
      cur->prev->next = cur->next;
    else
      head = cur->next;
    cur = find(target);
  }
}

template <typename T>
void Linkedlist<T>::forEach(std::function<void(T &)> f)
{
  for (auto cur = head; cur; cur = cur->next)
  {
    f(cur->data);
  }
}

template <typename T>
T Linkedlist<T>::operator[](int i)
{
  auto cur = head;
  for (int j = 0; j < i; ++j)
  {
    cur = cur->next;
  }
  return cur->data;
}

template <typename T>
void print(std::ostream &o, std::shared_ptr<Node<T>> cur)
{
  if (!cur) return;
  o << cur->data << ", ";
  print(o, cur->next);
}

template <typename T>
int sizeR(std::shared_ptr<Node<T>> cur)
{
  if (!cur) return 0;
  return 1 + sizeR(cur->next);
}

#endif
