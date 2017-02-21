#ifndef LINKEDLIST_HPP
#define LINKEDLIST_HPP

#include <functional>
#include <memory>
#include <sstream>

template <typename T>
struct Node;

template <typename T>
class LinkedList
{
	struct Node
	{
		Node(T d,
			std::shared_ptr<Node> n = nullptr,
			std::shared_ptr<Node> p = nullptr) :
			data(d), next(n), prev(p) {}

		T data;
		std::shared_ptr<Node> next;
		std::shared_ptr<Node> prev;
	};

public:
	LinkedList() : head(nullptr) {}

	//deep copy
	LinkedList(LinkedList const & other) : head(clone(other.head, nullptr)) {}

	//shallow copy
	LinkedList(LinkedList const && other) : head(other.head) {}

	void push_back(T);
	void push_front(T);
	T pop();
	T back();
	void clear() { clearR(head); head = tail = nullptr; }
	void forEach(std::function<void(T &)>);
	std::string toString();
	int size() { return sizeR(head); }

	bool isEmpty();

private:
	std::shared_ptr<Node> head;
	std::shared_ptr<Node> tail;

	std::shared_ptr<Node> clone(std::shared_ptr<Node> curr, std::shared_ptr<Node> prev)
	{
		if (!curr) return nullptr;
		auto copy = std::make_shared<Node>(curr->data);
		copy->prev = prev;
		copy->next = clone(curr->next, copy);
		return copy;
	}

	int sizeR(std::shared_ptr<Node>);
	void clearR(std::shared_ptr<Node> curr);
};

//inserts element at the back
template <typename T>
void LinkedList<T>::push_back(T item)
{
	if (!head)
	{
		head = tail = std::make_shared<Node>(item);
		return;
	}

	tail->next = std::make_shared<Node>(item, nullptr, tail);
	tail = tail->next;
}

//inserts element at the front
template <typename T>
void LinkedList<T>::push_front(T item)
{
	if (!head)
	{
		head = tail = std::make_shared<Node>(item);
		return;
	}

	head->prev = std::make_shared<Node>(item, head, nullptr);
	head = head->prev;
}

//removes the top element
template <typename T>
T LinkedList<T>::pop()
{
	if (!head)
		throw std::out_of_range("No ladder was found.");

	auto t = head;
	if (!head->next)
	{
		head = tail = nullptr;
		return t->data;
	}
	head = head->next;
	head->prev = nullptr;
	return t->data;
}

template <typename T>
T LinkedList<T>::back()
{
	if (!tail)
		throw std::out_of_range("Can not return back element\nThe list is empty!");

	return tail->data;
}

template <typename T>
void LinkedList<T>::clearR(std::shared_ptr<Node> curr)
{
	if (!curr)
		return;
	curr->prev = nullptr;
	clearR(curr->next);
}

template <typename T>
void LinkedList<T>::forEach(std::function<void(T &)> f)
{
	for (auto curr = head; curr; curr = curr->next)
		f(curr->data);
}

template <typename T>
std::string LinkedList<T>::toString()
{
	std::stringstream ss;

	for (auto curr = head; curr != nullptr; curr = curr->next)
		ss << curr->data << " ";
	ss << std::endl;

	return ss.str();
}

template <typename T>
int LinkedList<T>::sizeR(std::shared_ptr<Node> curr)
{
	if (!curr) return 0;
	return 1 + sizeR(curr->next);
}

template <typename T>
bool LinkedList<T>::isEmpty()
{
	if (head == nullptr)
		return true;
	return false;
}

#endif
