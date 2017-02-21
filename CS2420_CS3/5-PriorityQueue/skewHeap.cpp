#include <memory>
#include <memory>
#include <sstream>

#include "skewHeap.hpp"

//merge two heaps
std::shared_ptr<node> SkewHeap::merge(std::shared_ptr<node> root1, std::shared_ptr<node> root2)
{
	if(!root1)
		return root2;
	if(!root2)
		return root1;

	if(compare(root1->data, root2->data))
	{
		root1->right = merge(root1->right, root2);
		std::swap(root1->right, root1->left);
		++mergeCount;
		return root1;
	}
	else
	{
		root2->right = merge(root2->right, root1);
		std::swap(root2->right, root2->left);
		++mergeCount;
		return root2;
	}
}

//remove and return the top of the tree (min/max)
int SkewHeap::pop()
{
	int temp = root->data;
	root = merge(root->right, root->left);
	--heapSize;
	return temp;
}

//returns a "pretty" string representation of the heap
void SkewHeap::toString(std::shared_ptr<node> curr, std::ostringstream &oss, std::string tab) const
{
	if (!curr) return;

	toString(curr->right, oss, tab+"  ");
	oss << tab << curr->data << std::endl;
	toString(curr->left, oss, tab+"  ");
}

//insert new element into the heap
void SkewHeap::insert(int data)
{
	root = merge(root, std::make_shared<node>(data));
	++heapSize;
}

void SkewHeap::clear()
{
	root = nullptr;
	heapSize = 0;
	mergeCount = 0;
}
