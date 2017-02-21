#ifndef AVL_TREE_HPP
#define AVL_TREE_HPP

#include <algorithm>
#include <iostream> 
#include <assert.h>
#include <string>
#include <sstream>

namespace
{
	std::string TAB = "| ";
}

// AvlTree class
//
// CONSTRUCTION: zero parameter
//
// ******************PUBLIC OPERATIONS*********************
// void insert( item )    --> Insert item
// void remove( target )  --> Remove target  
// bool contains( target )--> Return true if target is present
// Comparable findMin( )  --> Return smallest item
// Comparable findMax( )  --> Return largest item
// bool isEmpty( )        --> Return true if empty; else false
// void makeEmpty( )      --> Remove all items
// string toString(string msg )--> Print tree in sorted order
// Comparable  removeMin()--> You are not allowed simply to do a find Min and then a remove on that value.  
//                            That would require that you traverse the tree twice  (once to find and once to delete).  We want a true removeMin() that 
//                            only traverses in order to delete.
//
// ******************ERRORS********************************
//
//HINT:   The class you use as Comparable must have overloaded operators for > == and <.
//        nullptr is another way of saying NULL.  It has some interesting properties for implicit conversions.
//        && is an "rvalue reference".  They are beyond the scope of this class.  A good explanation is at http://thbecker.net/articles/rvalue_references/section_01.html

template <typename Comparable>
class AvlTree
{
public:
	AvlTree() : root{ nullptr }
	{
		size = 0;
	}

	int getSize() { return size; }

	AvlTree(const AvlTree & rhs) : root{ nullptr }
	{
		size = rhs.size;
		root = clone(rhs.root);
	}

	AvlTree(AvlTree && rhs) : root{ rhs.root }
	{
		rhs.root = nullptr;
	}

	~AvlTree()
	{
		makeEmpty();
	}

	/* Deep copy. */
	AvlTree & operator=(const AvlTree & rhs)
	{
		AvlTree copy = rhs;
		std::swap(*this, copy);
		return *this;
	}

	/* Move. */
	AvlTree & operator=(AvlTree && rhs)
	{
		std::swap(root, rhs.root);
		return *this;
	}

	/* Find the smallest item in the tree. Throw UnderflowException if empty. */
	const Comparable & findMin() const
	{
		assert(!isEmpty());
		return findMin(root)->data;
	}

	/* Find the largest item in the tree.	 */
	const Comparable & findMax() const
	{
		assert(!isEmpty());
		return findMax(root)->data;
	}

	/* Returns true if target is found in the tree. */
	bool contains(const Comparable & target) const
	{
		return contains(target, root);
	}

	/* Test if the tree is logically empty.	 */
	bool isEmpty() const
	{
		return root == nullptr;
	}

	/* String contains the tree contents in sorted order. */
	std::string toString(std::string msg) const
	{
		std::ostringstream oss;
		oss << msg << std::endl;
		if (isEmpty())
			oss << "Empty tree" << std::endl;
		else
			toString(root, oss, "");
		oss << "END " << msg << std::endl;
		return oss.str();
	}

	/*String contains the inorder traversal of the tree. */
	std::string toStringList(std::string msg) const
	{
		std::ostringstream oss;
		oss << msg << std::endl;
		if (isEmpty())
			oss << "Empty tree" << std::endl;
		else
			toStringList(root, oss);
		oss << "END " << msg << std::endl;
		return oss.str();
	}

	/* Make the tree logically empty. */
	void makeEmpty()
	{
		makeEmpty(root);
	}

	/*	Insert item into the tree; 	 */
	void insert(const Comparable & item)
	{
		++size;
		insert(item, root);
	}

	/**
	 * Insert item into the tree;
	 * && is termed an rvalue reference.
	 * a good explanation is at http://thbecker.net/articles/rvalue_references/section_01.html
	 */
	void insert(Comparable && item)
	{
		++size;
		insert(std::move(item), root);
	}

	/* Remove target from the tree. Nothing is done if target is not found. */
	void remove(const Comparable & item)
	{
		--size;
		remove(item, root);
	}
	/* remove smallest element from the tree.  Return the value found there */
	Comparable removeMin()
	{
		if (isEmpty())
			throw std::domain_error("No ladder was found");
		--size;
		AvlNode * min = removeMin(root);
		assert(min != nullptr);
		return min->data;
	}

private:
	struct AvlNode
	{
		Comparable data;
		AvlNode    *left;
		AvlNode    *right;
		int        height;

		AvlNode(const Comparable & d, AvlNode *lt, AvlNode *rt, int h = 0)
			: data{ d }, left{ lt }, right{ rt }, height{ h } { }

		AvlNode(Comparable && ele, AvlNode *lt, AvlNode *rt, int h = 0)
			: data{ std::move(ele) }, left{ lt }, right{ rt }, height{ h } { }
	};

	AvlNode *root;
	int size;


	/**
	 * Internal method to insert into a subtree.
	 * item is the item to insert.
	 * t is the node that roots the subtree.
	 * Set the new root of the subtree.
	 */
	void insert(Comparable const & item, AvlNode * & t)
	{
		if (t == nullptr)
			t = new AvlNode{ item, nullptr, nullptr };
		else if (item < t->data)
			insert(item, t->left);
		else
			insert(item, t->right);

		balance(t);
	}

	/**
	 * Internal method to insert into a subtree.
	 * item is the item to insert.
	 * t is the node that roots the subtree.
	 * Set the new root of the subtree.
	 */
	void insert(Comparable && item, AvlNode * & t)
	{
		if (t == nullptr)
			t = new AvlNode{ std::move(item), nullptr, nullptr };
		else if (item < t->data)
			insert(std::move(item), t->left);
		else
			insert(std::move(item), t->right);

		balance(t);
	}

	/**
	 * Internal method to remove from a subtree.
	 * target is the item to remove.
	 * t is the node that roots the subtree.
	 * Set the new root of the subtree.
	 */
	void remove(const Comparable & target, AvlNode * & t)
	{
		if (t == nullptr)
			return;   // Item not found; do nothing

		if (target < t->data)
			remove(target, t->left);
		else if (t->data < target)
			remove(target, t->right);
		else if (t->left != nullptr && t->right != nullptr) // Two children
		{
			t->data = findMin(t->right)->data;
			remove(t->data, t->right);
		}
		else
		{
			AvlNode *oldNode = t;
			t = (t->left != nullptr) ? t->left : t->right;
			delete oldNode;
		}

		balance(t);
	}

	AvlNode * removeMin(AvlNode * & t)
	{
		if (!t) return nullptr;
		if (!t->left)
		{
			auto oldNode = t;
			if (t->right)
			{
				t = t->right;
				return oldNode;
			}
			else
			{
				t = nullptr;
				return oldNode;
			}
		}
		auto min = removeMin(t->left);
		balance(t);
		return min;
	}

	static const int ALLOWED_IMBALANCE = 1;

	// Assume t is balanced or within one of being balanced
	void balance(AvlNode * & t)
	{
		if (t == nullptr)
			return;

		if (height(t->left) - height(t->right) > ALLOWED_IMBALANCE)
			if (height(t->left->left) >= height(t->left->right))
				rotateWithLeftChild(t);
			else
				doubleWithLeftChild(t);
		else
			if (height(t->right) - height(t->left) > ALLOWED_IMBALANCE)
			{
				if (height(t->right->right) >= height(t->right->left))
					rotateWithRightChild(t);
				else
					doubleWithRightChild(t);
			}

		t->height = max(height(t->left), height(t->right)) + 1;
	}

	/**
	 * Internal method to find the smallest item in a subtree t.
	 * Return node containing the smallest item.
	 */
	AvlNode * findMin(AvlNode *t) const
	{
		if (t == nullptr)
			return nullptr;
		if (t->left == nullptr)
			return t;
		return findMin(t->left);
	}

	/**
	 * Internal method to find the largest item in a subtree t.
	 * Return node containing the largest item.
	 */
	AvlNode * findMax(AvlNode *t) const
	{
		if (t != nullptr)
			while (t->right != nullptr)
				t = t->right;
		return t;
	}


	/**
	 * Internal method to test if an item is in a subtree.
	 * target is item to search for.
	 * t is the node that roots the tree.
	 */
	bool contains(const Comparable & target, AvlNode *t) const
	{
		if (t == nullptr)
			return false;
		else if (target < t->data)
			return contains(target, t->left);
		else if (t->data < target)
			return contains(target, t->right);
		else
			return true;    // Match
	}

	/**
	 * Internal method to make subtree empty.
	 */
	void makeEmpty(AvlNode * & t)
	{
		if (t != nullptr)
		{
			makeEmpty(t->left);
			makeEmpty(t->right);
			delete t;
		}
		t = nullptr;
	}


	/**
	 * Internal method to create a string for a subtree rooted at t in preorder with tabbed indents.
	 */
	void toString(AvlNode *t, std::ostringstream &oss, std::string tab) const
	{
		if (!t) return;
		/*
		oss << tab << t->data << std::endl;
		toString(t->left, oss, tab + TAB);
		toString(t->right, oss, tab + TAB);
		*/
		toString(t->right, oss, tab+"  ");
		oss << tab << t->data << std::endl;
		toString(t->left, oss, tab+"  ");
	}

	/**
	 * Internal method to create a string for a subtree rooted at t in sorted order.
	 */
	void toStringList(AvlNode *t, std::ostringstream &oss) const
	{
		if (!t) return;
		toStringList(t->left, oss);
		oss << t->data << " " << std::endl;
		toStringList(t->right, oss);
	}

	/**
	 * Internal method to clone subtree.
	 */
	AvlNode * clone(AvlNode *t) const
	{
		if (t == nullptr)
			return nullptr;
		else
			return new AvlNode{ t->data, clone(t->left), clone(t->right), t->height };
	}

	// AVL manipulations
	/**
	 * Return the height of node t or -1 if nullptr.
	 */
	int height(AvlNode *t) const
	{
		return t == nullptr ? -1 : t->height;
	}

	int max(int lhs, int rhs) const
	{
		return lhs > rhs ? lhs : rhs;
	}

	/**
	 * Rotate binary tree node with left child.
	 * For AVL trees, this is a single rotation for case 1.
	 * Update heights, then set new root.
	 */
	void rotateWithLeftChild(AvlNode * & k2)
	{
		AvlNode *k1 = k2->left;
		k2->left = k1->right;
		k1->right = k2;
		k2->height = max(height(k2->left), height(k2->right)) + 1;
		k1->height = max(height(k1->left), k2->height) + 1;
		k2 = k1;
	}

	/**
	 * Rotate binary tree node with right child.
	 * For AVL trees, this is a single rotation for case 4.
	 * Update heights, then set new root.
	 */
	void rotateWithRightChild(AvlNode * & k1)
	{
		AvlNode *k2 = k1->right;
		k1->right = k2->left;
		k2->left = k1;
		k1->height = max(height(k1->left), height(k1->right)) + 1;
		k2->height = max(height(k2->right), k1->height) + 1;
		k1 = k2;
	}

	/**
	 * Double rotate binary tree node: first left child.
	 * with its right child; then node k3 with new left child.
	 * For AVL trees, this is a double rotation for case 2.
	 * Update heights, then set new root.
	 */
	void doubleWithLeftChild(AvlNode * & k3)
	{
		rotateWithRightChild(k3->left);
		rotateWithLeftChild(k3);
	}

	/**
	 * Double rotate binary tree node: first right child.
	 * with its left child; then node k1 with new right child.
	 * For AVL trees, this is a double rotation for case 3.
	 * Update heights, then set new root.
	 */
	void doubleWithRightChild(AvlNode * & k1)
	{
		rotateWithLeftChild(k1->right);
		rotateWithRightChild(k1);
	}
};

#endif
