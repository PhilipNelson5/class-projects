#include "Node.h"

//Feel free to change any declaraitons you want.  This is meant only as a guide
class Tree
{
	Node * root;
	public:
	Tree(void) { root = NULL; }
	void makeEmpty() { makeEmpty(root); }
	void makeEmpty(Node *& r);
	string printTree(string indent, Node * n);
	string printTree(string indent=""){ return printTree(indent, root); }
	string toPreorder(Node * n);
	string toPreorder();
	void buildFromPrefix(ifstream & inf);
	Node * buildNodeFromPrefix(ifstream & inf, Node * parent);
	Node * findWord(string word, Node * r);
	Node * findWord(string word) { return findWord(word, root); }
	void upCase(Node* n);
	void upCase(){ upCase(root); }
	Tree clone();
	Node * clone(Node * n, Node * p);
	int fringe() { return fringe(root); }
	int fringe(Node * n);
	int nodesInLevel(int level){ return nodesInLevel(level, root); };
	int nodesInLevel(int level, Node * r);
	bool isIsomorphic(Tree &t2) { return isIsomorphic(root, t2.root); }
	bool isIsomorphic(Node *r1, Node *r2);
	string commonAncestor(string s1, string s2);
	Node * commonAncestor(Node* n1, Node *n2);

};

