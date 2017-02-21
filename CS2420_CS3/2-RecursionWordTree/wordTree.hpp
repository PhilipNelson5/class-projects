#ifndef WORDTREE_HPP
#define WORDTREE_HPP

#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "Node.hpp"

class LLTree
{
	public:
		LLTree(std::shared_ptr<Node> r = nullptr) : root(r) {};
		void preorderBuild(std::ifstream & fin) { root = preorderBuildR(fin, nullptr); }
		std::string printTree() { std::stringstream ss; printTreeR(root, ss, ""); return ss.str(); }
		std::shared_ptr<Node> find(std::string target) { return findR(root, target); }
		std::string toPreorder() { std::stringstream ss; toPreorderR(root, ss); return ss.str(); }
		void upCase() { upCaseR(root); }
		void makeEmpty() { root = nullptr ;} //Smart pointers self deallocate once nothings is pointing to them.
		int fringe() { int leaf = 0; return fringeR(root, leaf); }
		bool isIsomorphic(LLTree);
		std::string getShape(){ std::stringstream ss; getShapeR(root, ss); return ss.str(); }
		LLTree clone() { return LLTree(cloneR(root, nullptr)); }
		std::string commonAncestor(std::string, std::string);
		int nodesInLevel(int tLevel) { int count = 0; return nodesInLevelR(root, tLevel, 0, count); }
		
	private:
		std::shared_ptr<Node> root;

		std::shared_ptr<Node> preorderBuildR(std::ifstream &, std::shared_ptr<Node> const);
		void printTreeR(std::shared_ptr<Node> const, std::stringstream &, std::string);
		std::shared_ptr<Node> findR(std::shared_ptr<Node>, std::string);
		void toPreorderR(std::shared_ptr<Node>, std::stringstream &) const;
		void upCaseR(std::shared_ptr<Node>);
		int fringeR(std::shared_ptr<Node>, int &);
		void getShapeR(std::shared_ptr<Node>, std::stringstream &);
		std::shared_ptr<Node> cloneR(std::shared_ptr<Node>, std::shared_ptr<Node>);
		void getAncestors(std::shared_ptr<Node> curr, std::vector<std::string> & lineage);
		int nodesInLevelR(std::shared_ptr<Node>, int, int, int &);

};

#endif
