#ifndef WORDLADDERTREE_HPP
#define WORDLADDERTREE_HPP

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Ladder.hpp"
#include "avlTree.hpp"

class WordLadderTree
{
	public:
		// ---------- [functions] ----------//
		WordLadderTree(std::string inFile) { dictionary.reserve(355000); importDict(inFile); }
		void findLadder(std::string, std::string);
		void findRandom();
		void findRandom(std::string);

	private:
		// ---------- [variables] ----------//
		AvlTree<Ladder> queue;
		std::string end; //end word in the word ladder
		std::vector<std::string> dictionary; //full dictionary
		std::vector<std::string> words; //sub-dictionary based on word length

		// ---------- [functions] ----------//
		void importDict(std::string inFile);
		void simplifyDictionary(unsigned int);
		bool isWord(std::vector<std::string> &, std::string);
		template <typename T>
			int searchR(std::vector<T> &, T, int, int);
		bool oneAway(std::vector<std::string>);
		int getScore(std::vector<std::string> const &);
};

template <typename T>
std::ostream& operator<< (std::ostream& o, std::vector<T> const & v) {
	for (auto & e : v)
		o << "[" << e << "]";

	return o;
}

#endif
