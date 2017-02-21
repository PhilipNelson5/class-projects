/**
 * WordLadderQueue class written by Philip Nelson USU CS2420-001
 */

#ifndef WORDLADDERQUEUE_HPP
#define WORDLADDERQUEUE_HPP

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "linkedList.hpp"

//------------------------------[WordLadder Class]------------------------------//
//
// CONSTRUCTION: std::string file location of dictionary
//
//------------------------------[Public Operations]-----------------------------//
//
// void findLadder( start word, end word ) --> Find ladder from two words
// void findRandom( )                      --> Finds two random words and initiates findLadder( )
// void findRandom( start word )           --> Finds one random word based on a starting word
//                                            and initiates findLadder( )
//
//------------------------------[Public Operations]-----------------------------//
//
//	void importDict( dictionary file path )--> Produces a dictionary vector from a .txt file
//	void simplifyDictionary(int);          --> Produces a sub-dictionary of n length words
//	bool isWord( dictionary , word );      --> Checks the dictionary for a given word
//	bool oneAway( ladder );                --> Creates ladders one away, pushes to queue
//	                                           and returns of the final ladder was encountered
//

class WordLadderQueue
{
	public:
		// ---------- [functions] ----------//
		WordLadderQueue(std::string inFile) { dictionary.reserve(355000); importDict(inFile); }
		void findLadder(std::string, std::string);
		void findRandom();
		void findRandom(std::string);

	private:
		// ---------- [variables] ----------//
		LinkedList<std::vector<std::string>> queue;
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
};

#endif
