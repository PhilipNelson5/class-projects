#include <chrono>
#include <sstream>
#include "avlTree.hpp"
#include "Ladder.hpp"
#include "wordLadderTree.hpp"

#define underline "\33[4m"
#define nounderline "\33[0m"

//Constants and counters
namespace
{
	const int RAND_LOW = 4;
	const int RAND_HIGH = 6;
	int enqueue;

}

//returns a good pseudo-random number from low to high
int rand(int low, int high)
{
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(low, high);
	return dist(mt);
}

//finds the word ladder given a beginning word (b) and ending word (e)
void WordLadderTree::findLadder(std::string b, std::string e)
{
	if (b.size() != e.size())
		throw std::domain_error("Words must be of the same length to generate a word ladder.");

	queue.makeEmpty();
	simplifyDictionary(b.size());

	if (!isWord(dictionary, b))
		throw std::domain_error("Beg must be English words");
	if (!isWord(dictionary, e))
		throw std::domain_error("End must be English words");

	std::cout << "searching for a ladder from [" << b << "] to [" << e << "]" << std::endl;

	end = e;
	int dequeue = 0;
	enqueue = 0;
	int treeMax = 0;
	double time = 0.0;

	std::vector<std::string> firstLadder;
	firstLadder.push_back(b);
	queue.insert(Ladder(firstLadder, getScore(firstLadder)));

	auto timeStart = std::chrono::high_resolution_clock::now();
	while (!oneAway(queue.removeMin().ladder)) { ++dequeue; queue.getSize() > treeMax ? treeMax = queue.getSize():treeMax; }
	auto timeEnd = std::chrono::high_resolution_clock::now();

	time = std::chrono::duration <double, std::milli>(timeEnd - timeStart).count();
	auto finalLadder = queue.removeMin();

	std::stringstream ss;
	ss << finalLadder.ladder;
	int chars = ss.str().size();

	for(int i = 0; i < chars; ++i)
		std::cout << " ";

	std::cout << underline << "\t\tEnqueue\tDequeue\tQ Size\tTime(ms)" << std::endl;
	std::cout << nounderline << finalLadder.ladder;
	std::cout << finalLadder.ladder.size() << "\tASTAR:\t" << enqueue << "\t" << dequeue << "\t" << treeMax << "\t" << time << std::endl;
}

//will choose a random word length and 2 words of that length
void WordLadderTree::findRandom()
{
	auto size = rand(RAND_LOW, RAND_HIGH);
	simplifyDictionary(size);

	findLadder(words[rand(RAND_LOW, words.size())], words[rand(RAND_LOW, words.size())]);
}

//will choose a random word of the same length as the beginning word (b)
void WordLadderTree::findRandom(std::string b)
{
	simplifyDictionary(b.size());
	findLadder(b, words[rand(0, words.size())]);
}

//imports a text file to the dictionary
void WordLadderTree::importDict(std::string inFile)
{
	std::ifstream fin(inFile);
	std::string word;

	while (!fin.eof())
	{
		fin >> word;
		dictionary.push_back(word);
	}
	std::cout << "dictionary imported from " << inFile << std::endl;
	std::cout << dictionary.size() << " words" << std::endl << std::endl;
}

//creates a separate "sub-dictionary" from a given word length (size)
void WordLadderTree::simplifyDictionary(unsigned int size)
{
	words.clear();
	for (auto && e : dictionary)
		if (e.size() == size)
			words.push_back(e);
}

//returns true or false if the word exists and deletes found words from the words sub-dictionary
bool WordLadderTree::isWord(std::vector<std::string> & dict, std::string target)
{
	auto result = searchR(dict, target, 0, dict.size() - 1);
	if (result >= 0)
	{
		if (words.size() >= dict.size())
			words.erase(words.begin() + result);
		return true;
	}
	return false;
}

//binary search on a given dictionary (dict) for a target. Returns the index of the found word, -1 if unfound.
	template <typename T>
int WordLadderTree::searchR(std::vector<T> & dict, T target, int start, int end)
{
	if (start > end)
		return -1;

	auto mid = (start + end) / 2;

	if (target == dict[mid])
		return mid;
	else if (target < dict[mid])
		return searchR(dict, target, start, mid - 1);
	else //(target > dict[mid])
		return searchR(dict, target, mid + 1, end);
}

//pushes to the queue vectors containing words one away from the last word on the current ladder
bool WordLadderTree::oneAway(std::vector<std::string> curr)
{
	for (auto i = 0u; i < curr.back().size(); ++i)
	{
		auto nextWord = curr.back();

		for (auto j = 97; j <= 122; ++j)
		{
			auto nextLadder = curr;
			nextWord[i] = static_cast<char>(j);
			if (isWord(words, nextWord))
			{
				nextLadder.push_back(nextWord);
				queue.insert(Ladder(nextLadder, getScore(nextLadder)));
				++enqueue;

				if (nextWord == end)
					return true;
			}
		}
	}
	return false;
}

int WordLadderTree::getScore(std::vector<std::string> const & curr)
{
	int score = curr.size();
	auto lastWord = curr.back();
	if(lastWord == end)
		return 0;
	for(unsigned int i = 0; i < lastWord.size(); ++i)
		if(lastWord[i] != end[i])
			++score;
	return score;
}
