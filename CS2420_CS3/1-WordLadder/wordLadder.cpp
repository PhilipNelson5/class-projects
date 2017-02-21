#include "wordLadder.hpp"

//returns a random number from 3 to size
int rand(int size)
{
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(3, size);
	return dist(mt);
}

//finds the word ladder given a beginning word (b) and ending word (e)
void WordLadder::findLadder(std::string b, std::string e)
{
	if (b.size() != e.size())
		throw std::domain_error("Words must be of the same length to generate a word ladder.");
	std::cout << "searching for a ladder from [" << b << "] to [" << e << "]" << std::endl;

	queue.clear();
	simplifyDictionary(b.size());

	if (!isWord(dictionary, b))
		throw std::domain_error("Beg must be English words");
	if (!isWord(dictionary, e))
		throw std::domain_error("End must be English words");

	end = e;

	std::vector<std::string> first;
	first.push_back(b);
	queue.push_back(first);

	while (!oneAway(queue.pop())){}

	std::cout << "The word ladder from [" << b << "] to [" << e << "] is " << queue.back().size() << " rungs: " << std::endl;
	std::cout << queue.back() << std::endl;
}

//will choose a random word length and 2 words of that length
void WordLadder::findRandom()
{
	auto size = rand(7);
	simplifyDictionary(size);

	findLadder(words[rand(words.size())], words[rand(words.size())]);
}

//will choose a random word of the same length as the beginning word (b)
void WordLadder::findRandom(std::string b)
{
	simplifyDictionary(b.size());
	findLadder(b, words[rand(words.size())]);
}

//imports a text file to the dictionary
void WordLadder::importDict(std::string inFile)
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

//creates a seperate "sub-dictionary" from a given word length (size)
void WordLadder::simplifyDictionary(unsigned int size)
{
	words.clear();
	for (auto && e : dictionary)
		if (e.size() == size)
			words.push_back(e);
}

//returns true or false if the word exists and deletes found words from the words sub-dictionary
bool WordLadder::isWord(std::vector<std::string> & dict, std::string target)
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

//binary search on a given dictionary (dict) for a target. returns the index of the found word, -1 if unfound.
template <typename T>
int WordLadder::searchR(std::vector<T> & dict, T target, int start, int end)
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
bool WordLadder::oneAway(std::vector<std::string> curr)
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
				queue.push_back(nextLadder);

				if (nextWord == end)
					return true;
			}
		}
	}
	return false;
}
