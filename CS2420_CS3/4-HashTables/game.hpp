#ifndef GAME_HPP
#define GAME_HPP

#include <fstream>
#include <string>

#include "hashTable.hpp"
#include "record.hpp"

class Game
{
	public:
		Game(std::vector<std::string> g): score(0), output_frequency(20)
	{
		games = g;
		for(auto&& f : games)
			fin.emplace_back(f);
	};
		void play();

	private:
		HashTable<std::string, Record> table;

		int score;
		int output_frequency;
		std::vector<std::string> games;
		std::vector<std::ifstream> fin;

		int getScore(std::string, int);
		int getLetterScore(std::string);
		int getLengthScore(std::string);
		int getBonusScore(int);
		void changeOutputFrequency();
		void play(int);
		std::string displayGames();
		void selectGame();
		void peak();
};

#endif
