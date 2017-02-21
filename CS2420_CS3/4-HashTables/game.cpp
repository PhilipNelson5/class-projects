#include <ctype.h>
#include <iostream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "game.hpp"
#include "hashTable.hpp"

//allows user to input new output_frequency
void Game::changeOutputFrequency()
{
	int response;
	std::cout << "How often would you like to see your score output?\n"
		<< "Every [     ] th word.\033[15D"; //move cursor back 15 columns
		//<< "Every nth word: ";

	std::cin >> response;

	output_frequency = response;
}

//lists all playable games for the user to choose from
std::string Game::displayGames()
{
	std::ostringstream oss;
	for (unsigned int i = 0; i < games.size(); ++i)
		oss << "   [" << i << "] " << games[i] << std::endl;

	oss << "[-1] to return" << std::endl;
	return oss.str();
}

//allows user to enter the number of elements they would like to see from the hash table
void Game::peak()
{
	std::string response = "";
	int num;
	std::cout << "Print the hashTable? [y/n]\n";
	std::cin >> response;

	if (tolower(response[0]) == 'y')
	{
		std::cout << "How many entries?\n";
		std::cin >> num;
		std::cout << "Word\t\tCount\tProbes" << std::endl;
		std::cout << table.toString(num);
		std::cout << std::endl;
	}
}

//menu where user is prompted to choose a game to be played.
//the selected game is then played
void Game::selectGame()
{
	while (true)
	{
		int response = -1;

		std::cout << "Which game would you like to play?\n"
			<< displayGames();
		std::cin >> response;

		if (response == -1)
			return;

		bool cont = true;
		while (cont)
		{
			if ((unsigned int)(response) < games.size())
				cont = false;
			else
			{
				std::cout << "Please choose a number 0 - " << games.size() - 1 << std::endl;
				std::cin >> response;
			}
		}
		play(response);
	}
}

//once a games is choosen here it is played and the score is given to the user
//he/she is then given more options to view the content of the hash table
void Game::play(int game)
{
	table.makeEmpty();
	score = 0;
	std::string nextWord = "";
	int words = 0;
	std::vector<int> length(100);

	std::cout << "Game: " << games[game] << std::endl;
	while (fin[game] >> nextWord)
	{
		++words;
		auto rec = table.insert(nextWord, new Record(nextWord));
		int wordScore = getScore(rec->word, rec->count);
		score += wordScore;
		if (words % output_frequency == 0)
			std::cout << "Words: " << words << " Score: " << score << std::endl;
		++length[nextWord.size()];
		++rec->count;
	}
	std::cout << "Score: " << score << std::endl;
	for (auto i = 0u; i < length.size(); ++i)
		if (length[i] != 0)
			std::cout << length[i] << " Words of Length " << i << std::endl;

	peak();
	//reset fin stream to the beginning of the file
	//allows a game to be played multiple times
	fin[game].clear();
	fin[game].seekg(0, fin[game].beg);
}

//main menue where the user can choose to edit the output frequency, play or exit
void Game::play()
{
	std::string response = "";
	std::cout << "Welcome to the Word Game!\n";
	bool cont = true;
	while (cont)
	{
		std::cout << "You can:" << std::endl
			<< "[d] change output frequency (" << output_frequency << ")" << std::endl
			<< "[p] play a game!" << std::endl
			<< "[e] exit" << std::endl;
		std::cin >> response;
		switch (tolower(response[0]))
		{
		case 'd':
			changeOutputFrequency();
			break;
		case 'p':
			selectGame();
			break;
		case 'e':
			cont = false;
			break;
		default:
			std::cout << "Your choice '" << static_cast<char>(tolower(response[0])) << "' does not exist...\n";
		}
	}

	for (auto&& is : fin)
		is.close();

	std::cout << "THANKS FOR PLAYING!!";
}

//return the compound score of the word
int Game::getScore(std::string word, int count)
{
	return getLetterScore(word) * getLengthScore(word) * getBonusScore(count);
}

//return the letter component of the score
int Game::getLetterScore(std::string word)
{
	const static std::vector<int> val = { 1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10 };

	int sum = 0;

	for (auto&& l : word)
		sum += val[l - 'a'];

	return sum;
}

//return the length component of the score
int Game::getLengthScore(std::string word)
{
	const static std::vector<int> val = { 0, 0, 1, 2, 3, 4, 5 };
	int length = word.length();

	if (length < 8) return val[length - 1];
	return 6;
}

//return the length component of the socre
int Game::getBonusScore(int count)
{
	const static std::vector<int> val = { 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2 };
	if (count <= 9)
		return val[count];
	return 1;
}
