#include "game.hpp"
#include "TicTacToeBoard.hpp"
#include "player.hpp"
#include <iostream>
#include <string>

bool ask(std::string q) //yes or no questions
{
	std::string a;
	std::cout << q << std::endl;
	while (true)
	{
		std::getline(std::cin, a);
		if (a.empty()){}
		else if (tolower(a.front()) == 'y') return true;
		else if (tolower(a.front()) == 'n') return false;
		std::cout << "Please enter [Y/N]" << std::endl;
	}
}

Player make_player(std::string player_name, char mark, location_state state)
{
	return Player(player_name, mark, state);
}

Player make_userdef_player(std::string player_, location_state state)
{
	std::string p1_name, p1_char;
	std::cout << player_ << " enter your name: " << std::endl;
	std::getline(std::cin, p1_name);
	std::cout << "Choose your symbol: " << std::endl;
	std::getline(std::cin, p1_char);

	return Player(p1_name, p1_char.front(), state);
}

void move(Board& b, Player p1, Player p2, Player& p){
	int move;
	std::string ans;
	std::cout << p.get_name() << " were do you want to play? " << std::endl;
	b.display_board(p1, p2);
	while (true){
		std::getline(std::cin, ans);
		if (ans.empty()){ std::cout << "invalid play location, choose again." << std::endl; continue; }
		move = ans.front() - '0';
		if (move < 10 && move > 0 && b.is_valid_move(move)){
			b.mark_board(p, move);
			break;
		}
		std::cout << "invalid play location, choose again." << std::endl;
	}
}

void turn(bool current_player, Board& b, Player& p1, Player& p2)
{
	if (current_player) move(b,p1, p2, p1);
	else move(b,p1, p2, p2);
}

bool next_player(bool current_player)
{
	if (current_player) current_player = false;
	else current_player = true;

	return current_player;
}

void game(Player p1, Player p2) //game logic
{
	Board b;
	bool current_player = true;//p1 = true

	while (b.moves_left() && !b.is_there_winner())
	{
		
		turn(current_player, b, p1, p2);
		
		current_player = next_player(current_player);
		
	}

	std::cout << std::endl << "WINNER: " << b.who_won(p1, p2) << " !" << std::endl;
	b.display_board(p1, p2);
	//std::cin.ignore();

}