#ifndef TICTACTOE_HPP
#define TICTACTOE_HPP
#include "player.hpp"
#include <string>

class Board
{
public:
	Board();
	void display_board(Player p1, Player p2);
	//void Board::display_instruction_board(Player p1, Player p2);
	bool is_there_winner();
	std::string who_won(Player& p1, Player& p2);
	bool is_valid_move(int move);
	void mark_board(Player p, int move);
	bool moves_left();

private:
	location_state winner;
	location_state board[3][3];

	char location_state_to_char(location_state s, Player& p1, Player& p2);
};

#endif
