#include "game.hpp"

int main(void)
{
	Player p1 = make_player("Player1", 'X', location_state::ONE);
	Player p2 = make_player("Player2", 'O', location_state::TWO);
	//Player p1 = make_userdef_player("Player 1", location_state::ONE);
	//Player p2 = make_userdef_player("Player 2", location_state::TWO);
	while (ask("Do you want to play TicTacToe?")){
		game(p1, p2);
	}
}
