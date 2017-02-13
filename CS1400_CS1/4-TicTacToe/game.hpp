#ifndef GAME_HPP
#define GAME_HPP
#include <string>
#include "player.hpp"

bool ask(std::string q);
void game(Player p1, Player p2);
bool next_player();
Player make_userdef_player(std::string player_, location_state state);
Player make_player(std::string player_name, char mark, location_state state);

#endif