#include "player.hpp"
#include <string>

Player::Player(std::string name, char mark, location_state state) : name(name), mark(mark), state(state){};

//getters
std::string  Player::get_name(){ return name; }
char Player::get_mark(){ return mark; }
location_state Player::get_state(){ return state; }

