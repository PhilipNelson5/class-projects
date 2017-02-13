#ifndef PLAYER_HPP
#define PLAYER_HPP
#include <string>

enum class location_state{
	EMPTY, ONE, TWO
};

class Player
{
public:
	Player(std::string name, char mark, location_state state);
	std::string get_name();
	char get_mark();
	location_state get_state();

private:
	std::string name;
	char mark;
	location_state state;
};

#endif
