#ifndef LADDER_HPP
#define LADDER_HPP

#include <iostream>
#include <string>
#include <vector>

struct Ladder
{
	Ladder(std::vector<std::string> const & l, int s) : ladder(l), score(s) {}
	std::vector<std::string> ladder;
	int score;

	friend bool operator< (Ladder const & a, Ladder const & b)
	{
		return a.score < b.score;
	}

	friend bool operator<= (Ladder const & a, Ladder const & b)
	{
		return a.score <= b.score;
	}

	friend bool operator> (Ladder const & a, Ladder const & b)
	{
		return b < a;
	}

	friend bool operator>= (Ladder const & a, Ladder const & b)
	{
		return b <= a;
	}

	friend bool operator== (Ladder const & a, Ladder const & b)
	{
		return a.score == b.score;
	}

	friend bool operator!= (Ladder const & a, Ladder const & b)
	{
		return !(a == b);
	}

	friend std::ostream& operator<< (std::ostream& o, Ladder const & l) {
		for (auto & e : l.ladder)
			o << "[" << e << "]\n";
		std::cout << l.score << std::endl;

		return o;
	}
};

#endif
