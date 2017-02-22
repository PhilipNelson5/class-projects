#ifndef NODE_HPP
#define NODE_HPP

#include <string>
#include <vector>
#include "edge.hpp"

struct Node
{
	Node(int id = -1) : ID(id), visited(false) {};
	int ID;
	std::vector<Edge> adj;//adjacent list
	bool visited;
};

#endif
