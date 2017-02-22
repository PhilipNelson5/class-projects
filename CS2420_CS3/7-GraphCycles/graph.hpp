#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <deque>
#include <iostream>
#include <string>
#include <vector>

#include "edge.hpp"
#include "node.hpp"

class Graph
{
	public:
		Graph(std::string f) : numVert(0), numEdge(0), tourAvail(true) { init(f); }
		std::string toString() const;
		std::string printAdj() const;
		std::string printEdges() const;
		std::string printVertDeg() const;
		std::string printAdjList() const;
		std::string printTour() const;
		std::string printTourFormatted() const;
		bool isConnected();
		std::vector<int> tour;

	private:
		int numVert;//# of vertices in the graph
		int numEdge;//# of edges in the graph
		bool tourAvail;//true if a tour is possible
		std::vector<Edge> edges;//array of edges
		std::vector<Node> nodes;//array of nodes
		std::vector<std::vector<int>> adj;//adjacency matrix

		void init(std::string);//creates the adjacency matrix and vector of edges
		void buildGraph(std::ifstream &);
		bool checkVert();
		void buildAdjList();
		void addSuccessors(std::deque<Node> &, int &);
		void findCycles();
		void cycle(int, int, int, int, int);
		int getNextEdge(int, int);
		void mergeCycles();
		void mergeCycles(std::deque<int> &, int, int, int);
		int getNextTourEdge(std::deque<int> &, int, int, int);
		int getPriority(std::deque<int> &, int, int) const;
};

#endif
