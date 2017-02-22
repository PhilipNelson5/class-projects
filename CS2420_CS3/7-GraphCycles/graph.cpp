#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include "graph.hpp"

// convert char to int
int toInt (char c) { return c - 'A'; }

//convert int to char
char toChar(int i) { return i + 'A'; }

//print all information from graph
std::string Graph::toString() const
{
	std::ostringstream oss;

	oss
		<< printAdj()
		<< printEdges()
		<< printVertDeg()
		//<< printAdjList()
		;

	oss << std::endl;
	return oss.str();
}

//print adjacency matrix
std::string Graph::printAdj() const
{
	std::ostringstream oss;

	oss << std::endl << "[Adjacency Matrix]" << std::endl;

	if(numEdge < 10)
		for(auto && r : adj)
		{
			for (auto && e : r)
			{
				if(e == -1)
					oss << ". ";
				else
					oss << e << " ";
			}
			oss << std::endl;
		}

	else
		for(auto && r : adj)
		{
			for (auto && e : r)
			{
				if(e == -1)
					oss << " . ";
				else
					if(e < 10)
						oss << " " << e << " ";
					else oss << e << " ";
			}
			oss << std::endl;
		}
	return oss.str();
}

//print edges
std::string Graph::printEdges() const
{
	std::ostringstream oss;

	oss << std::endl << "[Edges]" << std::endl;
	for(auto && e : edges)
		oss << e.toString() << std::endl;

	oss << std::endl;
	return oss.str();
}

//print vertex degrees
std::string Graph::printVertDeg() const
{
	std::ostringstream oss;

	oss << std::endl << "[Vertex Degree]" << std::endl;
	int sum;
	for(int i = 0; i < numVert; ++i)
	{
		sum = 0;
		for(int j = 0; j < numVert; ++j)
			if(adj[i][j] != -1)
				++sum;

		oss << "vert: " << toChar(i) << " - " << sum << std::endl;
	}

	oss << std::endl;
	return oss.str();
}

//print adjacency list for each node
std::string Graph::printAdjList() const
{
	std::ostringstream oss;

	oss << std::endl << "[Adjacency Lists]" << std::endl;
	for(auto && n : nodes)
	{
		oss << static_cast<char>(n.ID + 'A') << ": ";
		for(auto && e : n.adj)
			oss << e.toString() << ", ";
		oss << std::endl;
	}

	oss << std::endl;
	return oss.str();
}

//print final tour
std::string Graph::printTour() const
{
	std::ostringstream oss;
	int n = edges[tour[0]].fromNode;
	for(auto && e : tour)
	{	
		oss << toChar(n) << " ";
		n = edges[e].getOther(n);
	}
	return oss.str();
}

//prints the final tour in a border
std::string Graph::printTourFormatted() const
{
	std::ostringstream oss;
	std::string t= printTour();
	std::string line(t.length()+5, '-');

	oss << "[Euler Tour]" << std::endl
		<< line << "\n-> " << t<< "<-\n" << line;

	return oss.str();
}

//find cycles in the graph
void Graph::findCycles()
{
	std::cout << "finding cycles..." << std::endl;
	int numCycles = 0;
	for(unsigned int i = 0; i < edges.size(); ++i)
		if(edges[i].cycleID == -1)
			cycle(i, i, edges[i].toNode, ++numCycles, 0);
	std::cout << std::endl << numCycles << " Cycles" << std::endl;
}

//recursively find a single cycle
void Graph::cycle(int startE, int currE, int currN, int ID, int o)
{
	if(startE == currE && o!=0) return;

	edges[currE].order = o;
	edges[currE].cycleID = ID;

	std::cout << edges[currE].toString() << std::endl;

	int nextE = getNextEdge(startE, currN);

	cycle(startE, nextE, edges[nextE].getOther(currN), ID, ++o);
}

//find a legal next edge to follow
int Graph::getNextEdge(int startE, int currN)
{
	for(auto && e : adj[currN])
		if(e != -1 && edges[e].cycleID == -1)
			return e;

	return startE;
}

/*        [Priority]
 * 1) edge from a new cycle
 * 2) *edge from current cycle
 * 3) *edge from old cycle
 */

//gets the priority of an edge
int Graph::getPriority(std::deque<int> & q, int currE, int nextE) const
{
	if(edges[nextE].cycleID == edges[currE].cycleID)
		return 2;
	for(auto && c : q)
		if(c == edges[nextE].cycleID)
			return 3;
	return 1;
}

//gets the next edge in the tour
int Graph::getNextTourEdge(std::deque<int> & q, int startE, int currE, int currN)
{
	int bestPrio = 4;
	int bestEdge = -1;
	int prio;

	for(auto && e : adj[currN])
		if(e != -1 && !edges[e].used)
		{
			prio = getPriority(q, currE, e);
			if(prio == 1)
			{
				q.push_front(edges[e].cycleID);
				return e;
			}
			else if(prio < bestPrio)
			{
				bestPrio = prio;
				bestEdge = e;
			}
		}

	if(bestEdge != -1)
		return bestEdge;
	return startE;
}

//merges the cycles recursively
void Graph::mergeCycles(std::deque<int> & q, int startE, int currE, int currN) //TODO
{
	int nextE = getNextTourEdge(q, startE, currE, currN);
	if(nextE == startE && tour.size() > 1) return;
	tour.push_back(nextE);
	edges[nextE].used = true;
	mergeCycles(q, startE, nextE, edges[nextE].getOther(currN));
}

//begins the merging process
void Graph::mergeCycles()
{
	std::deque<int> cycleQ;
	cycleQ.push_front(edges[0].cycleID);

	tour.push_back(0);
	edges[0].used = true;

	mergeCycles(cycleQ, 0, 0, edges[0].toNode);
}

//check ifstream to ensure file exists
bool checkStream(std::ifstream & fin)
{
	if(!fin)
	{
		std::cout << "...[BAD FILE]..." << std::endl;
		return false;
	}

	return true;
}

//create an empty matrix size [m x n]
void emptyMatrix(std::vector<std::vector<int>> & v, int m, int n, int val)
{
	v.resize(m);
	for(auto && r : v)
	{
		r.resize(n);
		for(auto && e : r)
		{
			e = val;
		}
	}
}

//build adjacency matrix, edge list and node list;
void Graph::buildGraph(std::ifstream & fin)
{
	int ct = 0, vert1 = 0, vert2 = 0;
	char vert;

	for(int i = 0; i < numEdge; ++i)
	{
		fin >> vert;
		vert1 = toInt(vert);
		fin >> vert;
		vert2 = toInt(vert);

		adj[vert1][vert2] = ct;
		adj[vert2][vert1] = ct;

		edges.emplace_back(vert1, vert2);
		++ct;
	}

	for(int i = 0; i < numVert; ++i)
		nodes.emplace_back(i);
}

//check degree of vertices for odd degree (impossibility of Euler tour)
bool Graph::checkVert()
{
	int sum;
	for(int i = 0; i < numVert; ++i)
	{
		sum = 0;
		for(int j = 0; j < numVert; ++j)
			if(adj[i][j] != -1)
				++sum;

		if(sum % 2 != 0)
		{
			std::cout << "No Euler Tour Available" << std::endl;
			return false;
		}
	}
	return true;
}

//build adjacency list
void Graph::buildAdjList()
{
	for(auto && e : edges)
	{
		nodes[e.fromNode].adj.push_back(e);	
		nodes[e.toNode].adj.push_back(e);	
	}
}

//create graph and find cycles
void Graph::init(std::string file)
{
	int num;
	std::ifstream fin(file);

	//check ifstream
	if(!checkStream(fin)) { tourAvail = false; return; }

	fin >> num;
	numVert = num;

	fin >> num;
	numEdge = num;

	//create an empty matrix size [numVert x numVert]
	emptyMatrix(adj, numVert, numVert, -1);

	//build adjacency matrix, edge list and node list
	buildGraph(fin);

	//check degree of vertices for odd degree (impossibility of Euler tour)
	if(!checkVert()) { tourAvail = false; return; }

	//build adjacency list
	buildAdjList();

	//find the cycles in the graph
	findCycles();

	//merge cycles into an Euler Tour
	mergeCycles();

	std::cout << printTourFormatted() << std::endl;
}

//checks graph for connectivity
bool Graph::isConnected()
{
	std::deque<Node> queue;
	int vertCt = 0;

	queue.push_back(nodes[0]);

	while(!queue.empty())
		addSuccessors(queue, vertCt);

	return vertCt == numVert;
}

//adds all successors of a node to the queue
void Graph::addSuccessors(std::deque<Node> & queue, int & vertCt)
{
	Node n = queue.front();
	queue.pop_front();

	nodes[n.ID].visited = true;
	++vertCt;

	for(auto && e : n.adj)
	{
		if(!nodes[e.toNode].visited && n.ID != e.toNode)
		{
			queue.push_back(nodes[e.toNode]);
			nodes[e.toNode].visited = true;
		}

		if(!nodes[e.fromNode].visited && n.ID != e.fromNode)
		{
			queue.push_back(nodes[e.fromNode]);
			nodes[e.fromNode].visited = true;
		}
	}
}
