#ifndef EDGE_HPP
#define EDGE_HPP

#include <assert.h>
#include <sstream>

class Edge
{
  public:
    int fromNode; // Subscript of other endpoint in node array
    int toNode;   // Subscript of one endpoint in node array.  Nodes are stored as numbers, but printed as characters.
    int cycleID;  // Cycle which the edge is a member of, -1 if it is included in no cycle
    int order;
    bool used;    // true if edge is used in final tour

    Edge(char f, char t): cycleID(-1), order(-1), used(false) { set(f, t); }//constructor with vertex letters
    Edge(int f, int t): cycleID(-1), order(-1), used(false)  { set(f, t); }//constructor with vertex numbers

    // Create a string version of Edge
    // Edge endpoints are stored as numbers, but printed as characters.
    std::string toString() const
    {
      std::ostringstream os;  // allows string to act like stream to use stream operations
      char f = fromNode + 'A';
      char t = toNode + 'A';
      os << f << "-" << t << " (" << cycleID << ") (" << order << ")  ";
      return os.str();
    }

    // if one Node is an endpoint, return other endpoint
    int getOther(int oneNode) const
    {
      if (fromNode == oneNode) return toNode;
      assert(toNode == oneNode);
      return fromNode;
    }

    // Set initial values of an edge from Node f to Node t (char)
    void set(char f, char t)
    {
      fromNode = f - 'A';
      toNode = t - 'A';
      cycleID = -1;
      used = false;
      //std::cout << "creating Edge " << toString() << std::endl;
    }

    // Set initial values of an edge from Node f to Node t (int)
    void set(int f, int t)
    {
      fromNode = f;
      toNode = t;
      cycleID = -1;

      used = false;
      //std::cout << "creating Edge " << toString() << std::endl;
    }

    friend bool operator== (Edge const & a, Edge const & b)
    {
      return (a.fromNode == b.fromNode && a.toNode == b.toNode);
    }
};

#endif
