#ifndef UNION_FIND_HPP
#define UNION_FIND_HPP

#include <vector>

class UnionFind
{
public:
  UnionFind() : bigGroup(1), finds(0), touches(0) {}
  int getFinds() const { return finds; }
  int getTouches() const { return touches; }
  void init(int);
  bool isComplete() const;
  bool _union(int, int);
  int findH(int);
  int findS(int);

private:
  std::vector<int> arrH; // stores by height
  std::vector<int> arrS; // stores by size
  int bigGroup;
  int finds;
  int touches;
  bool unionHeight(int, int);
  bool unionSize(int, int);
};

#endif
