#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "unionFind.hpp"

class Network
{
public:
  Network() : size(0), days(0) {}
  void simulate(int);

private:
  UnionFind network;
  int size;
  int days;

  void init();
  void makeFriend();
  void report();
};

#endif
