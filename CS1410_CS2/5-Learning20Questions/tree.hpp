#ifndef TREE_HPP
#define TREE_HPP

#include <iostream>
#include <memory>
#include <string>

bool isTrue(std::string);

struct Node
{
  Node(std::string d) : data(d) {}
  virtual ~Node() = default;
  virtual void doWork(std::shared_ptr<Node> &) = 0;
  virtual void print(std::ostream &) = 0;
  std::string data;
};

struct Question : Node
{
  Question(std::string d, std::shared_ptr<Node> y = nullptr, std::shared_ptr<Node> n = nullptr)
    : Node(d), yes(y), no(n)
  {
  }

  void doWork(std::shared_ptr<Node> &) override;
  void print(std::ostream &) override;
  std::shared_ptr<Node> yes;
  std::shared_ptr<Node> no;
};

struct Answer : Node
{
  Answer(std::string d) : Node(d) {}
  void doWork(std::shared_ptr<Node> &) override;
  void print(std::ostream &) override;
};

class Tree
{
public:
  Tree() : root(nullptr){};
  Tree(std::string s) : root(std::make_shared<Answer>(s)) {}
  void start() { root->doWork(root); }
  void print(std::ostream &o) { root->print(o); }
  void read(std::istream &);

private:
  std::shared_ptr<Node> root;
};

#endif
