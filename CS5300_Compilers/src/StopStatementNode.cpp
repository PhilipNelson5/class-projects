#include "StopStatementNode.hpp"

#include <iostream> // for operator<<, basic_ostream, char_traits, cout

StopStatementNode::StopStatementNode() {}

void StopStatementNode::emitSource(std::string indent)
{
  std::cout << indent << "stop;" << '\n';
}

void StopStatementNode::emit()
{
  std::cout << "li $v0, 10"
            << " # load exit instruction" << '\n';
  std::cout << "syscall" << '\n';
}
