#include "RepeatStatementNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "SymbolTable.hpp"

#include <iostream>

RepeatStatementNode::RepeatStatementNode(ListNode<StatementNode>*& statements,
                                         ExpressionNode*& expr)
  : statements(ListNode<StatementNode>::makeVector(statements))
  , expr(expr)
{}


void RepeatStatementNode::emitSource(std::string indent)
{
  std::cout << indent << "repeat" << '\n';

  for (auto&& statement : statements)
  {
    statement->emitSource(indent + "  ");
  }

  std::cout << indent << "until ";
  expr->emitSource("");
  std::cout << "\n\n";
}

void RepeatStatementNode::emit()
{
  std::cout << "# repeat\n";
  auto lblStart = symbol_table.newLabel("start");

  std::cout << lblStart << ":\n";
  for (auto&& statement : statements)
  {
    statement->emit();
  }

  auto r_expr = expr->emit().getTheeIntoARegister();

  fmt::print("beq {}, $0, {}", r_expr, lblStart);
}
