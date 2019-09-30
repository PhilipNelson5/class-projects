#include "WhileStatementNode.hpp"

#include "SymbolTable.hpp"

WhileStatementNode::WhileStatementNode(ExpressionNode*& expr,
                                       ListNode<StatementNode>*& statements)
  : expr(expr)
  , statements(ListNode<StatementNode>::makeVector(statements))
{}

void WhileStatementNode::emitSource(std::string indent)
{
  std::cout << indent << "while ";
  expr->emitSource("");
  std::cout << " do\n";
  for (auto&& statement : statements)
  {
    statement->emitSource(indent + "  ");
  }
  std::cout << indent << "end;\n";
}

void WhileStatementNode::emit()
{
  std::cout << "# while ";
  expr->emitSource("");
  std::cout << '\n';

  auto lblStart = symbol_table.newLabel("start");
  auto lblEnd = symbol_table.newLabel("end");

  std::cout << lblStart << ":\n";
  auto r_expr = expr->emit().getTheeIntoARegister();

  fmt::print("beq {}, $0, {}\n", r_expr, lblEnd);

  for (auto&& statement : statements)
  {
    statement->emit();
  }

  fmt::print("j {}\n", lblStart);

  std::cout << lblEnd << ":\n\n";
}
