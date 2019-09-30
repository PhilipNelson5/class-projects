#include "IfStatementNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "ExpressionNode.hpp"        // for ExpressionNode
#include "ListNode.hpp"              // for ListNode
#include "SymbolTable.hpp"           // for SymbolTable, symbol_table

#include <iostream> // for operator<<, cout, ostream, basi...

std::vector<IfStatement> makeElseIfStatements(ListNode<IfStatement>*& elseIfStatements)
{
  if (elseIfStatements == nullptr)
  {
    return {};
  }
  std::vector<IfStatement> list;
  for (auto cur = std::shared_ptr<ListNode<IfStatement>>(elseIfStatements); cur != nullptr;
       cur = cur->next)
  {
    list.emplace_back(cur->data->first, cur->data->second);
  }
  return list;
}

IfStatementNode::IfStatementNode(ExpressionNode*& ifExpr,
                                 ListNode<StatementNode>*& ifStatements,
                                 ListNode<IfStatement>*& elseIfStatements,
                                 ListNode<StatementNode>*& elseStatement)
  : ifStatement(std::make_pair<std::shared_ptr<ExpressionNode>,
                               std::vector<std::shared_ptr<StatementNode>>>(
      std::shared_ptr<ExpressionNode>(ifExpr),
      ListNode<StatementNode>::makeVector(ifStatements)))
  , elseIfStatements(makeElseIfStatements(elseIfStatements))
  , elseStatement(elseStatement != nullptr ? ListNode<StatementNode>::makeVector(elseStatement)
                                           : std::vector<std::shared_ptr<StatementNode>>())
{}

void IfStatementNode::emitSource(std::string indent)
{
  // --
  // if
  // --
  std::cout << indent << "if ";
  ifStatement.first->emitSource("");
  std::cout << " then\n";
  for (auto&& statement : ifStatement.second)
  {
    statement->emitSource(indent + "  ");
  }

  // -------
  // else if
  // -------

  for (auto&& elseIfStatement : elseIfStatements)
  {
    std::cout << indent << "elseif ";
    elseIfStatement.first->emitSource("");
    std::cout << " then\n";
    for (auto&& statement : elseIfStatement.second)
    {
      statement->emitSource(indent + "  ");
    }
  }

  // ----
  // else
  // ----
  if (elseStatement.size() != 0)
  {
    std::cout << indent << "else\n";
    for (auto&& statement : elseStatement)
    {
      statement->emitSource(indent + "  ");
    }
  }

  // ---
  // end
  // ---
  std::cout << indent << "end;\n";
}

void IfStatementNode::emit()
{
  // --
  // if
  // --
  auto lblElseN = symbol_table.newLabel("else");
  auto lblEnd = symbol_table.newLabel("end");
  auto r_ifExpr = ifStatement.first->emit().getTheeIntoARegister();
  fmt::print("beq {}, $0, {}\n", r_ifExpr, lblElseN);

  for (auto&& statement : ifStatement.second)
  {
    statement->emit();
  }

  fmt::print("j {}\n\n", lblEnd);

  // -------
  // else if
  // -------
  for (auto&& elseIfStatement : elseIfStatements)
  {
    std::cout << lblElseN << ":\n";
    lblElseN = symbol_table.newLabel("else");

    auto r_expr = elseIfStatement.first->emit().getTheeIntoARegister();
    fmt::print("beq {}, $0, {}\n", r_expr, lblElseN);
    for (auto&& statement : elseIfStatement.second)
    {
      statement->emit();
    }

    fmt::print("j {}\n\n", lblEnd);
  }

  // ----
  // else
  // ----
  std::cout << lblElseN << ":\n";
  for (auto&& statement : elseStatement)
  {
    statement->emit();
  }

  // ---
  // end
  // ---
  std::cout << lblEnd << ":\n\n";
}
