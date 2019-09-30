#include "SuccessorExpressionNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "SymbolTable.hpp"
#include "Type.hpp"
#include "log/easylogging++.h"

#include <iostream>

SuccessorExpressionNode::SuccessorExpressionNode(ExpressionNode*& expr)
  : ExpressionNode()
  , expr(expr)
{}

bool SuccessorExpressionNode::isConstant() const
{
  return expr->isConstant();
}

const std::shared_ptr<Type> SuccessorExpressionNode::getType()
{
  if (type == nullptr)
  {
    type = expr->getType();
  }
  return type;
}

std::variant<std::monostate, int, char, bool> SuccessorExpressionNode::eval() const
{
  auto var_expr = expr->eval();

  if (var_expr.index() == 0)
  {
    return {};
  }
  if (std::holds_alternative<int>(var_expr))
  {
    return std::get<int>(var_expr) + 1;
  }
  if (std::holds_alternative<char>(var_expr))
  {
    return std::get<char>(var_expr) + 1;
  }
  if (std::holds_alternative<bool>(var_expr))
  {
    return !std::get<bool>(var_expr);
  }
  return {};
}

void SuccessorExpressionNode::emitSource(std::string indent)
{
  std::cout << indent << "succ(";
  expr->emitSource("");
  std::cout << ")";
}

Value SuccessorExpressionNode::emit()
{
  std::cout << "# ";
  emitSource("");
  std::cout << '\n';

  if (expr->getType() == BooleanType::get())
  {
    auto r_expr = expr->emit().getTheeIntoARegister();
    fmt::print("xori {}, 1\n", r_expr);
    return r_expr;
  }
  else if (expr->getType() == IntegerType::get() || expr->getType() == CharacterType::get())
  {
    auto r_expr = expr->emit().getTheeIntoARegister();
    fmt::print("addi {0}, {0}, 1\n", r_expr);
    return r_expr;
  }
  else
  {
    LOG(ERROR) << fmt::format("succ is not defined for type {}", expr->getType()->name());
    exit(EXIT_FAILURE);
  }
}
