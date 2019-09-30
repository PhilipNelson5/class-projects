#include "PredecessorExpressionNode.hpp"

#include "Type.hpp"

#include <iostream> // for operator<<, char_traits, cout, ostream, basic_os...

PredecessorExpressionNode::PredecessorExpressionNode(ExpressionNode*& expr)
  : ExpressionNode()
  , expr(expr)
{}

bool PredecessorExpressionNode::isConstant() const
{
  return expr->isConstant();
}

const std::shared_ptr<Type> PredecessorExpressionNode::getType()
{
  if (type == nullptr)
  {
    type = expr->getType();
  }
  return type;
}

std::variant<std::monostate, int, char, bool> PredecessorExpressionNode::eval() const
{
  auto var_expr = expr->eval();

  if (var_expr.index() == 0)
  {
    return {};
  }
  if (std::holds_alternative<int>(var_expr))
  {
    return std::get<int>(var_expr) - 1;
  }
  if (std::holds_alternative<char>(var_expr))
  {
    return std::get<char>(var_expr) - 1;
  }
  if (std::holds_alternative<bool>(var_expr))
  {
    return !std::get<bool>(var_expr);
  }
  return {};
}

void PredecessorExpressionNode::emitSource(std::string indent)
{
  std::cout << indent << "pred(";
  expr->emitSource("");
  std::cout << ")";
}

Value PredecessorExpressionNode::emit()
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
    fmt::print("addi {0}, {0}, -1\n", r_expr);
    return r_expr;
  }
  else
  {
    LOG(ERROR) << fmt::format("pred is not defined for type {}", expr->getType()->name());
    exit(EXIT_FAILURE);
  }
}
