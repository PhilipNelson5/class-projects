#include "UnaryMinusNode.hpp"

#include "Type.hpp"

#include <iostream>

UnaryMinusNode::UnaryMinusNode(ExpressionNode*& right)
  : ExpressionNode(IntegerType::get())
  , rhs(right)
{}

bool UnaryMinusNode::isConstant() const
{
  return rhs->isConstant();
}

std::variant<std::monostate, int, char, bool> UnaryMinusNode::eval() const
{
  auto var_rhs = rhs->eval();

  if (var_rhs.index() == 0)
  {
    return {};
  }
  if (std::holds_alternative<int>(var_rhs))
  {
    return -std::get<int>(var_rhs);
  }
  if (std::holds_alternative<char>(var_rhs))
  {
    return -std::get<char>(var_rhs);
  }
  if (std::holds_alternative<bool>(var_rhs))
  {
    return -std::get<bool>(var_rhs);
  }
  return {};
}

void UnaryMinusNode::emitSource(std::string indent)
{
  std::cout << indent;
  std::cout << "-";
  rhs->emitSource("");
}

Value UnaryMinusNode::emit()
{
  auto v_rhs = rhs->emit();
  auto r_rhs = v_rhs.getTheeIntoARegister();
  RegisterPool::Register result;

  fmt::print("subu {}, $0, {}\n", result, r_rhs);

  return result;
}
