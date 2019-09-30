#include "EqualExpressionNode.hpp"

#include "RegisterPool.hpp" // for operator<<, Register
#include "Type.hpp"         // for BooleanType

#include <iostream> // for operator<<, ostream, cout, basic_ostream

EqualExpressionNode::EqualExpressionNode(ExpressionNode*& lhs, ExpressionNode*& rhs)
  : ExpressionNode(BooleanType::get())
  , lhs(lhs)
  , rhs(rhs)
{}

bool EqualExpressionNode::isConstant() const
{
  return lhs->isConstant() && rhs->isConstant();
}

std::variant<std::monostate, int, char, bool> EqualExpressionNode::eval() const
{
  auto var_lhs = lhs->eval();
  auto var_rhs = rhs->eval();

  if (var_lhs.index() != var_rhs.index())
  {
    LOG(ERROR) << fmt::format(
      "Type mismatch, can not compare {} = {}", lhs->getType()->name(), rhs->getType()->name());
    exit(EXIT_FAILURE);
  }
  if ((var_lhs.index() == 0) || (var_rhs.index() == 0))
  {
    return {};
  }
  if (std::holds_alternative<int>(var_lhs))
  {
    return std::get<int>(var_lhs) == std::get<int>(var_rhs);
  }
  if (std::holds_alternative<char>(var_lhs))
  {
    return std::get<char>(var_lhs) == std::get<char>(var_rhs);
  }
  if (std::holds_alternative<bool>(var_lhs))
  {
    return std::get<bool>(var_lhs) == std::get<bool>(var_rhs);
  }
  return {};
}

void EqualExpressionNode::emitSource(std::string indent)
{
  std::cout << indent;
  lhs->emitSource("");
  std::cout << "=";
  rhs->emitSource("");
}

Value EqualExpressionNode::emit()
{
  std::cout << "# ";
  emitSource("");
  std::cout << '\n';

  auto v_lhs = lhs->emit();
  auto v_rhs = rhs->emit();
  auto r_lhs = v_lhs.getTheeIntoARegister();
  auto r_rhs = v_rhs.getTheeIntoARegister();
  RegisterPool::Register result;

  fmt::print("seq {}, {}, {}\n", result, r_lhs, r_rhs);

  return result;
}
