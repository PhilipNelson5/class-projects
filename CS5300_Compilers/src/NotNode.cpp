#include "NotNode.hpp"

#include "Type.hpp" // for IntegerType

#include <iostream> // for operator<<, cout, ostream

NotNode::NotNode(ExpressionNode*& right)
  : ExpressionNode(IntegerType::get())
  , rhs(right)
{}

bool NotNode::isConstant() const
{
  return rhs->isConstant();
}

std::variant<std::monostate, int, char, bool> NotNode::eval() const
{
  auto var_rhs = rhs->eval();

  if (var_rhs.index() == 0)
  {
    return {};
  }
  if (std::holds_alternative<int>(var_rhs))
  {
    return !std::get<int>(var_rhs);
  }
  if (std::holds_alternative<char>(var_rhs))
  {
    return !std::get<char>(var_rhs);
  }
  if (std::holds_alternative<bool>(var_rhs))
  {
    return !std::get<bool>(var_rhs);
  }
  return {};
}

void NotNode::emitSource(std::string indent)
{
  std::cout << indent;
  std::cout << "~";
  rhs->emitSource("");
}

Value NotNode::emit()
{
  auto v_rhs = rhs->emit();
  auto r_rhs = v_rhs.getTheeIntoARegister();
  RegisterPool::Register result;

  fmt::print("not {}, {}\n", result, r_rhs);

  return result;
}
