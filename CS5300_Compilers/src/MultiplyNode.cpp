#include "MultiplyNode.hpp"

#include "../fmt/include/fmt/core.h" // for print, format
#include "RegisterPool.hpp"          // for Register
#include "Type.hpp"                  // for IntegerType, Type
#include "log/easylogging++.h"       // for Writer, CERROR, LOG

#include <iostream> // for operator<<, cout, ostream

MultiplyNode::MultiplyNode(ExpressionNode*& left, ExpressionNode*& right)
  : ExpressionNode(IntegerType::get())
  , lhs(left)
  , rhs(right)
{}

bool MultiplyNode::isConstant() const
{
  return lhs->isConstant() && rhs->isConstant();
}

void MultiplyNode::emitSource(std::string indent)
{
  std::cout << indent;
  lhs->emitSource("");
  std::cout << "*";
  rhs->emitSource("");
}

std::variant<std::monostate, int, char, bool> MultiplyNode::eval() const
{
  auto var_lhs = lhs->eval();
  auto var_rhs = rhs->eval();

  if (var_lhs.index() != var_rhs.index())
  {
    LOG(ERROR) << fmt::format("mismatched types in multiply expression: {} and {}",
                              lhs->getType()->name(),
                              rhs->getType()->name());
    exit(EXIT_FAILURE);
  }
  if ((var_lhs.index() == 0) || (var_rhs.index() == 0))
  {
    return {};
  }
  if (std::holds_alternative<int>(var_lhs))
  {
    return std::get<int>(var_lhs) * std::get<int>(var_rhs);
  }

  LOG(ERROR) << "can not multiply non integer types";
  exit(EXIT_FAILURE);
}

Value MultiplyNode::emit()
{
  if (lhs->getType() != rhs->getType())
  {
    LOG(ERROR) << fmt::format("mismatched types in multiply expression: {} and {}",
                              lhs->getType()->name(),
                              rhs->getType()->name());
    exit(EXIT_FAILURE);
  }

  if (lhs->getType() != IntegerType::get())
  {
    LOG(ERROR) << "can not multiply non integer types";
    exit(EXIT_FAILURE);
  }

  std::cout << "# ";
  emitSource("");
  std::cout << '\n';

  auto v_lhs = lhs->emit();
  auto v_rhs = rhs->emit();
  auto r_lhs = v_lhs.getTheeIntoARegister();
  auto r_rhs = v_rhs.getTheeIntoARegister();
  RegisterPool::Register result;

  fmt::print("mult {}, {}\n", r_lhs, r_rhs);
  fmt::print("mflo {} # ", result);
  emitSource("");
  std::cout << '\n';

  return result;
}
