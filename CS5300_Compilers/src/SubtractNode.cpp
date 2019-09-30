#include "SubtractNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "IntegerLiteralNode.hpp"    // for IntegerLiteralNode
#include "RegisterPool.hpp"          // for operator<<, Register
#include "Type.hpp"                  // for IntegerType, Type
#include "log/easylogging++.h"       // for Writer, CERROR, LOG

#include <iostream> // for operator<<, ostream, cout, basic_o...

SubtractNode::SubtractNode(ExpressionNode*& left, ExpressionNode*& right)
  : ExpressionNode(IntegerType::get())
  , lhs(left)
  , rhs(right)
{}

bool SubtractNode::isConstant() const
{
  return lhs->isConstant() && rhs->isConstant();
}

std::variant<std::monostate, int, char, bool> SubtractNode::eval() const
{
  auto var_lhs = lhs->eval();
  auto var_rhs = rhs->eval();

  if (var_lhs.index() != var_rhs.index())
  {
    LOG(ERROR) << fmt::format("mismatched types in subtract expression: {} and {}",
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
    return std::get<int>(var_lhs) - std::get<int>(var_rhs);
  }

  LOG(ERROR) << "can not subtract non integer types";
  exit(EXIT_FAILURE);
}

void SubtractNode::emitSource(std::string indent)
{
  std::cout << indent;
  lhs->emitSource("");
  std::cout << "-";
  rhs->emitSource("");
}

Value SubtractNode::emit()
{
  if (lhs->getType() != rhs->getType())
  {
    LOG(ERROR) << fmt::format("mismatched types in subtract expression: {} and {}",
                              lhs->getType()->name(),
                              rhs->getType()->name());
    exit(EXIT_FAILURE);
  }

  if (lhs->getType() != IntegerType::get())
  {
    LOG(ERROR) << "can not subtract non integer types";
    exit(EXIT_FAILURE);
  }

  std::cout << "# ";
  emitSource("");
  std::cout << '\n';

  if (lhs->isConstant())
  {
    auto lhs_const = dynamic_cast<IntegerLiteralNode*>(lhs.get());
    auto v_rhs = rhs->emit();
    auto r_rhs = v_rhs.getTheeIntoARegister();
    RegisterPool::Register result;

    fmt::print("addi {}, {}, {} # ", result, r_rhs, -lhs_const->value);
    emitSource("");
    std::cout << '\n';

    return result;
  }
  else if (rhs->isConstant())
  {
    auto rhs_const = dynamic_cast<IntegerLiteralNode*>(rhs.get());
    auto v_lhs = lhs->emit();
    auto r_lhs = v_lhs.getTheeIntoARegister();
    RegisterPool::Register result;

    fmt::print("addi {}, {}, {} # ", result, r_lhs, -rhs_const->value);
    emitSource("");
    std::cout << '\n';

    return result;
  }
  else
  {
    auto v_lhs = lhs->emit();
    auto v_rhs = rhs->emit();
    auto r_lhs = v_lhs.getTheeIntoARegister();
    auto r_rhs = v_rhs.getTheeIntoARegister();
    RegisterPool::Register result;

    fmt::print("sub {}, {}, {} # ", result, r_lhs, r_rhs);
    emitSource("");
    std::cout << '\n';

    return result;
  }
}
