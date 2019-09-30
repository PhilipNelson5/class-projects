#include "AddNode.hpp"

#include "../fmt/include/fmt/core.h" // for print, format
#include "IntegerLiteralNode.hpp"    // for IntegerLiteralNode
#include "RegisterPool.hpp"          // for Register
#include "Type.hpp"                  // for IntegerType, Type, CharacterType
#include "log/easylogging++.h"       // for Writer, CERROR, LOG

#include <iostream> // for operator<<, cout, ostream

AddNode::AddNode(ExpressionNode*& left, ExpressionNode*& right)
  : ExpressionNode(IntegerType::get())
  , lhs(left)
  , rhs(right)
{}

bool AddNode::isConstant() const
{
  return lhs->isConstant() && rhs->isConstant();
}

std::variant<std::monostate, int, char, bool> AddNode::eval() const
{
  auto var_lhs = lhs->eval();
  auto var_rhs = rhs->eval();

  if (var_lhs.index() != var_rhs.index())
  {
    LOG(ERROR) << fmt::format("mismatched types in add expression: {} and {}",
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
    return std::get<int>(var_lhs) + std::get<int>(var_rhs);
  }
  if (std::holds_alternative<char>(var_lhs))
  {
    return std::get<char>(var_lhs) + std::get<char>(var_rhs);
  }
  if (std::holds_alternative<bool>(var_lhs))
  {
    return std::get<bool>(var_lhs) + std::get<bool>(var_rhs);
  }
  return {};
}

void AddNode::emitSource(std::string indent)
{
  std::cout << indent;
  lhs->emitSource("");
  std::cout << "+";
  rhs->emitSource("");
}

Value AddNode::emit()
{
  if (lhs->getType() != rhs->getType())
  {
    LOG(ERROR) << fmt::format("mismatched types in add expression: {} and {}",
                              lhs->getType()->name(),
                              rhs->getType()->name());
    exit(EXIT_FAILURE);
  }

  if (lhs->getType() != IntegerType::get() && lhs->getType() != CharacterType::get()
      && lhs->getType() != BooleanType::get())
  {
    LOG(ERROR) << fmt::format("can not add {}s", lhs->getType()->name());
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

    fmt::print("addi {0}, {1}, {2} # ", result, r_rhs, lhs_const->value);

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

    fmt::print("addi {0}, {1}, {2} # ", result, r_lhs, rhs_const->value);

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

    fmt::print("add {0}, {1}, {2} # ", result, r_lhs, r_rhs);

    emitSource("");
    std::cout << '\n';

    return result;
  }
}
