#include "SubscriptOperatorNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "ExpressionNode.hpp"        // for ExpressionNode
#include "RegisterPool.hpp"          // for operator<<, Register
#include "Type.hpp"                  // for ArrayType, Type
#include "cout_redirect.hpp"
#include "log/easylogging++.h" // for Writer, CERROR, LOG

#include <iostream> // for cout
#include <sstream>  // for operator<<, ostream, basic_ostream
#include <stdlib.h> // for exit, EXIT_FAILURE
#include <utility>  // for move, pair
#include <variant>  // for get

std::string SubscriptOperatorNode::getId() const
{
  std::stringstream s_expr;
  { // remporarily redirect stdout to the s_expr stringstream
    cout_redirect redirect(s_expr.rdbuf());
    expr->emitSource("");
  }
  return fmt::format("{}[{}]", lValue->getId(), s_expr.str());
}

std::shared_ptr<Type> getArrayType(std::shared_ptr<LvalueNode> lValue)
{
  if (ArrayType* array = dynamic_cast<ArrayType*>(lValue->getType().get()))
  {
    return array->getElementType();
  }

  LOG(ERROR) << lValue->getId() << " is not an array type, can not use subscript operator[]";
  exit(EXIT_FAILURE);
}

SubscriptOperatorNode::SubscriptOperatorNode(LvalueNode* lValue, ExpressionNode* expr)
  : LvalueNode()
  , lValue(std::shared_ptr<LvalueNode>(lValue))
  , expr(std::shared_ptr<ExpressionNode>(expr))
{}

bool SubscriptOperatorNode::isConstant() const
{
  return lValue->isConstant();
}

const std::shared_ptr<Type> SubscriptOperatorNode::getType()
{
  if (type == nullptr)
  {
    type = getArrayType(lValue);
  }
  return type;
}

std::variant<std::monostate, int, char, bool> SubscriptOperatorNode::eval() const
{
  return {};
}

void SubscriptOperatorNode::emitSource(std::string indent)
{
  (void)indent;
  lValue->emitSource(indent);
  std::cout << '[';
  expr->emitSource("");
  std::cout << ']';
}

Value SubscriptOperatorNode::emit()
{
  if (ArrayType* array = dynamic_cast<ArrayType*>(lValue->getType().get()))
  {
    auto v_lval = lValue->emit();

    std::cout << "# ";
    emitSource("");
    std::cout << '\n';

    auto r_index = expr->emit().getTheeIntoARegister();
    RegisterPool::Register elementSize;

    fmt::print("addi {0}, {0}, {1}", r_index, -array->getlb());
    std::cout << " # adjust the index ( indx - lb )\n";

    fmt::print("li {}, {}", elementSize, array->getElementType()->size());
    std::cout << " # get array element size\n";

    fmt::print("mult {}, {}\n", r_index, elementSize);
    fmt::print("mflo {}", r_index);
    std::cout << " # multiply the number of elements by the element size\n";

    if (v_lval.isLvalue())
    {
      auto [offset, memoryLocation] = std::get<std::pair<int, int>>(v_lval.value);

      fmt::print("addi {0}, {0}, {1}", r_index, offset);
      std::cout << " # add the offset to the index\n";

      fmt::print("add {0}, {0}, ${1}", r_index, memoryLocation);
      std::cout << " # add the memory location to the index\n";

      return {std::move(r_index), Value::RegisterIs::ADDRESS};
    }
    if (v_lval.isRegister())
    {
      auto r_lval = v_lval.getRegister();

      fmt::print("add {0}, {0}, {1}", r_index, r_lval);
      std::cout << " # add the index to the memory address\n";

      return {std::move(r_index), Value::RegisterIs::ADDRESS};
    }
    LOG(ERROR) << lValue->getId() << " is not an Lvalue, no memory associated";
    exit(EXIT_FAILURE);
  }
  else
  {
    LOG(ERROR) << lValue->getId() << " is not an array type";
    exit(EXIT_FAILURE);
  }
}
