#include "AssignmentStatementNode.hpp"

#include "ExpressionNode.hpp"  // for ExpressionNode
#include "LvalueNode.hpp"      // for LvalueNode
#include "RegisterPool.hpp"    // for operator<<, Register
#include "Type.hpp"            // for RecordType, ArrayType
#include "log/easylogging++.h" // for Writer, CERROR, LOG

#include <iostream> // for operator<<, ostream, basic_ostream
#include <stdlib.h> // for exit, EXIT_FAILURE
#include <utility>  // for pair
#include <variant>  // for get

AssignmentStatementNode::AssignmentStatementNode(LvalueNode*& identifier, ExpressionNode* expr)
  : identifier(identifier)
  , expr(expr)
{}

void AssignmentStatementNode::emitSource(std::string indent)
{
  std::cout << indent;
  identifier->emitSource("");
  std::cout << " := ";
  expr->emitSource("");
  std::cout << ";\n";
}

void AssignmentStatementNode::emit()
{
  std::cout << "\n# ";
  emitSource("");

  auto v_id = identifier->emit();
  auto v_expr = expr->emit();
  // auto size = expr->getType()->size();
  // assign(v_expr, v_id, size);
  if (v_id.isLvalue())
  {
    if (ArrayType* array = dynamic_cast<ArrayType*>(expr->getType().get()))
    {
      array->init();
      auto size = array->size();
      auto r_expr = v_expr.getRegister();
      auto [offset, memoryLocation] = std::get<std::pair<int, int>>(v_id.value);
      RegisterPool::Register tmp;
      std::cout << "# Deep Copy Array\n";
      for (auto i = 0; i < size; i += 4)
      {
        fmt::print("lw {}, {}({}) # copy\n", tmp, i, r_expr);
        fmt::print("sw {}, {}(${}) # paste\n", tmp, offset + i, memoryLocation);
      }
    }
    else if (RecordType* record = dynamic_cast<RecordType*>(expr->getType().get()))
    {
      auto size = record->size();
      auto r_expr = v_expr.getRegister();
      auto [offset, memoryLocation] = std::get<std::pair<int, int>>(v_id.value);
      RegisterPool::Register tmp;
      std::cout << "# Deep Copy Record\n";
      for (auto i = 0; i < size; i += 4)
      {
        fmt::print("lw {}, {}({}) # copy\n", tmp, i, r_expr);
        fmt::print("sw {}, {}(${}) # copy\n", tmp, offset + i, r_expr);
      }
    }
    else
    {
      auto r_expr = v_expr.getTheeIntoARegister();
      fmt::print("sw {}, {}", r_expr, v_id.getLocation());
    }
  }
  else if (v_id.isRegister())
  {
    if (RecordType* record = dynamic_cast<RecordType*>(expr->getType().get()))
    {
      auto size = record->size();
      auto r_expr = v_expr.getRegister();
      auto r_id = v_id.getRegister();
      RegisterPool::Register tmp;
      std::cout << "# Deep Copy Record\n";
      for (auto i = 0; i < size; i += 4)
      {
        fmt::print("lw {}, {}({}) # copy\n", tmp, i, r_expr);
        fmt::print("sw {}, {}({}) # paste\n", tmp, i, r_id);
      }
    }
    else
    {
      auto r_id = v_id.getRegister();
      auto r_expr = v_expr.getTheeIntoARegister();
      fmt::print("sw {}, 0({}) # paste\n", r_expr, r_id);
    }
  }
  else
  {
    LOG(ERROR) << identifier->getId() << " is not an Lvalue";
    exit(EXIT_FAILURE);
  }
  std::cout << " # ";
  emitSource("");
}
