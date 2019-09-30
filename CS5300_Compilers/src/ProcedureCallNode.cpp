#include "ProcedureCallNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "ExpressionNode.hpp"        // for ExpressionNode
#include "RegisterPool.hpp"          // for spill, unspill, Register
#include "SymbolTable.hpp"           // for SymbolTable, symbol_table
#include "Type.hpp"                  // for Type
#include "Value.hpp"                 // for Value
#include "log/easylogging++.h"       // for Writer, LOG, CERROR, CINFO
#include "stacktrace.hpp"            // for get_stacktrace

#include <iostream> // for operator<<, cout, ostream, basi...

void ProcedureCallNode::emitSource(std::string indent)
{
  std::cout << indent << "PROCEDURE" << id << "(";
  if (args.size() > 0)
  {
    if (args.size() > 1)
      for (auto i = 0u; i < args.size() - 1; ++i)
      {
        args[i]->emitSource("");
        std::cout << ", ";
      }
    args.back()->emitSource("");
  }
  std::cout << ");\n";
}

void ProcedureCallNode::emit()
{
  std::cout << "\n# ";
  emitSource("");

  auto info = symbol_table.lookupFunction(id);
  if (info == nullptr)
  {
    LOG(ERROR) << id << " not defined as a function";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Spill registers
  auto reg_in_use = RegisterPool::Register::getRegistersInUse();
  RegisterPool::spill(reg_in_use);

  // Setup procedure arguments
  int args_size = 0;
  for (auto&& arg : args)
  {
    args_size += arg->getType()->size();
  }
  fmt::print("addi $sp, $sp, -{}\n", args_size);

  // Store arguments on stack
  // TODO make sure these are put in the right place below FP
  std::vector<Value> vals_args;
  int offset = 4;
  for (auto&& arg : args)
  {
    auto reg = arg->emit().getTheeIntoARegister();
    fmt::print("sw {}, {}($sp)\n", reg, offset);
    offset += arg->getType()->size();
  }

  // Make the call
  fmt::print("jal {}\n", id);

  // Remove arguments
  fmt::print("addi $sp, $sp, {}\n", args_size);

  // Unspill registers
  RegisterPool::unspill(reg_in_use);
}
