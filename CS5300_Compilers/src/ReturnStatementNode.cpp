#include "ReturnStatementNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "SymbolTable.hpp"           // for SymbolTable, symbol_table
#include "Value.hpp"                 // for Value

#include <iostream> // for operator<<, cout, ostream, basi...

void ReturnStatementNode::emitSource(std::string indent)
{
  if (expr != nullptr)
  {
    std::cout << indent << "return ";
    expr->emitSource("");
    std::cout << ";\n";
  }
  else
  {
    std::cout << indent << "return;\n";
  }
}

void ReturnStatementNode::emit()
{
  if (expr != nullptr)
  {
    auto reg_expr = expr->emit().getTheeIntoARegister();
    auto [baseRegister, offset] = symbol_table.getReturnValueLocation();
    fmt::print("sw {} {}(${})\n", reg_expr, offset, baseRegister);
  }
  auto epilogueLbl = symbol_table.getepilogueLable();
  fmt::print("j {}\n", epilogueLbl);
}

