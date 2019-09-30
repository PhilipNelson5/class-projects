#include "ProcedureDeclarationNode.hpp"

#include "FormalParameter.hpp" // for FormalParameter
#include "SymbolTable.hpp"

#include <ext/alloc_traits.h> // for __alloc_traits<>::value_type
#include <iostream>           // for operator<<, cout, ostream, basic_ostream

void ProcedureDeclarationNode::emitSource(std::string indent)
{
  std::cout << indent << "PROCEDURE " << id << "(";

  if (parameters.size() > 0)
  {
    if (parameters.size() > 1)
      for (auto i = 0u; i < parameters.size() - 1; ++i)
      {
        parameters[i]->emitSource("");
        std::cout << "; ";
      }
    parameters.back()->emitSource("");
  }

  std::cout << ");\n";

  if (body == nullptr)
  {
    std::cout << "forward;\n";
  }
  else
  {
    std::cout << indent << "BEGIN\n";
    body->emitSource(indent + "  ");
    std::cout << indent << "END;\n";
  }
}

void ProcedureDeclarationNode::emit()
{
  std::vector<Parameter> params;
  for (auto&& param : parameters)
  {
    params.emplace_back(param->ids,
                        param->type->id,
                        (param->passby == FormalParameter::PassBy::REF)
                          ? Parameter::PassBy::REF
                          : Parameter::PassBy::VAL);
  }

  // Forward Declaration
  if (body == nullptr)
  {
    symbol_table.storeFunction(id, params, "__no_return__", Function::Declaration::FORWARD);
  }
  else
  {
    // if there was no forward declaration, add it to the current scope
    if (symbol_table.lookupFunction(id) == nullptr)
    {
      symbol_table.storeFunction(id, params, "__no_return__", Function::Declaration::FORWARD);
    }

    symbol_table.enter_scope();

    symbol_table.storeFunction(id, params, "__no_return__", Function::Declaration::DEFINITION);

    fmt::print("{}:\n", id);

    body->emit();

    fmt::print("jr $ra\n");

    symbol_table.exit_scope();
  }
}
