#include "FunctionDeclarationNode.hpp"

#include "FormalParameter.hpp" // for FormalParameter
#include "SymbolTable.hpp"
#include "Type.hpp" // for Type

#include <iostream> // for operator<<, basic_ostream, cout, ostream

void FunctionDeclarationNode::emitSource(std::string indent)
{
  std::cout << indent << "FUNCTION " << id << "(";

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

  std::cout << ") :" << getType()->name() << "; ";

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

void FunctionDeclarationNode::emit()
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
    symbol_table.storeFunction(id, params, returnTypeNode->id, Function::Declaration::FORWARD);
  }
  else
  {
    // if there was no forward declaration, add it to the current scope
    if (symbol_table.lookupFunction(id) == nullptr)
    {
      symbol_table.storeFunction(
        id, params, returnTypeNode->id, Function::Declaration::FORWARD);
    }

    symbol_table.enter_scope();

    symbol_table.storeFunction(
      id, params, returnTypeNode->id, Function::Declaration::DEFINITION);

    auto epilogueLbl = symbol_table.newLabel(id + "_epilogue");
    symbol_table.setepilogueLable(epilogueLbl);

    auto paramSize = 0;
    for (auto&& param : params)
    {
      for (auto&& id : param.ids)
      {
        (void)id;
        paramSize += param.getType()->size();
      }
    }

    symbol_table.setReturnValueLocation(symbol_table.FP, paramSize);

    auto localSize = 0;
    for (auto&& var : body->varDecls)
    {
      for (auto&& id : var->m_ids)
      {
        (void)id;
        localSize += var->m_typeNode->getType()->size();
      }
    }

    // Function Label
    fmt::print("{}:\n", id);

    body->emit();

    fmt::print("{}:\n", epilogueLbl);
    if (localSize > 0) fmt::print("addi $sp, $sp, {}\n", localSize);
    fmt::print("jr $ra\n");

    symbol_table.exit_scope();
  }
}

std::shared_ptr<Type> FunctionDeclarationNode::getType()
{
  if (returnType == nullptr)
  {
    returnType = returnTypeNode->getType();
  }
  return returnType;
}
