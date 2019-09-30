#include "BodyNode.hpp"

#include "../fmt/include/fmt/core.h"
#include "SymbolTable.hpp"

void BodyNode::emitSource(std::string indent)
{
  if (!constDecls.empty())
  {
    for (auto&& constDecl : constDecls)
    {
      constDecl->emitSource(indent);
    }
    std::cout << '\n';
  }

  if (!typeDecls.empty())
  {
    for (auto&& typeDecl : typeDecls)
    {
      typeDecl->emitSource(indent);
    }
    std::cout << '\n';
  }

  if (!varDecls.empty())
  {
    for (auto&& varDecl : varDecls)
    {
      varDecl->emitSource(indent);
    }
    std::cout << '\n';
  }

  if (!block.empty())
  {
    for (auto&& statement : block)
    {
      statement->emitSource(indent);
    }
  }
}

void BodyNode::emit()
{
  if (!constDecls.empty())
  {
    for (auto&& constDecl : constDecls)
    {
      constDecl->emit();
    }
  }

  if (!typeDecls.empty())
  {
    for (auto&& typeDecl : typeDecls)
    {
      typeDecl->emit();
    }
  }

  if (!varDecls.empty())
  {
    for (auto&& varDecl : varDecls)
    {
      varDecl->emit();
    }
  }

  fmt::print("or {}, {}, {} # Set $FP = $SP\n", "$fp", "$sp", "$0");

  int localSize = 0;
  for (auto&& var : varDecls)
  {
    localSize += var->m_ids.size() * var->m_typeNode->getType()->size();
  }

  if (localSize > 0)
  {
    fmt::print("addi {0}, {0}, {1} # Set $SP for local variables", "$sp", -localSize);
  }

  if (!block.empty())
  {
    for (auto&& statement : block)
    {
      statement->emit();
    }
  }
}
