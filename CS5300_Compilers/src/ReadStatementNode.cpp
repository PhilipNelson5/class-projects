#include "ReadStatementNode.hpp"

#include "../fmt/include/fmt/core.h" // for format, print
#include "ListNode.hpp"              // for ListNode
#include "LvalueNode.hpp"            // for LvalueNode
#include "Type.hpp"                  // for CharacterType, IntegerType, Type
#include "log/easylogging++.h"       // for Writer, CERROR, LOG

#include <ext/alloc_traits.h> // for __alloc_traits<>::value_type
#include <iostream>           // for operator<<, basic_ostream, cout
#include <stdlib.h>           // for exit, EXIT_FAILURE

ReadStatementNode::ReadStatementNode(ListNode<LvalueNode>*& identifiers)
  : identifiers(ListNode<LvalueNode>::makeVector(identifiers))
{}

void ReadStatementNode::emitSource(std::string indent)
{
  std::cout << indent << "read(";
  for (auto i = 0u; i < identifiers.size() - 1; ++i)
  {
    identifiers[i]->emitSource("");
    std::cout << ", ";
  }
  identifiers.back()->emitSource("");
  std::cout << ");\n";
}

void ReadStatementNode::emit()
{
  std::cout << "\n# ";
  emitSource("");

  for (auto&& identifier : identifiers)
  {
    auto v_id = identifier->emit();
    if (!v_id.isLvalue())
    {
      LOG(ERROR) << fmt::format("{} is not an Lvalue", identifier->getId());
      exit(EXIT_FAILURE);
    }

    if (identifier->getType() == IntegerType::get())
    {
      std::cout << "li $v0, 5"
                << " # load read integer instruction" << '\n';
    }
    else if (identifier->getType() == CharacterType::get())
    {
      std::cout << "li $v0, 12"
                << " # load read character instruction" << '\n';
    }
    else
    {
      LOG(ERROR) << fmt::format("type {} can not be read into", identifier->getType()->name());
      exit(EXIT_FAILURE);
    }

    std::cout << "syscall" << '\n';
    fmt::print("sw $v0, {}", v_id.getLocation());
    fmt::print(" # {} = user input\n\n", identifier->getId());
  }
}

