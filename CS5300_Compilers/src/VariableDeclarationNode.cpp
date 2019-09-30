#include "VariableDeclarationNode.hpp"

#include "ListNode.hpp"    // for ListNode
#include "SymbolTable.hpp" // for SymbolTable, symbol_table
#include "Type.hpp"        // for Type
#include "TypeNode.hpp"    // for TypeNode

#include <iostream> // for operator<<, cout, ostream, basic_ostream
#include <string>   // for string, operator<<, char_traits

VariableDeclarationNode::VariableDeclarationNode(ListNode<std::string>* identList,
                                                 TypeNode*& typeNode)
  : m_ids(ListNode<std::string>::makeDerefVector(identList))
  , m_typeNode(typeNode)
{}

void VariableDeclarationNode::emitSource(std::string indent)
{
  emit();
  std::cout << indent;
  for (auto i = 0u; i < m_ids.size() - 1; ++i)
  {
    std::cout << m_ids[i] << ", ";
  }
  std::cout << m_ids.back() << " : ";
  m_typeNode->getType()->emitSource(indent + "  ");
  std::cout << ";" << '\n';
}

void VariableDeclarationNode::emit()
{
  for (auto&& id : m_ids)
  {
    symbol_table.storeVariable(id, m_typeNode->getType());
  }
  // throw "VariableDeclarationNode::emit() should not be called";
}
