#include "TypeDeclarationNode.hpp"

#include "SymbolTable.hpp"
#include "Type.hpp"

#include <iostream>

TypeDeclarationNode::TypeDeclarationNode(std::string id, TypeNode*& typeNode)
  : m_id(id)
  , m_typeNode(typeNode)
{}

void TypeDeclarationNode::emitSource(std::string indent)
{
  emit();
  std::cout << indent << m_id << " = ";
  m_typeNode->getType()->emitSource(indent + "  ");
  std::cout << ";\n";
}

void TypeDeclarationNode::emit()
{
  symbol_table.storeType(m_id, m_typeNode->getType());
}
