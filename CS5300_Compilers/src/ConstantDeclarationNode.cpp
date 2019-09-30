#include "ConstantDeclarationNode.hpp"

#include "ExpressionNode.hpp" // for ExpressionNode
#include "LiteralNode.hpp"    // for LiteralNode
#include "SymbolTable.hpp"    // for SymbolTable, symbol_table

#include <iostream> // for operator<<, basic_ostream, cout, endl

ConstantDeclarationNode::ConstantDeclarationNode(std::string id, ExpressionNode* expr)
  : m_id(id)
  , m_expr(dynamic_cast<LiteralNode*>(expr))
{}

void ConstantDeclarationNode::emitSource(std::string indent)
{
  emit();
  std::cout << indent << m_id << " = ";
  m_expr->emitSource("");
  std::cout << ";\n";
}

void ConstantDeclarationNode::emit()
{
  symbol_table.storeConst(m_id, m_expr);
}
