#ifndef FACTORY_HPP
#define FACTORY_HPP

#include "../fmt/include/fmt/core.h"   // for format
#include "AddNode.hpp"                 // for AddNode
#include "BooleanLiteralNode.hpp"      // for BooleanLiteralNode
#include "CharacterLiteralNode.hpp"    // for CharacterLiteralNode
#include "ConstantDeclarationNode.hpp" // for ConstantDeclarationNode
#include "DivideNode.hpp"              // for DivideNode
#include "EqualExpressionNode.hpp"     // for EqualExpressionNode
#include "ExpressionNode.hpp"          // for ExpressionNode
#include "IntegerLiteralNode.hpp"      // for IntegerLiteralNode
#include "LiteralNode.hpp"             // for LiteralNode
#include "LvalueNode.hpp"              // for LvalueNode
#include "ModuloNode.hpp"              // for ModuloNode
#include "MultiplyNode.hpp"            // for MultiplyNode
#include "NotEqualExpressionNode.hpp"  // for NotEqualExpressionNode
#include "StringLiteralNode.hpp"       // for StringLiteralNode
#include "SubtractNode.hpp"            // for SubtractNode
#include "SymbolTable.hpp"             // for SymbolTable, symbol_table
#include "Type.hpp"                    // for BooleanType, CharacterType
#include "UnaryMinusNode.hpp"          // for UnaryMinusNode
#include "log/easylogging++.h"         // for Writer, CERROR, LOG

#include <memory>   // for shared_ptr, operator==, __sha...
#include <stdlib.h> // for exit, EXIT_FAILURE
#include <string>   // for string

template<typename LiteralType>
LiteralType* literalize(ExpressionNode* e)
{
  if (e->isLiteral())
  {
    return dynamic_cast<LiteralType*>(e);
  }
  else if (e->isConstant())
  {
    auto c1 = dynamic_cast<LvalueNode*>(e);
    return dynamic_cast<LiteralType*>(symbol_table.lookupConst(c1->getId()).get());
  }
  return nullptr;
}

template<typename NodeType, typename LiteralType, typename F>
ExpressionNode* makeBinaryExpressionNode(ExpressionNode* e1, ExpressionNode* e2, F f)
{
  // ----------------------------------------
  // Find expression 1 as literal or constant
  // ----------------------------------------
  LiteralType* lit1 = literalize<LiteralType>(e1);


  // ----------------------------------------
  // Find expression 2 as literal or constant
  // ----------------------------------------
  LiteralType* lit2 = literalize<LiteralType>(e2);

  // ---------------------------------
  // If both are found, constant fold;
  // ---------------------------------
  if (lit1 && lit2)
  {
    return new LiteralType(f(lit1->value, lit2->value));
  }
  else
  {
    return new NodeType(e1, e2);
  }
}

template<typename NodeType, typename LiteralType, typename F>
ExpressionNode* makeUnaryExpresisonNode(ExpressionNode* e, F f)
{
  // ----------------------------------------
  // Find expression as literal or constant
  // ----------------------------------------
  LiteralType* lit = literalize<LiteralType>(e);

  if (lit)
  {
    return new LiteralType(f(lit->value));
  }
  else
  {
    return new NodeType(e);
  }
}

ExpressionNode* makeUnaryMinusNode(ExpressionNode* e)
{
  return makeUnaryExpresisonNode<UnaryMinusNode, IntegerLiteralNode>(
    e, [](auto v) { return -v; });
}

ExpressionNode* makeAddNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<AddNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 + v2; });
}

ExpressionNode* makeSubtractNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<SubtractNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 - v2; });
}

ExpressionNode* makeMultiplyNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<MultiplyNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 * v2; });
}

ExpressionNode* makeDivideNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<DivideNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 / v2; });
}

ExpressionNode* makeModuloNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<ModuloNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 % v2; });
}

ExpressionNode* makeNotEqualNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<NotEqualExpressionNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 != v2; });
}

ExpressionNode* makeEqualNode(ExpressionNode* e1, ExpressionNode* e2)
{
  return makeBinaryExpressionNode<EqualExpressionNode, IntegerLiteralNode>(
    e1, e2, [](auto v1, auto v2) { return v1 == v2; });
}

ExpressionNode* makeLiteralNode(LvalueNode* e)
{
  auto literalExpr = symbol_table.lookupConst(e->getId());
  if (literalExpr != nullptr)
  {
    if (literalExpr->getType() == IntegerType::get())
    {
      return new IntegerLiteralNode(
        dynamic_cast<IntegerLiteralNode*>(literalExpr.get())->value);
    }
    if (literalExpr->getType() == BooleanType::get())
    {
      return new BooleanLiteralNode(
        dynamic_cast<BooleanLiteralNode*>(literalExpr.get())->value);
    }
    if (literalExpr->getType() == CharacterType::get())
    {
      return new CharacterLiteralNode(
        dynamic_cast<CharacterLiteralNode*>(literalExpr.get())->character);
    }
    if (literalExpr->getType() == StringType::get())
    {
      return new StringLiteralNode(
        dynamic_cast<StringLiteralNode*>(literalExpr.get())->string);
    }
    LOG(ERROR) << fmt::format(
      "{}:{} can not be turned into a literal!", e->getId(), e->getType()->name());
    exit(EXIT_FAILURE);
  }
  LOG(ERROR) << fmt::format("{} is not defined in the const symbol table", e->getId());
  exit(EXIT_FAILURE);
}

ConstantDeclarationNode* makeConstantDeclarationNode(std::string id, ExpressionNode* e)
{
  // if ( Expression is an IdentifierNode )
  if (LvalueNode* plval = dynamic_cast<LvalueNode*>(e))
  {
    return new ConstantDeclarationNode(id, makeLiteralNode(plval));
  }
  // if ( Expression is a Literal )
  else if (LiteralNode* plit = dynamic_cast<LiteralNode*>(e))
  {
    return new ConstantDeclarationNode(id, plit);
  }
  else
  {
    LOG(ERROR) << "Non-Const expression in Constant Declaration";
    exit(EXIT_FAILURE);
  }
}

#endif
