#ifndef BODY_NODE_HPP
#define BODY_NODE_HPP

#include "ConstantDeclarationNode.hpp"
#include "ListNode.hpp"
#include "StatementNode.hpp"
#include "TypeDeclarationNode.hpp"
#include "VariableDeclarationNode.hpp"

#include <memory>

class BodyNode
{
public:
  BodyNode(ListNode<ConstantDeclarationNode>*& constDecls,
           ListNode<TypeDeclarationNode>*& typeDecls,
           ListNode<VariableDeclarationNode>*& varDecls,
           ListNode<StatementNode>*& block)
    : constDecls(ListNode<ConstantDeclarationNode>::makeVector(constDecls))
    , typeDecls(ListNode<TypeDeclarationNode>::makeVector(typeDecls))
    , varDecls(ListNode<VariableDeclarationNode>::makeVector(varDecls))
    , block(ListNode<StatementNode>::makeVector(block))
  {}

  void emitSource(std::string indent);
  void emit();

  const std::vector<std::shared_ptr<ConstantDeclarationNode>> constDecls;
  const std::vector<std::shared_ptr<TypeDeclarationNode>> typeDecls;
  const std::vector<std::shared_ptr<VariableDeclarationNode>> varDecls;
  const std::vector<std::shared_ptr<StatementNode>> block;
};

#endif
