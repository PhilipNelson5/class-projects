#ifndef PROGRAM_NODE_HPP
#define PROGRAM_NODE_HPP

#include "Node.hpp" // for Node

#include <memory> // for shared_ptr
#include <string> // for string
#include <vector> // for vector
class ConstantDeclarationNode;
class StatementNode;
class TypeDeclarationNode;
class VariableDeclarationNode;
class ProcedureOrFunctionDeclarationNode;
template<typename T>
class ListNode;

class ProgramNode : public Node
{
public:
  ProgramNode(ListNode<ConstantDeclarationNode>*& cds,
              ListNode<TypeDeclarationNode>*& tds,
              ListNode<VariableDeclarationNode>*& vds,
              ListNode<ProcedureOrFunctionDeclarationNode>*& pfd,
              ListNode<StatementNode>*& mb);

  virtual void emitSource(std::string indent) override;
  void emit();

private:
  std::vector<std::shared_ptr<ConstantDeclarationNode>> constantDecls;
  std::vector<std::shared_ptr<TypeDeclarationNode>> typeDecls;
  std::vector<std::shared_ptr<VariableDeclarationNode>> varDecls;
  std::vector<std::shared_ptr<ProcedureOrFunctionDeclarationNode>> procedureAndFunctionDecls;
  std::vector<std::shared_ptr<StatementNode>> mainBlock;
};

#endif
