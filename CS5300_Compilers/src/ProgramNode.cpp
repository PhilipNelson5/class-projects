#include "ProgramNode.hpp"

#include "ConstantDeclarationNode.hpp" // for ConstantDeclarationNode
#include "ListNode.hpp"                // for ListNode
#include "ProcedureOrFunctionDeclarationNode.hpp"
#include "StatementNode.hpp"           // for StatementNode
#include "SymbolTable.hpp"             // for SymbolTable, symbol_table
#include "TypeDeclarationNode.hpp"     // for TypeDeclarationNode
#include "VariableDeclarationNode.hpp" // for VariableDeclarationNode

#include <iostream> // for operator<<, endl, basic_ostream
#include <memory>   // for shared_ptr, __shared_ptr_access

ProgramNode::ProgramNode(ListNode<ConstantDeclarationNode>*& cds,
                         ListNode<TypeDeclarationNode>*& tds,
                         ListNode<VariableDeclarationNode>*& vds,
                         ListNode<ProcedureOrFunctionDeclarationNode>*& pfd,
                         ListNode<StatementNode>*& mBlock)
{
  constantDecls = ListNode<ConstantDeclarationNode>::makeVector(cds);

  typeDecls = ListNode<TypeDeclarationNode>::makeVector(tds);

  varDecls = ListNode<VariableDeclarationNode>::makeVector(vds);

  procedureAndFunctionDecls = ListNode<ProcedureOrFunctionDeclarationNode>::makeVector(pfd);

  mainBlock = ListNode<StatementNode>::makeVector(mBlock);
}

void ProgramNode::emitSource(std::string indent)
{
  // Constant Declarations
  // ---------------------
  if (constantDecls.size() > 0u)
  {
    std::cout << indent << "CONST\n";
    for (auto&& constDecl : constantDecls)
    {
      constDecl->emitSource(indent + "  ");
    }
    std::cout << '\n';
  }

  // Type Declarations
  // ---------------------
  if (typeDecls.size() > 0)
  {
    std::cout << indent << "TYPE\n";
    for (auto&& typeDecl : typeDecls)
    {
      typeDecl->emitSource(indent + "  ");
    }
    std::cout << '\n';
  }

  // Variable Declarations
  // ---------------------
  if (varDecls.size() > 0)
  {
    std::cout << indent << "VAR\n";
    for (auto&& varDecl : varDecls)
    {
      varDecl->emitSource(indent + "  ");
    }
    std::cout << '\n';
  }

  // Procedure and Function Declarations
  // -----------------------------------
  if (procedureAndFunctionDecls.size() > 0)
  {
    for (auto&& procedureOrFunction : procedureAndFunctionDecls)
    {
      procedureOrFunction->emitSource(indent);
    }
    std::cout << '\n';
  }

  // Main Block
  // ---------------------
  std::cout << indent << "BEGIN\n";
  for (auto&& statement : mainBlock)
  {
    statement->emitSource(indent + "  ");
  }
  std::cout << indent << "END.\n";
}

void ProgramNode::emit()
{
  std::cout << ".text\n";
  std::cout << "la $gp, GLOBAL_AREA\n";
  std::cout << "or $fp, $sp, $0\n";
  std::cout << "j MAIN\n";

  // Constant Declarations
  // ---------------------
  if (constantDecls.size() > 0u)
  {
    for (auto&& constDecl : constantDecls)
    {
      constDecl->emit();
    }
  }

  // Type Declarations
  // -----------------
  if (typeDecls.size() > 0u)
  {
    for (auto&& typeDecl : typeDecls)
    {
      typeDecl->emit();
    }
  }

  // Variable Declarations
  // ---------------------
  if (varDecls.size() > 0u)
  {
    for (auto&& varDecl : varDecls)
    {
      varDecl->emit();
    }
  }

  // Procedure and Function Declarations
  // -----------------------------------
  if (procedureAndFunctionDecls.size() > 0u)
  {
    for (auto&& procedureOrFunctionDecl : procedureAndFunctionDecls)
    {
      procedureOrFunctionDecl->emit();
    }
  }

  // Main Block
  // ---------------------
  std::cout << "MAIN:\n";
  for (auto&& statement : mainBlock)
  {
    statement->emit();
  }

  std::cout << "\n.data\n";
  symbol_table.printStrings();
  std::cout << "\n.align 2\nGLOBAL_AREA:\n";
}
