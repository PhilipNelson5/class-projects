#include "ForStatementNode.hpp"

#include "SymbolTable.hpp"
#include "log/easylogging++.h"

ForStatementNode::ForStatementNode(IdentifierNode* identifier,
                                   ExpressionNode*& startExpr,
                                   ExpressionNode*& endExpr,
                                   ListNode<StatementNode>*& statements,
                                   Type type)
  : identifier(identifier)
  , startExpr(startExpr)
  , endExpr(endExpr)
  , statements(ListNode<StatementNode>::makeVector(statements))
  , type(type)
{
  LOG(DEBUG) << "NEW FOR_STATEMENT_NODE (" << identifier->getId() << ")";
}

void ForStatementNode::printForHeader(std::string indent)
{
  std::cout << indent << "for " << identifier->getId() << " := ";
  startExpr->emitSource("");
  std::cout << (type == Type::TO ? " to " : " downto ");
  endExpr->emitSource("");
  std::cout << " do\n";
}

void ForStatementNode::emitSource(std::string indent)
{
  symbol_table.enter_scope();

  printForHeader(indent);

  for (auto&& statement : statements)
  {
    statement->emitSource(indent + "  ");
  }

  std::cout << indent << "end;\n\n";

  symbol_table.exit_scope();
}

void ForStatementNode::emit()
{
  symbol_table.enter_scope();

  std::cout << "# ";
  printForHeader("");

  // ----------------------------
  // Declare counter if necessary
  // ----------------------------
  if (!symbol_table.lookupLval(identifier->getId()))
  {
    symbol_table.storeVariable(identifier->getId(), startExpr->getType());
  }

  // ----------------------------
  // Assign id = start expression
  // ----------------------------
  auto v_id = identifier->emit();
  {
    auto v_startExpr = startExpr->emit();
    auto r_expr = v_startExpr.getTheeIntoARegister();
    fmt::print("sw {}, {}\n", r_expr, v_id.getLocation());
  }

  auto lblStart = symbol_table.newLabel("start");
  auto lblEnd = symbol_table.newLabel("end");
  std::cout << lblStart << ":\n";

  // -------------------
  // Check end condition
  // -------------------
  {
    auto v_endExpr = endExpr->emit();
    auto r_id = v_id.getTheeIntoARegister();
    auto r_endExpr = v_endExpr.getTheeIntoARegister();
    RegisterPool::Register result;
    if (type == Type::TO)
    {
      fmt::print("sgt {}, {}, {}\n", result, r_id, r_endExpr);
    }
    else
    {
      fmt::print("slt {}, {}, {}\n", result, r_id, r_endExpr);
    }
    fmt::print("bne {}, $0, {}\n", result, lblEnd);
  }

  // ----------------------
  // execute statement list
  // ----------------------
  for (auto&& statement : statements)
  {
    statement->emit();
  }

  // --------------
  // Update Counter
  // --------------
  if (type == Type::TO)
  {
    auto r_id = v_id.getTheeIntoARegister();
    fmt::print("addi {0}, {0}, 1\n", r_id, r_id);
    fmt::print("sw {}, {}\n", r_id, v_id.getLocation());
  }
  else
  {
    auto r_id = v_id.getTheeIntoARegister();
    fmt::print("addi {0}, {0}, -1\n", r_id, r_id);
    fmt::print("sw {}, {}\n", r_id, v_id.getLocation());
  }

  // jump to start
  fmt::print("j {}\n", lblStart);

  // print end label
  std::cout << lblEnd << ":\n";

  symbol_table.exit_scope();
}
