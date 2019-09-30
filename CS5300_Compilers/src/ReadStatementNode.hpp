#ifndef READ_STATEMENT_NODE_HPP
#define READ_STATEMENT_NODE_HPP

#include "StatementNode.hpp" // for StatementNode
#include "Value.hpp"         // for Value

#include <memory> // for shared_ptr
#include <string> // for string
#include <vector> // for vector
class LvalueNode;
template<typename T>
class ListNode; // lines 14-15

class ReadStatementNode : public StatementNode
{
public:
  ReadStatementNode(ListNode<LvalueNode>*& lVals);
  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

private:
  const std::vector<std::shared_ptr<LvalueNode>> identifiers;
};

#endif
