#ifndef FORMAL_PARAMETER_HPP
#define FORMAL_PARAMETER_HPP

#include "ListNode.hpp" // for ListNode
#include "TypeNode.hpp" // for TypeNode

#include <memory> // for shared_ptr, __shared_ptr_access
#include <string> // for string
#include <vector> // for vector
class Type;

class FormalParameter
{
public:
  enum PassBy
  {
    REF,
    VAL
  };

  void emitSource(std::string indent);
  void emit();

  FormalParameter(ListNode<std::string>*& ids, TypeNode*& type, PassBy passBy = PassBy::VAL)
    : ids(ListNode<std::string>::makeDerefVector(ids))
    , type(type)
    , passby(passBy)
  {}

  std::shared_ptr<Type> getType() { return type->getType(); }

  const std::vector<std::string> ids;
  const std::shared_ptr<TypeNode> type;
  const PassBy passby;
};

#endif
