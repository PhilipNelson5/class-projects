#include "MemberAccessNode.hpp"

#include "RegisterPool.hpp"    // for operator<<
#include "Type.hpp"            // for RecordType
#include "log/easylogging++.h" // for Writer, CERROR, LOG

#include <iostream> // for operator<<, basic_ostream, ostream
#include <iterator> // for end
#include <map>      // for _Rb_tree_iterator, map
#include <stdlib.h> // for exit, EXIT_FAILURE
#include <utility>  // for move, pair
#include <variant>  // for get

std::string MemberAccessNode::getId() const
{
  return fmt::format("{}.{}", lValue->getId(), id);
}

std::shared_ptr<Type> lookupId(std::shared_ptr<LvalueNode> lValue, std::string id)
{
  if (RecordType* pRecord = dynamic_cast<RecordType*>(lValue->getType().get()))
  {
    auto member = pRecord->getTable().find(id);
    if (member != std::end(pRecord->getTable()))
    {
      return member->second.second;
    }
    else
    {
      LOG(ERROR) << fmt::format("Member {} does not exist in {}", id, lValue->getId());
      exit(EXIT_FAILURE);
    }
  }
  LOG(ERROR) << "Can not use member access on non record types";
  exit(EXIT_FAILURE);
}

MemberAccessNode::MemberAccessNode(LvalueNode* lValue, std::string id)
  : LvalueNode()
  , lValue(lValue)
  , id(id)
{}

bool MemberAccessNode::isConstant() const
{
  return lValue->isConstant();
}

const std::shared_ptr<Type> MemberAccessNode::getType()
{
  if (type == nullptr)
  {
    type = lookupId(lValue, id);
  }
  return type;
}

std::variant<std::monostate, int, char, bool> MemberAccessNode::eval() const
{
  return {};
}

void MemberAccessNode::emitSource(std::string indent)
{
  std::cout << indent << getId();
}

Value MemberAccessNode::emit()
{
  if (RecordType* record = dynamic_cast<RecordType*>(lValue->getType().get()))
  {
    auto v_lval = lValue->emit();
    if (v_lval.isLvalue())
    {
      auto [offset1, memoryLocation] = std::get<std::pair<int, int>>(v_lval.value);
      auto [offset2, type] = record->lookupId(id);

      return {offset1 + offset2, memoryLocation};
    }
    if (v_lval.isRegister())
    {
      auto r_lval = v_lval.getRegister();
      auto [offset2, type] = record->lookupId(id);

      fmt::print("addi {0}, {0}, {1}", r_lval, offset2);
      fmt::print(" # access member: {}\n", id);

      return {std::move(r_lval), Value::RegisterIs::ADDRESS};
    }

    LOG(ERROR) << lValue->getId() << " is not an lvalue";
    exit(EXIT_FAILURE);
  }
  else
  {
    LOG(ERROR) << lValue->getId() << " is not a record type";
    exit(EXIT_FAILURE);
  }
}
