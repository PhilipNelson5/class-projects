#ifndef TYPE_HPP
#define TYPE_HPP

#include "../fmt/include/fmt/core.h" // for print
#include "ExpressionNode.hpp"
#include "ListNode.hpp"        // for ListNode
#include "TypeNode.hpp"        // for TypeNode
#include "log/easylogging++.h" // for Writer, CERROR, LOG

#include <iostream> // for operator<<, cout, ostream, basi...
#include <iterator> // for end
#include <map>      // for map, map<>::iterator, _Rb_tree_...
#include <memory>   // for shared_ptr, make_shared, allocator
#include <numeric>  // for accumulate
#include <stdlib.h> // for exit, EXIT_FAILURE
#include <string>   // for string, operator<<, operator+
#include <utility>  // for pair, make_pair
#include <vector>   // for vector

//------------------------------------------------------------------------------
// Type
//------------------------------------------------------------------------------
class Type
{
public:
  virtual int size() { return 4; };
  virtual std::string name() = 0;
  virtual ~Type() = default;
  virtual void emitSource(std::string indent) = 0;
};

//------------------------------------------------------------------------------
// Integer Type
//------------------------------------------------------------------------------
class IntegerType : public Type
{
public:
  std::string name() override { return "integer"; }

  static std::shared_ptr<Type> get()
  {
    if (!pInt) pInt = std::make_shared<IntegerType>();
    return pInt;
  }

  virtual void emitSource(std::string indent) override
  {
    (void)indent;
    std::cout << name();
  }

private:
  static std::shared_ptr<Type> pInt;
};

//------------------------------------------------------------------------------
// Character Type
//------------------------------------------------------------------------------
class CharacterType : public Type
{
public:
  std::string name() override { return "char"; }

  static std::shared_ptr<Type> get()
  {
    if (!pChar) pChar = std::make_shared<CharacterType>();
    return pChar;
  }

  virtual void emitSource(std::string indent) override
  {
    (void)indent;
    std::cout << name();
  }

private:
  static std::shared_ptr<Type> pChar;
};

//------------------------------------------------------------------------------
// Boolean Type
//------------------------------------------------------------------------------
class BooleanType : public Type
{
public:
  std::string name() override { return "boolean"; }

  static std::shared_ptr<Type> get()
  {
    if (!pBool) pBool = std::make_shared<BooleanType>();
    return pBool;
  }

  virtual void emitSource(std::string indent) override
  {
    (void)indent;
    std::cout << name();
  }

private:
  static std::shared_ptr<Type> pBool;
};

//------------------------------------------------------------------------------
// String Type
//------------------------------------------------------------------------------
class StringType : public Type
{
public:
  int size() override { return 0; }

  virtual std::string name() override { return "string"; }

  static std::shared_ptr<Type> get()
  {
    if (!pStr) pStr = std::make_shared<StringType>();
    return pStr;
  }

  virtual void emitSource(std::string indent) override
  {
    (void)indent;
    std::cout << name();
  }

private:
  static std::shared_ptr<Type> pStr;
};

//------------------------------------------------------------------------------
// Array Type
//------------------------------------------------------------------------------
class ArrayType : public Type
{
public:
  ArrayType(ExpressionNode*& lb, ExpressionNode*& ub, TypeNode*& elementType)
    : lbExpr(lb)
    , ubExpr(ub)
    , elementTypeNode(elementType)
    , indexType(nullptr)
    , elementType(nullptr)
  {}

private:
  bool initialized = false;
  int lb, ub;
  const std::shared_ptr<ExpressionNode> lbExpr, ubExpr;
  const std::shared_ptr<TypeNode> elementTypeNode;
  std::shared_ptr<Type> indexType;
  std::shared_ptr<Type> elementType;

public:
  virtual std::string name() override { return "array"; }

  virtual void emitSource(std::string indent) override
  {
    (void)indent;
    init();
    fmt::print("array[{}:{}] of ", lb, ub);
    elementType->emitSource(indent + "  ");
  }

  virtual int size() override
  {
    init();
    return (ub - lb + 1) * elementType->size();
  }

  int getlb()
  {
    init();
    return lb;
  }

  const std::shared_ptr<Type> getElementType()
  {
    init();
    return elementType;
  }

  void init()
  {
    // static bool initialized = false;
    if (initialized) return;
    initialized = true;

    if (lbExpr->getType() != ubExpr->getType())
    {
      LOG(ERROR) << "array bounds must have the same type";
      exit(EXIT_FAILURE);
    }
    if ((!lbExpr->isConstant() && !lbExpr->isLiteral())
        || (!ubExpr->isConstant() && !ubExpr->isLiteral()))
    {
      LOG(ERROR) << "array bounds must be constant expressions";
      exit(EXIT_FAILURE);
    }

    auto var_lb = lbExpr->eval();
    auto var_ub = ubExpr->eval();

    if (lbExpr->getType() == IntegerType::get())
    {
      lb = std::get<int>(var_lb);
      ub = std::get<int>(var_ub);
      indexType = IntegerType::get();
    }
    else if (ubExpr->getType() == CharacterType::get())
    {
      lb = std::get<char>(var_lb);
      ub = std::get<char>(var_ub);
      indexType = IntegerType::get();
    }
    else
    {
      LOG(ERROR) << fmt::format(
        "array bounds must be integer or character type: {} not allowed",
        lbExpr->getType()->name());
      exit(EXIT_FAILURE);
    }

    elementType = elementTypeNode->getType();
  }
};

//------------------------------------------------------------------------------
// Record Type
//------------------------------------------------------------------------------
struct Field
{
  Field(ListNode<std::string>*& idList, TypeNode* typeNode)
    : ids(ListNode<std::string>::makeDerefVector(idList))
    , typeNode(typeNode)
  {}

  const std::shared_ptr<Type> getType()
  {
    if (type == nullptr)
    {
      type = typeNode->getType();
    }
    return type;
  }

  const std::vector<std::string> ids;
  const std::shared_ptr<TypeNode> typeNode;

private:
  std::shared_ptr<Type> type;
};

class RecordType : public Type
{
public:
  RecordType(ListNode<Field>*& fields)
    : fields(ListNode<Field>::makeVector(fields))
    , table()
  {}

private:
  std::vector<std::shared_ptr<Field>> fields;
  //          id                offset                type
  std::map<std::string, std::pair<int, std::shared_ptr<Type>>> table;

  bool initialized = false;

public:
  virtual std::string name() override { return "record"; }

  virtual void emitSource(std::string indent) override
  {
    init();
    std::cout << "record" << '\n';
    for (auto&& r : table)
    {
      std::cout << indent << r.first << " : ";
      r.second.second->emitSource(indent + "  ");
      std::cout << ";\n";
    }
    std::cout << std::string(indent.length() - 2, ' ') << "end";
  }

  std::pair<int, std::shared_ptr<Type>> lookupId(std::string id)
  {
    init();
    auto found = table.find(id);
    if (found == std::end(table))
    {
      LOG(ERROR) << id << "does not exist in record";
      exit(EXIT_FAILURE);
    }
    return found->second;
  }

  const auto& getTable()
  {
    init();
    return table;
  }

  virtual int size() override
  {
    init();
    return std::accumulate(table.begin(), table.end(), 0, [](int sum, auto e) {
      return sum + e.second.second->size();
    });
  }

  void init()
  {
    if (initialized) return;
    initialized = true;

    int offset = 0;
    for (auto&& f : fields)
    {
      for (auto&& id : f->ids)
      {
        table.emplace(id, std::make_pair(offset, f->getType()));
        offset += f->getType()->size();
      }
    }
  }
};

#endif
