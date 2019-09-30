#include "SymbolTable.hpp"

#include "../fmt/include/fmt/core.h" // for format
#include "BooleanLiteralNode.hpp"    // for BooleanLiteralNode
#include "LiteralNode.hpp"           // for LiteralNode
#include "Type.hpp"                  // for Type, BooleanType, CharacterType
#include "log/easylogging++.h"       // for Writer, LOG, CERROR, CDEBUG
#include "stacktrace.hpp"            // for print_stacktrace

#include <algorithm> // for max
#include <iostream>  // for cout
#include <sstream>   // for operator<<, basic_ostream, ostream
#include <stdio.h>   // for stderr
#include <stdlib.h>  // for exit, EXIT_FAILURE
#include <utility>   // for pair

SymbolTable symbol_table;

SymbolTable::SymbolTable()
{
  // Enter Predefined Scope
  scopes.emplace_back();
  auto itPredefines = scopes.rbegin();
  itPredefines->constants.emplace(std::string("true"), new BooleanLiteralNode(1));
  itPredefines->constants.emplace(std::string("TRUE"), new BooleanLiteralNode(1));
  itPredefines->constants.emplace(std::string("false"), new BooleanLiteralNode(0));
  itPredefines->constants.emplace(std::string("FALSE"), new BooleanLiteralNode(0));

  itPredefines->types.emplace(std::string("integer"), IntegerType::get());
  itPredefines->types.emplace(std::string("INTEGER"), IntegerType::get());
  itPredefines->types.emplace(std::string("char"), CharacterType::get());
  itPredefines->types.emplace(std::string("CHAR"), CharacterType::get());
  itPredefines->types.emplace(std::string("boolean"), BooleanType::get());
  itPredefines->types.emplace(std::string("BOOLEAN"), BooleanType::get());
  itPredefines->types.emplace(std::string("string"), StringType::get());
  itPredefines->types.emplace(std::string("STRING"), StringType::get());

  // Enter Global Scope
  enter_scope();
}

std::shared_ptr<Type> SymbolTable::getType(std::string id) const
{
  auto lval_info = lookupLval(id);
  if (lval_info != nullptr)
  {
    return lval_info->type;
  }

  auto const_info = lookupConst(id);
  if (const_info != nullptr)
  {
    return const_info->getType();
  }

  LOG(ERROR) << id << " is not defined";
  LOG(INFO) << get_stacktrace();
  exit(EXIT_FAILURE);
}

std::shared_ptr<Type> SymbolTable::lookupType(std::string id) const
{
  LOG(DEBUG) << "lookupType(" << id << ")";
  for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
  {
    auto found = scope->types.find(id);
    if (found != scope->types.end())
    {
      return found->second;
    }
  }
  LOG(ERROR) << id << " not defined as a type";
  LOG(INFO) << get_stacktrace();
  exit(EXIT_FAILURE);
}

std::shared_ptr<Function> SymbolTable::lookupFunction(std::string id) const
{
  LOG(DEBUG) << "lookupFunction(" << id << ")";
  for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
  {
    auto found = scope->functions.find(id);
    if (found != scope->functions.end())
    {
      return found->second;
    }
  }
  return nullptr;
}

const std::string SymbolTable::lookupString(std::string str)
{
  static int num = 0;
  const static std::string name = "string";
  auto found = strings.find(str);

  if (found == strings.end())
  {
    auto newLabel = name + std::to_string(num++);
    strings.emplace(str, newLabel);
    return newLabel;
  }

  return found->second;
}

const std::string SymbolTable::newLabel(std::string name)
{
  static unsigned n = 0u;
  return fmt::format("{}_{}", name, n++);
}

std::shared_ptr<Variable> SymbolTable::lookupLval(std::string id) const
{
  LOG(DEBUG) << "lookupLval(" << id << ")";
  for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
  {
    auto foundVar = scope->variables.find(id);
    if (foundVar != scope->variables.end())
    {
      return foundVar->second;
    }
  }
  LOG(DEBUG) << id << " not defined as variable";
  return nullptr;
}

std::shared_ptr<LiteralNode> SymbolTable::lookupConst(std::string id) const
{
  LOG(DEBUG) << "lookupConst(" << id << ")";
  for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
  {
    auto foundConst = scope->constants.find(id);
    if (foundConst != scope->constants.end())
    {
      return foundConst->second;
    }
  }
  LOG(DEBUG) << id << " not defined as constant";
  return nullptr;
}

void SymbolTable::storeVariable(std::string id, std::shared_ptr<Type> type)
{
  static int globalOffset = 0;

  if (type == nullptr)
  {
    LOG(ERROR) << id << "'s type is a nullptr!";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Find on top level - error if already defined
  auto top = scopes.rbegin();
  auto foundVar = top->variables.find(id);
  auto foundConst = top->constants.find(id);

  if (foundVar != top->variables.end())
  {
    LOG(ERROR) << id << " is already defined as a variable in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  if (foundConst != top->constants.end())
  {
    LOG(ERROR) << id << " is already defined as a constant in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  if (scopes.size() > 2) // Local Scope
  {
    // set the offset and increment by the type's size;
    auto var = std::make_shared<Variable>(id, type, FP, localOffset);
    localOffset -= var->type->size();

    // Insert in top level scope
    top->variables.emplace(id, var);
    LOG(DEBUG) << fmt::format("{}:{} stored in variable symbol table at scope {}, {}(${})",
                              id,
                              type->name(),
                              scopes.size() - 1,
                              localOffset + var->type->size(),
                              FP);
  }
  else // Global Scope
  {
    // set the offset and increment by the type's size;
    auto var = std::make_shared<Variable>(id, type, GP, globalOffset);
    globalOffset += var->type->size();

    // Insert in top level scope
    top->variables.emplace(id, var);
    LOG(DEBUG) << fmt::format("{}:{} stored in variable symbol table at scope {}, {}(${})",
                              id,
                              type->name(),
                              scopes.size() - 1,
                              globalOffset - var->type->size(),
                              GP);
  }
}
void SymbolTable::storeVariable(std::string id,
                                std::shared_ptr<Type> type,
                                int baseRegister,
                                int offset)
{
  if (type == nullptr)
  {
    LOG(ERROR) << id << "'s type is a nullptr!";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Find on top level - error if already defined
  auto top = scopes.rbegin();
  auto foundVar = top->variables.find(id);
  auto foundConst = top->constants.find(id);

  if (foundVar != top->variables.end())
  {
    LOG(ERROR) << id << " is already defined as a variable in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  if (foundConst != top->constants.end())
  {
    LOG(ERROR) << id << " is already defined as a constant in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // set the offset and increment by the type's size;
  auto var = std::make_shared<Variable>(id, type, baseRegister, offset);

  // Insert in top level scope
  top->variables.emplace(id, var);
  LOG(DEBUG) << fmt::format(
    "{}:{} stored in variable symbol table at scope {}", id, type->name(), scopes.size() - 1);
}

void SymbolTable::storeConst(std::string id, std::shared_ptr<LiteralNode> literal)
{
  if (literal == nullptr)
  {
    LOG(ERROR) << id << "'s LiteralNode is nullptr!";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Find on top level - error if already defined
  auto top = scopes.rbegin();
  auto foundVar = top->variables.find(id);
  auto foundConst = top->constants.find(id);

  if (foundVar != top->variables.end())
  {
    LOG(ERROR) << id << " is already defined as a variable in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  if (foundConst != top->constants.end())
  {
    LOG(ERROR) << id << " is already defined as a constant in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Insert in top level scope
  top->constants.emplace(id, literal);

  LOG(DEBUG) << fmt::format("{}:{} stored in constant symbol table at scope {}",
                            id,
                            literal->getType()->name(),
                            scopes.size() - 1);
}

void SymbolTable::storeType(std::string id, std::shared_ptr<Type> type)
{
  if (type == nullptr)
  {
    LOG(ERROR) << id << "'s type is a nullptr!";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }
  // Find on top level - error if already defined
  auto top = scopes.rbegin();
  auto foundType = top->types.find(id);

  if (foundType != top->types.end())
  {
    LOG(ERROR) << id << " is already defined as a type in the current scope\n";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Insert in top level scope
  top->types.emplace(id, type);
  LOG(DEBUG) << fmt::format(
    "{}:{} stored in type symbol table at scope {}", id, type->name(), scopes.size() - 1);
}

void SymbolTable::storeFunction(const std::string identifier,
                                std::vector<Parameter>& parameters,
                                std::string returnTypeName,
                                const Function::Declaration declaration)
{
  LOG(DEBUG) << "storeFunction(" << identifier << ")";
  for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope)
  {
    auto found = scope->functions.find(identifier);
    if (found != scope->functions.end())
    {
      if (declaration == Function::Declaration::FORWARD)
      {
        LOG(ERROR) << identifier << "already defined as function";
        LOG(INFO) << get_stacktrace();
        exit(EXIT_FAILURE);
      }
      else if (found->second->declaration == Function::Declaration::DEFINITION)
      {
        LOG(ERROR) << identifier << "already defined as function";
        LOG(INFO) << get_stacktrace();
        exit(EXIT_FAILURE);
      }
    }
  }

  // overwrites the forward declaration if it exists
  scopes.back().functions[identifier]
    = std::make_shared<Function>(identifier, parameters, returnTypeName, declaration);

  if (declaration == Function::Declaration::DEFINITION)
  {
    // TODO figure out how to deal with locals and parameters both being offset from FP
    int offset = 4;
    for (auto&& param : parameters)
    {
      for (auto&& id : param.ids)
      {
        // arguments are positive offsets from the FP
        storeVariable(id, param.getType(), FP, offset);
        fmt::print("# recorded {} -> {}($fp)\n", id, offset);
        offset += param.getType()->size();
      }
    }
  }

  LOG(DEBUG) << fmt::format("{}:{} stored in type symbol table at scope {}",
                            identifier,
                            returnTypeName,
                            scopes.size() - 1);
}

void SymbolTable::printStrings() const
{
  for (auto cur = strings.begin(); cur != strings.end(); ++cur)
  {
    std::cout << cur->second << ": .asciiz " << cur->first << '\n';
  }
}

void SymbolTable::enter_scope()
{
  scopes.emplace_back();
  localOffset = 0;
}

void SymbolTable::exit_scope()
{
  scopes.pop_back();
}

const std::shared_ptr<Type> Parameter::getType()
{
  if (type == nullptr)
  {
    type = symbol_table.lookupType(typeName);
  }
  return type;
}

std::shared_ptr<Type> Function::getType()
{
  init();
  return returnType;
}

void Function::init()
{
  if (initialized) return;
  initialized = true;

  if (returnTypeName != "__no_return__")
  {
    returnType = symbol_table.lookupType(returnTypeName);
  }

  int offset = 0;
  for (auto&& param : parameters)
  {
    for (auto&& id : param.ids)
    {
      table.emplace(id, std::make_pair(offset, param.getType()));
      offset += param.getType()->size();
    }
  }
}
