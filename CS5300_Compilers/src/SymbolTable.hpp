#ifndef SYMBOL_TABLE_HPP
#define SYMBOL_TABLE_HPP

#include <map>     // for map
#include <memory>  // for shared_ptr, __shared_ptr_access
#include <sstream> // for basic_ostream::operator<<, operator<<, stringstream
#include <string>  // for string
#include <utility> // for pair, make_pair
#include <vector>  // for vector
class LiteralNode;
class Type;

struct Variable
{
  std::string identifier;
  std::shared_ptr<Type> type;
  int memoryLocation;
  int offset = 0;

  Variable() = default;

  virtual ~Variable() = default;

  Variable(std::string& identifier,
           std::shared_ptr<Type> const& type,
           int memoryLocation,
           int offset = 0)
    : identifier(identifier)
    , type(type)
    , memoryLocation(memoryLocation)
    , offset(offset)
  {}

  std::string getLoc()
  {
    std::stringstream ss;
    ss << offset << "($" << memoryLocation << ")";
    return ss.str();
  }
};

struct Parameter
{
public:
  enum PassBy
  {
    VAL,
    REF
  };
  Parameter(const std::vector<std::string> ids,
            const std::string typeName,
            PassBy passBy = PassBy::VAL)
    : ids(ids)
    , typeName(typeName)
    , passBy(passBy)
    , type(nullptr)
  {}

  // Parameter(const std::vector<std::string> ids, std::shared_ptr<Type> type)
  //: ids(ids)
  //, typeName("")
  //, type(type)
  //{}

  const std::shared_ptr<Type> getType();

  const std::vector<std::string> ids;
  const std::string typeName;
  const PassBy passBy;

private:
  std::shared_ptr<Type> type;
};

struct Function
{
public:
  enum Declaration
  {
    FORWARD,
    DEFINITION
  };
  Function(const std::string identifier,
           const std::vector<Parameter>& parameters,
           const std::string returnTypeName,
           const Declaration declaration)
    : identifier(identifier)
    , parameters(parameters)
    , returnTypeName(returnTypeName)
    , returnType(nullptr)
    , declaration(declaration)
  {}

  const std::string identifier;
  std::vector<Parameter> parameters;
  const std::string returnTypeName;
  std::shared_ptr<Type> returnType;
  const Declaration declaration;
  std::map<std::string, std::pair<int, std::shared_ptr<Type>>> table;

  void init();
  std::shared_ptr<Type> getType();

private:
  bool initialized = false;
};

struct Scope
{
  std::map<std::string, std::shared_ptr<LiteralNode>> constants;
  std::map<std::string, std::shared_ptr<Variable>> variables;
  std::map<std::string, std::shared_ptr<Type>> types;
  std::map<std::string, std::shared_ptr<Function>> functions;
};

class SymbolTable
{
public:
  SymbolTable();

  std::pair<int, int> getReturnValueLocation() const
  {
    return {returnValueBaseRegister, returnValueOffset};
  }
  void setReturnValueLocation(int baseRegister, int offset)
  {
    returnValueBaseRegister = baseRegister;
    returnValueOffset = offset;
  }

  std::string getepilogueLable() const { return epilogueLbl; }
  void setepilogueLable(std::string lbl) { epilogueLbl = lbl; }

  std::shared_ptr<Type> getType(std::string id) const;
  std::shared_ptr<LiteralNode> lookupConst(std::string id) const;
  std::shared_ptr<Variable> lookupLval(std::string id) const;
  std::shared_ptr<Type> lookupType(std::string id) const;
  std::shared_ptr<Function> lookupFunction(std::string id) const;
  const std::string lookupString(std::string str);
  const std::string newLabel(std::string name);
  void storeConst(std::string id, std::shared_ptr<LiteralNode> expr);
  void storeVariable(std::string id, std::shared_ptr<Type> type);
  void storeVariable(std::string id, std::shared_ptr<Type> type, int baseRegister, int offset);
  void storeType(std::string id, std::shared_ptr<Type> type);
  void storeFunction(const std::string identifier,
                     std::vector<Parameter>&,
                     std::string returnTypeName,
                     const Function::Declaration declaration);
  void printStrings() const;
  void enter_scope();
  void exit_scope();

  static constexpr int GP = 28;
  static constexpr int SP = 29;
  static constexpr int FP = 30;

private:
  std::map<std::string, std::string> strings;
  std::vector<Scope> scopes;
  int localOffset;
  int returnValueBaseRegister;
  int returnValueOffset;
  std::string epilogueLbl;
};

extern SymbolTable symbol_table;

#endif
