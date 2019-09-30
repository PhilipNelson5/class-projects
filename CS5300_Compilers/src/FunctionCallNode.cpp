#include "FunctionCallNode.hpp"

#include "SymbolTable.hpp"
#include "Type.hpp"
#include "stacktrace.hpp"

void FunctionCallNode::emitSource(std::string indent)
{
  std::cout << indent << " " << id << "(";
  if (args.size() > 0)
  {
    if (args.size() > 1)
      for (auto i = 0u; i < args.size() - 1; ++i)
      {
        args[i]->emitSource("");
        std::cout << ", ";
      }
    args.back()->emitSource("");
  }
  std::cout << ")";
}

Value FunctionCallNode::emit()
{
  std::cout << "\n# ";
  emitSource("");

  auto info = symbol_table.lookupFunction(id);
  if (info == nullptr)
  {
    LOG(ERROR) << id << " not defined as a function";
    LOG(INFO) << get_stacktrace();
    exit(EXIT_FAILURE);
  }

  // Spill registers
  auto reg_in_use = RegisterPool::Register::getRegistersInUse();
  RegisterPool::spill(reg_in_use);

  // make space for the return value
  fmt::print("addi $sp, $sp, -4\n");

  // Setup procedure arguments
  int args_size = 0;
  for (auto&& arg : args)
  {
    args_size += arg->getType()->size();
  }
  fmt::print("addi $sp, $sp, -{}\n", args_size);

  // Store arguments on stack
  // TODO make sure these are put in the right place below FP
  fmt::print("# Store function arguments\n");
  std::vector<Value> vals_args;
  int offset = 4;
  for (auto&& arg : args)
  {
    std::cout << "# ";
    arg->emitSource("");
    std::cout << "\n";

    auto val_arg = arg->emit();
    auto size = arg->getType()->size();
    if (size == 4)
    {
      auto reg_arg = val_arg.getTheeIntoARegister();
      fmt::print("sw {}, {}($sp)\n", reg_arg, offset);
      offset += size;
    }
    else
    {
      RegisterPool::Register tmp;
      auto [offsetArg, baseRegister] = std::get<std::pair<int, int>>(val_arg.value);
      fmt::print("# {}, ${}\n", offsetArg, baseRegister);
      for (int i = 0; i < size; i += 4)
      {
        fmt::print("lw {}, {}(${})\n", tmp, i+offsetArg, baseRegister);
        fmt::print("sw {}, {}($sp)\n", tmp, offset);
        offset += 4;
      }
    }
  }

  // Make the call
  fmt::print("jal {}\n", id);

  // Remove arguments
  fmt::print("addi $sp, $sp, {}\n", args_size);

  // Get return value
  RegisterPool::Register returnValue;
  fmt::print("lw {}, 0($sp)\n", returnValue);
  fmt::print("addi $sp, $sp, 4\n");

  // Unspill registers
  RegisterPool::unspill(reg_in_use);
  return returnValue;
}

const std::shared_ptr<Type> FunctionCallNode::getType()
{
  if (type == nullptr)
  {
    auto func = symbol_table.lookupFunction(id);
    type = func->getType();
  }
  return type;
}

bool FunctionCallNode::isConstant() const
{
  return false;
}

std::variant<std::monostate, int, char, bool> FunctionCallNode::eval() const
{
  return {};
}
