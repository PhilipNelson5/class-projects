#include "FormalParameter.hpp"

#include "Type.hpp" // for Type

#include <iostream> // for operator<<, cout, ostream, basic_ostream

void FormalParameter::emitSource(std::string indent)
{
  (void)indent;

  if (passby == PassBy::VAL)
    std::cout << "var ";
  else
    std::cout << "ref ";

  if (ids.size() > 0)
  {
    if (ids.size() > 1)
      for (auto i = 0u; i < ids.size() - 1; ++i)
      {
        std::cout << ids[i] << ", ";
      }
    std::cout << ids.back() << " : " << type->getType()->name();
  }
}

void FormalParameter::emit() {}
