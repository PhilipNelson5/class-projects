#ifndef REGISTER_POOL_HPP
#define REGISTER_POOL_HPP

#include "../fmt/include/fmt/core.h" // for formatter

#include <ostream>  // for ostream
#include <stddef.h> // for size_t
#include <string>   // for operator+, to_string, string
#include <vector>   // for vector

namespace RegisterPool
{
class Register
{
public:
  // constructor
  Register();

  // move constructor
  Register(Register&& old);

  // move assignment operator
  Register& operator=(Register&& reg) = default;

  // destructor
  ~Register();

  // register name getter
  std::string toString() const { return "$" + std::to_string(name); }

  // get the registers in use
  const std::vector<int> static getRegistersInUse();

  // lowest number of registers in pool
  static size_t low;

private:
  // deleted copy constructor
  Register(const Register& reg) = delete;

  // deleted assignment operator
  Register& operator=(const Register& reg) = delete;

  // initialize the pool
  void static init();

  // Register name
  int name;

  // unmoved registers are valid
  bool valid;

  // pool of registers
  static std::vector<int> pool;

}; // class Register

// spill registers in use
void spill(std::vector<int> const& registers);

// spill registers in use
void unspill(std::vector<int> const& registers);

} // namespace RegisterPool

namespace fmt
{
template<>
struct formatter<RegisterPool::Register>
{
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<typename FormatContext>
  auto format(const RegisterPool::Register& r, FormatContext& ctx)
  {
    return format_to(ctx.begin(), "{}", r.toString());
  }
};
} // namespace fmt

std::ostream& operator<<(std::ostream& o, RegisterPool::Register const& r);

#endif
