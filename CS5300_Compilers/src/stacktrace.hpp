// Original stacktrace.h (c) 2008, Timo Bingmann from http://idlebox.net/
// https://panthema.net/2008/0901-stacktrace-demangled/
// published under the WTFPL v2.0
//
// Edits stacktrace.hpp (C) 2019 Philip Nelson
// published under the WTFPL v2.0
//
// Compile with -export-dynamic

#ifndef STACKTRACE_HPP
#define STACKTRACE_HPP

#include <cxxabi.h>
#include <execinfo.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// return a demangled stack backtrace of the caller function
static std::string get_stacktrace(unsigned int max_frames = 63)
{
  std::stringstream ss;
  ss << "stack trace:\n";

  // storage array for stack trace address data
  void* addrlist[max_frames + 1];

  // retrieve current stack addresses
  int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

  if (addrlen == 0)
  {
    ss << "  <empty, possibly corrupt>\n";
    return ss.str();
  }

  // resolve addresses into strings containing "filename(function+address)",
  // this array must be free()-ed
  char** symbollist = backtrace_symbols(addrlist, addrlen);

  // allocate string which will be filled with the demangled function name
  size_t funcnamesize = 256;
  char* funcname = (char*)malloc(funcnamesize);

  // iterate over the returned symbol lines. skip the first, it is the
  // address of this function.
  for (int i = 1; i < addrlen; i++)
  {
    char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

    // find parentheses and +address offset surrounding the mangled name:
    // ./module(function+0x15c) [0x8048a6d]
    for (char* p = symbollist[i]; *p; ++p)
    {
      if (*p == '(')
        begin_name = p;
      else if (*p == '+')
        begin_offset = p;
      else if (*p == ')' && begin_offset)
      {
        end_offset = p;
        break;
      }
    }

    if (begin_name && begin_offset && end_offset && begin_name < begin_offset)
    {
      *begin_name++ = '\0';
      *begin_offset++ = '\0';
      *end_offset = '\0';

      // mangled name is now in [begin_name, begin_offset) and caller
      // offset in [begin_offset, end_offset). now apply
      // __cxa_demangle():

      int status;
      char* ret
        = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
      if (status == 0)
      {
        funcname = ret; // use possibly realloc()-ed string
        ss << "  " << symbollist[i] << " : " << funcname << "+" << begin_offset
           << "\n";
      }
      else
      {
        // demangling failed. Output function name as a C function with
        // no arguments.
        ss << "  " << symbollist[i] << " : " << begin_name << "()+"
           << begin_offset << "\n";
      }
    }
    else
    {
      // couldn't parse the line? add the whole line.
      ss << "  " << symbollist[i] << "\n";
    }
  }

  free(funcname);
  free(symbollist);
  return ss.str();
}

#endif // STACKTRACE_HPP
