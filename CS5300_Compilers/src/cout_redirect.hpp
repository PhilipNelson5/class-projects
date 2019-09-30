#ifndef COUT_REDIRECT_HPP
#define COUT_REDIRECT_HPP

#include <iostream>

struct cout_redirect
{
  cout_redirect(std::streambuf* new_buffer)
    : old(std::cout.rdbuf(new_buffer))
  {}

  ~cout_redirect() { std::cout.rdbuf(old); }

private:
  std::streambuf* old;
};

#endif
