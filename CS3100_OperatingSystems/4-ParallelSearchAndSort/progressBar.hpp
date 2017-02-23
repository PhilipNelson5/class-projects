#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include "random.hpp"
#include <iomanip>
#include <string>

/******************\
\*****[COLORS]*****\
\******************\
black - 30
red - 31
green - 32
brown - 33
blue - 34
magenta - 35
cyan - 36
lightgray - 37
\*****************/

void printMsg(float prog, int total, std::string message)
{
  float progress = prog / total;
  int barWidth = 70;

  if (message != "")
  {
    std::cout << std::setw(barWidth + 10) << std::left << message << std::endl;
  }

  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i)
  {
    if (i < pos)
      std::cout << "\033[" + std::to_string(rand(30, 37)) + "m#\033[0m";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
}

void incProg(int &prog, int total, std::string message)
{
  ++prog;
  printMsg(prog, total, message);
}

#endif
