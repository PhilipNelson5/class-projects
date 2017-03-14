#include "command.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

Command::Command(std::string input)
  : line(input), infile(""), hasInFile(false), outfile(""), hasOutFile(false), cmd_c(0)
{
  auto parsed = parse(line);

  for (auto i = 0u; i < parsed.size(); ++i)
  {
    if (parsed[i][0] == "<")
    {
      infile = parsed[++i][0];
      hasInFile = true;
    }
    else if (parsed[i][0] == ">")
    {
      outfile = parsed[++i][0];
      hasOutFile = true;
    }
    else if (parsed[i][0] == "|")
    {
      continue;
    }
    else
    {
      cmd_v.push_back(parsed[i]);
      ++cmd_c;
    }
  }
  // std::cerr << "IN: " << infile << " OUT: " << outfile << " CMDS: " << cmd_c << std::endl;
}

std::string Command::toString()
{
  return line;
}

int Command::size()
{
  return cmd_c;
}

std::vector<std::vector<std::string>> Command::parse(std::string input)
{
  /***************************************
   *      Tokens of cmds and pipes
   **************************************/

  std::vector<std::string> cmds;
  // std::vector<char> iomanip;

  auto curr = input.end();
  auto begin = input.begin();
  while (true)
  {
    curr =
      std::find_if(begin, input.end(), [](char c) { return (c == '|' || c == '<' || c == '>'); });
    cmds.emplace_back(begin, curr);
    cmds.push_back(std::string(1, *curr));
    if (curr == input.end()) break;
    // iomanip.push_back(*curr);

    begin = curr + 2;
  }
  cmds.erase(cmds.end());

  /***************************************
   *     vector of commands as words
   **************************************/

  std::vector<std::vector<std::string>> words;
  for (auto i = 0u; i < cmds.size(); ++i)
  {
    std::istringstream iss(cmds[i]);
    std::vector<std::string> temp;
    std::string wrd;

    while (iss >> wrd)
      temp.push_back(wrd);
    words.push_back(temp);
  }
  return words;
}
