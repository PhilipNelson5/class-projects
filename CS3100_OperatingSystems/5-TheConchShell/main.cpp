#include "path.hpp"
#include "termColors.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
  std::string txtcolor = BLUE;
  std::vector<std::vector<std::string>> history;
  std::vector<std::vector<std::string>> alias;
  Timer t;
}

auto parse(std::string line)
{
  std::istringstream iss(line);
  std::string wrd;
  std::vector<std::string> words;

  while (iss >> wrd)
    words.push_back(wrd);

  return words;
}

void load()
{
  std::ifstream fin(".conchrc");
  std::string line;
  while (std::getline(fin, line))
    alias.push_back(parse(line));
}

char *const *buildAliasArgs(const std::vector<std::string> &words)
{
  char **args = new char *[words.size()];
  for (auto i = 1u; i < words.size(); ++i)
  {
    args[i - 1] = new char[words[i].size() + 1];
    strcpy(args[i - 1], words[i].c_str());
  }
  args[words.size() - 1] = nullptr;
  return args;
}

char *const *buildArgs(const std::vector<std::string> &words)
{
  char **args = new char *[words.size() + 1];
  for (auto i = 0u; i < words.size(); ++i)
  {
    args[i] = new char[words[i].size() + 1];
    strcpy(args[i], words[i].c_str());
  }
  args[words.size()] = nullptr;
  return args;
}

void exec(char *const *&args)
{
  int pid = fork();
  if (pid < 0)
    std::cout << "fork failed..." << std::endl;
  else if (pid == 0)
  {
    execvp(args[0], args);
    perror(args[0]);
    exit(EXIT_FAILURE);
  }
  else // (pid > 0)
  {
    int status;
    waitpid(pid, &status, 0);
    if (status != EXIT_SUCCESS)
    {
      std::cout << RED << "FAILURE TO EXECUTE PROPERLY" << txtcolor << std::endl;
    }
  }
}

void hist(const std::vector<std::vector<std::string>> &history)
{
  std::cout << GREEN << "[HISTORY]" << BLUE << std::endl;
  for (auto i = 0u; i < history.size(); ++i)
  {
    std::cout << i + 1 << ": ";
    for (auto &&e : history[i])
    {
      std::cout << e << " ";
    }
    std::cout << std::endl;
  }
}

void color(std::vector<std::string> &words)
{
  if (words.size() >= 2)
  {
    if (words[1] == "BLACK")
      txtcolor = BLACK;
    else if (words[1] == "RED")
      txtcolor = RED;
    else if (words[1] == "GREEN")
      txtcolor = GREEN;
    else if (words[1] == "YELLOW")
      txtcolor = YELLOW;
    else if (words[1] == "BLUE")
      txtcolor = BLUE;
    else if (words[1] == "MAGENTA")
      txtcolor = MAGENTA;
    else if (words[1] == "CYAN")
      txtcolor = CYAN;
    else if (words[1] == "WHITE")
      txtcolor = WHITE;
    else
      std::cout << "RED     GREEN     YELLOW     BLUE     MAGENTA     CYAN     WHITE \n";
  }
  else
    std::cout << "RED     GREEN     YELLOW     BLUE     MAGENTA     CYAN     WHITE \n";
}

void cd(std::vector<std::string> &words)
{
  if (words.size() < 2)
  {
    std::string dir("/home/");
    dir.append(getLogin());
    if (chdir(dir.c_str()) < 0) std::cout << words[0] << ": No such file or directory" << std::endl;
  }
  else if (chdir(words[1].c_str()) < 0)
    std::cout << words[0] << ": No such file or directory" << std::endl;
}

bool aliases(std::vector<std::string> &words)
{
  for (auto &&a : alias)
    if (words[0] == a[0])
    {
      auto args = buildAliasArgs(a);
      exec(args);
      return true;
    }
  return false;
}

void run(std::vector<std::string> words)
{
  auto args = buildArgs(words);

  history.push_back(words);
  if (words[0] == "exit")
    return exit(EXIT_SUCCESS);
  else if (words[0] == "history")
    hist(history);
  else if (words[0] == "^")
  {
    unsigned int hist = stoi(words[1]) - 1;
    if (stoi(words[1]) >= 0 && hist <= history.size())
    {
      run(history[hist]);
    }
    else
      std::cout << RED << "History out of bounds" << txtcolor << std::endl;
  }
  else if (words[0] == "color")
    color(words);
  else if (words[0] == "cd")
    cd(words);
  else if (words[0] == "ptime")
    std::cout << t.ptime() << std::endl;
  else if (aliases(words))
    return;
  else
  {
    exec(args);
  }
}

int main()
{
  load();
  for (;;)
  {
    std::cout << GREEN << getLogin() << "@" << getHostName() << WHITE << ":" << CYAN << getCWD()
              << WHITE << "$ " << txtcolor;
    std::string line;
    std::getline(std::cin, line);
    if (line == "") continue;
    auto words = parse(line);
    t.time([&]() { run(words); });
  }

  return EXIT_SUCCESS;
}
/**
 * system calls needed for pt2
 * dup2(oldfd, newfd) //after the fork, open a tile, get file descriptor, coppy to std::in or
 * std::out
 *
 * pipe(p)
 */
