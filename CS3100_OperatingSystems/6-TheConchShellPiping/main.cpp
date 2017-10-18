#include "command.hpp"
#include "path.hpp"
#include "termColors.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define READ_END 0
#define WRITE_END 1
namespace
{
  std::string txtcolor = BLUE;
  std::vector<Command> history;
  std::vector<Command> alias;
  Timer t;
}

void run(Command);

// load aliases
void load()
{
  std::ifstream fin(".conchrc");
  if (!fin) return;
  // std::cerr << "loading aliases" << std::endl;
  std::string line;
  while (std::getline(fin, line)) alias.push_back(Command(line));
}

char* const* buildArgs(const std::vector<std::string>& words)
{
  char** args = new char*[words.size() + 1];
  for (auto i = 0u; i < words.size(); ++i)
  {
    args[i] = new char[words[i].size() + 1];
    strcpy(args[i], words[i].c_str());
  }
  args[words.size()] = nullptr;
  return args;
}

void exec(char* const*& args)
{
  int pid = fork();
  if (pid < 0)
    perror("fork failed...");
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
      std::cout << RED << "FAILURE TO EXECUTE PROPERLY" << txtcolor
                << std::endl;
    }
  }
}

void hist(const std::vector<Command>& history)
{
  std::cout << GREEN << "[HISTORY]" << BLUE << std::endl;

  for (auto i = 0u; i < history.size(); ++i)
  {
    std::cout << i + 1 << ": ";
    std::cout << history[i].line << std::endl;
  }
}

// text color change
void color(std::vector<std::string>& words)
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
      std::cout << "RED     GREEN     YELLOW     BLUE     MAGENTA     CYAN     "
                   "WHITE \n";
  }
  else
    std::cout
      << "RED     GREEN     YELLOW     BLUE     MAGENTA     CYAN     WHITE \n";
}

void cd(std::vector<std::string>& words)
{
  if (words.size() < 2)
  {
    std::string dir("/home/");
    dir.append(getLogin());
    if (chdir(dir.c_str()) < 0)
      std::cout << words[0] << ": No such file or directory" << std::endl;
  }
  else if (chdir(words[1].c_str()) < 0)
    std::cout << words[0] << ": No such file or directory" << std::endl;
}

// aliases
bool aliases(std::vector<std::string>& words)
{
  for (auto&& a : alias)
    if (words[0] == a.cmd_v[0][0])
    {
      Command words = a;
      words.cmd_v.erase(words.cmd_v.begin());
      --words.cmd_c;
      run(words);
      return true;
    }
  return false;
}

void run(Command cmds)
{
  history.push_back(cmds);
  /*******************************************************************/
  /*                          piping                                 */
  /*******************************************************************/
  if (cmds.size() != 1 || cmds.infile != "" || cmds.outfile != "")
  {
    pid_t pid;
    int fd[2][2];

    int curr_in = 0;
    int curr_out = 1;

    // auto std_in = dup(STDIN_FILENO);
    // auto std_out = dup(STDOUT_FILENO);

    // make a new input pipe
    if (pipe(fd[curr_in]) < 0) perror("pipe failed");
    for (auto i = 0; i < cmds.size(); ++i)
    {
      // make a new output pipe
      if (pipe(fd[curr_out]) < 0) perror("pipe failed");

      auto args = buildArgs(cmds.cmd_v[i]);
      pid = fork();

      /*********************************************/
      /*                   child                   */
      /*********************************************/
      if (pid == 0)
      {
        /*********************************************/
        /*               first and last              */
        /*********************************************/
        if (cmds.cmd_c == 1)
        {
          std::cerr << "FIRSTLAST: " << cmds.cmd_v[i][0] << std::endl;

          if (cmds.hasInFile)
          {
            std::cerr << "IN FILE: " << cmds.infile << std::endl;
            int infd = open(cmds.infile.c_str(), O_RDWR, S_IWUSR | S_IRUSR);
            dup2(infd, STDIN_FILENO);
            // dup2(fd[curr_out][WRITE_END], STDOUT_FILENO);
            close(infd);
          }

          if (cmds.hasOutFile)
          {
            std::cerr << "OUT FILE: " << cmds.outfile << std::endl;
            int outfd =
              open(cmds.outfile.c_str(), O_RDWR | O_CREAT, S_IWUSR | S_IRUSR);
            dup2(outfd, STDOUT_FILENO);
            // dup2(fd[curr_in][READ_END], STDIN_FILENO);
            close(outfd);
          }
        }
        else
        {
          /*********************************************/
          /*               first command               */
          /*********************************************/
          if (i == 0)
          {
            std::cerr << "FIRST: " << cmds.cmd_v[i][0] << std::endl;
            if (cmds.hasInFile)
            {
              std::cerr << "IN FILE: " << cmds.infile << std::endl;
              int infd = open(cmds.infile.c_str(), O_RDWR, S_IWUSR | S_IRUSR);
              dup2(infd, STDIN_FILENO);
              dup2(fd[curr_out][WRITE_END], STDOUT_FILENO);
              close(infd);
            }
            else
            {
              std::cerr << "NO INFILE" << std::endl;
              dup2(fd[curr_out][WRITE_END], STDOUT_FILENO);
            }
          }

          /*********************************************/
          /*                last command               */
          /*********************************************/
          if (i == cmds.size() - 1)
          {
            std::cerr << "LAST: " << cmds.cmd_v[i][0] << std::endl;
            if (cmds.hasOutFile)
            {
              std::cerr << "OUT FILE: " << cmds.outfile << std::endl;
              int outfd =
                open(cmds.outfile.c_str(), O_RDWR | O_CREAT, S_IWUSR | S_IRUSR);
              dup2(outfd, STDOUT_FILENO);
              dup2(fd[curr_in][READ_END], STDIN_FILENO);
              close(outfd);
            }
            else
            {
              std::cerr << "NO OUTFILE" << std::endl;
              dup2(fd[curr_in][READ_END], STDIN_FILENO);
            }
          }

          /*********************************************/
          /*                mid command                */
          /*********************************************/
          if (i > 0 && i < cmds.size() - 1)
          {
            std::cerr << "MID CMD: " << cmds.cmd_v[i][0] << std::endl;
            dup2(fd[curr_in][READ_END], STDIN_FILENO);
            dup2(fd[curr_out][WRITE_END], STDOUT_FILENO);
            close(fd[curr_in][WRITE_END]); //
            close(fd[curr_out][WRITE_END]);
          }

          execvp(args[0], args);
          perror(args[0]);
          exit(EXIT_FAILURE);
        }
      }

      /*********************************************/
      /*                  parent                   */
      /*********************************************/
      else
      {
        int status;
        waitpid(pid, &status, 0);
        close(fd[curr_out][WRITE_END]);
        std::swap(curr_in, curr_out);
      }
    }
    // dup2(std_in, STDIN_FILENO);
  }

  /*******************************************************************/
  /*                    single command execution                     */
  /*******************************************************************/
  else
  {
    auto words = cmds.cmd_v[0];
    auto args = buildArgs(words);
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
}

void handle(int){}
int main()
{
  load();
  for (;;)
  {
    std::cout << GREEN << getLogin() << "@" << getHostName() << WHITE << ":"
              << CYAN << getCWD() << WHITE << "$ " << txtcolor;
    std::string line = "cat timer.hpp | grep // | grep return | grep total";

    signal(SIGINT, SIG_IGN);

    if (!std::getline(std::cin, line))
    {
      std::cerr << "Unexpected end of file" << std::endl;
      return EXIT_SUCCESS;
    }

    if (line == "") continue;
    Command cmds(line);
    t.time([&]() { run(cmds); });
  }

  return EXIT_SUCCESS;
}
