extern int yyparse();
extern int yydebug;

#include "src/ProgramNode.hpp"     // for ProgramNode
#include "src/RegisterPool.hpp"    // for Register, Register::low
#include "src/log/easylogging++.h" // for Writer, LOG, DefaultLogBuilder

#include <cstring>  // for strcmp
#include <iostream> // for operator<<, cout, ostream, endl
#include <memory>   // for allocator, shared_ptr, __shared_p...
#include <stdlib.h> // for exit, EXIT_FAILURE, EXIT_SUCCESS
#include <string>   // for string, operator+

INITIALIZE_EASYLOGGINGPP

std::shared_ptr<ProgramNode> programNode;

void initEasyLogging(int argc, char** argv);
void showHelp();

int main(int argc, char** argv)
{
  initEasyLogging(argc, argv);

  yydebug = 0;
  auto f_source = false;
  auto f_parseOnly = false;

  for (auto i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "-h") == 0 | strcmp(argv[i], "--help") == 0)
    {
      showHelp();
      return EXIT_SUCCESS;
    }
    if (strcmp(argv[i], "-d") == 0 | strcmp(argv[i], "--debug") == 0) yydebug = 1;
    if (strcmp(argv[i], "-p") == 0 | strcmp(argv[i], "--parse-only") == 0) f_parseOnly = true;
    if (strcmp(argv[i], "-s") == 0 | strcmp(argv[i], "--source") == 0) f_source = true;
  }

  LOG(DEBUG) << "Parsing";
  yyparse();
  LOG(DEBUG) << "Parsing Complete";

  if (!f_parseOnly)
  {
    if (f_source)
    {
      LOG(DEBUG) << "Emitting Source";
      programNode->emitSource("");
    }
    else
    {
      LOG(DEBUG) << "Emitting Assembly";
      try
      {
        programNode->emit();
      }
      catch (const char* msg)
      {
        LOG(ERROR) << msg << std::endl;
        exit(EXIT_FAILURE);
      }
      LOG(INFO) << "most registers used: " << 18 - RegisterPool::Register::low;
    }
  }
}

void setLogLevel(std::string& config, std::string const& name, bool const& enabled)
{
  config += "* " + name + ":\n";
  if (enabled)
    config += "    ENABLED = true\n";
  else
    config += "    ENABLED = false\n";
}

void initEasyLogging(int argc, char* argv[])
{
  bool log_debug = false;
  bool log_warn = false;
  bool log_info = true;
  bool log_error = true;
  for (auto i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "-ld") == 0 | strcmp(argv[i], "--log-debug") == 0) log_debug = true;
    if (strcmp(argv[i], "-lw") == 0 | strcmp(argv[i], "--log-warning") == 0) log_warn = true;
    if (strcmp(argv[i], "-li") == 0 | strcmp(argv[i], "--no-log-info") == 0) log_error = false;
    if (strcmp(argv[i], "-le") == 0 | strcmp(argv[i], "--no-log-error") == 0)
      log_error = false;
  }
  /* clang-format off */
  std::string config = R"(
* GLOBAL:
    FORMAT               =  "[%level][%fbase:%line] %msg"
    ENABLED              =  true
    TO_FILE              =  false
    TO_STANDARD_OUTPUT   =  true
    MILLISECONDS_WIDTH   =  6
    PERFORMANCE_TRACKING =  true
    MAX_LOG_FILE_SIZE    =  2097152 ## 2MB
    LOG_FLUSH_THRESHOLD  =  1 ## Flush after every log
)";
  setLogLevel(config, "DEBUG", log_debug);
  setLogLevel(config, "INFO", log_info);
  setLogLevel(config, "WARNING", log_warn);
  setLogLevel(config, "ERROR", log_error);
  /* clang-format on */

  START_EASYLOGGINGPP(argc, argv);

  el::Configurations conf;
  conf.parseFromText(config);
  el::Loggers::setDefaultConfigurations(conf);
  el::Loggers::reconfigureAllLoggers(conf);
}

void showHelp()
{
  std::cout << "\n";
  std::cout << "     ______   _______    ______   __\n";
  std::cout << "    /      \\ |       \\  /      \\ |  \\\n";
  std::cout << "   |  $$$$$$\\| $$$$$$$\\|  $$$$$$\\| $$\n";
  std::cout << "   | $$   \\$$| $$__/ $$| $$___\\$$| $$\n";
  std::cout << "   | $$      | $$    $$ \\$$    \\ | $$\n";
  std::cout << "   | $$   __ | $$$$$$$  _\\$$$$$$\\| $$\n";
  std::cout << "   | $$__/  \\| $$      |  \\__| $$| $$_____\n";
  std::cout << "    \\$$    $$| $$       \\$$    $$| $$     \\\n";
  std::cout << "     \\$$$$$$  \\$$        \\$$$$$$  \\$$$$$$$$\n\n";

  std::cout << "-d\t--debug\t\tset yydebug to a positive value\n";
  std::cout << "-h\t--help\t\tshow this help message\n";
  std::cout << "-p\t--parse-only\tskip all assembly or source code emission\n";
  std::cout << "-s\t--source\tparse and emit source code\n";
  std::cout << "-ld\t--log-debug\tenable debugging level logging\n";
  std::cout << "-lw\t--log-warn\tenable warning level logging\n";
  std::cout << "-li\t--no-log-info\tdisable info level logging\n";
  std::cout << "-le\t--no-log-error\tdisable error level logging\n";
  std::cout << std::endl;
}
