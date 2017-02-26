#ifndef COMMAND_HPP
#define COMMAND_HPP
#include <string>
#include <vector>

class Command
{
public:
  Command(std::string input);
	std::string line;
  std::vector<std::vector<std::string>> cmd_v;
  std::string infile;
  std::string outfile;
	int cmd_c;

	std::string toString();
	int size();
private:
	std::vector<std::vector<std::string>> parse(std::string line);
};

#endif
