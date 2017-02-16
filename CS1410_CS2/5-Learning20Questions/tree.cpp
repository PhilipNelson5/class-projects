#include "tree.hpp"
#include <iostream>
#include <vector>

bool isTrue(std::string s)
{
  std::cout << s << std::endl;
  while (true)
  {
    std::getline(std::cin, s);
    if (std::tolower(s[0]) == 'y') return true;
    if (std::tolower(s[0] == 'n')) return false;
    std::cout << "please enter [y]es or [n]o" << std::endl;
  }
}

void Question::doWork(std::shared_ptr<Node> &cur)
{
  if (isTrue(cur->data))
    yes->doWork(yes);
  else
    no->doWork(no);
}

void Answer::doWork(std::shared_ptr<Node> &cur)
{
  if (isTrue("Are you " + data + " ?"))
  {
    std::cout << "I win =D" << std::endl;
  }
  else
  {
    std::string rightAnswer, newQuestion;
    std::cout << "What were you thinking of?" << std::endl;
    std::getline(std::cin, rightAnswer);
    std::cout << "What could I have asked to know you were " << rightAnswer << " not " << data
              << " ?" << std::endl;

    std::getline(std::cin, newQuestion);

    cur = std::make_shared<Question>(
      newQuestion, std::make_shared<Answer>(rightAnswer), std::make_shared<Answer>(cur->data));
  }
}

void Question::print(std::ostream &o)
{
  no->print(o);
  yes->print(o);
  o << "Q" << data << std::endl;
}

void Answer::print(std::ostream &o)
{
  o << "A" << data << std::endl;
}

void Tree::read(std::istream &i)
{
  std::string s;
  std::vector<std::shared_ptr<Node>> stack;
  while (std::getline(i, s))
  {
    if (s[0] == 'A')
      stack.push_back(std::make_shared<Answer>(s.substr(1)));
    else
    {
      auto first = stack.back();
      stack.pop_back();
      auto second = stack.back();
      stack.pop_back();
      stack.push_back(std::make_shared<Question>(s.substr(1), first, second));
    }
  }
  root = stack.back();
}
