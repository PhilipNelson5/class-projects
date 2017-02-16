#include "Rational.hpp"
#include "rpnCalc.hpp"
#include <exception>
#include <iostream>
#include <string>

void instructions()
{
  std::cout << "Enter numbers and operators separated by spaces or new lines" << std::endl;
  std::cout << "Integers are supported and rationals numbers in the form [int]/[int]" << std::endl;
  std::cout << "Enter 'r' to remove the last number" << std::endl;
  std::cout << "Enter 'c' to clear the stack" << std::endl;
  std::cout << "Enter 'e' to end" << std::endl;
  std::cout << "Enter 'h' for help" << std::endl;
}
int main()
{
  auto cont = true;
  std::string input = "";
  Calc calc;

  std::cout << "Welcome to the RPN calculator" << std::endl;
  instructions();

  while (cont)
  {
    std::getline(std::cin, input);
    std::string temp;
    std::stringstream ss{input};
    while (ss >> temp)
    {
      try
      {
        switch (temp[0])
        {
        case '+':
          calc.operation([](Rational a, Rational b) { return a + b; });
          break;
        case '-':
          if (!isdigit(temp[1]))
            calc.operation([](Rational a, Rational b) { return a - b; });
          else
            calc.push(temp);
          break;
        case '*':
        case 'x':
          calc.operation([](Rational a, Rational b) { return a * b; });
          break;
        case '/':
        case '\\':
          calc.operation([](Rational a, Rational b) { return a / b; });
          break;
        case 'r':
          calc.remove();
          break;
        case 'c':
          calc.clear();
          break;
        case 'h':
          instructions();
          break;
        case 'e':
          cont = false;
          break;
        default:
          calc.push(temp);
        }
      }
      catch (std::exception &e)
      {
        std::cerr << e.what() << std::endl;
      }
    }
    calc.print();
  }
}
