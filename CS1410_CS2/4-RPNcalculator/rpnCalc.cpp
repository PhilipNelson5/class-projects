#include <string>
#include "rpnCalc.hpp"

void Calc::print(){
	std::cout << stack << std::endl;
}

void Calc::push (std::string input){
	std::string temp;
	std::stringstream ss { input };
	while(ss>>temp){
		stack.emplace_back(temp);
	}
}

void Calc::remove(){
	stack.pop_back();
}

void Calc::clear(){
	stack.clear();
}
