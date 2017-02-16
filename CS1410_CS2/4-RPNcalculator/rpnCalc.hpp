#ifndef RPNCALC_HPP
#define RPNCALC_HPP

#include <exception>
#include <vector>
#include "Rational.hpp"

class Calc {
	public:
		void print();
		template <typename F>
			void operation(F);
		void push(std::string);
		void remove();
		void clear();
	private:
		std::vector<Rational> stack;
};

template <typename T>
std::ostream& operator<< (std::ostream& o, std::vector<T> const & v){
	for(auto & e:v)
		o << e << ", ";
	return o;
}

template <typename F>
void Calc::operation(F f){
	if(stack.size() < 2) 
		throw std::out_of_range("Can not operate on fewer than 2 numbers\nPlease enter more numbers first.");
	auto second =	stack.back();
	stack.pop_back();
	auto first =	stack.back();
	stack.pop_back();
	this -> stack.push_back(f(first, second));
}

#endif
