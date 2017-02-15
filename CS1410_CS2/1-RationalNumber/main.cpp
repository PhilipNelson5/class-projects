#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include "Rational.hpp"

std::string isTrue(bool b) {
	if (b) return "true";
	return "false";
}

int main(void) {

	Rational a;
	Rational b;
	Rational c;
	Rational d(3);
	Rational e(4, 6);
	Rational f(5, 7);
	Rational g(1, 2);
	Rational h(2, 3);
	Rational i(3, 5);
	Rational j(8, 2);

	std::cout << "Enter a rational number in this format [int]/[int]: ";
	std::cin >> a;
	std::cout << std::endl << "Enter another rational number: ";
	std::cin >> b;
	std::cout << std::endl;

	std::cout << std::endl << "a = " << a << ", " << "b = " << b << std::endl;
	std::cout << "a + b = " << a + b << std::endl;
	a += b;
	std::cout << "a +=  b" << ": a = " << a << std::endl;

	std::cout << std::endl << "c = " << c << ", " << "b = " << d << std::endl;
	std::cout << "c - d = " << c - d << std::endl;
	c -= d;
	std::cout << "c -=  d" << ": c = " << c << std::endl;

	std::cout << std::endl << "e = " << e << ", " << "f = " << f << std::endl;
	std::cout << "e * f = " << e * f << std::endl;
	e *= f;
	std::cout << "e *=  f" << ": e = " << e << std::endl;

	std::cout << std::endl << "g = " << g << ", " << "h = " << h << std::endl;
	std::cout << "g / h = " << g / h << std::endl;
	g /= h;
	std::cout << "g /=  h" << ": g = " << g << std::endl;

	std::cout << std::endl << "i = " << i << ", " << "j = " << j << std::endl;
	std::cout << "i to float = " << static_cast<float>(i) << std::endl;
	std::cout << std::setprecision(17);
	std::cout << "j to double = " << static_cast<double>(j) << std::endl;

	std::cout << std::endl << "a = " << a << ", " << "b = " << b << std::endl;
	std::cout << "a <  b: " << isTrue(a < b) << std::endl;
	std::cout << "a <= b: " << isTrue(a <= b) << std::endl;
	std::cout << "a >  b: " << isTrue(a > b) << std::endl;
	std::cout << "a >= b: " << isTrue(a >= b) << std::endl;
	std::cout << "a == b: " << isTrue(a == b) << std::endl;
	std::cout << "a != b: " << isTrue(a != b) << std::endl;

	return EXIT_SUCCESS;
}
