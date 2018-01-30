#ifndef RATIONAL_HPP
#define RATIONAL_HPP

#include <iostream>
#include <string>
#include <sstream>

class Rational;

std::ostream& operator<< (std::ostream& o, Rational const & a);
std::istream& operator>> (std::istream& i, Rational& a);

class Rational {

public:
	Rational(int n = 0, int d = 1);
	Rational(std::string s){std::stringstream ss { s }; ss>>*this;}
	int getNumerator() const;
	int getDenominator() const;
	void setNumerator(int n);
	void setDenominator(int d);
	void simplify();

	operator float();
	operator double();

private:
	int numerator;
	int denominator;

};

Rational operator- (Rational const & a);
Rational operator+ (Rational const & a, Rational const & b);
Rational operator+= (Rational& a, Rational const & b);
Rational operator- (Rational const & a, Rational const & b);
Rational operator-= (Rational& a, Rational const & b);
Rational operator* (Rational const & a, Rational const & b);
Rational operator*= (Rational& a, Rational const & b);
Rational operator/ (Rational const & a, Rational const & b);
Rational operator/= (Rational& a, Rational const & b);

bool operator<  (Rational const & a, Rational const & b);
bool operator<= (Rational const & a, Rational const & b);
bool operator>  (Rational const & a, Rational const & b);
bool operator>= (Rational const & a, Rational const & b);
bool operator== (Rational const & a, Rational const & b);
bool operator!= (Rational const & a, Rational const & b);

#endif
