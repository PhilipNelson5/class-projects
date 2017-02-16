#include "Rational.hpp"

/*-------------------------------------------------------------------------------*/
/*                                 [constructor]                                 */
/*-------------------------------------------------------------------------------*/
Rational::Rational(int n, int d) :numerator(n), denominator(d) {
	if (d == 0) 
		denominator = 1;
	this->simplify();
};

/*-------------------------------------------------------------------------------*/
/*                                   [get/set]                                   */
/*-------------------------------------------------------------------------------*/
int Rational::getNumerator() const { return numerator; }

int Rational::getDenominator() const { return denominator; }

void Rational::setNumerator(int n) { numerator = n; }

void Rational::setDenominator(int d) { denominator = d; }

/*-------------------------------------------------------------------------------*/
/*                               [simplification]                                */
/*-------------------------------------------------------------------------------*/
int GCD(int a, int b) {
	if (b == 0) return a;
	return GCD(b, a%b);
}

void Rational::simplify() {
	int gcd = GCD(denominator, numerator);
	numerator /= gcd;
	denominator /= gcd;

	if (denominator < 0){
		numerator = -numerator;
		denominator = -denominator;
	}
}

/*-------------------------------------------------------------------------------*/
/*                                   [casting]                                   */
/*-------------------------------------------------------------------------------*/
Rational::operator float() {
	return static_cast<float>(numerator) / denominator;
}

Rational::operator double() {
	return static_cast<double>(numerator) / denominator;
}

/*-------------------------------------------------------------------------------*/
/*                          [extraction and insertion]                           */
/*-------------------------------------------------------------------------------*/

std::ostream& operator<< (std::ostream& o, Rational const & a) {
	if (a.getNumerator() == 0)
		o << 0;
	else if (a.getDenominator() == 1)
		o << a.getNumerator();
	else
		o << a.getNumerator() << "/" << a.getDenominator();

	return o;
}

std::istream& operator>> (std::istream& i, Rational& a) {
	int t = 0;
	char ignore = ' ';
	i >> t;
	a.setNumerator(t);
	if(i >> ignore){
		i >> t;
		if (t == 0)
			a.setDenominator(1);
		else
			a.setDenominator(t);
	}
	else
		a.setDenominator(1);
	a.simplify();
	return i;
}

/*-------------------------------------------------------------------------------*/
/*                                    [cmath]                                    */
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
/*                            [arithmetic operators]                             */
/*-------------------------------------------------------------------------------*/
Rational operator-(Rational const & a) {
	return Rational(-a.getNumerator(), a.getDenominator());
}

Rational operator+ (Rational const & a, Rational const & b) {
	Rational c(
			a.getNumerator()*b.getDenominator() + b.getNumerator()*a.getDenominator(),
			a.getDenominator()*b.getDenominator());
	c.simplify();
	return c;
}

Rational operator+= (Rational& a, Rational const & b) {
	return a = a + b;
}

Rational operator- (Rational const & a, Rational const & b) {
	Rational c = a + (-b);
	c.simplify();
	return c;
}

Rational operator-= (Rational& a, Rational const & b) {
	return a = a - b;
}

Rational operator* (Rational const & a, Rational const & b) {
	Rational c(
			a.getNumerator()*b.getNumerator(),
			a.getDenominator()*b.getDenominator());
	c.simplify();
	return c;
}

Rational operator*= (Rational& a, Rational const & b) {
	return a = a * b;
}

Rational operator/ (Rational const & a, Rational const & b) {
	Rational c(b.getDenominator(), b.getNumerator());
	return a*c;
}

Rational operator/= (Rational& a, Rational const & b) {
	return a = a / b;
}

/*-------------------------------------------------------------------------------*/
/*                              [logic operators]                                */
/*-------------------------------------------------------------------------------*/
bool operator<  (Rational const & a, Rational const & b) {
	return a.getNumerator()*b.getDenominator() < b.getNumerator()*a.getDenominator();
}

bool operator<= (Rational const & a, Rational const & b) {
	return a.getNumerator()*b.getDenominator() <= b.getNumerator()*a.getDenominator();
}

bool operator>  (Rational const & a, Rational const & b) {
	return b < a;
}

bool operator>= (Rational const & a, Rational const & b) {
	return b <= a;
}

bool operator== (Rational const & a, Rational const & b) {
	return (a.getNumerator() == b.getNumerator() && a.getDenominator() == b.getDenominator());
}

bool operator!= (Rational const & a, Rational const & b) {
	return !(a == b);
}
