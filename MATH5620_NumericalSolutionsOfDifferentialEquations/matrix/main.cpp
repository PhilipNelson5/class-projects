#include "matrix.hpp"
#include "matrix_util.hpp"
#include "termColors.hpp"
#include <iostream>
#include <string>

template <typename T, typename R>
void test(T a, R r, std::string name)
{
  std::cout << GREEN << "[RUN     ] " << RESET << name << std::endl;
  if (a == r)
    std::cout << GREEN << "[      OK] " << RESET << std::endl;
  else
    std::cout << RED << "[    FAIL] " << RESET << std::endl;
}

// template <typename F, typename R>
// void test(F f, R r, std::string name)
// {
// std::cout << "[RUN     ] " << name << std::endl;
// if (f() == r)
// std::cout << "[      OK] " << name << std::endl;
// else
// std::cout << "[    FAIL] " << name << std::endl;
// }

int main()
{
  Matrix<int, 2, 2> a({{4, 0}, {1, -9}});
  Matrix<int, 2, 2> _a({{8, 0}, {2, -18}});
  Matrix<int, 2, 3> b({{1, 2, 3}, {4, 5, 6}});
  Matrix<int, 3, 2> c({{7, 8}, {9, 10}, {11, 12}});
  Matrix<int, 2, 2> _bc({{58, 64}, {139, 154}});
  Matrix<int, 2, 2> d({{3, 8}, {4, 6}});
  Matrix<int, 2, 2> e({{4, 0}, {1, -9}});
  Matrix<int, 2, 2> _de_add({{7, 8}, {5, -3}});
  Matrix<int, 2, 2> _de_sub({{-1, 8}, {3, 15}});
  Matrix<int, 2, 2> _e_neg({{-4, 0}, {-1, 9}});
  Matrix<int, 3, 3> _id({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
  Matrix<int, 3, 3> f({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
  Matrix<int, 1, 3> g({{2, 4, 6}});
  Matrix<int, 3, 1> h({{7}, {9}, {11}});
  Matrix<int, 3, 3> i({{6, 1, 1}, {4, -2, 5}, {2, 8, 7}});
  Matrix<int, 2, 3> _i_row({{6, 1, 1}, {2, 8, 7}});
  Matrix<int, 3, 2> _i_col({{6, 1}, {4, 5}, {2, 7}});

  test(d + e, _de_add, "matrix addition");
  test(d - e, _de_sub, "matrix subtraction");
  test(b * c, _bc, "matrix multiplication");
  test(2 * a, _a, "scalar * matrix");
  test(a * 2, _a, "matrix * scalar");
  test(-e, _e_neg, "unary minus, negation");
  test(identity<double, 3>(), _id, "identity construction");
  test(determinant(f), -306, "determinant");
  test(dotProduct(g, h), 116, "dot product");
  test(removeRow(i, 1), _i_row, "remove row");
  test(removeCol(i, 1), _i_col, "remove col");
}
