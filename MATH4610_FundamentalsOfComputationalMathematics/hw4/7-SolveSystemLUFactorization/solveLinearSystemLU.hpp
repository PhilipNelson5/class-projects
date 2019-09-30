#ifndef SOLVE_L_U_FACTORIZATION_HPP
#define SOLVE_L_U_FACTORIZATION_HPP

#include "../../utils/matrixUtils.hpp"
#include "../6-LUFactorization/LUFactorization.hpp"
#include "../3-ForwardSubstitution/forwardSubstitution.hpp"
#include "../4-BackSubstitution/backSubstitution.hpp"
#include <iostream>
#include <tuple>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 * Solve a system of equations Ax = b
 *
 * @tparam T The type of the elements of A and b
 * @param  A The matrix of linear systems of equations 
 * @param  b The right hand side
 * @return A vector of solutions x
 */
template <typename T>
std::vector<T> solve_linear_system_LU(Matrix<T> A, std::vector<T> b)
{
  auto [L, U, P] = LU_factorize(A);
  auto y = forward_substitution(L, P * b);
  auto x = back_substitution(U, y);
  return x;
}

#endif
