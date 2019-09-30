#ifndef SOLVE_LINEAR_SYSTEM_CHOLESKY_HPP_HPP
#define SOLVE_LINEAR_SYSTEM_CHOLESKY_HPP_HPP

#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../hw4/3-ForwardSubstitution/forwardSubstitution.hpp"
#include "../../hw4/4-BackSubstitution/backSubstitution.hpp"
#include "../8-CholeskyFactorization/choleskyFactorization.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> solve_linear_system_cholesky(Matrix<T> A, std::vector<T> b)
{
  auto L = cholesky_factorization(A);
  auto U = transpose(L);
  auto y = forward_substitution(L, b);
  auto x = back_substitution(U, y);
  return x;
}

#endif
