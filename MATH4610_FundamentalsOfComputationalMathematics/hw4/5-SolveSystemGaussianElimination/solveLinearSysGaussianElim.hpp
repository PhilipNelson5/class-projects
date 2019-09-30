#ifndef SOLVE_LINEAR_SYSTEM_GAUSSIAN_ELIMINATION_HPP
#define SOLVE_LINEAR_SYSTEM_GAUSSIAN_ELIMINATION_HPP

#include "../1-GaussianElimination/gaussianElimination.hpp"
#include "../3-ForwardSubstitution/forwardSubstitution.hpp"
#include "../4-BackSubstitution/backSubstitution.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> solve_linear_system_gaussian_elimination(Matrix<T>& m,
                                                        std::vector<T>& b)
{
  // create augmented matrix
  for (auto i = 0u; i < m.size(); ++i)
    m[i].push_back(b[i]);

  // perform Gaussian elimination of the augmented matrix
  gaussian_emlimination(m);

  // remove augmentation
  for (auto i = 0u; i < m.size(); ++i)
  {
    b[i] = m[i].back();
    m[i].pop_back();
  }

  // do back substitution
  auto x = back_substitution(m, b);

  // return the answer
  return x;
}

#endif
