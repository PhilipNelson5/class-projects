#ifndef ORTHOGONAL_BASIS_HPP
#define ORTHOGONAL_BASIS_HPP

#include "../1-vectorNorms/vectorNorms.hpp"
#include "../3-vectorOperations/vectorOperations.hpp"
#include <cmath>
#include <tuple>
#include <vector>

/**
 * Create an orthogonal basis from two vectors
 *
 * @tparam T Type of the elements of vector a and b
 * @param a  The first vector
 * @param b  The second vector
 * @return   The two vectors that form an orthogonal basis
 */
template <typename T>
std::tuple<std::vector<T>, std::vector<T>> orthogonal_basis(std::vector<T> a,
                                                            std::vector<T> b)
{
  // check that the vectors are in R2
  if (a.size() != 2 || b.size != 2)
  {
    std::cerr << "[ERROR] vectors not in R2 in orthogonal_basis" << std::endl;
  }

  // normalize v2
  auto bn = b / p_norm(b, 2);

  // the projection of v1 onto v2
  std::vector<T> proj = inner_product(a, bn) * bn;

  // find the vector orthogonal to v2 which points to v1
  std::vector<T> ortho = a - proj;

  // normalize
  std::vector<T> u1 = ortho / p_norm(ortho, 2);
  std::vector<T> u2 = bn;

  // return the two orthogonal vectors
  return {u1, u2};
}

#endif
