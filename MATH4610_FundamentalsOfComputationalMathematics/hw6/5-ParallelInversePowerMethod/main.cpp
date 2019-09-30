#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../hw5/2-MatrixGenerator/matrixGenerator.hpp"
#include "parallelInverseIteration.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);
  // Matrix<double> A = {{1, -3, 3}, {3, -5, 3}, {6, -6, 4}};

  auto eigval = parallel_inverse_power_iteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Smallest Eigenvalue\n" << eigval << std::endl;
}
