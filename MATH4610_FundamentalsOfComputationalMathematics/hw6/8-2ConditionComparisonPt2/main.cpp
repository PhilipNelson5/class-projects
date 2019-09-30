#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../hw5/2-MatrixGenerator/matrixGenerator.hpp"
#include "../3-Condition2Estimate/condition2Estimate.hpp"
#include "../6-Parallel2ConditionEstimate/parallelCondition2Estimate.hpp"
#include <chrono>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

int main()
{
  for (auto i = 40u; i <= 640u; i *= 2)
  {
    auto A = generate_square_symmetric_diagonally_dominant_matrix(i);

    auto start = std::chrono::high_resolution_clock::now();
    auto conditionNum1 = condition_2_estimate(A, 50u);
    auto end = std::chrono::high_resolution_clock::now();
    auto result1 =
      std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto conditionNum2 = parallel_condition_2_estimate(A, 50u);
    end = std::chrono::high_resolution_clock::now();
    auto result2 =
      std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << i << " x " << i << std::endl;
    std::cout << "serial algorithm:\n  result: " << conditionNum1
              << "\n  time: " << result1 << std::endl
              << std::endl;

    std::cout << "parallel algorithm:\n  result: " << conditionNum2
              << "\n  time: " << result2 << std::endl
              << std::endl;
  }
}
