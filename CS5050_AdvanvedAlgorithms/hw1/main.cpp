#include <iostream>
#include <vector>

std::vector<int> fill(std::vector<int> A, int K)
{
  std::vector<int> knap;
  int sum = 0;
  // do a linear scan looking for K/2 <= Ai <= K (single element solution)
  // place all elements < K/2 in array B

  for (int i = 0; i < A.size(); ++i)
  {
    if (K / 2 <= A[i] && A[i] <= K) // check for one element solution
    {
      return {A[i]};
    }
    if (A[i] < K / 2) // all elements < K/2
    {
      sum += A[i];
      knap.push_back(A[i]);
      if (sum >= K/2)
        return knap;
    }
  }
}

int main()
{
  int K = 20;
  // std::vector<int> A = {9, 24, 14, 5, 8, 17};
  std::vector<int> A = {9, 24, 5, 8};

  std::cout << "{";
  for (auto&& e : fill(A, K)) std::cout << e << ", ";

  std::cout << "}" << std::endl;
}
