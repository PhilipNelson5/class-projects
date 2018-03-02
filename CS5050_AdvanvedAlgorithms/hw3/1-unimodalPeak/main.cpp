#include <iostream>
#include <vector>

template <typename T>
int peak(std::vector<T> A, int start, int end)
{
  if (start > end)
    return -1;
  auto mid = (start + end) / 2;
  if (A[mid - 1] < A[mid] && A[mid] > A[mid + 1])
    return mid;

  if (A[mid - 1] < A[mid] && A[mid] < A[mid + 1])
    return peak(A, mid + 1, end);

  if (A[mid - 1] > A[mid] && A[mid] > A[mid + 1])
    return peak(A, start, mid - 1);

  return -1;
}

int main()
{
  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8, 10, 9};
  auto p = peak(A, 0, A.size() - 1);
  if (p > 0)
    std::cout << "peak at: " << p << "\nvalue:\t" << A[p] << std::endl;
  else
    std::cout << "FAILURE" << std::endl;
}
