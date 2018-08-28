#include <iomanip>
#include <iostream>
#include <vector>

bool fill(unsigned int M, std::vector<unsigned int> A)
{
  auto N = A.size();
  bool f[M + 1][N + 1];

  for (auto i = M; i >= M; --i)
    for (auto k = 0u; k <= N; ++k)
      if (i == 0)
        f[i][k] = true;
      else
        f[i][k] = false;

  // ------
  // PRINT
  // ------
  for (auto i = 0u; i < N; ++i)
  {
    std::cout << A[i] << " :";
    for (auto k = 0u; k < M; ++k)
    {
      std::cout << std::setw(3) << f[i][k];
    }

    std::cout << "\n";
  }
    std::cout << "   ";
  for (auto i = 1u; i <= M; ++i)
  {
    std::cout << std::setw(3) << i;
  }
  std::cout << "\n\n";
  // ---------
  // END PRINT
  // ---------

  for (auto i = 1u; i < N; ++i)
  {
    for (auto k = 1u; k < M; ++k)
    {
      if (k >= A[i])
        f[i][k] = std::max(f[i - 1][k], f[i - 1][k - A[i]]);
      else
        f[i][k] = f[i - 1][k];
    }
  }

  // ------
  // PRINT
  // ------
  std::cout << '\n';
  for (auto i = 1u; i <= M; ++i)
  {
    std::cout << std::setw(3) << i;
  }
  std::cout << '\n';

  for (auto i = 0u; i < N; ++i)
  {
    std::cout << A[i] << " :";
    for (auto k = 0u; k < M; ++k)
    {
      std::cout << std::setw(3) << f[i][k];
    }

    std::cout << "\n\n";
  }
  // ---------
  // END PRINT
  // ---------

  return f[N][M];
}

int main()
{

  std::vector<unsigned int> A = {1, 2, 3};
  auto M = 5;
  std::cout << fill(M, A) << std::endl;
  // std::cout << std::boolalpha << std::setw(5) << fill(M, A) << std::endl;
}
