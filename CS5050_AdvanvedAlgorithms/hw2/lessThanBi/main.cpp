#include <iostream>
#include <vector>

int mergeList(std::vector<int>& v, int start, int mid, int end)
{
  int inv = 0;
  std::vector<int> merged;
  int i = start;
  int j = mid + 1;
  while (i <= mid && j <= end)
  {
    if (v[i] < v[j])
      merged.push_back(v[i++]);
    else
    {
      if (i < j)
        inv += mid - i + 1;

      merged.push_back(v[j++]);
    }
  }
  while (i <= mid) merged.push_back(v[i++]);

  while (j <= end) merged.push_back(v[j++]);

  for (unsigned int x = start, y = 0; y < merged.size(); ++x, ++y) v[x] = merged[y];

  return inv;
}

std::vector<int> mergeSort(std::vector<int>& v, int start, int end)
{
  if (end - start == 0) return 0;

  auto mid = (start + end) / 2;

  return mergeSort(v, start, mid) +
    mergeSort(v, mid + 1, end) +
    mergeList(v, start, mid, end);
}

std::vector<int> mergeSort(std::vector<int>& a, std::vector<int>& b)
{
  return mergeSort(a, 0, a.size() - 1, b, 0, b.size()-1);
}

int main()
{
  std::vector<int> a = {3, 7, 8, 1, 9, 5, 2};
  std::vector<int> b = {3, 7, 8};

  std::vector<int> inv = mergeSort(a, b);

  std::cout << "inversions: " << inv << std::endl;
}
