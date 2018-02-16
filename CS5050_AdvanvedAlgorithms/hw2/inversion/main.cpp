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

int mergeSort(std::vector<int>& v, int start, int end)
{
  if (end - start == 0) return 0;

  auto mid = (start + end) / 2;

  return mergeSort(v, start, mid) +
    mergeSort(v, mid + 1, end) +
    mergeList(v, start, mid, end);
}

int mergeSort(std::vector<int>& list)
{
  return mergeSort(list, 0, list.size() - 1);
}

int main()
{
  std::vector<int> a = {3, 7, 8, 1, 9, 5, 2};
  // std::vector<int> a = {5, 4, 3, 2, 1, 0};
  std::cout << "{ ";
  for (auto e : a) std::cout << e << " ";
  std::cout << "}" << std::endl;

  int inv = mergeSort(a);

  std::cout << "inversions: " << inv << std::endl;
}
