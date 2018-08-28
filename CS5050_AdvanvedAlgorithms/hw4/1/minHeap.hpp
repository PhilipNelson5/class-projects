#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
class minHeap
{
public:
  void insert(T t)
  {
    heap.push_back(t);
    auto pos = heap.size() - 1;

    while (heap[pos] < heap[pos / 2])
    {
      std::swap(heap[pos], heap[pos / 2]);
      pos /= 2;
    }
  }

  bool kthLessThanX(unsigned int k, T x)
  {
    std::queue<unsigned int> q;
    auto pos = 0u, ct = 0u, c1 = 0u, c2 = 0u;
    q.push(pos);
    while (ct <= k && q.size() != 0)
    {
      pos = q.front();
      q.pop();
      if (heap[pos] < x)
      {
        ++ct;
        c1 = pos * 2 + 1;
        c2 = pos * 2 + 2;
        if (c1 < heap.size()) q.push(c1);
        if (c2 < heap.size()) q.push(c2);
      }
    }

    if (ct >= k) return true;
    return false;
  }

  std::string toString(unsigned int i = 0u, std::string tab = "")
  {
    if (i > heap.size() - 1) return "";
    std::stringstream ss;

    ss << toString(i * 2 + 2, tab + "      ");
    ss << tab << "(" << i << ")" << heap[i] << "\n";
    ss << toString(i * 2 + 1, tab + "      ");

    return ss.str();
  }

  std::string serialize()
  {
    std::stringstream ss;
    for (auto&& e : heap)
    {
      ss << e << " ";
    }
    return ss.str();
  }

private:
  std::vector<T> heap;
};
