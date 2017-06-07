#ifndef CS3100_SCHEDULER_FIFO_CACHE_ALGORITHM_HPP
#define CS3100_SCHEDULER_FIFO_CACHE_ALGORITHM_HPP

#include "CacheAlgorithm.hpp"
#include <vector>
namespace cs3100
{
  class FifoCacheAlgorithm : public CacheAlgorithm
  {
  public:
    FifoCacheAlgorithm(int s) : CacheAlgorithm(s), cache(size, -1), cur(0) {}
    bool in(int);
    void load(int);

  private:
    std::vector<int> cache;
    unsigned int cur;

    void increment();
  };
}
#endif
