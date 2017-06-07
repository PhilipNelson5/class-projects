#ifndef CS3100_SCHEDULER_SECOND_CHANCE_CACHE_ALGORITHM_HPP
#define CS3100_SCHEDULER_SECOND_CHANCE_CACHE_ALGORITHM_HPP

#include "CacheAlgorithm.hpp"
#include <vector>
namespace cs3100
{
  class SecondChanceCacheAlgorithm : public CacheAlgorithm
  {
  public:
    SecondChanceCacheAlgorithm(int s)
      : CacheAlgorithm(s), cache(size, -1), mark(size, false), cur(0)
    {
    }
    bool in(int)override;
    void load(int) override;

  private:
    std::vector<int> cache;
    std::vector<bool> mark;
    unsigned int cur;

    void increment();
  };
}
#endif
