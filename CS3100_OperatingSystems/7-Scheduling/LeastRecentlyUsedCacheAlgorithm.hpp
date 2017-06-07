#ifndef CS3100_SCHEDULER_LEAST_RECENTLY_USED_CACHE_ALGORITHM_HPP
#define CS3100_SCHEDULER_LEAST_RECENTLY_USED_CACHE_ALGORITHM_HPP

#include "CacheAlgorithm.hpp"
#include <vector>
namespace cs3100
{
  class LeastRecentlyUsedCacheAlgorithm : public CacheAlgorithm
  {
  public:
    LeastRecentlyUsedCacheAlgorithm(int s) : CacheAlgorithm(s), cache(size, -1) {}
    bool in(int)override;
    void load(int) override;

  private:
    std::vector<int> cache;
  };
}
#endif
