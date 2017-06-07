#ifndef CS3100_SCHEDULER_MOST_RECENTLY_USED_CACHE_ALGORITHM_HPP
#define CS3100_SCHEDULER_MOST_RECENTLY_USED_CACHE_ALGORITHM_HPP

#include "CacheAlgorithm.hpp"
#include <vector>
namespace cs3100
{
  class MostRecentlyUsedCacheAlgorithm : public CacheAlgorithm
  {
  public:
    MostRecentlyUsedCacheAlgorithm(int s) : CacheAlgorithm(s), cache(), cur(0)
    {
      for (auto i = 0; i < s; ++i)
        cache.push_back(-1);
    }
    bool in(int) override;
    void load(int) override;

  private:
    std::vector<int> cache;
    unsigned int cur;
  };
}
#endif
