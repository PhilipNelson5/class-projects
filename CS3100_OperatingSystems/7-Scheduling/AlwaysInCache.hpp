#ifndef CS3100_SCHEDULER_ALWAYS_IN_CACHE_HPP
#define CS3100_SCHEDULER_ALWAYS_IN_CACHE_HPP

#include "CacheAlgorithm.hpp"
namespace cs3100
{

  class AlwaysInCache : public CacheAlgorithm
  {
  public:
    AlwaysInCache(int s) : CacheAlgorithm(s) {}
    bool in(int)override { return true; }
    void load(int) override {}
  };
}

#endif
