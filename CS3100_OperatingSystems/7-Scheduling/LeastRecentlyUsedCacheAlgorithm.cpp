#include "LeastRecentlyUsedCacheAlgorithm.hpp"
#include <algorithm>
#include <iterator>
#include <vector>

bool cs3100::LeastRecentlyUsedCacheAlgorithm::in(int page)
{
  auto target = std::find(cache.begin(), cache.end(), page);
  if (target == cache.end()) return false;
  auto val = *target;
  cache.erase(target);
  cache.push_back(val);
  return true;
}

void cs3100::LeastRecentlyUsedCacheAlgorithm::load(int page)
{
  if (cache.size() < size)
    cache.push_back(page);
  else
  {
    cache.erase(cache.begin());
    cache.push_back(page);
  }
}
