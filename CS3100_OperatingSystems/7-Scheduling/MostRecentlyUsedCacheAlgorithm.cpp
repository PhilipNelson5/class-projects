#include "MostRecentlyUsedCacheAlgorithm.hpp"
#include <algorithm>
#include <iterator>
#include <vector>

bool cs3100::MostRecentlyUsedCacheAlgorithm::in(int page)
{
  auto target = std::find(cache.begin(), cache.end(), page);
  if (target == cache.end()) return false;
  cur = std::distance(cache.begin(), target);
  return true;
}

void cs3100::MostRecentlyUsedCacheAlgorithm::load(int page)
{
  if (cache.size() < size)
	{
    cache.push_back(page);
		cur = cache.size()-1;
	}
  else
    cache[cur] = page;
}
