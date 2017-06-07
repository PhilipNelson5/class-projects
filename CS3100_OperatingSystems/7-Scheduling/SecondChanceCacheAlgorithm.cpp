#include "SecondChanceCacheAlgorithm.hpp"
#include <algorithm>
#include <iterator>
#include <vector>

bool cs3100::SecondChanceCacheAlgorithm::in(int page)
{
  auto target = std::find(cache.begin(), cache.end(), page);
  if (target == cache.end()) return false;
  mark[std::distance(cache.begin(), target)] = true;
  return true;
}

void cs3100::SecondChanceCacheAlgorithm::increment()
{
  if (++cur == cache.size()) cur = 0;
}

void cs3100::SecondChanceCacheAlgorithm::load(int page)
{
  if (mark[cur] == false)
	{
    cache[cur] = page;
		increment();
	}
  else
  {
    mark[cur] = false;
    increment();
    load(page);
  }
}
