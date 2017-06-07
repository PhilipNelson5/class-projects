#include "FifoCacheAlgorithm.hpp"
#include <algorithm>
#include <vector>

bool cs3100::FifoCacheAlgorithm::in(int page)
{
  return std::find(cache.begin(), cache.end(), page) != cache.end();
}

void cs3100::FifoCacheAlgorithm::increment()
{
  if (++cur == cache.size()) cur = 0;
}

void cs3100::FifoCacheAlgorithm::load(int page) 
{
  cache[cur] = page;
  increment();
}
