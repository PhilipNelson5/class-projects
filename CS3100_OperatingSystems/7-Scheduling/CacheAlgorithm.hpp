#ifndef CS3100_SCHEDULER_CACHE_ALGORITHM_HPP
#define CS3100_SCHEDULER_CACHE_ALGORITHM_HPP
namespace cs3100
{
  class CacheAlgorithm
  {
  public:
    CacheAlgorithm(int s) : size(s){};
    virtual bool in(int) = 0;
    virtual void load(int) = 0;
    unsigned int size;
  };
}
#endif
