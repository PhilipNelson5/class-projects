#pragma once
#include <condition_variable>
#include <mutex>

/**
 * @brief A thread syncronization object
 * Initialize the barrier with the number of threads that will be synchronized.
 * Each thread calls wait and blocks until all threads are waiting.
 * The barrier protects agains spurious wakeup.
 * The barrier can be reused to synchronize the _same_ number of threads.
 * 
 */
class Barrier
{
public:
  /**
   * @brief Construct a new Barrier object
   * 
   * @param n number of threads to be synchronized
   */
  explicit Barrier(const int n): n(n), count(n), generation(0) {} 

  /**
   * @brief blocks until all threads have waited
   * 
   */
  void wait()
  {
    std::unique_lock<std::mutex> lk(m);
    const int last_gen = generation;
    if (--count == 0)
    {
      ++generation;
      count = n;
      cv.notify_all(); 
    }
    else
    {
      cv.wait(lk, [this, last_gen]{ return last_gen != generation; }); 
    }
  }

private:
  const int n;
  int count;
  int generation;
  std::condition_variable cv;
  std::mutex m;
};
