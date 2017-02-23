#ifndef THREAD_POOL
#define THREAD_POOL

#include "threadSafeQueue.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

class ThreadPool
{
public:
  using func = std::function<void(void)>;
  ThreadPool(int n) : queue(), pool(n), tasks(0), done(false) {}

  ~ThreadPool()
  {
    for (auto &t : pool)
      t.join();
  }

  void post(func f)
  {
    queue.enqueue(f);
    {
      ++tasks;
    }
    // std::cout << "+" << tasks << std::endl;
  }

  void start()
  {
    for (auto &t : pool)
      t = std::thread([&]() { this->run(); });
  }

  void stop()
  {
    // std::cout << "stop" << std::endl;
    done = true;
  }

private:
  TSQ<func> queue;
  std::vector<std::thread> pool;
	std::atomic<int> tasks;
  std::atomic<bool> done;

  void run()
  {
    while (!done)
    {
      func task;
      if (queue.dequeue(task))
      {
        task();
        {
          --tasks;
        }
        if (tasks <= 0) stop();
      }
      if (done) return;
    }
  }
};

#endif
