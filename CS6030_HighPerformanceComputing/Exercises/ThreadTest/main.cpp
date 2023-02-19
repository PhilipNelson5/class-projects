#include <thread>
#include <vector>
#include <iostream>

int main() 
{
  const int thread_cnt = 4;
  std::vector<std::thread> threads;
  for (int i = 0; i < thread_cnt; ++i)
  {
    threads.emplace_back([](const int rank){
      std::cout << "rank: " << rank << " id: " << std::this_thread::get_id() << std::endl;
    }, i+1);
  }

  for (int i = 0; i < thread_cnt; ++i)
  {
    threads[i].join();
  }
}