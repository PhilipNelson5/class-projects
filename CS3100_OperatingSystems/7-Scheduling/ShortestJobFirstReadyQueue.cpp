#include "ShortestJobFirstReadyQueue.hpp"
#include "Simulation.hpp"
#include <algorithm>

namespace cs3100
{

  void ShortestJobFirstReadyQueue::associateSimulator(Simulation* s)
  {
    simulation = s;
  }

  void ShortestJobFirstReadyQueue::add(int job)
  {
    /*TODO Assignment 1: Correctly implement the Shortest Job First scheduling
     * algorithm*/
    queue.push_back(job);
  }

  int ShortestJobFirstReadyQueue::next()
  {
    /*TODO Assignment 1: Correctly implement the Shortest Job First scheduling
     * algorithm*/
    if (queue.empty()) return -1;
    auto minJob = queue[0];
    for (auto&& job : queue)
      if (simulation->jobs[job].getFutureDuration() <
          simulation->jobs[minJob].getFutureDuration())
        minJob = job;
    queue.erase(std::find(queue.begin(), queue.end(), minJob));
    return minJob;
  }
}
