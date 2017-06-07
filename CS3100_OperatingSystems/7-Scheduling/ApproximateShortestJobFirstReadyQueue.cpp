#include "ApproximateShortestJobFirstReadyQueue.hpp"
#include <algorithm>
#include "Simulation.hpp"
namespace cs3100
{

	void ApproximateShortestJobFirstReadyQueue::associateSimulator(Simulation* s)
	{
		simulation = s;
	}

  void ApproximateShortestJobFirstReadyQueue::add(int job)
  {
    /*TODO Assignment 1: Correctly implement the Shortest Job First scheduling
     * algorithm*/
    queue.push_back(job);
    // queue.emplace(job, simulation->jobs[job].getAverageDuration());
  }

  int ApproximateShortestJobFirstReadyQueue::next()
  {
    /*TODO Assignment 1: Correctly implement the Shortest Job First scheduling
     * algorithm*/
    // auto nextJob = queue.top().job;
    // queue.pop();
    if (queue.empty()) return -1;
    auto minJob = queue[0];
    for (auto&& job : queue)
      if (simulation->jobs[job].getAverageDuration() <
          simulation->jobs[minJob].getAverageDuration())
        minJob = job;
    queue.erase(std::find(queue.begin(), queue.end(), minJob));
    return minJob;
    return 0;
  }
}
