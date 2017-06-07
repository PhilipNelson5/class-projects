#ifndef APPROXIMATE_SHORTEST_JOB_READY_QUEUE_HPP
#define APPROXIMATE_SHORTEST_JOB_READY_QUEUE_HPP

#include "ReadyQueue.hpp"
#include <vector>
namespace cs3100
{
  class Simulation;


  class ApproximateShortestJobFirstReadyQueue : public ReadyQueue
  {
  public:
	ApproximateShortestJobFirstReadyQueue():queue(), simulation(nullptr){}
    // Approximate Shortest Job First needs to be able to look inside current
    // simulation
    void associateSimulator(Simulation* s);

    void add(int) override;
    int next() override;

  private:
    //struct priorityJob
    //{
    //  int job;
    //  float priority;
		//	priorityJob(int j, float p):job(j), priority(p){}
    //  inline bool operator<(priorityJob const& b)
    //  {
    //    return this->priority < b.priority;
    //  }
    //  inline bool operator>(priorityJob const& b)
    //  {
    //    return this->priority > b.priority;
    //  }
    //};
    //std::priority_queue<priorityJob,
    //                    std::vector<priorityJob>,
    //                    std::greater<priorityJob>>
    //  queue;
		std::vector<int> queue;
    Simulation* simulation;
  };
}
#endif
