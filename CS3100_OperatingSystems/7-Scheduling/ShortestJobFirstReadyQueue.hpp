#ifndef SHORTEST_JOB_FIRST_READY_QUEUE_HPP
#define SHORTEST_JOB_FIRST_READY_QUEUE_HPP

#include "ReadyQueue.hpp"
#include <vector>

namespace cs3100
{
  class Simulation;

  class ShortestJobFirstReadyQueue : public ReadyQueue
  {
  public:
    ShortestJobFirstReadyQueue() : queue(), simulation(nullptr) {}
    // Shortest Job First needs to be able to look inside current simulation
    void associateSimulator(Simulation* s);
    void add(int) override;
    int next() override;

  private:
    std::vector<int> queue;
    Simulation* simulation;
  };
}
#endif
