#include "AlwaysInCache.hpp"
#include <string>
#include "ApproximateShortestJobFirstReadyQueue.hpp"
#include "FifoCacheAlgorithm.hpp"
#include "FifoReadyQueue.hpp"
#include "LeastRecentlyUsedCacheAlgorithm.hpp"
#include "MostRecentlyUsedCacheAlgorithm.hpp"
#include "SecondChanceCacheAlgorithm.hpp"
#include "ShortestJobFirstReadyQueue.hpp"
#include "Simulation.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>

namespace
{
  std::vector<double> efficiency;
  std::vector<double> latency;
  std::vector<double> response;

  void clear()
  {
    efficiency.clear();
    latency.clear();
    response.clear();
  }

  std::ofstream fout;

  void report(cs3100::Simulation& s)
  {
    /*TODO create a report based on the results in s*/
    double eff = s.getEfficiency();
    double lat = 0;
    double res = 0;

    // std::cout << "Efficiency : " << eff << std::endl;
    // std::cout << "Task\tLatency\tResponseTime" << std::endl;
    for (size_t i = 0; i < s.getJobs(); ++i)
    {
      lat = s.rawLatency(i);
      res = s.rawResponseTime(i);
      // std::cout << i << "\t";
      // std::cout << lat << "\t";
      // std::cout << res;
      // std::cout << std::endl;

      latency.push_back(lat);
      response.push_back(res);
    }
    efficiency.push_back(eff);
  }

  void averageData(std::string name)
  {
    double aveEff = std::accumulate(efficiency.begin(), efficiency.end(), 0.0) / efficiency.size();
    double aveLat = std::accumulate(latency.begin(), latency.end(), 0.0) / latency.size();
    double aveRes = std::accumulate(response.begin(), response.end(), 0.0) / response.size();
    fout << name << std::endl
         << "Efficiency,Latency,Response" << std::endl
         << aveEff << ',' << aveLat << ',' << aveRes << std::endl;
    clear();
  }

  template <typename ReadyType, typename CacheType>
  void runSimulation(cs3100::SimulationParameters const& p)
  {
    cs3100:: Simulation s(
      p, std::make_unique<ReadyType>(), std::make_unique<CacheType>(p.cacheSize));
    s.run();
    report(s);
  }
}

int main()
{
	const std::string outputFile = "output.csv";
  const int TEST = 10;
	fout.open(outputFile);
  /*TODO vary the simulation parameters to get richer results for your report*/
  cs3100::SimulationParameters fifo;
  fifo.cpus = 1;
  fifo.devices = 2;
  fifo.cacheSize = 100;
  fifo.contextSwitchCost = 0.1f;
  fifo.cacheMissCost = 1.0f;
  fifo.maximumTimeSlice = std::numeric_limits<float>::max();
  fifo.jobs = 10;
  fifo.meanTimeBetweenJobs = 10.0f;
  fifo.stddevTimeBetweenJobs = 2.0f;
  // create simulation with specific parameters and algorithms
  for (auto i = 0u; i < TEST; ++i)
    runSimulation<cs3100::FifoReadyQueue, cs3100::FifoCacheAlgorithm>(fifo);
  averageData("Fifo");
  for (auto i = 0u; i < TEST; ++i)
    runSimulation<cs3100::FifoReadyQueue, cs3100::LeastRecentlyUsedCacheAlgorithm>(fifo);
  averageData("Least Recent");
  for (auto i = 0u; i < TEST; ++i)
    runSimulation<cs3100::FifoReadyQueue, cs3100::MostRecentlyUsedCacheAlgorithm>(fifo);
  averageData("Most Recent");
  for (auto i = 0u; i < TEST; ++i)
    runSimulation<cs3100::FifoReadyQueue, cs3100::SecondChanceCacheAlgorithm>(fifo);
  averageData("Second Chance");

	std::cout << "Data saved to: " << outputFile << std::endl;
  // std::cout << std::endl << std::endl << "Round Robin";
  // cs3100::SimulationParameters roundRobin;
  // roundRobin.cpus = 1;
  // roundRobin.devices = 2;
  // roundRobin.cacheSize = 0;
  // roundRobin.contextSwitchCost = 0.1f;
  // roundRobin.cacheMissCost = 1.0f;
  // roundRobin.maximumTimeSlice = 5.0f;
  // roundRobin.jobs = 10;
  // roundRobin.meanTimeBetweenJobs = 10.0f;
  // roundRobin.stddevTimeBetweenJobs = 2.0f;
  // // create simulation with specific parameters and algorithms
  // runSimulation<cs3100::FifoReadyQueue, cs3100::AlwaysInCache>(roundRobin);

  // std::cout << std::endl << std::endl << "Shortest";
  // cs3100::SimulationParameters sjf; // Shortest Job First
  // sjf.cpus = 1;
  // sjf.devices = 2;
  // sjf.cacheSize = 0;
  // sjf.contextSwitchCost = 0.1f;
  // sjf.cacheMissCost = 1.0f;
  // sjf.maximumTimeSlice = 5.0f;
  // sjf.jobs = 10;
  // sjf.meanTimeBetweenJobs = 10.0f;
  // sjf.stddevTimeBetweenJobs = 2.0f;
  // // create simulation with specific parameters and algorithms
  // runSimulation<cs3100::ShortestJobFirstReadyQueue, cs3100::AlwaysInCache>(sjf);

  // std::cout << std::endl << std::endl << "Approx";
  // cs3100::SimulationParameters asjf; // Shortest Job First
  // asjf.cpus = 1;
  // asjf.devices = 2;
  // asjf.cacheSize = 0;
  // asjf.contextSwitchCost = 0.1f;
  // asjf.cacheMissCost = 1.0f;
  // asjf.maximumTimeSlice = 5.0f;
  // asjf.jobs = 10;
  // asjf.meanTimeBetweenJobs = 10.0f;
  // asjf.stddevTimeBetweenJobs = 2.0f;
  // // create simulation with specific parameters and algorithms
  // runSimulation<cs3100::ApproximateShortestJobFirstReadyQueue, cs3100::AlwaysInCache>(asjf);
}
