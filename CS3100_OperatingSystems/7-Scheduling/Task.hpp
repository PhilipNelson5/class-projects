#ifndef CS3100_SCHEDULER_TASK_HPP
#define CS3100_SCHEDULER_TASK_HPP

#include <vector>
namespace cs3100
{

  struct Task
  {
    const float duration;
    float progress;
    float completionTime;
    const int device;
    enum Type
    {
      CPU,
      IO
    };
    const Type type;
    bool isDone() { return progress >= duration; }
    Task(float dur, int dev, Type t)
      : duration(dur),
        progress(0.0f),
        completionTime(-1.0f),
        device(dev),
        type(t)
    {
    }
    Task(Task const& o) = default;
  };

  struct Job
  {
    Job(float, int, int);
    const float creationTime;
    unsigned int cur;
    std::vector<Task> tasks;

    float getFutureDuration()
    {
      float dur = 0;
      for (auto i = cur; i < tasks.size(); ++i)
        dur += tasks[i].duration;
      return dur;
    }

    float getAverageDuration()
    {
      if (cur == 0) return 0;
      float dur = 0;
      for (auto i = 0u; i < cur; ++i)
        if (tasks[i].type == Task::Type::CPU) dur += tasks[i].duration;
      return dur / cur;
    }

    float getCpuDuration()
    {
      float dur = 0;
      for (auto&& task : tasks)
        if (task.type == Task::CPU) dur += task.duration;
      return dur;
    }
  };
}

#endif
