#include "random.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <queue>
#include <thread>

enum TAG
{
  NEW_WORK,
  TOKEN,
  ACTION,
  CONTINUE,
  FINALIZE
};

enum TOKEN
{
  WHITE,
  BLACK
};

void process(double task)
{
  int work = (int)(task*1000);
  std::this_thread::sleep_for(std::chrono::milliseconds(work));
}

void getNewWork(std::queue<double>& tasks, int& workRecv)
{
  int flag;
  MPI_Status stat;
  do
  {
    MPI_Iprobe(MPI_ANY_SOURCE, TAG::NEW_WORK, MPI_COMM_WORLD, &flag, &stat);
    if (flag)
    {
      double task;
      MPI_Recv(&task,
               1,
               MPI_DOUBLE,
               stat.MPI_SOURCE,
               stat.MPI_TAG,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      tasks.push(task);
      ++workRecv;
    }
  } while (flag);
}

void makeMoreWork(std::queue<double>& tasks,
                  int& workMade,
                  int maxWorkMade,
                  int workMin,
                  int workMax)
{
  if (workMade < maxWorkMade)
  {
    auto newWork = random_int(1, 3);
    for (auto i = 0; i < newWork; ++i)
    {
      tasks.push(random_double(workMin, workMax));
      ++workMade;
    }
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int workMin = 1, workMax = 3, maxWorkLoad = 16, amntToSend = 2,
            maxWorkMade = random_int(1024, 2048);
  int workMade = 0, workDone = 0, workSent = 0, workRecv = 0;
  enum TOKEN token = TOKEN::WHITE;

  std::queue<double> tasks;
  if (rank == 0)
  {
    tasks.push(random_double(workMin, workMax));

    do // while token == black
    {
      token = TOKEN::WHITE;
      while (!tasks.empty())
      {
        process(tasks.front());
        tasks.pop();
        ++workDone;

        /* Create more work */
        makeMoreWork(tasks, workMade, maxWorkMade, workMin, workMax);

        /* Check for incoming work */
        getNewWork(tasks, workRecv);

        /* Send work to other processes */
        if (tasks.size() > maxWorkLoad)
        {
          for (auto i = 0; i < amntToSend; ++i)
          {
            int dest = random_int(rank + 1, rank + world_size - 1) % world_size;
            MPI_Send(&tasks.front(),
                     1,
                     MPI_DOUBLE,
                     dest,
                     TAG::NEW_WORK,
                     MPI_COMM_WORLD);
            tasks.pop();
            ++workSent;
          }
        }
      }

      /* Send Token */
      MPI_Send(
        &token, 1, MPI_INT, rank + 1 % world_size, TAG::TOKEN, MPI_COMM_WORLD);

      std::cout << rank << " -- send token -> "
                << (token == TOKEN::BLACK ? "BLACK" : "WHITE") << '\n';

      /* Receive Token */
      MPI_Recv(&token,
               1,
               MPI_INT,
               world_size - 1,
               TAG::TOKEN,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      std::cout << rank << " <- recv token -- "
                << (token == TOKEN::BLACK ? "BLACK" : "WHITE") << '\n';

      if (token == TOKEN::BLACK)
      {
        enum TAG action = TAG::CONTINUE;
        for (auto i = 1; i < world_size; ++i)
        {
          MPI_Send(&action, 1, MPI_INT, i, TAG::ACTION, MPI_COMM_WORLD);
        }
      }

    } while (token == TOKEN::BLACK);

    /* Kill all processes */
    enum TAG action = TAG::FINALIZE;
    for (auto i = 1; i < world_size; ++i)
    {
      MPI_Send(&action, 1, MPI_INT, i, TAG::ACTION, MPI_COMM_WORLD);
    }
  }
  else // SLAVE
  {
    tasks.push(random_double(workMin, workMax));
    enum TAG action;
    do // while action == continue
    {
      token = TOKEN::WHITE;
      while (!tasks.empty())
      {
        process(tasks.front());
        tasks.pop();
        ++workDone;

        /* Create more work */
        makeMoreWork(tasks, workMade, maxWorkMade, workMin, workMax);

        /* Check for incoming work */
        getNewWork(tasks, workRecv);

        /* Send work to other processes */
        if (tasks.size() > maxWorkLoad)
        {
          for (auto i = 0; i < amntToSend; ++i)
          {
            int dest = random_int(rank + 1, rank + world_size - 1) % world_size;
            if (dest < rank) token = TOKEN::BLACK;
            MPI_Send(&tasks.front(),
                     1,
                     MPI_DOUBLE,
                     dest,
                     TAG::NEW_WORK,
                     MPI_COMM_WORLD);
            tasks.pop();
            ++workSent;
          }
        }
      }

      enum TOKEN recvToken;
      /* Receive Token */
      MPI_Recv(&recvToken,
               1,
               MPI_INT,
               rank - 1,
               TAG::TOKEN,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      if (recvToken == TOKEN::BLACK)
      {
        token = TOKEN::BLACK;
      }
      std::cout << rank << " <- recv token -- "
                << (recvToken == TOKEN::BLACK ? "BLACK" : "WHITE") << '\n';

      /* Send Token */
      MPI_Send(&token,
               1,
               MPI_INT,
               (rank + 1) % world_size,
               TAG::TOKEN,
               MPI_COMM_WORLD);

      std::cout << rank << " -- send token -> "
                << (token == TOKEN::BLACK ? "BLACK" : "WHITE") << '\n';

      MPI_Recv(
        &action, 1, MPI_INT, 0, TAG::ACTION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } while (action == TAG::CONTINUE);
  }

  std::cout << rank << " -- FINALIZE -- " << '\n';
  std::cout << rank << " -- work done -- " << workDone << '\n';

  int totalWorkDone;
  MPI_Allreduce(&workDone, &totalWorkDone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  double percentWorkDone = (double)workDone / totalWorkDone * 100;
  std::cout << rank << "-- percent done -- " << percentWorkDone << '\n';

  if (rank == 0) std::cout << "Total Work Done " << totalWorkDone << '\n';
  MPI_Finalize();

  return EXIT_SUCCESS;
}
