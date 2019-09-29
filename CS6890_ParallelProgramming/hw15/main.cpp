#include "random.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <thread>
#include <utility>

struct Data
{
  int data;
  int tag;
};

enum Job
{
  PE,
  SW,
  NONE
};

enum Tag
{
  DATA,
  END
};

std::pair<int, int> getSize(int size)
{
  int PEs = 2;
  int SWs = log2(PEs) * PEs / 2;
  int processes = PEs + SWs;
  while (processes <= size)
  {
    PEs *= 2;
    SWs = log2(PEs) * PEs / 2;
    processes = PEs + SWs;
  }
  PEs /= 2;
  SWs = log2(PEs) * PEs / 2;
  processes = PEs + SWs;

  return {PEs, SWs};
}

Job getJob(int PEs, int SWs, int rank)
{
  if (rank < PEs)
    return Job::PE;
  else if (rank < SWs + PEs)
    return Job::SW;
  else
    return Job::NONE;
}

std::pair<int, int> getDimenstionsSW(int PEs)
{
  int stages = log2(PEs);
  int rows = PEs / 2;
  return {stages, rows};
}

std::pair<int, int> getOutputs(int rank, int stage, int rows, int row)
{
  int outH, outL;
  if (stage == 0) // Output to PEs
  {
    outH = row * 2;
    outL = outH + 1;
  }
  else
  {
    int n = std::pow(2, stage - 1);
    if ((row / n) % 2 == 0) // TOP
    {
      outH = rank + rows;
      outL = outH + n;
    }
    else // BOTTOM
    {
      outL = rank + rows;
      outH = outL - n;
    }
  }
  return {outH, outL};
}
std::pair<int, int> getInputs(int rank,
                              int PEs,
                              int stages,
                              int stage,
                              int rows,
                              int row)
{
  int inH, inL;
  if (stage == stages - 1) // Input from PEs
  {
    inH = rank - PEs;
    inL = rank - PEs / 2;
  }
  else
  {
    int n = std::pow(2, stage);
    if ((row / n) % 2 == 0) // TOP
    {
      inH = rank - rows;
      inL = inH + n;
    }
    else // BOTTOM
    {
      inL = rank - rows;
      inH = inL - n;
    }
  }
  return {inH, inL};
}

void printInfoPE(int rank, int dest)
{
  std::cout << rank << "\tPE --> " << dest << '\n';
}
void printInfoSW(int rank,
                 int stage,
                 int row,
                 int inH,
                 int inL,
                 int outH,
                 int outL)
{
  (void)row;

  /* clang-format off */
  std::cout
    << rank << " SW Stage:" << stage
    << "\tin high: " << inH
    << "\tin low: " << inL
    << "\tout high: " << outH
    << "\tout low "  << outL << '\n';
  /* clang-format on */
}

enum Routing_xor
{
  STRAIGHT,
  CROSS
};

Routing_xor getRoutingXor(int stage, int tag)
{
  int mask = 1 << stage;
  return (mask & tag) == mask ? Routing_xor::CROSS : Routing_xor::STRAIGHT;
}

void doPE(int PEs, int rank, int out)
{
  // printInfoPE(rank, out);
  MPI_Comm PE_Comm;
  MPI_Comm_split(MPI_COMM_WORLD, (int)Job::PE, rank, &PE_Comm);
  int dataDest;
  if (rank == 0)
  {
    std::vector<int> sendto(PEs);
    std::iota(std::begin(sendto), std::end(sendto), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(std::begin(sendto), std::end(sendto), g);
    MPI_Scatter(sendto.data(), 1, MPI_INT, &dataDest, 1, MPI_INT, 0, PE_Comm);
  }
  else
  {
    MPI_Scatter(0, 0, 0, &dataDest, 1, MPI_INT, 0, PE_Comm);
  }

  int tag = rank ^ dataDest;
  Data data{rank, tag};
  std::cout << rank << " --> " << dataDest << std::endl;
  MPI_Send(&data, 2, MPI_INT, out, Tag::DATA, MPI_COMM_WORLD);
  MPI_Status status;
  MPI_Recv(
    &data, 2, MPI_INT, MPI_ANY_SOURCE, Tag::DATA, MPI_COMM_WORLD, &status);
  std::cout << rank << " got " << data.data << std::endl;
}

void doSW(int rank, int stage, int row, int inH, int inL, int outH, int outL)
{
  (void)row;
  // printInfoSW(rank, stage, row, inH, inL, outH, outL);
  MPI_Comm SW_Comm;
  MPI_Comm_split(MPI_COMM_WORLD, Job::SW, rank, &SW_Comm);

  MPI_Status status;
  do
  {
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (status.MPI_TAG == Tag::DATA)
    {
      Data data;
      MPI_Recv(&data,
               2,
               MPI_INT,
               status.MPI_SOURCE,
               status.MPI_TAG,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      Routing_xor route = getRoutingXor(stage, data.tag);
      if (route == Routing_xor::STRAIGHT)
      {
        if (status.MPI_SOURCE == inH)
        {
          std::cout << inH << " -> " << rank << " (=) -> " << outH << std::endl;
          MPI_Send(&data, 2, MPI_INT, outH, Tag::DATA, MPI_COMM_WORLD);
        }
        else
        {
          std::cout << inL << " -> " << rank << " (=) -> " << outL << std::endl;
          MPI_Send(&data, 2, MPI_INT, outL, Tag::DATA, MPI_COMM_WORLD);
        }
      }
      else
      {
        if (status.MPI_SOURCE == inH)
        {
          std::cout << inH << " -> " << rank << " (X) -> " << outL << std::endl;
          MPI_Send(&data, 2, MPI_INT, outL, Tag::DATA, MPI_COMM_WORLD);
        }
        else
        {
          std::cout << inL << " -> " << rank << " (X) -> " << outH << std::endl;
          MPI_Send(&data, 2, MPI_INT, outH, Tag::DATA, MPI_COMM_WORLD);
        }
      }
    }
  } while (status.MPI_TAG != Tag::END);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto [PEs, SWs] = getSize(size);
  const Job job = getJob(PEs, SWs, rank);

  if (job == Job::NONE)
  {
    MPI_Comm SW_Comm;
    MPI_Comm_split(MPI_COMM_WORLD, Job::SW, rank, &SW_Comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  const auto [stages, rows] = getDimenstionsSW(PEs);

  if (job == Job::PE)
  {
    const int dest = rank % (PEs / 2) + PEs;
    doPE(PEs, rank, dest);
  }
  else if (job == Job::SW) // NEW
  {
    const int stage = stages - (rank - PEs) / rows - 1;
    const int row = (rank - PEs) % rows;
    const auto [inH, inL] = getInputs(rank, PEs, stages, stage, rows, row);
    const auto [outH, outL] = getOutputs(rank, stage, rows, row);
    doSW(rank, stage, row, inH, inL, outH, outL);
  }

  if (rank < PEs) std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (rank == 0)
    for (int i = PEs; i < PEs + SWs; ++i)
    {
      MPI_Send(nullptr, 0, MPI_INT, i, Tag::END, MPI_COMM_WORLD);
    }


  MPI_Finalize();
  return EXIT_SUCCESS;
}
