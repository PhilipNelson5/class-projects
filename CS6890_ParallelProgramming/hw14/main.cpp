#include <cmath>
#include <iostream>
#include <mpi.h>
#include <utility>

enum class Job
{
  PE,
  SW,
  NONE
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
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  const auto [stages, rows] = getDimenstionsSW(PEs);

  if (job == Job::PE)
  {
    const int dest = rank % (PEs / 2) + PEs;
    printInfoPE(rank, dest);
  }
  else if (job == Job::SW) // NEW
  {
    const int stage = stages - (rank - PEs) / rows - 1;
    const int row = (rank - PEs) % rows;
    const auto [inH, inL] = getInputs(rank, PEs, stages, stage, rows, row);
    const auto [outH, outL] = getOutputs(rank, stage, rows, row);
    printInfoSW(rank, stage, row, inH, inL, outH, outL);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
