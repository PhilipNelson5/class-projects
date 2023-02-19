#include <iostream>
#include <mpi.h>
#include <vector>
#include <string>
#include <algorithm>

#include "random.hpp"
#include "print.hpp"

#define MCW MPI_COMM_WORLD

std::vector<int> count_histogram(std::vector<int> data, const int min, const int max, const int n_bins)
{
  const double bin_size = (max - min+1) / (double)n_bins;
  std::vector<int> histogram;
  histogram.resize(n_bins, 0);
  for (auto i = 0u; i < data.size(); ++i)
  {
    const int bin = (data[i] - min) / bin_size; 
    ++histogram[bin];
  }
  return histogram;
}

int main(int argc, char** argv)
{
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MCW, &rank);
  MPI_Comm_size(MCW, &size);
  const int n_elements = argc > 1 ? std::stoi(argv[1]) : 1000;
  const int n_bins = argc > 2 ? std::stoi(argv[2]) : 10;
  const int n_elem_per_process = n_elements / (size-1);
  
  if (rank == 0)
  {
    std::vector<int> data;
    data.resize(n_elements);
    random_int_fill(data.begin(), data.end(), 0, 100);
    const auto [min, max] = std::minmax_element(data.begin(), data.end());
    // print(data);
    
    for (int i = 1; i < size; ++i)
    {
      MPI_Send(&data[(i-1)*n_elem_per_process], n_elem_per_process, MPI_INT, i, 0, MCW);
      MPI_Send(&*min, 1, MPI_INT, i, 0, MCW);
      MPI_Send(&*max, 1, MPI_INT, i, 0, MCW);
    }
    
    std::vector<int> histogram;
    if (n_elements % n_elem_per_process != 0)
    {
      histogram = count_histogram(data, *min, *max, n_bins);
    }
    else
    {
      histogram.resize(n_bins, 0);
    }
    
    std::vector<int> recv;
    recv.resize(n_bins);
    for (int i = 1; i < size; ++i)
    {
      MPI_Recv(recv.data(), n_bins, MPI_INT, i, 0, MCW, MPI_STATUS_IGNORE);
      for (int i = 0; i < n_bins; ++i)
      {
        histogram[i] += recv[i];
      }
    }
    
    print("histogram", histogram) << '\n';
    std::for_each(histogram.begin(), histogram.end(), [] (const int n) {
      print('|', std::string(n/5, '*')) << '\n';
    });
  }
  else
  {
    std::vector<int> data;
    int min, max;
    data.resize(n_elem_per_process);
    MPI_Recv(data.data(), n_elem_per_process, MPI_INT, 0, 0, MCW, MPI_STATUS_IGNORE);
    MPI_Recv(&min, 1, MPI_INT, 0, 0, MCW, MPI_STATUS_IGNORE);
    MPI_Recv(&max, 1, MPI_INT, 0, 0, MCW, MPI_STATUS_IGNORE);
    // print(data);
    
    std::vector<int> histogram = count_histogram(data, min, max, n_bins);
    print(histogram) << '\n';

    MPI_Send(histogram.data(), n_bins, MPI_INT, 0, 0, MCW);
  }

  MPI_Finalize();
}