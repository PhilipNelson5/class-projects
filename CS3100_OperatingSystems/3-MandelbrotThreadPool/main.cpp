#include "mandelbrot.hpp"
#include "threadPool.hpp"
#include "threadSafeQueue.hpp"
#include "timer.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

//* 1 pixel
//* multiple pixels but less than 1 row
//* 1 row
//* multiple rows but less than evenly divided chunks
//* rows/n size chunks

int rand(int start, int end)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(start, end);
  return dist(mt);
}

void runThreadedRows(int threads, int rows)
{
  ThreadPool pool(threads);

  for (int i = 0; i < height(); i += rows)
  {
    if (i + rows > height())
      pool.post([=]() { render(i, height()); });
    else
      pool.post([=]() { render(i, i + rows - 1); });
  }

  pool.start();
}

void runThreadedPixels(int threads, int pixels)
{
  ThreadPool pool(threads);

  for (int i = 0; i < imagesize() - 1; i += pixels)
  {
    if (i + pixels > imagesize())
      pool.post([=]() { renderPixels(i, imagesize()); });
    else
      pool.post([=]() { renderPixels(i, i + pixels - 1); });
  }

  pool.start();
}

void runThreadedRandom(int threads, int start, int end)
{
  ThreadPool pool(threads);

  int pixels = rand(start, end);
  for (int i = 0; i < imagesize() - 1; i += pixels)
  {
    pixels = rand(start, end);
    if (i + pixels > imagesize())
      pool.post([=]() { renderPixels(i, imagesize()); });
    else
      pool.post([=]() { renderPixels(i, i + pixels - 1); });
  }

  pool.start();
}

int main(int argc, char *argv[])
{
  // check if settings.txt exists
  if (!import("settings.txt")) return EXIT_FAILURE;
  std::ofstream fout("data.csv");

  Timer t;
  t.reset();
  auto tests = 10;
  auto n = std::thread::hardware_concurrency();
  auto threads = n + 1;

  // specify number of tests per thread
  if (argc > 1)
  {
    tests = atoi(argv[1]);
  }
  std::cout << n << " concurrent threads are supported.\n";
  std::cout << "threads: " << threads << std::endl;

  // one  pixel
  fout << "1 pixel" << std::endl;
  for (int j = 1; j < 9; ++j)
  {
    threads = j;
    std::cout << "threads: " << threads << std::endl;

    // fout << "Threads,Average,Standard Dev" << std::endl;

    for (int i = 0; i < tests; ++i)
    {
      t.time([=]() { runThreadedPixels(threads, 1); });
    }
    auto stdev = t.getStdDev();
    auto ave = t.getAverage();

    std::cout << "           Average: " << t.getAverage() << std::endl;
    std::cout << "Standard Deviation: " << t.getStdDev() << std::endl;
    fout << threads << "," << ave << "," << stdev << "," << std::endl;
    t.reset();
  }

  fout << "multiple pixels, less than one row" << std::endl;
  for (int j = 1; j < 9; ++j)
  {
    threads = j;
    std::cout << "threads: " << threads << std::endl;

    // fout << "Threads,Average,Standard Dev" << std::endl;

    for (int i = 0; i < tests; ++i)
    {
      t.time([=]() { runThreadedPixels(threads, 250); });
    }
    auto stdev = t.getStdDev();
    auto ave = t.getAverage();

    std::cout << "           Average: " << t.getAverage() << std::endl;
    std::cout << "Standard Deviation: " << t.getStdDev() << std::endl;
    fout << threads << "," << ave << "," << stdev << "," << std::endl;
    t.reset();
  }

  fout << "one row" << std::endl;
  for (int j = 1; j < 9; ++j)
  {
    threads = j;
    std::cout << "threads: " << threads << std::endl;

    // fout << "Threads,Average,Standard Dev" << std::endl;

    for (int i = 0; i < tests; ++i)
    {
      t.time([=]() { runThreadedRows(threads, 1); });
    }
    auto stdev = t.getStdDev();
    auto ave = t.getAverage();

    std::cout << "           Average: " << t.getAverage() << std::endl;
    std::cout << "Standard Deviation: " << t.getStdDev() << std::endl;
    fout << threads << "," << ave << "," << stdev << "," << std::endl;
    t.reset();
  }

  fout << "multiple rows unevenly divided" << std::endl;
  for (int j = 1; j < 9; ++j)
  {
    threads = j;
    std::cout << "threads: " << threads << std::endl;

    // fout << "Threads,Average,Standard Dev" << std::endl;

    for (int i = 0; i < tests; ++i)
    {
      t.time([=]() { runThreadedRandom(threads, width() + 1, width() * 5); });
    }
    auto stdev = t.getStdDev();
    auto ave = t.getAverage();

    std::cout << "           Average: " << t.getAverage() << std::endl;
    std::cout << "Standard Deviation: " << t.getStdDev() << std::endl;
    fout << threads << "," << ave << "," << stdev << "," << std::endl;
    t.reset();
  }

  fout << "rows divided by n chunks" << std::endl;
  for (int j = 1; j < 9; ++j)
  {
    threads = j;
    std::cout << "threads: " << threads << std::endl;

    // fout << "Threads,Average,Standard Dev" << std::endl;

    for (int i = 0; i < tests; ++i)
    {
      t.time([=]() { runThreadedRows(threads, height() / threads); });
    }
    auto stdev = t.getStdDev();
    auto ave = t.getAverage();

    std::cout << "           Average: " << t.getAverage() << std::endl;
    std::cout << "Standard Deviation: " << t.getStdDev() << std::endl;
    fout << threads << "," << ave << "," << stdev << "," << std::endl;
    t.reset();
  }
  write("SAVE.ppm");
}
