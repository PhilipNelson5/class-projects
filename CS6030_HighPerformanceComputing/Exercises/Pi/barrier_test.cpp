#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include "Barrier.hpp"

const int MAX_THREADS = 1024;
const int MSG_MAX = 100;
std::shared_ptr<Barrier> barrier = nullptr;

/* Global variables:  accessible to all threads */
int thread_count;
std::vector<std::string> messages;

void Usage(char* prog_name);
void Send_msg(int rank);  /* Thread function */

int main(int argc, char* argv[])
{

  if (argc != 2) Usage(argv[0]);
  thread_count = std::stoi(argv[1]);
  if (thread_count <= 0 || thread_count > MAX_THREADS) Usage(argv[0]);

  messages.resize(thread_count);
  barrier = std::make_shared<Barrier>(thread_count);
  std::vector<std::thread> threads;

  for (int rank = 0; rank < thread_count; rank++)
  {
    threads.emplace_back(Send_msg, rank);
  }

  for (auto & thread : threads)
  {
    thread.join();
  }

  return EXIT_SUCCESS;
}


/*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name)
{

  fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
  exit(EXIT_FAILURE);
}


/*-------------------------------------------------------------------
 * Function:       Send_msg
 * Purpose:        Create a message and ``send'' it by copying it
 *                 into the global messages array.  Receive a message
 *                 and print it.
 * In arg:         rank
 * Global in:      thread_count
 * Global in/out:  messages
 * Return val:     Ignored
 * Note:           The my_msg buffer is freed in main
 */
void Send_msg(const int rank)
{
  const int dest = (rank + 1) % thread_count;
  const int source = (rank + thread_count - 1) % thread_count;
  std::stringstream msg;

  msg << "Hello to " << dest << " from " << rank;
  messages[dest] = msg.str();
  
  barrier->wait();

  if (messages[rank] != "") 
    std::cout << "Thread " << rank << " > " << messages[rank] << std::endl;
  else
    std::cout << "Thread " << rank << " > No message from " << source << std::endl;
}