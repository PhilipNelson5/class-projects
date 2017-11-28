# Concurrency

* exploit concurrent hardware
  * multi-core systems
  * i/o bound processes
    * network card sending data
* concurrent design fits some domains
* distributed processing

### Threads and Processes
* Thread, sequence of executed statements
* A process is a collection of threads

### Synchronization
* Co-operative
  * Producer-Consumer
* Competitive
  * Resource Contention

### Race Condition
* \> 2 threads
* Accessing the same resource
* At least one is modifying the resource

### Synchronization Quadrants
```
                 Multi Thread
                      ^
                      |
        Race          |    Functional
        Condition     |    Programming
                      |
                      |
Mutable <---------------------------> Immutable
                      |
        Assignment    |    Functional
        ~90%          |    Programming
                      |
                      |
                      v
                Single Thread
```

### Liveness and Deadlock
#### Mutexes are the anti thread
* Everyone is wasting for everyone

### Semaphores
* The Anti-thread
* Low Level - at the end, everything is a semaphore
* Considered unsafe

### Monitors  (java & c#)
* Mandatory mutex
* Language inserts locks and unlocks as needed
* Generally more conservative

### Message Passing
* Relies on a thread-safe queue
* No shared state outside of the queue

### Statement Level Concurency
* Statements that express parallelism
  * FORTRAN parallel for loop
* Such statements can be automatically mapped to concurrent forms
* FORTRAN, openMP
