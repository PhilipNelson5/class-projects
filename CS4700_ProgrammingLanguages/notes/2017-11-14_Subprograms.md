# Sub Programs

### Sub Programs
* 

### Definition
* the Definition includes
  * interface
  * actions

### Call and Return
* Call - the request to enter a sub program
* return - the resumption of the calling program
* A subprogram is active between call and return

### Procedures and Functions
* Procedures
  * do not return
  * they are intend as extension points for statements in the language
  * Mostly a feature of older languages
* Functions
  * Return
  * Modeled on mathematical functions
  * Generally should not have side effects

### Side effects
* The ways in which CS functions are not mathematical functions
* context
  * global variables
  * static local variables
* error
* Non-determinism
* Destruction
  * i/o

### Co-routines
* Includes _Yield_ and _Resume_
* Yield returns a value but maintains the current sate
* Resume restarts co-routine after last yield
* Call and Return still exist and define the lifetime

### Referencing environment
* set of bindings that a statement can see

### Closure
* a subprogram and it's referencing environment

### Return Values
* What are the types of return
* What are the number of returns

### Formal and Actual
* The parameter definitions in the header are called formal
* The parameter values in a call sire are called actual

### Positional or Keyword
* Positional - If the matching between formal and actual parameters is based only on the order
* Keyword
  * `foo ( bar = 42 )`

### Passing Parameters
* Pass by value
  * Only the value is passed (copy)
* Pass by result
  * A local variable is created
  * The local variable's value is copied back to the callers variable
* Pass by value result
  * Copy passed to function
  * Value is copied back to to caller
  * Also called pass by copy
* Pass by reference
  * creates an alias to caller's memory
* Pass by name
  * as if parameter was textually substituted
  * Referencing environment must also be included for name lookups

### Type checking parameters
* Do formal parameters have type?
* Do formal and actual parameters have to match?

### multidimensional arrays as parameters
* A language needs to be able to build the array mapping
* This complicates passing arrays
  * Sends pointer and do pointer arithmetic
  * Less flexible functions (specific array size and layout)
  * More complex (build in arrays)

### Sub-programs as parameters
* How can subprograms be passed?
* What is the referencing environment?
  * Call statement - Shallow Bindings
  * passed by function definition - deep
  * specified at call site - ad hawk

### Indirect Subprograms
* Function points
* Delegates
* Virtual functions

### Overloaded Functions
* Subprograms with the same name and referencing environment

### Overloaded Operators
* Some languages (C++, Ada, Python, Ruby...) allow operators to be overloaded
* Usually there is special function name that is invoked by operator syntax

### Generic Subprograms
* Generic subprograms work on multiple types
* The concept of a parameter is what the generic subprogram expects
* A type is said to model the concept if it meets the requirements

### Prologue and Epilogue (Function overhead)
* Function call must
  * suspend caller
  * compute any parameters and pass them
  * pass return address
  * transfer control
* Return must
  * resolve out parameters 
  * pass return value 
  * return control
  * resume the caller

### Activation Records
* Data needed by every invocation of a function
* Stack allocated local variables
* Parameters
* Return address
* Dynamic link (pointer to Activation Record of caller)
* Static link (pointer to Activation Record of functions own static scope)

### Dynamic Scope
* Deep access - lookup names using dynamic links
* Shallow access - maintain a stack for each name
* Semantics are equal
