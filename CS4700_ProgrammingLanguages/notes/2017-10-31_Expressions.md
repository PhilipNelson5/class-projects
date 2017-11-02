## Expressions

### Expression
* means of expressing computation
* Some combination of values and operators that has a value

### Operator Overloading
* Can users overload operators?
* Does the language overload operators?
* How does this affect the language

### Side Effects
* An observable change of global state
* ex:
  * output parameters
  * global variables
  * class variables
  * IO

### Refferential Transparency
* If an expression that has no side effects it can be thought of as a reference to it's value
* Functions are pure if they have this property
* Four major causes of impurity
  * Error/Failure
  * Non-determinism
    * Random Values
  * Context
    * Global Variables
    * Class Variables
  * Destruction 
    * If the function can not be undone
    * output can't be undone
    * assignment

### Short Circuit Evaluation
* Determining the value of an expression without evaluating the whole expression is called short circuit evaluation
* Boolean algebra

### Lazy vs. Eager Evaluation
* Lazy
  * expressions become values at the last possible moment
  * very common in functional languages
  * allows expressions of infinite objects - but not their evaluation
* Eager
  * expressing become values at the earliest opportunity

### Arithmetic Expressions
* Unary
* Binary
  * infix
  * prefix - unambiguous
  * postfix - unambiguous
* Ternary

### Boolean Expressions
* Comparisons
  * Two-way operator
    * True | False
  * Three-way operator
    * Less | Equal | Greater
* Boolean Algebra

### Assignment
* Procedural Languages - Write to memory, always a side effect
* Functional and Logic - Creates a new name binding to a constant value

### Type conversoins - Casting
* Narrowing
  * the target can not exactly represent all instances of the source type
  * Float <-> Int
* Widening
* Casting - explicit type conversion
* Which are allowed in the language?
* Which are implicit and explicit?
* Are mixed mode expressions allowed?
  * 1 + 1.0
