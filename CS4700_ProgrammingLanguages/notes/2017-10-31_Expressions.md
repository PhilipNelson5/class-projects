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

