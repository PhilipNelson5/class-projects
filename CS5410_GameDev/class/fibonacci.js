// memoized fibonacci

let memory = {0: 1, 1: 1};

let fib =
    function Fibonacci(n) {
  if (memory.hasOwnProperty(n)) {
    return memory[n];
  } else {
    memory[n] = fib(n - 1) + fib(n - 2);
    return memory[n];
  }
}

    console.log(fib(30));
