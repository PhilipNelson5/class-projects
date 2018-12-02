
function factorial(n) {
    if (n == 0) return 1;

    return n * factorial(n - 1);
}

function factorialItr(n) {
    let stack = [];

    let result = 1;
    stack.push({ n: 0, result: 1 });
    while (stack.length > 0) {
        let item = stack.pop();
        result = item.result;

        if (item.n < n) {
            stack.push({
                n: item.n + 1,
                result: item.result * (item.n + 1)
            });
        }
    }

    return result;
}


console.log("factorial of 10: ", factorial(12));
console.log("factorial of 10: ", factorialItr(12));
