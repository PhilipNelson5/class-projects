// According to this blog, one class of WAT behaviors in JS arise from the fact
// that in this language { curly braces } serve double-duty as delimiters for code
// blocks as well as object literals.
// 
//                http://2ality.com/2012/01/object-plus-object.html
// 
// Also, the addition operator + converts its complex arguments to a "primitive"
// type before performing the arithmetic. Moreover, because + means both
// "arithmetic addition" and "string concatenation", certain expressions can yield
// unexpected results if the types of the operands don't match.


    1 + 2 + 3

    "1" + "2" + "3"

    1 + "2" + 3

    1 + 2 + 3 + 4 + 5 + "6" + 7 + 8 + 9

// Subtraction seems to turn strings back into numbers:

    1 + 2 + 3 + 4 + 5 + "6" + 7 + 8 + 9 - 2

// But division doesn't:

    1 + 2 + 3 + 4 + 5 + "6" + 7 + 8 + 9 / 2

// I mean, you can even divide strings:

    "3" / "4"

    "577" / "408"


// Oh, I get it - it converts strings back to numbers when it feels like it!
// 
// Actually it has to do with operator precedence - the division happens before
// the addition/concatenation, like this:

    1 + 2 + 3 + 4 + 5 + "6" + 7 + 8 + (9 / 2)

// The important thing to realize is that sometimes you have to take matters into
// your own hands and manually convert values to or from strings just to be sure
// you're doing the right operation.



// Equality vs. Strict Equality
// ============================
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Comparison_Operators
// 
// We must be careful when we compare two values because, like the addition
// operator +, the equality comparison operators (== and !=) convert their
// operands before performing the comparison.
// 
// This means that we cannot detect the difference between a string and a number
// with == or !=:

    "3" == 3

    "4" != 4

// For this reason JavaScript provides "strict" versions of these same operators.
// They are written with an extra = at the end:

    "3" === 3

    "4" !== 4

// We can also ask JavaScript what the type of any value is in a similar way to
// Python's type() function. In JavaScript it is an operator called typeof, and
// returns a string describing its input's type:

    typeof 3

    typeof("3") // it may be used with or without parentheses

    typeof(Math)

    typeof({})

// So far, so good!

    typeof([])

    typeof(NaN)

// Ok, well, nothing is ever perfect... hey, look over there!



// Interesting facts about NaN
// ===========================
// NaN means "Not a Number", and is not ordinarily used by programmers; it is most
// often seen as the result of failure of a numeric function, in particular the
// functions in the Math object:

Math.sqrt(-1)

parseInt("one")

parseFloat("three-point-one-four-one-five-nine")

// NaN also has the interesting property that it is unequal to itself.

    NaN == NaN 
    
// Therefore, 

    Math.sqrt(-1) == parseFloat("i")


// Speaking of NaN...
// ==================
// What happens if we spell it "NAN" or "nan" or even "nAn"?

    NAN

    nan

    nAn

// JavaScript, like most modern programming languages, is case sensitive. This
// extends to the names of properties within objects.

/

// Objects and their properties
// ============================
// This is an empty object (no properties)

    var empty_object = {};


// This object has five properties - one of which is an object of two properties:

    var today = {
        day: "Wednesday",
        month: "April",
        year: 2014,

        weather: { //objects can contain nested objects like this one
            morning: "sunny",
            afternoon: "cloudy"
        },

        say_cheese: function() {
            console.log("Cheese!")
        },
    }

// We can ask JavaScript to help us count the number of properties

    Object.keys(empty_object).length

    Object.keys(today).length

    Object.keys(today.weather).length


// Each property of an object has a unique name. Duplicates get overwritten:

    var different = {
        name: "Henry Cavill",
        namE: "Ben Affleck",
        naMe: "Gal Gadot",
        naME: "Jason Momoa",
        nAme: "Ray Fisher",
        nAmE: "Ezra Miller",
    }

    var same = {
        name: "Henry Cavill",
        name: "Ben Affleck",
        name: "Gal Gadot",
        name: "Jason Momoa",
        name: "Ezra Miller",
        name: "Ray Fisher",
    }

Object.keys(different).length

Object.keys(same).length

console.log(same.name)
