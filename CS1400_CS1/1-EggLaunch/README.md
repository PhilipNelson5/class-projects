# cs1.1.eggLaunch
First CS assignment

An industrious USU freshman wishes to conduct an ad hoc experiment wherein a 0.065 Kg raw egg is to be launched over the top of Old Main Hall using a slingshot with ideal elastics and delivered to an otherwise preoccupied professor traveling across the quad.  

The mass of the egg, the elastic constant, and the gravitational constant are all known.   Given a value for D (the distance to the professor in meters) and the value for theta (the angle of elevation in degrees), what distance X (the draw length in meters) should be elastic be stretched to in order to correctly deliver the egg to the indicated destination?  

A hint is cleverly hidden somewhere in the diagram that may be helpful for you when calculating a solution for X.

For this assignment, you are to write a function GetX(float D, float theta) that accepts two floats, D and theta (in that order), and then returns the correct value for X.  

To aid you with this, you are given a file eggLaunch.cppPreview the documentView in a new window that contains a main() function that calls your GetX() function.  You should modify this file by adding your name, a-number, and section to the comments at the top, and then by adding your code between the "//start of your code" and "//end of your code" comments ONLY.  Make no code changes outside this section.

To be painfully clear, your code should NOT ask the user for input or write anything to the screen.

After compiling and running the resulting program you should get this output:

distance: 100   theta: 1   draw: 8.54457
distance: 100   theta: 15   draw: 2.25743
distance: 100   theta: 30   draw: 1.71528
distance: 100   theta: 45   draw: 1.59625
distance: 100   theta: 60   draw: 1.71528
distance: 100   theta: 75   draw: 2.25743
distance: 100   theta: 89   draw: 8.54428

 

Don't forget that the sin() function in the cmath library takes radians -- not degrees.

What to turn in:  a single .cpp file with your and the original code in it.  


