#!/usr/bin/env python

from sympy import *

init_printing()

A = Matrix([[3.4109, -0.2693, -1.0643],[1.5870, 1.5546, -5.5361],[0.2981, -0.2981, 1.2277]])
I = Matrix([[1,0,0],[0,1,0],[0,0,1]])
y = symbol('y')
lamba = symbol('lamba')

pprint(I)

eVals = A.eigenvals()

eVects = A.eigenvects()

#print(eVals)
pprint(A)
print("\nEigen Values:")
for val in list(eVals.keys()):
    print(val.evalf())

print("\nEigen Vectors:")
pprint(eVects)
