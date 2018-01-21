#!/usr/bin/env python

from sympy import *
import math

#init_printing()

A = Matrix([[3.4109, -0.2693, -1.0643],[1.5870, 1.5546, -5.5361],[0.2981, -0.2981, 1.2277]])
I = Matrix([[1,0,0],[0,1,0],[0,0,1]])
y = symbols('y')

#pprint(A - y*I)

eVals = A.eigenvals()
eVects = A.eigenvects()

#print(eVals)
pprint(A)
print("\nEigen Values:")
for val in list(eVals.keys()):
    print(val.evalf())

print("\nEigen Vectors:")
pprint(eVects)



print("\nEigen Value Matricies:")
print("-----------------------------------------------------")
#for val in list(eVals.keys()):
for val in [math.pi, 1.0/3.0, math.e]:
    eig = val#.evalf()
    m = A-I*eig
    print(eig)
    pprint(m)
    pprint(m.rref())
    print("-----------------------------------------------------")

