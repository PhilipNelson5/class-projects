#!/usr/bin/env python

import numpy.random as rnd
import matplotlib.pyplot as plt

def loadData(file, color):
    dataSet = []
    for line in open(file,"r"):
        raw = line.split()
        p = [1, float(raw[0]), float(raw[1])]
        plt.plot(p[1],p[2], "o")
        dataSet.append(p)
    return dataSet

def negate(set):
    for row in set:
        row[0] *= -1
        row[1] *= -1
        row[2] *= -1
    return set

def aTy(a, y):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*y[i]
    return sum

def gradDesc(a, rate, y):
    for i in range(len(a)):
        a[i] = a[i] + rate * y[i]
    return a

def main():
    lr = 1
    a = [rnd.random(), rnd.random(), rnd.random()]
    set1 = loadData("perceptrondat1", "blue")
    set2 = loadData("perceptrondat2", "red")

    trainingData = set1 + negate(set2)

    aOld = [0, 0, 0]
    while aOld != a:
        aOld = a
        for i in range(len(trainingData)):
            result = aTy(a, trainingData[i]) 
            if result < 0:
                a = gradDesc(a, lr, trainingData[i])

    m = a[0]/a[1]
    b = a[2]/a[1]
    print("Line: y = -" + str(m) + " x - " + str(b))


    plt.plot([-1,-1], [1,1], 'k-')
    plt.show()

main()
