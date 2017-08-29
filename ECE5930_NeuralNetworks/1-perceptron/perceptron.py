from graphics import *
import random

height = 500
width = 500
win = GraphWin("Drawing Test", height, width)

class Perceptron:
    def __init__(self, numInputs):
        self.weights = [0 for i in range(numInputs)]
        for i in range (numInputs):
            self.weights[i] = random.uniform(-1, 1)
    
    def activationFunc(self, sum):
        return 1 if sum >= 0 else -1

    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i]*self.weights[i]
        return self.activationFunc(sum)
         


def interp(x, x1, x2, y1, y2):
    return y1 + ((x-x1) * (y2-y1) / (x2-x1))

def plot(x, y, color):
    x = interp(x, -5, 5, 0, width)
    y = interp(y, -5, 5, 0, height)
    cir = Circle(Point(x, y), 3)
    cir.setOutline(color)
    cir.draw(win)

def loadData(file, color):
    dataSet = []
    for line in open(file,"r"):
        point = line.split()
        point = [float(point[0]), float(point[1])]
        plot(point[0], point[1], color)
        dataSet.append(point)
    return dataSet

#    for i in range(0,550,50):
#        line = Line(Point(i, 500), Point(500-i, 0))
#        line.draw(win)
#        win.getKey()   # Pause until key press to view result
#        line.undraw()


def main():

    win.setBackground("white")

    set1 = loadData("perceptrondat1", "blue")
    set2 = loadData("perceptrondat2", "black")

    p = Perceptron(2)

    for i in range(len(set1)):
        print(p.guess(set1[i]))

    win.getKey()   # Pause until key press to view result
    win.close()    # Close window when done

main()
