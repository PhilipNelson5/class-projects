from graphics import *
import random
import time

height = 700
width = 700
MIN = -5
MAX = 5
POINT = 0
LABEL = 1
X = 0
Y = 1
win = GraphWin("Drawing Test", height, width)

def interp(x, x1 = -5, x2 = 5, y1 = 0, y2 = height):
    return y1 + ((x-x1) * (y2-y1) / (x2-x1))

def plot(x, y, outline, fill, size):
    x = interp(x, -5, 5, 0, width)
    y = interp(y, -5, 5, 0, height)
    cir = Circle(Point(x, y), size)
    cir.setOutline(outline)
    cir.setFill(fill)
    cir.draw(win)

def loadData(file, target, color, dataSet = []):
    for line in open(file,"r"):
        point = line.split()
        point = [float(point[0]), float(point[1])]
        plot(point[0], point[1], color, "white", 6)
        dataSet.append([point, target])
    return dataSet

#    for i in range(0,550,50):
#        line = Line(Point(i, 500), Point(500-i, 0))
#        line.draw(win)
#        win.getKey()   # Pause until key press to view result
#        line.undraw()


class Perceptron:
    def __init__(self, numInputs):
        self.lr = 0.1 # Learning Rate
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

    def train(self, inputs, target):
        error = target - self.guess(inputs)
        if error == 0:
            plot(inputs[X], inputs[Y], "green", "green", 3)
        else :
            plot(inputs[X], inputs[Y], "red", "red", 3)

        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr

def main():

    win.setBackground("white")

    trainingData = loadData("perceptrondat1", 1, "black")
    trainingData = loadData("perceptrondat2", -1,"black", trainingData)

    p = Perceptron(2)

    #for i in range(len(trainingData)):
    index = 0;
    line = Line(Point(interp(-4), interp(-4)), Point(interp(4), interp(4)))
    line.draw(win)
    while True:
        p.train(trainingData[index][POINT], trainingData[index][LABEL])
        index += 1
        index %= len(trainingData)
        time.sleep(.01)



    win.getKey()   # Pause until key press to view result
    win.close()    # Close window when done

main()
