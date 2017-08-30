from graphics import *
import random
import time

height = 700
width = 700
minX = -5
maxX = 5
minY = -5
maxY = 5
X = 0
Y = 1
win = GraphWin("Drawing Test", height, width)

class MyPoint:
    def __init__(self, x_, y_, b_= 1, l_= 0):
        self.x = x_     # x coordinate
        self.y = y_     # y coordinate
        self.b = b_     # bias
        self.label = l_ # label of data set

    def mapX(self):
        return interp(self.x, minX, maxX, 0, width)

    def mapY(self):
        return interp(self.y, minY, maxY, height, 0)

def interp(x, x1, x2, y1, y2):
    return y1 + ((x-x1) * (y2-y1) / (x2-x1))

def plot(x, y, outline, fill, size):
    cir = Circle(Point(x, y), size)
    cir.setOutline(outline)
    cir.setFill(fill)
    cir.draw(win)

def loadData(file, target, color, dataSet = []):
    for line in open(file,"r"):
        raw = line.split()
        p = MyPoint(float(raw[0]), float(raw[1]), 1, target)
        plot(p.mapX(), p.mapY(), color, "white", 6)
        dataSet.append(p)
    return dataSet

class Perceptron:
    def __init__(self, numInputs):
        self.lr = 0.1 # Learning Rate
        self.weights = [0 for i in range(numInputs)]
        for i in range (numInputs):
            self.weights[i] = random.uniform(-1, 1)
        self.done = True
    
    def activationFunc(self, sum):
        return 1 if sum >= 0 else -1

    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i]*self.weights[i]
        return self.activationFunc(sum)

    def guessY(self, x):
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]

        return -(w2/w1) - (w0/w1) * x

    def train(self, inputs, target):
        error = target - self.guess(inputs)
        pt = MyPoint(inputs[X], inputs[Y])
        if error == 0:
            plot(pt.mapX(), pt.mapY(), "green", "green", 3)
        else :
            plot(pt.mapX(), pt.mapY(), "red", "red", 3)
            self.done = False

        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr

def main():

    win.setBackground("white")

    trainingData = loadData("perceptrondat1", 1, "black")
    trainingData = loadData("perceptrondat2", -1,"black", trainingData)

    p = Perceptron(3)

    #for i in range(len(trainingData)):
    index = 0;
    iters = 0;
    while True:
        ptA = MyPoint(minX,p.guessY(minX))
        ptB = MyPoint(maxX,p.guessY(maxX))
        line = Line(Point(ptA.mapX(), ptA.mapY()), Point(ptB.mapX(), ptB.mapY()))
        line.draw(win)

        x = trainingData[index].x
        y = trainingData[index].y
        b = trainingData[index].b
        l = trainingData[index].label
        p.train([x,y,b], l)

        index += 1
        if index == len(trainingData):
            index = 0
            iters += 1
            if p.done:
                print("iterations: " + str(iters))
                break
            p.done = True

        time.sleep(.05)
        line.undraw()



    win.getKey()   # Pause until key press to view result
    win.close()    # Close window when done

main()
