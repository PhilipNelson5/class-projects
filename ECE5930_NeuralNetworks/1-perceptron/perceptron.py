from graphics import *

height = 500
width = 500
win = GraphWin("Drawing Test", height, width)

def interp(x, x1, x2, y1, y2):
    return y1 + ((x-x1) * (y2-y1) / (x2-x1))

def plot(x, y, color):
    x = interp(x, -5, 5, 0, width)
    y = interp(y, -5, 5, 0, height)
    cir = Circle(Point(x, y), 3)
    cir.setOutline(color)
    cir.draw(win)

def plotData():
    infile = open("perceptrondat1","r")
    for line in infile:
        point = line.split()
        point = [float(point[0]), float(point[1])]
        plot(point[0], point[1], 'blue')

    infile = open("perceptrondat2","r")
    for line in infile:
        point = line.split()
        point = [float(point[0]), float(point[1])]
        plot(point[0], point[1], 'black')

#    for i in range(0,550,50):
#        line = Line(Point(i, 500), Point(500-i, 0))
#        line.draw(win)
#        win.getKey()   # Pause until key press to view result
#        line.undraw()


def main():

    win.setBackground("white")

    plotData()

    win.getKey()   # Pause until key press to view result
    win.close()    # Close window when done

main()
