from graphics import *

def interp(x, x1, x2, y1, y2):
    return y1 + ((x-x1) * (y2-y1) / (x2-x1))

def main():

    height = 500
    width = 500
    win = GraphWin("Drawing Test", height, width)
    win.setBackground("white")

    infile = open("perceptrondat1","r")

    for line in infile:
        point = line.split()
        point = [float(point[0]), float(point[1])]
        x = interp(point[0], -5, 5, 0, width)
        y = interp(point[1], -5, 5, 0, height)

        Circle(Point(x, y), 3).draw(win)

    for i in range(0,550,50):
        line = Line(Point(i, 500), Point(500-i, 0))
        line.draw(win)
        win.getKey()   # Pause until key press to view result
        line.undraw()

    win.getKey()   # Pause until key press to view result
    win.close()    # Close window when done

main()
