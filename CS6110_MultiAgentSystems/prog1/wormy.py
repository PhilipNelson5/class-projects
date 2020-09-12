# Dangernoodles and Noperopes (a Nibbles clone)
# By Philip Nelson
#  - Modified from Al Sweigart al@inventwithpython.com
#    http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license

import random, pygame, sys, math
from pygame.locals import *
from typing import List, Tuple

from Direction import Dir
from Snek import Snek
from Laser import Laser
import Colors as c
import settings as s
import Draw

Color = Tuple[int, int, int]
Key = int
Point = Tuple[int, int]

FPS: int = 10
CELLSIZE: int = 15
GRID_X: int = 75
GRID_Y: int = 50
WINDOWWIDTH: int = CELLSIZE * GRID_X
WINDOWHEIGHT: int = CELLSIZE * GRID_Y

assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."

CELLWIDTH: int = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT: int = int(WINDOWHEIGHT / CELLSIZE)

BGCOLOR: Color = c.BLACK

HEAD: int = 0
X: int = 0
Y: int = 1

def main() -> None:
    global FPSCLOCK, DISPLAYSURF, BASICFONT

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    pygame.display.set_caption('Nope Ropes and Danger Noodles')

    showStartScreen()
    while True:
        runGame()
        showGameOverScreen()


def runGame() -> None:
    # Create new Snek
    sneks: List[Snek] = [
        Snek(CELLWIDTH, CELLHEIGHT, c.DARKGREEN, c.GREEN, K_RIGHT, K_LEFT, K_UP, K_DOWN, K_RETURN),
        Snek(CELLWIDTH, CELLHEIGHT, c.BLUE, c.WHITE, K_d, K_a, K_w, K_s, K_SPACE)
    ]

    apples: List[Point] = [ getRandomLocation() for _ in range(s.numApples) ]
    lasers: List[Laser] = [ ]
    stones: List[Point] = [ ]

    while True: # main game loop
        for event in pygame.event.get(): # event handling loop
            if event.type == QUIT:
                terminate()
            if event.type == KEYDOWN:
                key: Key = event.key
                if key == K_ESCAPE:
                    terminate()
                for snek in sneks:
                    snek.checkTurn(key)
                    if key == snek.shootKey:
                        lasers.append(Laser(snek.dir, snek.locs[HEAD]))

        # snek hit edge
        for snek in sneks:
            if (snek.locs[HEAD][X] == -1 or
                    snek.locs[HEAD][X] == CELLWIDTH or
                    snek.locs[HEAD][Y] == -1 or
                    snek.locs[HEAD][Y] == CELLHEIGHT):
                return # game over

        # snek hit stone
        for snek in sneks:
            for stone in stones:
                if snek.locs[HEAD] == stone:
                    return # game over

        # snek hit own body
        for snek in sneks:
            for body in snek.locs[3:]:
                if body == snek.locs[HEAD]:
                    return # game over

        # snek hit other snek
        for snek1 in sneks:
            for snek2 in sneks:
                if snek1 is snek2: continue
                for loc in snek2.locs:
                    if snek1.locs[HEAD] == loc:
                        return # game over

        # laser hit snek
        lasers[:] = [ laser for laser in lasers if (laser.locs[HEAD][X] > -1 and
                                                    laser.locs[HEAD][X] < CELLWIDTH and
                                                    laser.locs[HEAD][Y] > -1 and
                                                    laser.locs[HEAD][Y] < CELLHEIGHT) ]

        for laser in lasers:
            for snek in sneks:
                for locS in snek.locs:
                    for locL in laser.locs:
                        if locL == locS:
                            if locS == HEAD:
                                return # game over
                            tail = snek.split(locL)
                            stones.extend(tail)
                            lasers.remove(laser)
                            continue


        # snek hit apple
        for snek in sneks:
            for apple in apples:
                try:
                    if snek.locs[HEAD] == apple:
                        apples.remove(apple)
                        apples.append(getRandomLocation())
                        snek.feed()
                except:
                    pass

        # move the sneks
        for snek in sneks:
            snek.move()

        # move the lasers
        for laser in lasers:
            laser.move()

        # draw scene
        DISPLAYSURF.fill(BGCOLOR)
        Draw.grid(DISPLAYSURF ,CELLSIZE, WINDOWWIDTH, WINDOWHEIGHT)
        for snek in sneks: Draw.snek(snek, DISPLAYSURF, CELLSIZE)
        for apple in apples: Draw.apple(apple, DISPLAYSURF, CELLSIZE)
        for laser in lasers: Draw.laser(laser, DISPLAYSURF, CELLSIZE)
        for stone in stones: Draw.stone(stone, DISPLAYSURF, CELLSIZE)
        for n in range(len(sneks)): Draw.score(len(sneks[n].locs) - 3, n, DISPLAYSURF, BASICFONT, WINDOWWIDTH)
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def checkForKeyPress():
    if len(pygame.event.get(QUIT)) > 0:
        terminate()

    keyUpEvents = pygame.event.get(KEYUP)
    if len(keyUpEvents) == 0:
        return None
    if keyUpEvents[0].key == K_ESCAPE:
        terminate()
    return keyUpEvents[0].key


def terminate() -> None:
    pygame.quit()
    sys.exit()


def spaceOccupied(loc) -> bool:
    for apple in apples:
        if apple == loc:
            return True

    for body in wormCoords:
        if body == loc:
            return True

    return False


def getRandomLocation() -> Point:
    return (random.randint(0, CELLWIDTH - 1), random.randint(0, CELLHEIGHT - 1))


def getRandomUnoccupiedLocation() -> bool:
    loc = getRandomLocation()
    while spaceOccupied(loc):
        loc = getRandomLocation()

    return loc
    

def showGameOverScreen() -> None:
    Draw.gameOver(DISPLAYSURF, pygame.font.Font('freesansbold.ttf', 150), WINDOWWIDTH, WINDOWHEIGHT)
    Draw.pressKeyMsg(DISPLAYSURF, BASICFONT, WINDOWWIDTH, WINDOWHEIGHT)
    pygame.display.update()
    pygame.time.wait(500)
    checkForKeyPress() # clear out any key presses in the event queue

    while True:
        if checkForKeyPress():
            pygame.event.get() # clear event queue
            return
        pygame.time.wait(500)


def showStartScreen() -> None:
    deg1: int = 0
    deg2: int = 0

    while True:
        Draw.fill(DISPLAYSURF, BGCOLOR)
        Draw.startScreen(DISPLAYSURF, BASICFONT, WINDOWWIDTH, WINDOWHEIGHT, deg1, deg2)

        if checkForKeyPress():
            pygame.event.get() # clear event queue
            return

        pygame.display.update()
        FPSCLOCK.tick(FPS)
        deg1 += 3 # rotate by 3 degrees each frame
        deg2 += 7 # rotate by 7 degrees each frame


if __name__ == '__main__':
    main()
