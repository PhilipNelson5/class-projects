# Roomba Simulation
# By Philip Nelson

import random, pygame, sys, math
from pygame.locals import *
from typing import List, Tuple, Optional

from Direction import Dir
from Object import Object
from Roomba import Roomba
from Charger import Charger
from Dirt import Dirt
from Wall import Wall
from Obstacle import Obstacle
from Dropoff import Dropoff
from Dog import Dog
import Colors as c
import settings
import Draw
import rendering as r

Color = Tuple[int, int, int]
Key = int
Point = Tuple[int, int]

r.cellsize = 15
WINDOWWIDTH: int = r.cellsize * settings.env_width_x
WINDOWHEIGHT: int = r.cellsize * settings.env_height_y

assert WINDOWWIDTH % r.cellsize == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % r.cellsize == 0, "Window height must be a multiple of cell size."

CELLWIDTH: int = int(WINDOWWIDTH / r.cellsize)
CELLHEIGHT: int = int(WINDOWHEIGHT / r.cellsize)

env:       List[List[Object]] = [ [ None for j in range(CELLHEIGHT) ] for i in range(CELLWIDTH) ]
roombas:   List[Roomba]   = [ ]
chargers:  List[Charger]  = [ ]
dirts:     List[Dirt]     = [ ]
walls:     List[Wall]     = [ ]
obstacles: List[Obstacle] = [ ]
dropoffs:  List[Dropoff]  = [ ]
dogs:      List[Dog]      = [ ]
totalDirt: int = 0

BGCOLOR: Color = c.ALICE_BLUE

HEAD: int = 0
X: int = 0
Y: int = 1

def main() -> None:
    global FPSCLOCK, BASICFONT

    # random.seed(0)
    # random.seed(6)
    # random.seed(7)

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    r.screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    pygame.display.set_caption('Roomba!')

    showStartScreen()
    while True:
        runGame()
        showGameOverScreen()


def runGame() -> None:
    global totalDirt

    # roombas and chargers
    for _ in range(settings.num_roombas):
        loc = getRandomOpenEdge()

        roomba = Roomba(c.DARK_TURQUOISE, c.SLATE_GRAY, loc, CELLWIDTH, CELLHEIGHT)
        charger = Charger(c.BLACK, c.YELLOW, loc)

        env[loc[X]][loc[Y]] = roomba

        roombas.append(roomba)
        chargers.append(charger)

    # walls
    for row in { 0, CELLHEIGHT-1 }:
        for col in range(CELLWIDTH):
            walls.append(Wall(c.LAVENDER, c.LAVENDER, (col, row)))

    for row in range(1, CELLHEIGHT-1):
        for col in [0, CELLWIDTH-1]:
            walls.append(Wall(c.LAVENDER, c.LAVENDER, (col, row)))


    # dirt
    for _ in range(settings.num_dirt):
        for loc in Dirt.generate(getOpenLocation(), CELLWIDTH, CELLHEIGHT):
            dirt = Dirt((210, 166, 121), (210, 166, 121), loc)
            if isinstance(env[loc[X]][loc[Y]], Dirt):
                env[loc[X]][loc[Y]].dirty()
            else:
                env[loc[X]][loc[Y]] = dirt
                dirts.append(dirt) 

    totalDirt = len(dirts)

    # obstacles
    for _ in range(settings.num_obstacles):
        loc = getOpenLocation(2, 2)
        for x in range(0,2):
            for y in range(0,2):
                obs = Obstacle(c.CRIMSON, c.CRIMSON, (loc[X]+x, loc[Y]+y))
                env[loc[X]+x][loc[Y]+y] = obs
                obstacles.append(obs)

    # dropoffs
    for _ in range(settings.num_dropoffs):
        loc = getOpenLocation(5, 5)
        for x in range(0,5):
            for y in range(0,5):
                obs = Dropoff(c.BLACK, c.BLACK, (loc[X]+x, loc[Y]+y))
                env[loc[X]+x][loc[Y]+y] = obs
                obstacles.append(obs)

    # dogs
    for _ in range(settings.num_dogs):
        loc = getOpenLocation()
        dog = Dog(c.DARK_BLUE, c.BROWN, loc)
        env[loc[X]][loc[Y]] = dog
        dogs.append(dog)

    while True: # main game loop
        render()

        # event handling loop
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            if event.type == KEYDOWN:
                key: Key = event.key
                if key == K_ESCAPE:
                    terminate()

        for roomba in roombas:
            loc: Point = roomba.move(getPercepts(roomba.loc))
            obj: Object = objectAt(loc)
            if isinstance(obj, Dirt):
                if obj.level == 0:
                    dirts.remove(obj)
                    roomba.cleaned += 1

        for dog in dogs:
            leaveDirt, loc = dog.move()
            obj = objectAt(dog.loc)
            if not isinstance(obj, Dog) and not isinstance(obj, Dirt):
                dog.loc = loc

            if leaveDirt:
                obj = objectAt(loc) 
                if isinstance(obj, Dirt):
                    obj.dirty()
                else:
                    dirts.append(Dirt((210, 166, 121), (210, 166, 121), loc))


        # input()
        pygame.display.update()
        FPSCLOCK.tick(settings.fps)


def render() -> None:
    Draw.fill(BGCOLOR)
    Draw.grid(r.screen, r.cellsize, WINDOWWIDTH, WINDOWHEIGHT)
    for dirt in dirts:
        dirt.draw()
    for wall in walls:
        wall.draw()
    for charger in chargers:
        charger.draw()
    for obs in obstacles:
        obs.draw()
    for dog in dogs:
        dog.draw()
    for i in range(len(roombas)):
        roomba = roombas[i]
        Draw.score(round(roomba.cleaned / totalDirt * 100), i, BASICFONT, WINDOWWIDTH)
        roomba.draw()



def getPercepts(loc: Point) -> List[List[Object]]:
#{
    percepts: List[List[Object]] = [[],[],[]]
    for row in range(-1, 2):
        for col in range(-1, 2):
            p: Point = (loc[X] + col, loc[Y] + row)
            if (p[X] >= 0
                    and p[X] < CELLWIDTH
                    and p[Y] >= 0
                    and p[Y] < CELLHEIGHT):
                percepts[row+1].append(objectAt(p))
            else:
                percepts[row+1].append(None)

    return percepts
            

def checkForKeyPress() -> None:
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


def objectAt(p: Point) -> Optional[Object]:
    global roombas, chargers, dirts

    for a in walls:
        if a.loc == p: return a

    for a in chargers:
        if a.loc == p: return a

    for a in obstacles:
        if a.loc == p: return a

    for a in dropoffs:
        if a.loc == p: return a

    for a in dirts:
        if a.loc == p: return a

    for a in roombas:
        if a.loc == p: return a

    for a in dogs:
        if a.loc == p: return a

    return None


def getRandomLocation(dx: int = 0, dy: int = 0) -> Point:
    return (random.randint(0 + dx, CELLWIDTH - dy - 1), random.randint(0 + dy, CELLHEIGHT - dy - 1))


def getOpenLocation(dx: int = 0, dy: int = 0) -> Point:
    p = getRandomLocation(dx, dy)

    while objectAt(p) is not None:
        p = getRandomLocation(dx, dy)

    return p


def getRandomOpenEdge() -> Point:
    wall = random.randint(0, 3)
    while True:
        p: Point = (0, 0)
        if wall == 0: # top
            p = (random.randint(0, CELLWIDTH - 1), 0)
        elif wall == 1: # bottom
            p = (random.randint(0, CELLWIDTH - 1), CELLHEIGHT - 1)
        elif wall == 2: # right
            p = (CELLWIDTH - 1, random.randint(0, CELLHEIGHT - 1))
        else: # wall == 3: # left
            p = (0, random.randint(0, CELLHEIGHT - 1))

        if objectAt(p) is None: break

    return p



def showGameOverScreen() -> None:
    Draw.gameOver(r.screen, pygame.font.Font('freesansbold.ttf', 150), WINDOWWIDTH, WINDOWHEIGHT)
    Draw.pressKeyMsg(r.screen, BASICFONT, WINDOWWIDTH, WINDOWHEIGHT)
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
    inc: int = 1

    while True:
        Draw.fill(BGCOLOR)
        Draw.startScreen(r.screen, BASICFONT, WINDOWWIDTH, WINDOWHEIGHT, deg1, deg2)

        if checkForKeyPress():
            pygame.event.get() # clear event queue
            return

        pygame.display.update()
        FPSCLOCK.tick(settings.fps)
        deg1 += 3
        deg2 += inc
        inc += 1


if __name__ == '__main__':
    main()
