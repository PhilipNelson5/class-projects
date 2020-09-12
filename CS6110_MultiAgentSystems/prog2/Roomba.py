import random
from random import randint
from typing import Tuple, List
from enum import Enum, unique
import pygame
from pygame.locals import *

from Object import Object
from Dirt import Dirt
from Wall import Wall
from Obstacle import Obstacle
from Direction import Dir
from Dropoff import Dropoff
from Dog import Dog
import rendering as r
import Colors as c

Color = Tuple[int, int, int]
Point = Tuple[int, int]

X: int = 0;
Y: int = 1;

debug: bool = False

@unique
class State(Enum):
    BACK_FORTH_RIGHT = 'back and forth right'
    BACK_FORTH_LEFT = 'back and forth left'
    BACK_FORTH_TOP = 'back and forth top'
    BACK_FORTH_BOT = 'back and forth bottom'


class Roomba(Object):
    def __init__(self, outerColor: Color, innerColor: Color, charger: Point, w: int, h: int):
        super().__init__(outerColor, innerColor, charger)

        self.charger: Point = charger
        self.nextMoves = []
        self.cleaned: int = 0
        self.moves: int = 0
        
        if self.loc[X] == 0:
            self.dir = Dir.RIGHT
        elif self.loc[X] == w - 1:
            self.dir = Dir.LEFT
        elif self.loc[Y] == 0:
            self.dir = Dir.DOWN
        elif self.loc[Y] == h - 1:
            self.dir = Dir.UP
        else:
            self.dir = random.choice(list(Dir))

        if self.dir is Dir.RIGHT or self.dir is Dir.LEFT:
            self.state: State = State.BACK_FORTH_TOP
        else:
            self.state : State = State.BACK_FORTH_RIGHT

        self.moveForward()



    def printPercepts(self, percepts: List[List[Object]]) -> None:
        print(" ---")
        for l in percepts:
            print("|", end="")
            for p in l:
                if p is None: print(" ", end="")
                elif isinstance(p, Dirt): print("D", end="")
                elif isinstance(p, Wall): print("W", end="")
                else: print("X", end="")
            print("|")
        print(" ---")



    def rectify(self, percepts: List[List[Object]]) -> None:
        if self.dir == Dir.UP: return
        if self.dir == Dir.DOWN: 
            for row in percepts:
                row.reverse()
            temp: List[Object] = percepts[0]
            percepts[0] = percepts[2]
            percepts[2] = temp
        if self.dir == Dir.LEFT: 
            tmp: Object = percepts[0][0]
            tmp2: Object = percepts[1][0]
            percepts[0][0] = percepts[2][0]
            percepts[1][0] = percepts[2][1]
            percepts[2][0] = percepts[2][2]
            percepts[2][1] = percepts[1][2]
            percepts[2][2] = percepts[0][2]
            percepts[1][2] = percepts[0][1]
            percepts[0][2] = tmp          
            percepts[0][1] = tmp2
        if self.dir == Dir.RIGHT: 
            tmp: Object = percepts[0][0]
            tmp2: Object = percepts[0][1]
            percepts[0][0] = percepts[0][2]
            percepts[0][1] = percepts[1][2]
            percepts[0][2] = percepts[2][2]
            percepts[1][2] = percepts[2][1]
            percepts[2][2] = percepts[2][0]
            percepts[2][1] = percepts[1][0]
            percepts[2][0] = tmp          
            percepts[1][0] = tmp2


    def turnRight(self) -> None:
        if debug: print("  turn right")
        if self.dir == Dir.RIGHT:
            self.dir = Dir.DOWN
        elif self.dir == Dir.DOWN:
            self.dir = Dir.LEFT
        elif self.dir == Dir.LEFT:
            self.dir = Dir.UP
        elif self.dir == Dir.UP:
            self.dir = Dir.RIGHT


    def turnLeft(self) -> None:
        if debug: print("  turn left")
        if self.dir == Dir.RIGHT:
            self.dir = Dir.UP
        elif self.dir == Dir.UP:
            self.dir = Dir.LEFT
        elif self.dir == Dir.LEFT:
            self.dir = Dir.DOWN
        elif self.dir == Dir.DOWN:
            self.dir = Dir.RIGHT


    def turnAround(self) -> None:
        if debug: print("  turn around")
        if self.dir == Dir.RIGHT:
            self.dir = Dir.LEFT
        elif self.dir == Dir.LEFT:
            self.dir = Dir.RIGHT
        elif self.dir == Dir.UP:
            self.dir = Dir.DOWN
        elif self.dir == Dir.DOWN:
            self.dir = Dir.UP


    def turnRandomly(self) -> None:
        rnd = randint(0,20)
        if rnd == 0:
            self.turnRight()
        elif rnd == 1:
            self.turnLeft()
        elif rnd == 2:
            self.turnAround()
        else:
            if debug: print("  don't turn")


    def moveForward(self) -> None:
        if debug: print("  move forward")

        self.moves += 1

        if self.dir == Dir.RIGHT:
            self.loc = (self.loc[X] + 1, self.loc[Y])
        elif self.dir == Dir.LEFT:
            self.loc = (self.loc[X] - 1, self.loc[Y])
        elif self.dir == Dir.UP:
            self.loc = (self.loc[X], self.loc[Y] - 1)
        elif self.dir == Dir.DOWN:
            self.loc = (self.loc[X], self.loc[Y] + 1)


    def move(self, percepts: List[List[Object]]) -> Point:

        self.rectify(percepts)
        if debug: print("=============")
        if debug: print(self.loc, self.state.name)
        if debug: self.printPercepts(percepts)


        if isinstance(percepts[1][1], Dirt):
            percepts[1][1].clean()
            return self.loc

        #  dirt in front 
        if isinstance(percepts[0][1], Dirt):
            if debug: print("dirt in front")
            self.moveForward()

        #  dirt on right 
        elif isinstance(percepts[1][2], Dirt):
            if debug: print("dirt on right")
            self.turnRight()

        #  dirt on left 
        elif isinstance(percepts[1][0], Dirt):
            if debug: print("dirt on left")
            self.turnLeft()

        #  dirt on front-right 
        elif isinstance(percepts[0][2], Dirt):
            if debug: print("dirt on front-right")
            self.moveForward()

        #  dirt on front-left 
        elif isinstance(percepts[0][0], Dirt):
            if debug: print("dirt on front-left")
            self.moveForward()

        #  dirt on back-right 
        elif isinstance(percepts[2][2], Dirt):
            if debug: print("dirt on back-right")
            self.turnRight()

        #  dirt on back-left 
        elif isinstance(percepts[2][0], Dirt):
            if debug: print("dirt on back-left")
            self.turnLeft()

        #  wall / obstacle in front 
        elif (isinstance(percepts[0][1], Wall)
                or isinstance(percepts[0][1], Obstacle)
                or isinstance(percepts[0][1], Roomba)
                or isinstance(percepts[0][1], Dog)
                or isinstance(percepts[0][1], Dropoff)):
        #{
            if debug: print("wall / obstacle in front")
            if (not isinstance(percepts[1][0], Wall)
                    and not isinstance(percepts[1][2], Wall)
                    and not isinstance(percepts[1][0], Obstacle)
                    and not isinstance(percepts[1][2], Obstacle)
                    and not isinstance(percepts[1][0], Dropoff)
                    and not isinstance(percepts[1][2], Dropoff)):
            #{
                turn = None
                if self.dir is Dir.RIGHT:
                    if self.state is State.BACK_FORTH_TOP:
                        turn = (self.turnLeft)
                    else:# self.state is State.BACK_FORTH_BOT:
                        turn = (self.turnRight)

                elif self.dir is Dir.LEFT:
                    if self.state is State.BACK_FORTH_TOP:
                        turn = (self.turnRight)
                    else:# self.state is State.BACK_FORTH_BOT:
                        turn = (self.turnLeft)

                elif self.dir is Dir.UP:
                    if self.state is State.BACK_FORTH_RIGHT:
                        turn = (self.turnRight)
                    else:# self.state is State.BACK_FORTH_LEFT:
                        turn = (self.turnLeft)

                elif self.dir is Dir.DOWN:
                    if self.state is State.BACK_FORTH_RIGHT:
                        turn = (self.turnLeft)
                    else:# self.state is State.BACK_FORTH_LEFT:
                        turn = (self.turnRight)

                turn()
                self.nextMoves.append((self.moveForward))
                self.nextMoves.append((self.moveForward))
                self.nextMoves.append(turn)
            #}

            # in corner
            else:
                if debug: print("  corner")
                # top left
                if self.loc[X] == 1 and self.loc[Y] == 1:
                    if debug: print("    top left")
                    self.turnLeft()
                    self.nextMoves.clear()
                    self.nextMoves.append((self.moveForward))

                    if self.state is State.BACK_FORTH_LEFT:
                        self.state = State.BACK_FORTH_BOT
                    if self.state is State.BACK_FORTH_TOP:
                        self.state = State.BACK_FORTH_RIGHT

                # top right
                elif self.loc[Y] == 1:
                    if debug: print("    top right")
                    self.turnLeft()
                    self.nextMoves.clear()
                    self.nextMoves.append((self.moveForward))

                    if self.state is State.BACK_FORTH_RIGHT:
                        self.state = State.BACK_FORTH_BOT
                    if self.state is State.BACK_FORTH_TOP:
                        self.state = State.BACK_FORTH_LEFT

                # bottom left
                elif self.loc[X] == 1:
                    if debug: print("    bottom left")
                    self.turnRight()
                    self.nextMoves.clear()
                    self.nextMoves.append((self.moveForward))

                    if self.state is State.BACK_FORTH_LEFT:
                        self.state = State.BACK_FORTH_TOP
                    if self.state is State.BACK_FORTH_BOT:
                        self.state = State.BACK_FORTH_RIGHT

                # bottom right
                else:
                    if debug: print("    bottom right")
                    self.turnLeft()
                    self.nextMoves.clear()
                    self.nextMoves.append((self.moveForward))

                    if self.state is State.BACK_FORTH_BOT:
                        self.state = State.BACK_FORTH_LEFT
                    if self.state is State.BACK_FORTH_RIGHT:
                        self.state = State.BACK_FORTH_TOP

        #}

        # pre-chosen moves
        elif len(self.nextMoves) > 0:
            if debug: print("next move")
            self.nextMoves.pop(0)()

        elif self.state is State.BACK_FORTH_TOP and self.dir not in { Dir.RIGHT, Dir.LEFT }:
            self.turnRight()
        elif self.state is State.BACK_FORTH_BOT and self.dir not in { Dir.RIGHT, Dir.LEFT }:
            self.turnLeft()
        elif self.state is State.BACK_FORTH_RIGHT and self.dir not in { Dir.UP, Dir.DOWN }:
            self.turnRight()
        elif self.state is State.BACK_FORTH_LEFT and self.dir not in { Dir.UP, Dir.DOWN }:
            self.turnLeft()

        # move forward
        else:
            if debug: print("move forward")
            self.moveForward()

        return self.loc


    def draw(self):
        x, y = (n * r.cellsize for n in self.loc)

        outside = pygame.Rect(x, y, r.cellsize, r.cellsize)
        pygame.draw.rect(r.screen, self.outerColor, outside)

        inside = pygame.Rect(x + 4, y + 4, r.cellsize - 8, r.cellsize - 8)
        pygame.draw.rect(r.screen, self.innerColor, inside)

        if self.dir == Dir.RIGHT:
            front = pygame.Rect(x + 8, y, r.cellsize - 8, r.cellsize)
        elif self.dir == Dir.LEFT:
            front = pygame.Rect(x, y, r.cellsize - 8, r.cellsize)
        elif self.dir == Dir.UP:
            front = pygame.Rect(x, y, r.cellsize, r.cellsize - 8)
        elif self.dir == Dir.DOWN:
            front = pygame.Rect(x, y + 8, r.cellsize, r.cellsize - 8)

        pygame.draw.rect(r.screen, c.FOREST_GREEN, front)
