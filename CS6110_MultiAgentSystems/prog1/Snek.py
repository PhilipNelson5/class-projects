import pygame
from pygame.locals import *
from typing import Tuple, List

from random import randint
from Direction import Dir

Color = Tuple[int, int, int]
Point = Tuple[int, int]
Key = int

HEAD = 0
X = 0
Y = 1

class Snek:
    def __init__(self, w: int, h: int, outerColor: Color, innerColor: Color, rightKey: Key, leftKey: Key, upKey: Key, downKey: Key, shootKey: Key):
        self.dir = Dir.RIGHT
        x: int = randint(0, w - 6)
        y: int = randint(0, h - 6)
        self.locs = [ (x - n, y) for n in range(3) ]
            # (x,     y),
            # (x - 1, y),
            # (x - 2, y)
        # ]
        self.outerColor = outerColor
        self.innerColor = innerColor

        self.rightKey = rightKey
        self.leftKey = leftKey
        self.upKey = upKey
        self.downKey = downKey

        self.shootKey = shootKey

        self.fed = 0
        self.turned = False


    def feed(self) -> None:
        self.fed += 1


    def checkTurn(self, key: Key) -> None:
        if self.turned: return
        if key in (self.rightKey, K_KP6) and self.dir != Dir.LEFT:
            self.dir = Dir.RIGHT
            self.turned = True
        elif key in (self.leftKey, K_KP4) and self.dir != Dir.RIGHT:
            self.dir = Dir.LEFT
            self.turned = True
        elif key in (self.upKey, K_KP8) and self.dir != Dir.DOWN:
            self.dir = Dir.UP
            self.turned = True
        elif key in (self.downKey, K_KP2) and self.dir != Dir.UP:
            self.dir = Dir.DOWN
            self.turned = True


    def move(self) -> None:
        if self.dir == Dir.UP:
            newHead = (self.locs[HEAD][X],      self.locs[HEAD][Y] - 1)
        elif self.dir == Dir.DOWN:
            newHead = (self.locs[HEAD][X],      self.locs[HEAD][Y] + 1)
        elif self.dir == Dir.LEFT:
            newHead = (self.locs[HEAD][X] - 1,  self.locs[HEAD][Y]    )
        elif self.dir == Dir.RIGHT:
            newHead = (self.locs[HEAD][X] + 1,  self.locs[HEAD][Y]    )

        if self.fed > 0:
            self.fed -= 1
        else:
            del self.locs[-1]

        self.locs.insert(0, newHead)

        self.turned = False


    def split(self, loc: Point) -> List[Point]:
        try:
            hit: int = self.locs.index(loc)
        except:
            return []
        tail: List[Point] = self.locs[hit:]
        self.locs = self.locs[:hit]
        return tail
