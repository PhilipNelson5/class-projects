import random
from typing import Tuple, List
import pygame
from pygame.locals import *

from Object import Object
from Direction import Dir
import rendering as r

Color = Tuple[int, int, int]
Point = Tuple[int, int]

X: int = 0
Y: int = 1
R: int = 0
G: int = 1
B: int = 2


def clamp(v: int, MIN: int, MAX: int):
    return MIN if v < MIN else MAX if v > MAX else v


class Dirt(Object):
    def __init__(self, outerColor: Color, innerColor: Color, loc: Point):
        super().__init__(outerColor, innerColor, loc)


    def clean(self) -> bool:
        self.level -= 1
        return self.level == 0


    def dirty(self) -> None:
        if self.level < 5:
            self.level += 1


    @staticmethod
    def generate(loc: Point, cellwidth: int, cellheight: int) -> List[Point]:
        q: List[Point] = [ loc ]
        ret: list[Point] = []
        spread: float = .75
        while len(q) > 0:
            p = q.pop(0)
            if (p[X] > 0
                    and p[X] < cellwidth
                    and p[Y] > 0
                    and p[Y] < cellheight):
                ret.append(p)

                if random.random() > spread:
                    q.append((p[X] + 1, p[Y]))
                if random.random() > spread:
                    q.append((p[X] - 1, p[Y]))
                if random.random() > spread:
                    q.append((p[X], p[Y] + 1))
                if random.random() > spread:
                    q.append((p[X], p[Y] - 1))


        return ret

    
    def draw(self):
        factor = 20
        x, y = (n * r.cellsize for n in self.loc)

        color = (clamp(self.outerColor[R]-self.level*factor, 0, 255),
                 clamp(self.outerColor[G]-self.level*factor, 0, 255),
                 clamp(self.outerColor[B]-self.level*factor, 0, 255))

        outside = pygame.Rect(x, y, r.cellsize, r.cellsize)
        pygame.draw.rect(r.screen, color, outside)

        inside = pygame.Rect(x + 4, y + 4, r.cellsize - 8, r.cellsize - 8)
        pygame.draw.rect(r.screen, color, inside)
