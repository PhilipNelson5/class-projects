from typing import Tuple, List
import pygame
from pygame.locals import *
import rendering as r

Point = Tuple[int, int]

class Object:
    def __init__(self, outerColor: Color, innerColor: Color, loc: Point = (0,0)):
        self.loc: Point = loc
        self.outerColor: Color = outerColor
        self.innerColor: Color = innerColor
        self.level = 1


    def draw(self):
        x, y = (n * r.cellsize for n in self.loc)

        outside = pygame.Rect(x, y, r.cellsize, r.cellsize)
        pygame.draw.rect(r.screen, self.outerColor, outside)

        inside = pygame.Rect(x + 4, y + 4, r.cellsize - 8, r.cellsize - 8)
        pygame.draw.rect(r.screen, self.innerColor, inside)


