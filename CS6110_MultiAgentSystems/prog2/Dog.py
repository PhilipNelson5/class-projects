import random
from typing import Tuple, List

from Object import Object

Color = Tuple[int, int, int]
Point = Tuple[int, int]

X: int = 0
Y: int = 1

debug: bool = False

class Dog(Object):
    def __init__(self, outerColor: Color, innerColor: Color, loc: Point):
        super().__init__(outerColor, innerColor, loc)
        self.dropChance: float = .75

    def move(self):
        oldLoc = self.loc

        x: int = random.choice([-1,0,0,0,0,0,0,1])
        y: int = random.choice([-1,0,0,0,0,0,0,1])

        self.loc = (self.loc[X] + x, self.loc[Y] + y)
        if debug: print(oldLoc, self.loc)
        return random.random() > self.dropChance, oldLoc



