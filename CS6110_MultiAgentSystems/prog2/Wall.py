from typing import Tuple, List

from Object import Object

Color = Tuple[int, int, int]
Point = Tuple[int, int]

class Wall(Object):
    def __init__(self, outerColor: Color, innerColor: Color, loc: Point):
        super().__init__(outerColor, innerColor, loc)


