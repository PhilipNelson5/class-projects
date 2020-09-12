from typing import Tuple

from Direction import Dir

Point = Tuple[int, int]

X = 0
Y = 1

class Laser:
    def __init__(self, dir: Dir, loc: Point):

        self.speed = 2
        self.dir = dir

        if dir == Dir.UP:
            self.locs = [(loc[X], loc[Y] - 1), (loc[X], loc[Y] - 2)]
        elif dir == Dir.DOWN:
            self.locs = [(loc[X], loc[Y] + 1), (loc[X], loc[Y] + 2)]
        elif dir == Dir.LEFT:
            self.locs = [(loc[X] - 1, loc[Y]), (loc[X] - 2, loc[Y])]
        else: #if dir == Dir.RIGH    T                     
            self.locs = [(loc[X] + 1, loc[Y]), (loc[X] + 2, loc[Y])]


    def move(self) -> None:
        if self.dir == Dir.UP:
            self.locs = [(x,y - self.speed) for x,y in self.locs]
        elif self.dir == Dir.DOWN:
            self.locs = [(x,y + self.speed) for x,y in self.locs]
        elif self.dir == Dir.LEFT:
            self.locs = [(x - self.speed,y) for x,y in self.locs]
        else: #if dir == Dir.RIGHT                     
            self.locs = [(x + self.speed,y) for x,y in self.locs]
        
