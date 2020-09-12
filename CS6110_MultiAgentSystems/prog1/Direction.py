from enum import Enum, unique

@unique
class Dir(Enum):
    UP    = 'up'
    DOWN  = 'down'
    LEFT  = 'left'
    RIGHT = 'right'

