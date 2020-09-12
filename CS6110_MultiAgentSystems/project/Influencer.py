from __future__ import annotations
from pygame.locals import *
from typing import Tuple, List
from Boid import Boid
import Colors as c
import math
import pygame
import rendering as r
import settings
from Math import angle_between, rotate_vec, cross_prod

Color = Tuple[int, int, int]
Point = Tuple[int, int]
PointF = Tuple[int, int]
Vec2dF = Tuple[float, float]

X: int = 0
Y: int = 1

class Influencer(Boid):
    def __init__(self, pos: PointF, vel: Vec2dF, color: Color, debug: bool):
        Boid.__init__(self, pos, vel, color, debug)
    
    
    def update(self, dt: float, boids: List[Boid]) -> None:
        Boid.update_position(self, dt)
        self.vel = rotate_vec(self.vel, math.radians(dt))


