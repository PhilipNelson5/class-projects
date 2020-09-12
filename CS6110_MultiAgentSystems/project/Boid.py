from __future__ import annotations
from pygame.locals import *
from typing import Tuple, List
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

speed = 5
vision = 150
collision_dist = 25
collision_avoid_strength = 100
vel_align_stren = 10
center_of_mass_align = 50
field_of_view = math.radians(150)


class Boid:
    def __init__(self, pos: PointF, vel: Vec2dF, color: Color, debug: bool):
        self.pos: PointF = pos
        self.vel: Vec2dF = vel
        self.color: Color = color
        self.debug = debug
        self.seen = []
        self.com = (0, 0)
        self.id = id


    def update_velocity(self, dt: float, boids: List[Boid]) -> None:
        # Collision detection
        seen = []
        if self.debug: self.seen = []
        for boid in boids:
            if boid == self: continue

            x = self.pos[X] - boid.pos[X]
            y = self.pos[Y] - boid.pos[Y]
            d = math.sqrt(x * x + y * y)
            if d > vision: continue

            boid_v = (boid.pos[X] - self.pos[X], boid.pos[Y] - self.pos[Y])
            th = angle_between(boid_v, self.vel)
            if th < field_of_view:
                seen.append((boid, boid_v, th, d))
                if self.debug: self.seen.append(boid.pos)

        # Collision avoidance
        for boid, boid_v, th, d in seen:
            if d > collision_dist: continue
            # determine rotation direction
            if cross_prod(boid_v, self.vel) < 0: th = -th
            self.vel = rotate_vec(self.vel, th * dt / collision_avoid_strength)
        
        # Velocity alignment
        if len(seen) > 0:
            vel_avg = (0, 0)
            com = (0, 0)
            for boid, boid_v, th, d in seen:
                vel_avg = (vel_avg[X] + boid.vel[X], vel_avg[Y] + boid.vel[Y])
                com = (com[X] + boid.pos[X], com[Y] + boid.pos[Y])
            vel_avg = (vel_avg[X]/len(seen), vel_avg[Y]/len(seen))
            com = (com[X]/len(seen), com[Y]/len(seen))
            th = angle_between(vel_avg, self.vel)

            # determine rotation direction
            if cross_prod(vel_avg, self.vel) > 0: th = -th
            self.vel = rotate_vec(self.vel, th * dt / vel_align_stren)
            self.com = com

        # Center of mass alignment
        if len(seen) > 0:
            com_v = (com[X] - self.pos[X], com[Y] - self.pos[Y])
            th = angle_between(com_v, self.vel)
            
            # determine rotation direction
            if cross_prod(com_v, self.vel) > 0: th = -th
            self.vel = rotate_vec(self.vel, th * dt / center_of_mass_align)


    def update_position(self, dt: float) -> None:
        # move forward
        self.pos = (
            self.pos[X] + self.vel[X] * dt * speed,
            self.pos[Y] + self.vel[Y] * dt * speed
        )

        # wrap around the edges
        if self.pos[X] > settings.width: self.pos = (0, self.pos[Y])
        if self.pos[X] < 0: self.pos = (settings.width, self.pos[Y])
        if self.pos[Y] > settings.height: self.pos = (self.pos[X], 0)
        if self.pos[Y] < 0: self.pos = (self.pos[X], settings.height)


    def update(self, dt: float, boids: List[Boid]) -> None:
        self.update_velocity(dt, boids)
        self.update_position(dt)
    

    def draw(self) -> None:
        th = math.atan(self.vel[Y] / self.vel[X])
        if self.vel[X] < 0: th -= math.pi

        # Triangle at the origin
        verts: List[PointF] = [(10.0, 0.0), (-10.0, 5.0), (-10.0, -5.0)]

        # Rotate
        sin_th = math.sin(th)
        cos_th = math.cos(th)

        for i in range(len(verts)):
            x = verts[i][X] * cos_th - verts[i][Y] * sin_th
            y = verts[i][X] * sin_th + verts[i][Y] * cos_th
            verts[i] = (x, y)
        
        # Translate
        for i in range(len(verts)):
            verts[i] = (verts[i][X] + self.pos[X], verts[i][Y] + self.pos[Y])
        
        # Draw
        pygame.draw.polygon(r.screen, self.color, verts)

        if self.debug:
            pygame.draw.circle(r.screen, c.RED, (int(self.pos[X]), int(self.pos[Y])), vision, 1)
            pygame.draw.circle(r.screen, c.BLUE, (int(self.pos[X]), int(self.pos[Y])), collision_dist, 1)
            if len(self.seen) > 0:
                pygame.draw.circle(r.screen, c.RED, (int(self.com[X]), int(self.com[Y])), 7, 1)
                for boid in self.seen:
                    pygame.draw.line(
                        r.screen,
                        c.RED,
                        (int(self.pos[X]), int(self.pos[Y])),
                        (int(boid[X]), int(boid[Y]))
                    )

