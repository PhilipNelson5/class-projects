# Boid Simulation
# By Philip Nelson

from pygame.locals import *
from tqdm import tqdm
from typing import List, Tuple, Optional
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pygame
import random
import sys
import time

from Math import dist, add, div
import MakeBoids
import Boid
import Colors as c
import Draw
import Influencer
import rendering as r
import settings

Color = Tuple[int, int, int]
Key = int
Point = Tuple[int, int]

X = 0
Y = 1

BGCOLOR: Color = c.ALICE_BLUE
BASICFONT: pygame.font.Font = None

fps: int = 0
com: Point = (0, 0)
boids: List[Boid.Boid] = []
densities = {}


def density() -> float:
    global boids
    d = 0
    for i in range(settings.num_boids):
        for j in range(i, settings.num_boids):
            d += dist(boids[i].pos, boids[j].pos)

    d /= settings.num_boids ** 2
    return d


def density_2() -> float:
    global boids
    com = (0,0)
    for i in range(settings.num_boids):
        com = add(com, boids[i].pos)

    com = div(com, settings.num_boids)

    dis = 0
    for i in range(settings.num_boids):
        d = dist(com, boids[i].pos)
        dis += d
    
    d /= settings.num_boids
    return d


def main() -> None:
    global FPSCLOCK, BASICFONT

    if settings.render:
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        r.screen = pygame.display.set_mode((settings.width, settings.height))
        BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        pygame.display.set_caption('Boid Simulation')

    simulate()


def generate():
    global boids

    boids = MakeBoids.flock()
    boids[0].debug = True

    if settings.influencer_method == settings.InfluencerMethod.RANDOM:
        boids.extend(MakeBoids.random_placement())
    if settings.influencer_method == settings.InfluencerMethod.GRID:
        boids.extend(MakeBoids.grid())
    if settings.influencer_method == settings.InfluencerMethod.BORDER:
        boids.extend(MakeBoids.border())
    if settings.influencer_method == settings.InfluencerMethod.GRAPH:
        boids.extend(MakeBoids.graph(boids))


def simulate() -> None:
    global boids, densities


    if settings.render:
        generate()
        last_time = pygame.time.get_ticks() + 16
        while True:
            cur_time = pygame.time.get_ticks()
            dt = cur_time - last_time
            last_time = cur_time

            render()
            handle_input()
            update(dt)

            # refresh screen
            pygame.display.update()
            FPSCLOCK.tick(settings.fps)
    else:
        dt = 16
        for metric in [1,2]:
            settings.desnity_metric = metric
            for n in [int(x) for x in np.linspace(5, 25, 5)]:
                print(f"{n} influencers")
                for data in settings.InfluencerMethod:
                    settings.num_influencers = n
                    generate()
                    densities[n] = []
                    random.seed(0)
                    for _ in tqdm(range(settings.tests)):
                        densities[n].append([])
                        settings.influencer_method = data
                        for _ in range(settings.iterations):
                            if settings.desnity_metric == 1:
                                densities[n][-1].append(density())
                            else:
                                densities[n][-1].append(density_2())
                            update(dt)
                    plot(n)
                plt.clf()


def render() -> None:
    global fps, com

    Draw.fill(BGCOLOR)
    for boid in boids:
        boid.draw()

    Draw.text(f'FPS {fps}', BASICFONT, (1,0))
    Draw.text(f'speed (s/a) {Boid.speed}', BASICFONT, (1,15))
    Draw.text(f'vision (v/c) {Boid.vision}', BASICFONT, (1,30))
    Draw.text(f'collision dist (x/z) {Boid.collision_dist}', BASICFONT, (1,45))
    Draw.text(f'collision avoid (9/7) {Boid.collision_avoid_strength}', BASICFONT, (1,60))
    Draw.text(f'vel align (6/4) {Boid.vel_align_stren}', BASICFONT, (1,75))
    Draw.text(f'center of mass (3/1) {Boid.center_of_mass_align}', BASICFONT, (1,90))
    Draw.circle((int(com[X]), int(com[Y])), 3)


def update(dt: float) -> None:
    global fps, com
    fps = int(1e3/dt)
    dt = dt / 3e1

    # Calculate the center of mass of the flock
    for boid in boids:
        com = (com[X] + boid.pos[X], com[Y] + boid.pos[Y])
    com = (com[X]/len(boids), com[Y]/len(boids))

    # upate boids
    for boid in boids:
        boid.update(dt, boids)


def handle_input():
    for event in pygame.event.get():
        if event.type == QUIT:
            terminate()
        if event.type == KEYDOWN:
            key: Key = event.key
            if key == K_ESCAPE:
                terminate()
            if key == K_s:
                Boid.speed += 1
            if key == K_a:
                Boid.speed -= 1
            if key == K_v:
                Boid.vision += 10
            if key == K_c:
                Boid.vision -= 10
            if key == K_x:
                Boid.collision_dist += 5
            if key == K_z:
                if Boid.collision_dist > 5: Boid.collision_dist -= 5
            if key == K_KP9:
                Boid.collision_avoid_strength += 10
            if key == K_KP7:
                if Boid.collision_avoid_strength > 10: Boid.collision_avoid_strength -= 10
            if key == K_KP6:
                Boid.vel_align_stren += 5
            if key == K_KP4:
                if Boid.vel_align_stren > 5: Boid.vel_align_stren -= 5
            if key == K_KP3:
                Boid.center_of_mass_align += 10
            if key == K_KP1:
                if Boid.center_of_mass_align > 10: Boid.center_of_mass_align -= 10


def plot(n: int):
    arr = np.array(densities[n])
    d = arr.mean(axis=0)
    plt.plot(d, label=settings.influencer_method.name)
    plt.title(f"{settings.num_influencers} influencing agents")
    if settings.desnity_metric == 1:
        plt.ylabel("Average distance between flock memebers")
    else:
        plt.ylabel("Average distance from center of mass")
    plt.xlabel("Iterations of Simulation (16 ms)")
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # plt.show()
    now = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    plt.savefig(f"./images/{now}_{settings.influencer_method.name}({settings.num_influencers}).png")


def terminate() -> None:
    global densities
    if settings.render:
        pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
