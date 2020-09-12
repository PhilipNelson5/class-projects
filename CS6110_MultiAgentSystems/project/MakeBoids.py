import random
import math
import numpy as np
from typing import List, Tuple
from Math import dist
from operator import itemgetter

import Colors as c
import Boid
import Influencer
import settings

X = 0
Y = 1

def rand_dir() -> Tuple[float, float]:
    x: float = random.uniform(-1, 1)
    y: float = random.uniform(-1, 1)
    mag: float = math.sqrt(x*x + y*y)
    return (x/mag, y/mag)


def random_circle(numPoints: int):
    for i in range(numPoints):
        a = random.random() * 2 * math.pi
        r = math.sqrt(random.random())

        x = r * math.cos(a)
        y = r * math.sin(a)

        yield (x, y)


def uniform_circle(numPoints: int):
    for i in range(numPoints):
        dst = math.pow(i / (numPoints - 1), 0.5)
        angle = 2 * math.pi * 1.618 * i

        x = dst * math.sin(angle)
        y = dst * math.cos(angle)

        yield (x, y)


def flock() -> List[Boid.Boid]:
    scale = 100
    boids: List[Boid.Boid] = []
    cs = np.linspace(100, 255, settings.num_boids)
    i = 0
    for (x, y) in uniform_circle(settings.num_boids):
      x = (x * scale + 700)
      y = (y * scale + 200)
      boids.append(Boid.Boid((x,y), (1,0), (0, 0, cs[i]), False))
      i += 1
    return boids


def random_placement() -> List[Influencer.Influencer]:
    boids = []
    vel = (1,0)
    scale = 100
    influencers: List[Influencer.Influencer] = []
    for (x, y) in random_circle(settings.num_influencers):
      x = (x * scale + 700)
      y = (y * scale + 200)
      boids.append(Influencer.Influencer((x,y), vel, c.GREEN, False))
    return boids


def grid() -> List[Influencer.Influencer]:
    vel = (1,0)
    n = math.ceil(settings.num_influencers/4)
    influencers: List[Influencer.Influencer] = []
    for x in np.linspace(450, 850, n):
        for y in np.linspace(100, 400, n):
            influencers.append(Influencer.Influencer((x,y), vel, c.GREEN, False))
    return influencers


def border() -> List[Influencer.Influencer]:
    vel = (1,0)
    n = math.ceil(settings.num_influencers/4)
    influencers: List[Influencer.Influencer] = []
    for x in np.linspace(450, 850, n):
        influencers.append(Influencer.Influencer((x,100), vel, c.GREEN, False))
    del(influencers[-1])
    for y in np.linspace(100, 400, n):
        influencers.append(Influencer.Influencer((850,y), vel, c.GREEN, False))
    del(influencers[-1])
    for x in np.linspace(850, 450, n):
        influencers.append(Influencer.Influencer((x,400), vel, c.GREEN, False))
    del(influencers[-1])
    for y in np.linspace(400, 100, n):
        influencers.append(Influencer.Influencer((450,y), vel, c.GREEN, False))
    del(influencers[-1])

    return influencers


def graph(boids: List[Boid.Boid]) -> List[Influencer.Influencer]:
    pos = 0
    con = 1

    vel = (1,0)
    g = []
    influencers = []
    locations = []

    # make graph
    for i in range(len(boids)-2):
        for j in range(i+1, len(boids)-1):
            d = dist(boids[i].pos, boids[j].pos)
            if d > Boid.vision: continue
            g.append((d, boids[i].pos, boids[j].pos))

    # determine all possible placements
    for d, p1, p2 in g:
        x = (p1[X] + p2[X]) / 2
        y = (p1[Y] + p2[Y]) / 2
        locations.append(((x, y), 0))
    
    # calculate direct connecions
    for i in range(len(locations)-1):
        for b in boids:
            d = dist(locations[i][pos], b.pos)
            if d <= Boid.vision:
                locations[i] = (locations[i][pos], locations[i][con]+1)

    # sort possible positions by number of direct direct connections
    locations = sorted(locations, key=itemgetter(con), reverse=True)

    for i in range(settings.num_influencers):
        influencers.append(Influencer.Influencer(locations[i][pos], vel, c.GREEN, False))
    
    return influencers
        
        