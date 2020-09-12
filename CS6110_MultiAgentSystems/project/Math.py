import math
from typing import Tuple, List

Color = Tuple[int, int, int]
Point = Tuple[int, int]
PointF = Tuple[int, int]
Vec2dF = Tuple[float, float]

X: int = 0
Y: int = 1


def angle_between(v1: Vec2dF, v2: Vec2dF) -> float:
    m1 = mag(v1)
    m2 = mag(v2)

    try: 
        return math.acos(
            (v1[X] * v2[X] + v1[Y] * v2[Y])
            /
            (m1 * m2)
        )
    except:
        return 0.0


def rotate_vec(v: Vec2dF, th: float) -> Vec2dF:
    sin_th = math.sin(th)
    cos_th = math.cos(th)
    return(
        v[X] * cos_th - v[Y] * sin_th,
        v[X] * sin_th + v[Y] * cos_th
    )


def cross_prod(v1: Vec2dF, v2: Vec2dF) -> float:
    return v1[X] * v2[Y] - v1[Y] * v2[X] 


def dist(p1: Point, p2: Point) -> float:
    x = p1[X] - p2[X]
    y = p1[Y] - p2[Y]
    return math.sqrt(x * x + y * y)


def mag(v: Vec2dF) -> float:
    return math.sqrt(v[X] * v[X] + v[Y] * v[Y])


def norm(v: Vec2dF) -> Vec2dF:
    m = mag(v)
    return (v[X] / m, v[Y] / m)


def set_mag(v: Vec2dF, m: float) -> Vec2dF:
    v = norm(v)
    return (v[X] * m, v[Y] * m)


def sub(v1: Vec2dF, v2: Vec2dF) -> Vec2dF:
    return (v1[X] - v2[X], v1[Y] - v2[Y])


def add(v1: Vec2dF, v2: Vec2dF) -> Vec2dF:
    return (v1[X] + v2[X], v1[Y] + v2[Y])


def mul(v: Vec2dF, s: float) -> Vec2dF:
    return (v[X] * s, v[Y] * s)


def div(v: Vec2dF, s: float) -> Vec2dF:
    return (v[X] / s, v[Y] / s)


def limit(v: Vec2dF, m: float) -> Vec2dF:
    if mag(v) < m: return v

    return mul(norm(v), m)
