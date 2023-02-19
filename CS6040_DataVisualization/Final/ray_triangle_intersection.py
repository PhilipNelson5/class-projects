from typing import Tuple
from ray import Ray
import numpy as np
import numpy.typing as npt
import numba as nb

Triangle = Tuple[npt.NDArray, npt.NDArray, npt.NDArray]

# https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
@nb.njit(cache=True, nogil=True)
def ray_triangle_intersection(
    ray_d: npt.NDArray,
    ray_o: npt.NDArray,
    vert0: npt.NDArray,
    vert1: npt.NDArray,
    vert2: npt.NDArray,
    epsilon=0.000001,
) -> Tuple[bool, npt.NDArray]:
    t = u = v = 0.0
    edge1 = vert1 - vert0
    edge2 = vert2 - vert0

    pvec = np.cross(ray_d, edge2)

    det = np.dot(edge1, pvec)

    if det < epsilon:
        return (False, np.array([t, u, v]))

    tvec = ray_o - vert0

    u = np.dot(tvec, pvec)
    if u < 0.0 or u > det:
        return (False, np.array([t, u, v]))

    qvec = np.cross(tvec, edge1)

    v = np.dot(ray_d, qvec)
    if v < 0.0 or u + v > det:
        return (False, np.array([t, u, v]))

    t = np.dot(edge2, qvec)

    inv_det = 1.0 / det
    t = t * inv_det
    u = u * inv_det
    v = v * inv_det

    return (True, np.array([t, u, v]))


@nb.njit(cache=True)
def ray_triangle_intersection_class(
    ray: Ray, triangle: Triangle, epsilon=0.000001, cull=False
) -> Tuple[bool, float, float, float]:
    t = u = v = 0.0
    vert0, vert1, vert2 = triangle
    edge1 = np.subtract(vert1, vert0)
    edge2 = np.subtract(vert2, vert0)

    pvec = np.cross(ray.direction, edge2)

    det = np.dot(edge1, pvec)

    if cull:
        if det < epsilon:
            return (False, t, u, v)

        tvec = np.subtract(ray.origin, vert0)

        u = np.dot(tvec, pvec)
        if u < 0.0 or u > det:
            return (False, t, u, v)

        qvec = np.cross(tvec, edge1)

        v = np.dot(ray.direction, qvec)
        if v < 0.0 or u + v > det:
            return (False, t, u, v)

        t = np.dot(edge2, qvec)

        inv_det = 1.0 / det
        t = t * inv_det
        u = u * inv_det
        v = v * inv_det

        return (True, t, u, v)

    else:
        if det > -epsilon and det < epsilon:
            return (False, t, u, v)

        inv_det = 1.0 / det

        tvec = np.subtract(ray.origin, vert0)

        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return (False, t, u, v)

        qvec = np.cross(tvec, edge1)

        v = np.dot(ray.direction, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return (False, t, u, v)

        t = np.dot(edge1, qvec) * inv_det

        return (True, t, u, v)
