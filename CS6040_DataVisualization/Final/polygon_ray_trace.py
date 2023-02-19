from typing import Generator, Tuple
import numba as nb
import numpy.typing as npt
import numpy as np
from tqdm import tqdm
from camera import Camera

from image import Image
from ray import Ray
from ray_triangle_intersection import ray_triangle_intersection
from make_primary_rays import make_primary_rays
from utils import norm


@nb.njit(cache=True, nogil=True)
def test_polygons(ray: Ray, verts: npt.NDArray, faces: npt.NDArray, norms: npt.NDArray):
    hit = False
    ray_int = np.array([np.inf, np.inf, np.inf])
    N = np.empty(3)
    for i in range(len(faces)):
        v0 = verts[faces[i][0]][:3]
        v1 = verts[faces[i][1]][:3]
        v2 = verts[faces[i][2]][:3]

        result = ray_triangle_intersection(ray.direction, ray.origin, v0, v1, v2)
        if result[0] and np.linalg.norm(result[1]) < np.linalg.norm(ray_int):
            hit = True
            ray_int = result[1]
            N = norms[i]  # [:3]

    return (hit, ray_int, N)


@nb.njit(cache=True, nogil=True)
def cast_ray(
    ray: Ray,
    verts: npt.NDArray,
    faces: npt.NDArray,
    norms: npt.NDArray,
    lights: npt.NDArray,
    viewer: npt.NDArray,
) -> npt.NDArray:
    hit, ray_int, N = test_polygons(ray, verts, faces, norms)
    # https://en.wikipedia.org/wiki/Phong_reflection_model
    if hit:
        color = np.array([0.0, 0.0, 0.0])

        k_s = 0.15
        k_d = 0.3
        k_a = 1 - k_d - k_s

        # ambient
        i_a = np.array([1.0, 1.0, 1.0])
        color_a = k_a * i_a

        color = color + color_a

        for i in range(len(lights)):
            # diffuse
            light_location = lights[i][0]
            i_d = lights[i][1]
            L = norm(light_location - ray_int)
            color_d = k_d * np.dot(L, N) * i_d

            # specular
            i_s = lights[i][1]
            V = norm(viewer - ray_int)
            R = 2.0 * np.dot(N, L) * N - L
            color_s = k_s * np.power(np.dot(R, V), 5) * i_s

            color = color + color_d / len(lights) + color_s / len(lights)

        color = color.clip(0.0, 1.0)
        return (255 * color).astype(np.uint8)

    return np.array([0, 0, 0], dtype=np.uint8)
    # return np.array([100, 149, 237], dtype=np.uint8)


# @nb.njit()
def polygon_ray_trace(
    scene: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    lights,
    camera: Camera,
    image: Image,
) -> Image:
    """_summary_

    Args:
        scene (Tuple[np.array, np.array]): Tuple of faces and vertices

    Returns:
        Image: rendered image
    """
    verts, faces, norms = scene
    viewer = (camera.transformation_matrix @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    # for (row, col, ray) in make_primary_rays(
    #     camera, image.width, image.height, image.aspect_ratio
    # ):
    for (row, col, ray) in tqdm(
        make_primary_rays(camera, image.width, image.height, image.aspect_ratio),
        total=image.width * image.height,
        leave=False,
    ):
        image[row, col] = cast_ray(ray, verts, faces, norms, lights, viewer)

    return image


if __name__ == "__main__":
    from utils import calculate_normals, normalize_array
    import data_readers.ply as ply_reader
    from transformations import *

    # verts, faces = ply_reader.read("./data/bun_zipper_res4.ply")
    verts, faces = ply_reader.read("./data/bunny/reconstruction/bun_zipper_res2.ply")
    # verts, faces = ply_reader.read("./data/dodecahedron.ply")
    # verts, faces = ply_reader.read("./data/cube.ply")

    verts = normalize_array(verts)
    mat = translate(-2.0, 1.75, -3.0) @ rot_x(215) @ rot_y(195)
    verts = np.array(list(map(lambda vert: mat @ vert, verts)), dtype=verts.dtype)

    norms = calculate_normals(verts, faces)
    lights = np.array([[0, 0, 0]])

    # image = Image(width=192, height=108)
    # image = Image(width=500, height=350)
    image = Image(width=1920, height=1080)
    camera = Camera(
        fov=np.double(50),
        aspect_ratio=image.aspect_ratio,
        # move the camera around, affects where the rays are generated from
        transformation_matrix=translate(0.0, 0.0, -1.0),
    )

    polygon_ray_trace((verts, faces, norms), lights, camera, image)

    image.write_ppm("polygon.ppm")
