from typing import Generator, Tuple
import numba as nb
import numpy as np

from camera import Camera
from ray import Ray
from utils import norm

# https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays
@nb.njit()
def make_primary_rays(
    camera: Camera, image_width, image_height, image_aspect_ratio
) -> Generator[Tuple[int, int, Ray], None, None]:
    """generate primary rays

    Args:
        camera (Camera): the camera
        image_width (number): image width
        image_height (number): image height
        image_aspect_ratio (number): aspect ratio

    Yields:
        Generator[Tuple[int, int, Ray], None, None]: primary rays
    """
    scale = np.tan(np.deg2rad(camera.fov) * 0.5)
    for row in range(image_height):
        for col in range(image_width):
            # normalized device coordinates
            pixNDCx = (col + 0.5) / image_width
            pixNDCy = (row + 0.5) / image_height

            # screen coordinates
            pixScreenX = 2 * pixNDCx - 1
            pixScreenY = 1 - 2 * pixNDCy

            # camera coordinates
            pixCameraX = (2 * pixScreenX - 1) * image_aspect_ratio * scale
            pixCameraY = 1 - 2 * pixScreenY * scale

            # world coordinates
            pixel = camera.transformation_matrix @ np.array(
                [pixCameraX, pixCameraY, -1, 1], dtype=np.double
            )
            origin = camera.transformation_matrix @ np.array(
                [0, 0, 0, 1], dtype=np.double
            )
            ray = Ray(origin[:3], norm(pixel[:3] - origin[:3]))
            yield (row, col, ray)
