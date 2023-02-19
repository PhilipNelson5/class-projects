import numpy.typing as npt
import numpy as np
import numba as nb
from numba.experimental import jitclass

spec = [
    ("fov", nb.float64),
    ("aspect_ratio", nb.float64),
    ("transformation_matrix", nb.float64[:, :]),
]


@jitclass(spec)
class Camera:
    def __init__(
        self,
        fov: np.number,
        aspect_ratio: np.number,
        transformation_matrix: npt.NDArray,
    ):
        """Camera model

        Args:
            fov (np.number): camera field of view (degrees)
            aspect_ratio (np.number): camera ratio of width to height
            transformation_matrix (npt.NDArray): transformation matrix used to move camera in scene
        """
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.transformation_matrix = transformation_matrix
