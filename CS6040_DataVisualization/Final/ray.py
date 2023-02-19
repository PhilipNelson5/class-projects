import numpy.typing as npt
import numba as nb
from numba.experimental import jitclass

spec = [
    ("origin", nb.float64[:]),
    ("direction", nb.float64[:]),
]


@jitclass(spec)
class Ray:
    def __init__(self, origin: npt.NDArray, direction: npt.NDArray) -> None:
        self.origin = origin
        self.direction = direction
