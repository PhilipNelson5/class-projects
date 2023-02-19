import numpy.typing as npt
import numpy as np
from numpy import sin, cos


def rot_x(deg) -> npt.NDArray:
    """rotation matrix around x axis

    Args:
        deg (number): number of degrees to rotate

    Returns:
        npt.NDArray: rotation matrix
    """
    th = np.deg2rad(deg)
    # fmt: off
    mat = np.array([
        1, 0, 0, 0,
        0, cos(th), -sin(th), 0,
        0, sin(th), cos(th), 0,
        0, 0, 0, 1,
    ])
    # fmt: on
    mat.shape = (4, 4)
    return mat


def rot_y(deg) -> npt.NDArray:
    """rotation matrix around y axis

    Args:
        deg (number): number of degrees to rotate

    Returns:
        npt.NDArray: rotation matrix
    """
    th = np.deg2rad(deg)
    # fmt: off
    mat = np.array([
        cos(th), 0, sin(th), 0,
        0, 1, 0, 0,
        -sin(th), 0, cos(th), 0,
        0, 0, 0, 1,
    ])
    # fmt: on
    mat.shape = (4, 4)
    return mat


def rot_z(deg) -> npt.NDArray:
    """rotation matrix around z axis

    Args:
        deg (number): number of degrees to rotate

    Returns:
        npt.NDArray: rotation matrix
    """
    th = np.deg2rad(deg)
    # fmt: off
    mat = np.array([
        cos(th), -sin(th), 0, 0,
        sin(th), cos(th), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ])
    # fmt: on
    mat.shape = (4, 4)
    return mat


def translate(dx, dy, dz) -> npt.NDArray:
    """translation matrix

    Args:
        dx (number): x offset
        dy (number): y offset
        dz (number): z offset

    Returns:
        npt.NDArray: translation matrix
    """
    # fmt: off
    mat = np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1,
    ])
    # fmt: on
    mat.shape = (4, 4)
    return mat
