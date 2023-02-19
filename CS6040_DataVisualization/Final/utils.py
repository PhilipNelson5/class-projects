import numpy.typing as npt
import numpy as np
import numba as nb


@nb.njit(cache=True)
def norm(x: npt.NDArray) -> npt.NDArray:
    return x / np.sqrt((x**2).sum())


@nb.njit(cache=True)
def normalize_array(verts: npt.NDArray) -> npt.NDArray:
    normed = np.empty(verts.shape)
    _max = np.max(verts[:3, :3])
    _min = np.min(verts[:3, :3])
    for i in range(len(verts)):
        normed[i] = (verts[i] - _min) * 2 / (_max - _min) - 1
        normed[i, 3] = 1

    return normed


@nb.njit(cache=True)
def calculate_normals(verts: npt.NDArray, faces: npt.NDArray) -> npt.NDArray:
    norms = np.empty((len(faces), 3))
    for i, face in enumerate(faces):
        v0 = verts[face[0]][:3]
        v1 = verts[face[1]][:3]
        v2 = verts[face[2]][:3]

        direction = np.cross(v1 - v0, v2 - v0)
        norms[i] = norm(direction)

    return norms
