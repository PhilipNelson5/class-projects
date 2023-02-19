from __future__ import annotations
from abc import ABC, abstractmethod
from numpy import number
from typing import SupportsRound, Tuple, Callable
import math
import numpy as np
import numpy.typing as npt


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, key) -> number:
        pass


class UniformGrid(Dataset):
    """Abstraction of a uniform grid dataset

    The dataset is indexed via nearest neighbors
    """

    @staticmethod
    def load(filename: str, reader: Callable) -> UniformGrid:
        """Static method to read a dataset from a file and construct a UniformGrid dataset

        Args:
            filename (str): filename of the dataset
            reader (Callable): a function to read the dataset and return a numpy ndarray with the data

        Returns:
            UniformGrid: UniformGrid dataset
        """
        return UniformGrid(reader(filename))

    def __init__(self, data: npt.NDArray) -> None:
        self.data = data
        self.shape = data.shape

    def __getitem__(
        self, key: Tuple[SupportsRound, SupportsRound, SupportsRound]
    ) -> number:
        """get element via nearest neighbor

        Args:
            key (Tuple[Number, Number, Number]): (x,y,z) tuple

        Returns:
            Number: element at specified location
        """
        x, y, z = key
        x0 = round(x)
        y0 = round(y)
        z0 = round(z)
        if (
            x0 > self.data.shape[0] - 1
            or x0 < 0
            or y0 > self.data.shape[1] - 1
            or y0 < 0
            or z0 > self.data.shape[2] - 1
            or z0 < 0
        ):
            return np.nan
        return self.data[round(x), round(y), round(z)]


class UniformGridInterpolated(UniformGrid):
    def __init__(self, data: npt.NDArray) -> None:
        UniformGrid.__init__(self, data)

    def __getitem__(
        self, key: Tuple[SupportsRound, SupportsRound, SupportsRound]
    ) -> number:
        """get element via trilinear interpolation

        Args:
            key (Tuple[Number, Number, Number]): (x,y,z) tuple

        Returns:
            Number: element at specified location
        """
        x, y, z = key
        x0 = math.floor(x)
        y0 = math.floor(y)
        z0 = math.floor(z)
        x1 = x0 + 1  # math.ceil(x + np.spacing(x))
        y1 = y0 + 1  # math.ceil(y + np.spacing(y))
        z1 = z0 + 1  # math.ceil(z + np.spacing(z))
        # verify values are in data range
        if (
            x0 > self.data.shape[0] - 1
            or x0 < 0
            or x1 > self.data.shape[0] - 1
            or x1 < 0
            or y0 > self.data.shape[1] - 1
            or y0 < 0
            or y1 > self.data.shape[1] - 1
            or y1 < 0
            or z0 > self.data.shape[2] - 1
            or z0 < 0
            or z1 > self.data.shape[2] - 1
            or z1 < 0
        ):
            return np.nan
        xd = (x - x0) / (x1 - x0)
        yd = (y - y0) / (y1 - y0)
        zd = (z - z0) / (z1 - z0)
        c000 = self.data[x0, y0, z0]
        c001 = self.data[x0, y0, z1]
        c010 = self.data[x0, y1, z0]
        c011 = self.data[x0, y1, z1]
        c100 = self.data[x1, y0, z0]
        c101 = self.data[x1, y0, z1]
        c110 = self.data[x1, y1, z0]
        c111 = self.data[x1, y1, z1]
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c = c0 * (1 - zd) + c1 * zd
        return c

    @staticmethod
    def load(filename: str, reader: Callable) -> UniformGrid:
        """Static method to read a dataset from a file and construct a UniformGridInterpolated dataset

        Args:
            filename (str): filename of the dataset
            reader (Callable): a function to read the dataset and return a numpy ndarray with the data

        Returns:
            UniformGridInterpolated: UniformGridInterpolated dataset
        """
        return UniformGridInterpolated(reader(filename))


if __name__ == "__main__":
    import data_readers.vol as vol_reader
    import matplotlib.pyplot as plt

    print("testing uniform grid")
    # dataset = UniformGrid.load("data/C60Small.vol", vol_reader.read)
    dataset = UniformGrid.load("data/Skull.vol", vol_reader.read)
    print(dataset.shape)

    plt.figure()
    plt.imshow(dataset.data[30, :, :])

    print("testing uniform grid trilinear")
    dataset = UniformGridInterpolated.load("data/C60Small.vol", vol_reader.read)
    data = dataset.data[30, :, :]
    factor = 5
    up_scaled = np.zeros((data.shape[0] * factor, data.shape[1] * factor))
    for r, row in enumerate(np.linspace(0, dataset.shape[0], up_scaled.shape[0])):
        for c, col in enumerate(np.linspace(0, dataset.shape[1], up_scaled.shape[1])):
            try:
                up_scaled[r, c] = dataset[30, row, col]
            except:
                pass

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title("native")
    plt.imshow(dataset.data[30, :, :])

    plt.subplot(1, 2, 2)
    plt.title("5x upscale")
    plt.imshow(up_scaled)

    plt.show()
