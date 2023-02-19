from typing import Tuple
import numpy.typing as npt
import numpy as np
import numba as nb
from numba.experimental import jitclass

Color = Tuple[int, int, int]

spec = [
    ("width", nb.int32),
    ("height", nb.int32),
    ("aspect_ratio", nb.int32),
    ("image", nb.uint8[:]),
]


# @jitclass(spec)
class Image:
    def __init__(self, width: int, height: int) -> None:
        """construct empty image with given width and height

        Args:
            width (int): image width in pixels
            height (int): image height in pixels
        """
        self.width: int = width
        self.height: int = height
        self.aspect_ratio = np.double(self.width / self.height)
        self.image: npt.NDArray = np.empty(shape=width * height * 3, dtype=np.uint8)

    def write_ppm(self, filename: str) -> None:
        """write the image to a binary ppm file

        Args:
            filename (str): image file name
        """
        if filename[-4:] != ".ppm":
            filename = filename + ".ppm"

        f = open(filename, "w")
        f.write("P6\n")
        f.write(f"{self.width} {self.height}\n")
        f.write("255\n")
        f.close()

        f = open(filename, "ab")
        f.write(self.image.tobytes())
        f.close()

    def write_ppm_ascii(self, filename: str) -> None:
        """write the image to an ascii ppm file

        This is mainly for debugging, ascii files are larger
        and slower to write

        Args:
            filename (str): image file name
        """
        if filename[-4:] != ".ppm":
            filename = filename + ".ppm"

        with open(filename, "w") as f:
            f.write("P3\n")
            f.write(f"{self.width} {self.height}\n")
            f.write("255\n")
            for i in range(0, len(self.image), 3):
                if i % self.width == 0:
                    f.write("\n")
                f.write(
                    f"{self.image[i]:<3} {self.image[i+1]:<3} {self.image[i+2]:<3} "
                )

    def fill(self, color: Color) -> None:
        """fill the image with a single color

        Args:
            color (Color): (r, g, b) tuple
        """
        for i in range(0, len(self.image), 3):
            self.image[i] = color[0]
            self.image[i + 1] = color[1]
            self.image[i + 2] = color[2]

    def __getitem__(self, key: Tuple[int, int]) -> Color:
        """get the pixel at a specific row and column

        Args:
            key (Tuple[int, int]): (row, col) tuple

        Returns:
            Color: (r, g, b) tuple
        """
        i = 3 * (key[0] * self.width + key[1])
        return (self.image[i], self.image[i + 1], self.image[i + 2])

    def __setitem__(self, key: Tuple[int, int], value: Color) -> None:
        """set the pixel at a specific row and column

        Args:
            key (Tuple[int, int]): (row, col) tuple
            value (Color): (r, g, b) tuple
        """
        i = 3 * (key[0] * self.width + key[1])
        self.image[i] = value[0]
        self.image[i + 1] = value[1]
        self.image[i + 2] = value[2]


if __name__ == "__main__":
    print("Testing PPM")
    image = Image(255, 255)
    image.fill((0, 0, 0))
    for i in range(image.height):
        for j in range(image.width):
            image[i, j] = (i, j, (i + j) // 2)
    image.write_ppm("out.ppm")
