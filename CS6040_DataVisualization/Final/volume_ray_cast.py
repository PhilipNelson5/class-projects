from camera import Camera
from tqdm import tqdm
from typing import Callable
import matplotlib as mpl
import numba as nb
import numpy as np
import numpy.typing as npt

from dataset import Dataset, UniformGrid, UniformGridInterpolated
from image import Image
from polygon_ray_trace import make_primary_rays
from transformations import rot_x, translate

"""
Ray casting on volumetric data. 
Starting with a uniform grid, we will directly take the data and create an image. 
We will start with nearest neighbor approach to determine our voxels. 
Then we will create the images using a naive approach and different compositing methods.
"""


@nb.njit(cache=True)
def get_percent(x: float) -> float:
    a = 0.1
    b = 2.0
    return 1.0 / (x + b) ** 2 - a


def make_color_transfer_function(
    samples: npt.NDArray, cmap: mpl.cm.ScalarMappable
) -> Callable:
    """construct a color transfer function for a set of samples

    Args:
        samples (npt.NDArray): samples of a volume
        cmap (mpl.cm.ScalarMappable): a matplotlib colormap

    Returns:
        Callable: color transfer function for the set
    """
    _min = np.min(samples)
    _max = np.max(samples)
    norm = mpl.colors.Normalize(_min, _max)
    return lambda x: np.array(cmap(norm(x)))[:3]


def get_color_proximity(samples: npt.NDArray, ctf: Callable) -> npt.NDArray:
    """assign more color from points closer to the camera and less to samples further away

    Args:
        samples (npt.NDArray): samples of a volume
        ctf (Callable): color transfer function

    Returns:
        npt.NDArray: rgb color
    """
    color = [0.0, 0.0, 0.0]
    n = len(samples)
    for i, sample in enumerate(samples):
        p = get_percent(i / n)
        color = color + p * ctf(sample)
    color = np.clip(color, 0, 1)
    rgb = (255 * color).astype(np.uint8)
    return rgb


def get_color_avg(samples: npt.NDArray, ctf: Callable) -> npt.NDArray:
    """use the average of the sample values

    Args:
        samples (npt.NDArray): samples of a volume
        ctf (Callable): color transfer function

    Returns:
        npt.NDArray: rgb color
    """
    if len(samples) == 0:
        return [0, 0, 0]

    avg_value = sum(samples) / len(samples)
    color = ctf(avg_value)
    color = np.clip(color, 0, 1)
    rgb = (255 * color).astype(np.uint8)
    return rgb


def get_color_max(samples: npt.NDArray, ctf: Callable) -> npt.NDArray:
    """use the maximum sample value

    Args:
        samples (npt.NDArray): samples of a volume
        ctf (Callable): color transfer function

    Returns:
        npt.NDArray: rgb color
    """
    if len(samples) == 0:
        return [0, 0, 0]
    max_value = np.max(samples)
    color = ctf(max_value)
    color = np.clip(color, 0, 1)
    rgb = (255 * color).astype(np.uint8)
    return rgb


def get_color_min(samples: npt.NDArray, ctf: Callable) -> npt.NDArray:
    """use the minimum sample value

    Args:
        samples (npt.NDArray): samples of a volume
        ctf (Callable): color transfer function

    Returns:
        npt.NDArray: rgb color
    """
    if len(samples) == 0:
        return [0, 0, 0]
    min_value = np.min(samples)
    color = ctf(min_value)
    color = np.clip(color, 0, 1)
    rgb = (255 * color).astype(np.uint8)
    return rgb


def volume_ray_cast(
    dataset: Dataset,
    camera: Camera,
    image: Image,
    composition_fn: Callable[[npt.NDArray, Callable], npt.NDArray],
    cmap: mpl.cm.ScalarMappable,
) -> Image:
    """directly visualize a volumetric dataset

    Args:
        dataset (Dataset): volumetric dataset
        camera (Camera): camera model
        image (Image): image to render on to
        composition_fn (Callable[[npt.NDArray, Callable], npt.NDArray]): compositing scheme
        cmap (mpl.cm.ScalarMappable): color map

    Returns:
        Image: rendered image
    """
    step_size = 1.0
    max_steps = dataset.shape[0] * 2

    primary_rays = make_primary_rays(
        camera, image.width, image.height, image.aspect_ratio
    )

    # for each ray, take steps along the ray
    samples = np.empty(shape=(image.height, image.width, max_steps))
    for (row, col, ray) in tqdm(primary_rays, total=image.height * image.width):
        location = ray.origin

        for step in range(0, max_steps):
            location = location + ray.direction * step_size
            try:
                sample = dataset[location]
            except:
                sample = np.nan
            samples[row, col, step] = sample

    ctf = make_color_transfer_function(samples[~np.isnan(samples)], cmap)

    for row in tqdm(range(image.height)):
        for col in range(image.width):
            image[row, col] = composition_fn(
                list(
                    filter(
                        lambda x: not np.isnan(x) and (x < -1 or x > 1),
                        samples[row, col],
                    )
                ),
                ctf,
            )
    return image


if __name__ == "__main__":
    import data_readers.vol as vol_reader

    dataset = UniformGrid.load("data/C60Small.vol", vol_reader.read)
    image = Image(width=500, height=500)
    camera = Camera(
        fov=np.double(120),
        aspect_ratio=image.aspect_ratio,
        transformation_matrix=translate(32.0, 32.0, -8) @ rot_x(180),
    )
    cmap = mpl.cm.viridis
    volume_ray_cast(
        dataset=dataset,
        camera=camera,
        image=image,
        composition_fn=get_color_avg,
        cmap=cmap,
    )
    image.write_ppm("volume.ppm")
