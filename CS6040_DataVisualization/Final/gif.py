from tqdm import tqdm
from polygon_ray_trace import polygon_ray_trace
import data_readers.ply as ply_reader
from transformations import *
from utils import calculate_normals, normalize_array
from image import Image
from camera import Camera

verts, faces = ply_reader.read("./data/dodecahedron.ply")

verts_ = normalize_array(verts)

lights = np.array(
    [
        [[0, 0, 0], [1, 1, 1]],
        [[10, 0, 0], [0, 0, 1]],
        # [[-5, 5, -5], [1, 1, 1]],
    ]
)

# image = Image(width=650, height=500)
image = Image(width=500, height=500)

camera = Camera(
    fov=np.double(60),
    aspect_ratio=image.aspect_ratio,
    transformation_matrix=translate(0.0, 0.0, 0.0),
)

start = 0
stop = 360 + 1
step = 10
n = (stop - start) // step
for i, deg in tqdm(enumerate(range(start, stop, step)), total=n):
    # camera = Camera(
    #     fov=np.double(60),
    #     aspect_ratio=image.aspect_ratio,
    #     transformation_matrix=translate(0, -0.5, i * 0.1),
    # )

    mat = translate(-3.0, 5.0, -5.0) @ rot_y(deg)

    verts = np.array(list(map(lambda vert: mat @ vert, verts_)), dtype=verts.dtype)
    # norms = np.array(list(map(lambda norm: mat @ norm, norms_)), dtype=verts.dtype)

    norms = calculate_normals(verts, faces)

    # verts = np.array(list(map(lambda vert: mat_r @ vert, verts_)), dtype=verts.dtype)
    # verts = np.array(list(map(lambda vert: mat_t @ vert, verts)), dtype=verts.dtype)

    polygon_ray_trace((verts, faces, norms), lights, camera, image)

    image.write_ppm(f"_images/out{i}.ppm")
