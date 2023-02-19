from camera import Camera
from image import Image
from polygon_ray_trace import polygon_ray_trace
from marching_cube import marching_cube
import data_readers.vol as vol_reader

from utils import calculate_normals, normalize_array
from transformations import *

dataset = vol_reader.read("./data/Skull.vol")

vertices, triangles, vert_count, tri_count = marching_cube(
    data=dataset, isovalue=10, step=10
)

verts = np.array([np.array([v.x, v.y, v.z]) for v in vertices])
faces = np.array([np.array([t.vertex_1, t.vertex_2, t.vertex_3]) for t in triangles])
# generate_ply_file(vertices, triangles, vert_count, tri_count, "skull_test_iso10.ply")

verts = normalize_array(verts)
# mat = translate(-2.0, 1.75, -3.0) @ rot_x(215) @ rot_y(195)
# verts = np.array(list(map(lambda vert: mat @ vert, verts)), dtype=verts.dtype)

norms = calculate_normals(verts, faces)
lights = np.array([[0, 0, 0], [1.0, 1.0, 1.0]])

# image = Image(width=192, height=108)
image = Image(width=200, height=200)
# image = Image(width=500, height=350)
# image = Image(width=1920, height=1080)
camera = Camera(
    fov=np.double(90),
    aspect_ratio=image.aspect_ratio,
    # move the camera around, affects where the rays are generated from
    transformation_matrix=translate(-1, 3.5, -2.0),
)

polygon_ray_trace((verts, faces, norms), lights, camera, image)

image.write_ppm(f"pipeline.ppm")
