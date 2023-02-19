from typing import Tuple
from numpy import number
from tqdm import tqdm
from typing import Tuple
import numpy as np
import numpy.typing as npt

from dataset import Dataset
from lookup_tables import edgeTable, triTable


class Vertex:
    def __init__(self, x=None, y=None, z=None, val=None):
        self.x = x
        self.y = y
        self.z = z
        self.v = val

    def __str__(self):
        return "{0} {1} {2}".format(self.x, self.y, self.z)
        # return "{0} {1} {2} {3}".format(self.x, self.y, self.z, self.v)


class Triangle:
    def __init__(self, vertex1=None, vertex2=None, vertex3=None) -> None:
        self.vertex_1 = vertex1
        self.vertex_2 = vertex2
        self.vertex_3 = vertex3

    def __str__(self):
        return "3 {0} {1} {2}".format(int(self.vertex_1), int(self.vertex_2), int(self.vertex_3))


class Cube:
    def __init__(self) -> None:
        self.corners = np.array([Vertex() for _ in range(0, 8)], dtype=Vertex)

    def __str__(self):
        temp = ""
        for corn in self.corners:
            temp += str(corn) + " "
        return temp


def make_grid(data: Dataset, x, y, z, threshold):
    if (
        x + threshold >= data.shape[0]
        or y + threshold >= data.shape[1]
        or z + threshold >= data.shape[2]
    ):
        return None

    # fmt: off
    grid = Cube()
    grid.corners[0] = Vertex(x, y, z, data[x][y][z])
    grid.corners[1] = Vertex(x + threshold, y, z, data[x + threshold][y][z])
    grid.corners[2] = Vertex(x + threshold, y + threshold, z, data[x + threshold][y + threshold][z])
    grid.corners[3] = Vertex(x, y + threshold, z, data[x][y + threshold][z])
    grid.corners[4] = Vertex(x, y, z + threshold, data[x][y][z + threshold])
    grid.corners[5] = Vertex(x + threshold, y, z + threshold, data[x + threshold][y][z + threshold])
    grid.corners[6] = Vertex(x + threshold, y + threshold, z + threshold, data[x + threshold][y + threshold][z + threshold])
    grid.corners[7] = Vertex(x, y + threshold, z + threshold, data[x][y + threshold][z + threshold])
    # fmt: on

    return grid


def generate_ply_file(vertices, triangles, vcount, tcount, filename):
    with open("./data/" + filename, "w+") as f:
        f.write("ply \n")
        f.write("format ascii 1.0 \n")
        f.write("element vertex {0}\n".format(vcount))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face {0}\n".format(tcount))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in vertices:
            f.write(str(i))
            f.write("\n")
        for j in triangles:
            f.write(str(j))
            f.write("\n")
        f.close()


# https://github.com/alvin-yang68/Marching-Cubes/blob/main/Marching_Cubes.ipynb
# https://graphics.stanford.edu/~mdfisher/Code/MarchingCubes/MarchingCubes.cpp
# http://paulbourke.net/geometry/polygonise/


def interpolation(
    vertex_1: Vertex, vertex_2: Vertex, val_1: number, val_2: number, threshold: number
) -> Vertex:
    if abs(threshold - val_1.astype(float)) < 0.00001:
        return vertex_1
    if abs(threshold - val_2.astype(float)) < 0.00001:
        return vertex_2
    if abs(val_1.astype(float) - val_2.astype(float)) < 0.00001:
        return vertex_1
    m = (threshold - val_1.astype(float)) / (val_2.astype(float) - val_1.astype(float))

    temp_point = Vertex()
    temp_point.x = vertex_1.x + m * (vertex_2.x - vertex_1.x)
    temp_point.y = vertex_1.y + m * (vertex_2.y - vertex_1.y)
    temp_point.z = vertex_1.z + m * (vertex_2.z - vertex_1.z)

    return temp_point


def check_vertices(all_vertices, new_vertex):
    for (index, i) in enumerate(all_vertices):
        if str(i) == str(new_vertex):
            return index
    return False


def process_single_cube(grid: Cube, threshold: number, vertex_count: int, all_vertices):
    cube_index = 0
    temp_vertices = np.array([Vertex() for _ in range(0, 12)], dtype=Vertex)
    triangles = []

    # Process the vertices
    # fmt: off
    if (grid.corners[0].v < threshold): cube_index |= 1
    if (grid.corners[1].v < threshold): cube_index |= 2
    if (grid.corners[2].v < threshold): cube_index |= 4
    if (grid.corners[3].v < threshold): cube_index |= 8
    if (grid.corners[4].v < threshold): cube_index |= 16
    if (grid.corners[5].v < threshold): cube_index |= 32
    if (grid.corners[6].v < threshold): cube_index |= 64
    if (grid.corners[7].v < threshold): cube_index |= 128

    if edgeTable[cube_index] == 0: return None, None, None, None
    if edgeTable[cube_index] & 1: 
        temp_vertices[0] = interpolation(grid.corners[0], grid.corners[1], grid.corners[0].v, grid.corners[1].v, threshold)
    if edgeTable[cube_index] & 2:
        temp_vertices[1] = interpolation(grid.corners[1], grid.corners[2], grid.corners[1].v, grid.corners[2].v, threshold)
    if edgeTable[cube_index] & 4:
        temp_vertices[2] = interpolation(grid.corners[2], grid.corners[3], grid.corners[2].v, grid.corners[3].v, threshold)
    if edgeTable[cube_index] & 8:
        temp_vertices[3] = interpolation(grid.corners[3], grid.corners[0], grid.corners[3].v, grid.corners[0].v, threshold)
    if edgeTable[cube_index] & 16:
        temp_vertices[4] = interpolation(grid.corners[4], grid.corners[5], grid.corners[4].v, grid.corners[5].v, threshold)
    if edgeTable[cube_index] & 32:
        temp_vertices[5] = interpolation(grid.corners[5], grid.corners[6], grid.corners[5].v, grid.corners[6].v, threshold)
    if edgeTable[cube_index] & 64:
        temp_vertices[6] = interpolation(grid.corners[6], grid.corners[7], grid.corners[6].v, grid.corners[7].v, threshold)
    if edgeTable[cube_index] & 128:
        temp_vertices[7] = interpolation(grid.corners[7], grid.corners[4], grid.corners[7].v, grid.corners[4].v, threshold)
    if edgeTable[cube_index] & 256:
        temp_vertices[8] = interpolation(grid.corners[0], grid.corners[4], grid.corners[0].v, grid.corners[4].v, threshold)
    if edgeTable[cube_index] & 512:
        temp_vertices[9] = interpolation(grid.corners[1], grid.corners[5], grid.corners[1].v, grid.corners[5].v, threshold)
    if edgeTable[cube_index] & 1024:
        temp_vertices[10] = interpolation(grid.corners[2], grid.corners[6], grid.corners[2].v, grid.corners[6].v, threshold)
    if edgeTable[cube_index] & 2048:
        temp_vertices[11] = interpolation(grid.corners[3], grid.corners[7], grid.corners[3].v, grid.corners[7].v, threshold)
    # fmt: on

    # Vertices handling
    vertices = []
    temp_mapping = np.empty((12), dtype=np.int32)
    temp_mapping.fill(-1)
    for i in range(0, len(triTable[cube_index]), 1):
        if triTable[cube_index][i] != -1:
            if temp_mapping[triTable[cube_index][i]] == -1:
                # Check if vertex exists already
                located = check_vertices(
                    all_vertices, temp_vertices[triTable[cube_index][i]]
                )
                if not located:
                    vertices.append(temp_vertices[triTable[cube_index][i]])
                    temp_mapping[triTable[cube_index][i]] = vertex_count
                    vertex_count += 1
                else:
                    temp_mapping[triTable[cube_index][i]] = located

    # Create triangles
    count = 0
    for i in range(0, len(triTable[cube_index]), 3):
        if triTable[cube_index][i] != -1:
            triangles.append(
                Triangle(
                    temp_mapping[triTable[cube_index][i]],
                    temp_mapping[triTable[cube_index][i + 1]],
                    temp_mapping[triTable[cube_index][i + 2]],
                )
            )
            count += 1
    return vertices, triangles, count, vertex_count


def marching_cube(
    data: Dataset, isovalue: number, step: number
) -> Tuple[npt.NDArray, npt.NDArray, int, int]:
    """Extract an iso surface from a volumetric dataset

    Args:
        data (Dataset): A volumetric dataset
        isovalue (number): isovalue to extract
        step (number): size of cube to march

    Returns:
        Tuple[npt.NDArray, npt.NDArray, int, int]: list of vertices, list of edges
    """

    vertices = np.array([])
    triangles = np.array([])
    tri_count = 0
    vert_count = 0

    for x in tqdm(range(0, data.shape[0], step)):
        for y in range(0, data.shape[1], step):
            for z in range(0, data.shape[2], step):
                cube = make_grid(data, x, y, z, step)
                if cube == None:
                    continue
                (
                    temp_vertices,
                    temp_triangles,
                    temp_tri_count,
                    temp_vert_count,
                ) = process_single_cube(cube, isovalue, vert_count, vertices)
                if temp_vertices == None or temp_triangles == None:
                    continue
                tri_count += temp_tri_count
                vert_count = temp_vert_count
                vertices = np.concatenate((vertices, temp_vertices))
                triangles = np.concatenate((triangles, temp_triangles))
    return vertices, triangles, vert_count, tri_count


if __name__ == "__main__":
    import data_readers.vol as vol_reader

    # dataset = vol_reader.read("./data/C60Small.vol")

    # step = 4
    # vertices, triangles, vert_count, tri_count = marching_cube(dataset, 100, step)
    # generate_ply_file(vertices, triangles, vert_count, tri_count, 'bucky_iso100_step4.ply')

    # f = "C60Small"
    f = "C60Small"
    dataset = vol_reader.read(f"./data/{f}.vol")

    # for i in range(10, 101, 10):
    iso = 75
    # iso = i
    step = 3
    print(iso, step)
    vertices, triangles, vert_count, tri_count = marching_cube(
        data=dataset, isovalue=iso, step=step
    )
    generate_ply_file(
        vertices, triangles, vert_count, tri_count, f"{f}_i{iso}_s{step}.ply"
    )
